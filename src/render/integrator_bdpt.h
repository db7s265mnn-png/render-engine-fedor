// Bidirectional path tracing (Veach 1997) with the full multiple-importance
// weighting over all (s,t) strategies (PBRT-v3 style pdfFwd/pdfRev remap).
// CPU / Embree only — included from embree_device.cpp.
//
// Strategies: t = 1 light-tracing splats onto the camera (the workhorse for
// caustics from small lights: light → delta glass chain → diffuse → camera),
// s = 0 (eye path hits an emitter), s = 1 (light resampled toward the eye
// vertex, NEE-like), s >= 2 (surface↔surface connections), all t >= 2.
// Dome / distant lights contribute via s ∈ {0,1} with the power heuristic
// (no light subpath from them). Optional OpenPGL guiding mixes into eye-path
// BSDF sampling; the mixture pdf is the true forward pdf so MIS stays valid.
#pragma once

#include "core/rng.h"
#include "render/framebuffer.h"
#include "render/integrator.h"
#include "render/lights.h"
#include "render/shading.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPENPGL
#include "render/cpu/path_guiding.h"
#endif

namespace sol {
namespace bdpt {

constexpr int kMaxVerts = 16;

enum class VType : uint8_t { Camera, Light, Surface };

struct Vert {
    Vec3 p{0.0f};
    Vec3 ng{0.0f, 1.0f, 0.0f};
    Vec3 ns{0.0f, 1.0f, 0.0f};
    Vec3 beta{1.0f};      // throughput up to (and including arrival at) this vertex
    Vec3 wo{0.0f};        // unit direction toward the previous vertex
    Material mat{};
    float pdfFwd = 0.0f;  // area pdf of generating this vertex from the previous one
    float pdfRev = 0.0f;  // area pdf of generating it from the next one (filled for MIS)
    int lightIndex = -1;
    VType type = VType::Surface;
    bool delta = false;       // delta BSDF vertex (or delta light origin)
    bool connectable = true;  // has a non-delta lobe to connect through
};

SR_INL float remap0(float f) { return f > 0.0f ? f : 1.0f; }

SR_INL float toAreaPdf(float pdfSa, Vec3 from, Vec3 to, Vec3 nTo) {
    Vec3 d = to - from;
    const float dist2 = lengthSquared(d);
    if (dist2 < 1e-12f || pdfSa <= 0.0f) return 0.0f;
    d = d * (1.0f / sqrtf(dist2));
    return pdfSa * fabsf(dot(nTo, d)) / dist2;
}

SR_INL float geometryTerm(Vec3 a, Vec3 na, Vec3 b, Vec3 nb) {
    Vec3 d = b - a;
    const float dist2 = lengthSquared(d);
    if (dist2 < 1e-12f) return 0.0f;
    d = d * (1.0f / sqrtf(dist2));
    const float cosA = fabsf(dot(na, d));
    const float cosB = fabsf(dot(nb, -d));
    if (cosA < 1e-7f || cosB < 1e-7f) return 0.0f;
    return (cosA * cosB) / dist2;
}

// Solid-angle BSDF pdf for wo→wi at a surface vertex (both world-space).
SR_INL float bsdfPdfSa(const Material& mat, Vec3 ns, Vec3 woW, Vec3 wiW) {
    const Frame frame(ns);
    const BsdfEval e = bsdfEvalLocal(mat, frame.toLocal(woW), frame.toLocal(wiW));
    return srIsFinite(e.pdf) ? e.pdf : 0.0f;
}

SR_INL Vec3 bsdfF(const Material& mat, Vec3 ns, Vec3 woW, Vec3 wiW) {
    const Frame frame(ns);
    const BsdfEval e = bsdfEvalLocal(mat, frame.toLocal(woW), frame.toLocal(wiW));
    return isFinite(e.f) ? e.f : Vec3(0.0f);
}

SR_INL bool lightIsFinite(const LightData& l) {
    return l.type == kLightRect || l.type == kLightDisk || l.type == kLightSphere ||
           l.type == kLightPoint;
}

SR_INL float lightArea(const LightData& l) {
    switch (l.type) {
        case kLightRect: return rectLightArea(l);
        case kLightDisk: return diskLightArea(l);
        case kLightSphere: {
            const float r = sphereLightRadius(l);
            return 4.0f * kPi * r * r;
        }
        default: return 0.0f;
    }
}

// Area pdf of sampling the emission origin on this light (light selection included).
SR_INL float pdfLightOrigin(const SceneView& scene, const LightData& l, int lightIndex) {
    const float select = lightSelectionPdfIndex(scene, lightIndex);
    if (l.type == kLightPoint) return select;  // delta position — only used in ratios
    const float area = lightArea(l);
    return area > 1e-12f ? select / area : 0.0f;
}

// Solid-angle pdf of the light emitting from `onLight` toward `dir`.
SR_INL float pdfLightDirSa(const LightData& l, Vec3 lightNormal, Vec3 dir) {
    switch (l.type) {
        case kLightRect:
        case kLightDisk:
        case kLightSphere: {
            const float c = dot(lightNormal, dir);
            const float cosT = l.twoSided ? fabsf(c) : srMax(0.0f, c);
            return cosT * kInvPi;  // cosine-hemisphere emission
        }
        case kLightPoint: return kInv4Pi;
        default: return 0.0f;
    }
}

// --------------------------------------------------------------------------
// Camera importance (pinhole / thin-lens center) for light-tracing splats.
// --------------------------------------------------------------------------
struct CameraProj {
    Mat4 worldToCam;
    Vec3 camPos{0.0f};
    float focal = 50.0f;      // mm
    float sensorW = 36.0f;    // mm
    float sensorH = 24.0f;
    float resX = 1.0f;
    float resY = 1.0f;
    float pixelArea = 1.0f;   // sensor mm² per pixel
    bool valid = false;
};

SR_INL CameraProj buildCameraProj(const SceneView& scene) {
    CameraProj c;
    const CameraData& cam = scene.camera;
    c.worldToCam = inverse(cam.cameraToWorld);
    c.camPos = transformPoint(cam.cameraToWorld, Vec3(0.0f, 0.0f, 0.0f));
    c.focal = srMax(1e-3f, cam.focalLength);
    c.resX = float(srMax(1, scene.settings.resolutionX));
    c.resY = float(srMax(1, scene.settings.resolutionY));
    c.sensorW = cam.sensorWidth;
    c.sensorH = cam.sensorWidth * (c.resY / c.resX);
    c.pixelArea = (c.sensorW / c.resX) * (c.sensorH / c.resY);
    c.valid = c.pixelArea > 1e-12f;
    return c;
}

// Projects a world point onto the raster. Returns false when outside the
// frustum. cosTheta is measured against the optical axis.
SR_INL bool projectToPixel(const CameraProj& proj, Vec3 pWorld, float& px, float& py, float& cosTheta,
                           float& dist2) {
    const Vec3 pc = transformPoint(proj.worldToCam, pWorld);
    if (pc.z >= -1e-5f) return false;  // behind the pinhole
    dist2 = lengthSquared(pc);
    cosTheta = -pc.z / sqrtf(srMax(1e-12f, dist2));
    const float sx = pc.x * (proj.focal / -pc.z);
    const float sy = pc.y * (proj.focal / -pc.z);
    px = (sx / proj.sensorW + 0.5f) * proj.resX;
    py = (0.5f - sy / proj.sensorH) * proj.resY;
    return px >= 0.0f && px < proj.resX && py >= 0.0f && py < proj.resY;
}

// Solid-angle pdf of the camera generating a ray toward direction with cosTheta
// (uniform sampling over the pixel; matches generateCameraRay's density).
SR_INL float cameraPdfOmega(const CameraProj& proj, float cosTheta) {
    const float c = srMax(1e-4f, cosTheta);
    return (proj.focal * proj.focal) / (proj.pixelArea * c * c * c);
}

// --------------------------------------------------------------------------
// Subpath generation
// --------------------------------------------------------------------------

// Start a light subpath: position + emission direction on a finite light.
SR_INL bool startLightPath(const SceneView& scene, Rng& rng, Vert& v0, Vec3& emitDir, float& pdfDirSa) {
    float selectPdf = 0.0f;
    const int li = sampleLightIndex(scene, rng.nextFloat(), selectPdf);
    if (li < 0 || selectPdf <= 0.0f) return false;
    const LightData& l = scene.lights[li];
    if (!lightIsFinite(l)) return false;  // env / distant handled by s∈{0,1}

    v0 = Vert{};
    v0.type = VType::Light;
    v0.lightIndex = li;

    if (l.type == kLightPoint) {
        v0.p = lightOrigin(l);
        v0.ng = v0.ns = Vec3(0.0f, 1.0f, 0.0f);
        // Origin delta-ness is passed separately into MIS (lightOriginDelta) so the
        // s'=1 strategy still counts; vertex delta stays false like PBRT light verts.
        v0.delta = false;
        v0.connectable = true;
        v0.pdfFwd = selectPdf;
        emitDir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
        pdfDirSa = kInv4Pi;
        // Throughput folds BOTH the position and the direction pdf (I / (p_A·p_ω));
        // there is no cosine at a point emitter.
        v0.beta = l.emittedRadiance() / srMax(1e-12f, v0.pdfFwd * pdfDirSa);
        return true;
    }

    float area = 0.0f;
    if (l.type == kLightRect) {
        const Vec3 pLocal((rng.nextFloat() - 0.5f) * l.width, (rng.nextFloat() - 0.5f) * l.height, 0.0f);
        v0.p = transformPoint(l.xform, pLocal);
        v0.ng = v0.ns = areaLightNormal(l);
        area = rectLightArea(l);
    } else if (l.type == kLightDisk) {
        const Vec2 d = sampleConcentricDisk(rng.nextFloat(), rng.nextFloat());
        v0.p = transformPoint(l.xform, Vec3(d.x * l.radius, d.y * l.radius, 0.0f));
        v0.ng = v0.ns = areaLightNormal(l);
        area = diskLightArea(l);
    } else {  // sphere
        const Vec3 center = lightOrigin(l);
        const float radius = srMax(1e-5f, sphereLightRadius(l));
        const Vec3 dir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
        v0.p = center + dir * radius;
        v0.ng = v0.ns = dir;
        area = 4.0f * kPi * radius * radius;
    }
    if (area <= 1e-12f) return false;
    v0.pdfFwd = selectPdf / area;

    // Cosine-hemisphere emission around the light normal.
    Vec3 nEmit = v0.ns;
    if (l.twoSided && rng.nextFloat() < 0.5f) nEmit = -nEmit;
    const Frame frame(nEmit);
    const Vec3 local = sampleCosineHemisphere(rng.nextFloat(), rng.nextFloat());
    emitDir = normalize(frame.toWorld(local));
    pdfDirSa = fabsf(dot(nEmit, emitDir)) * kInvPi * (l.twoSided ? 0.5f : 1.0f);
    if (pdfDirSa <= 0.0f) return false;
    v0.beta = lightRadiance(l) * fabsf(dot(v0.ns, emitDir)) / srMax(1e-12f, v0.pdfFwd * pdfDirSa);
    return true;
}

struct WalkConfig {
    bool eyePath = false;
    int heroChannel = -1;  // chromatic dispersion hero channel (-1 = off)
#if SOLSTICE_HAVE_OPENPGL
    PathGuiding::ThreadState* guiding = nullptr;
#endif
};

// Extend a subpath by BSDF sampling.
template <typename Tracer>
SR_INL int randomWalk(const SceneView& scene, const Tracer& tracer, Rng& rng, Vert* path, int count,
                      Vec3 origin, Vec3 dir, float pdfDirSa, int maxVerts, const WalkConfig& cfg) {
    Vec3 beta = path[count - 1].beta;
    float pdfSaFwd = pdfDirSa;
    int passThrough = 0;

    while (count < maxVerts) {
        Vert& prev = path[count - 1];
        RayHit hit;
        if (!tracer.intersect(origin, dir, kFloatMax, hit)) {
            // Escaped. For eye paths record an environment pseudo-vertex.
            if (cfg.eyePath && scene.domeLightIndex >= 0) {
                Vert v{};
                v.type = VType::Light;
                v.lightIndex = scene.domeLightIndex;
                v.p = origin + dir * 1.0e6f;
                v.ng = v.ns = -dir;
                v.wo = -dir;
                v.beta = beta;
                v.pdfFwd = pdfSaFwd;  // solid-angle pdf (env special case)
                v.connectable = false;
                path[count++] = v;
            }
            break;
        }
        SurfaceInteraction si;
        if (!buildSurfaceInteraction(scene, hit, origin, dir, si)) break;

        if (si.lightIndex >= 0) {
            if (cfg.eyePath) {
                Vert v{};
                v.type = VType::Light;
                v.lightIndex = si.lightIndex;
                v.p = si.p;
                v.ng = v.ns = si.ng;
                v.wo = -dir;
                v.beta = beta;
                v.pdfFwd = toAreaPdf(pdfSaFwd, prev.p, si.p, si.ng);
                v.connectable = false;
                path[count++] = v;
            }
            break;  // light geometry terminates both subpaths
        }

        Material mat = si.materialIndex >= 0 && si.materialIndex < scene.materialCount
                           ? scene.materials[si.materialIndex]
                           : defaultMaterial();
        mat = evaluateTexturedMaterial(scene, mat, si.uv, si.ns, si.pObject, si.nObject, si.uvFilterWidth);
        applyDispersion(mat, cfg.heroChannel);

        // Stochastic cutout — pass through without creating a vertex.
        if (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity) {
            origin = offsetRayOrigin(si.p, si.ng, dir);
            if (++passThrough > 16) break;
            continue;
        }

        Vert v{};
        v.type = VType::Surface;
        v.p = si.p;
        v.ng = si.ng;
        v.ns = si.ns;
        v.mat = mat;
        v.wo = -dir;
        v.beta = beta;
        v.pdfFwd = toAreaPdf(pdfSaFwd, prev.p, si.p, si.ns);
        {
            const LobeWeights lw = computeLobes(mat);
            v.delta = lw.delta && lw.diffuse < 1e-4f;
            v.connectable = !v.delta;
        }
        path[count++] = v;
        if (count >= maxVerts) break;
        Vert& cur = path[count - 1];

        // Sample the next direction (guided mixture on eye paths).
        const Frame frame(cur.ns);
        const Vec3 woLocal = frame.toLocal(cur.wo);
        BsdfSample bs{};
        bool haveSample = false;
        Vec3 wiWorld;
#if SOLSTICE_HAVE_OPENPGL
        bool guideReady = false;
        if (cfg.eyePath && cfg.guiding && cfg.guiding->active() && !cur.delta) {
            guideReady = cfg.guiding->prepare(cur.p, cur.ns, rng);
            if (guideReady && rng.nextFloat() < cfg.guiding->guideProbability()) {
                float gPdf = 0.0f;
                if (cfg.guiding->sample(rng.nextFloat(), rng.nextFloat(), wiWorld, gPdf) && gPdf > 0.0f) {
                    const Vec3 wiLocal = frame.toLocal(wiWorld);
                    const BsdfEval ev = bsdfEvalLocal(cur.mat, woLocal, wiLocal);
                    if (ev.pdf > 0.0f && !isBlack(ev.f)) {
                        const float pg = cfg.guiding->guideProbability();
                        const float mixPdf = pg * gPdf + (1.0f - pg) * ev.pdf;
                        bs.wi = wiLocal;
                        bs.pdf = mixPdf;
                        bs.weight = ev.f * (fabsf(wiLocal.z) / mixPdf);
                        bs.specular = false;
                        bs.transmitted = wiLocal.z < 0.0f;
                        haveSample = true;
                    }
                }
            }
        }
#endif
        if (!haveSample) {
            bs = bsdfSampleLocal(cur.mat, woLocal, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                 rng.nextFloat());
            if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;
            wiWorld = normalize(frame.toWorld(bs.wi));
#if SOLSTICE_HAVE_OPENPGL
            if (cfg.eyePath && cfg.guiding && guideReady && !bs.specular) {
                const float pg = cfg.guiding->guideProbability();
                const float gPdf = cfg.guiding->pdf(wiWorld);
                const float mixPdf = pg * gPdf + (1.0f - pg) * bs.pdf;
                if (mixPdf > 0.0f) {
                    bs.weight = bs.weight * (bs.pdf / mixPdf);
                    bs.pdf = mixPdf;
                }
            }
#endif
        }

        // Reverse pdf of the segment we just travelled (for MIS).
        {
            const float revSa = bs.specular ? 0.0f : bsdfPdfSa(cur.mat, cur.ns, wiWorld, cur.wo);
            prev.pdfRev = toAreaPdf(revSa, cur.p, prev.p, prev.type == VType::Surface ? prev.ns : prev.ng);
        }

#if SOLSTICE_HAVE_OPENPGL
        if (cfg.eyePath && cfg.guiding && cfg.guiding->active()) {
            cfg.guiding->recordBounce(cur.ns, wiWorld, bs.pdf, bs.weight, bs.specular, cur.mat.roughness,
                                      computeLobes(cur.mat).eta, 1.0f);
        }
#endif

        beta = beta * bs.weight;
        if (!isFinite(beta) || isBlack(beta)) break;
        // Delta segments carry pdf 0 → remap0() treats them as unit ratios in MIS
        // and the delta flags keep those strategies out of the sums (PBRT convention).
        pdfSaFwd = bs.specular ? 0.0f : bs.pdf;
        origin = offsetRayOrigin(cur.p, cur.ng, wiWorld);
        dir = wiWorld;
        passThrough = 0;
    }
    return count;
}

// --------------------------------------------------------------------------
// MIS weight (PBRT MISWeight with pdfRev overrides for the connection ends)
// --------------------------------------------------------------------------
struct MisOverride {
    float eyeLastRev = -1.0f;   // pdfRev of eye[t-1]
    float eyePrevRev = -1.0f;   // pdfRev of eye[t-2]
    float lightLastRev = -1.0f; // pdfRev of light[s-1]
    float lightPrevRev = -1.0f; // pdfRev of light[s-2]
    bool lightOriginDelta = false;
    bool splatStrategy = false;  // t=1 light tracing is being sampled this pass
    // False when the s'=0 strategy (eye path hits the light) has been handed to
    // light tracing for this path family (delta chain adjacent to an area light):
    // formal Veach weights cannot express the lens Jacobian of delta chains, so
    // the family is partitioned deterministically instead of MIS-blended.
    bool s0Sampled = true;
};

SR_INL float misWeight(const Vert* eye, int t, const Vert* light, int s, const MisOverride& ov) {
    if (s + t == 2) return 1.0f;  // only one strategy for a length-2 path

    float sumRi = 0.0f;

    // Eye side: hypothetical strategies where eye[i..t-1] came from the light side
    // (i == 1 corresponds to the light-tracing splat strategy t' = 1 — counted
    // only when splats are actually being rendered).
    {
        float ri = 1.0f;
        for (int i = t - 1; i >= 1; --i) {
            float rev = eye[i].pdfRev;
            if (i == t - 1 && ov.eyeLastRev >= 0.0f) rev = ov.eyeLastRev;
            if (i == t - 2 && ov.eyePrevRev >= 0.0f) rev = ov.eyePrevRev;
            ri *= remap0(rev) / remap0(eye[i].pdfFwd);
            if (i == 1 && !ov.splatStrategy) break;  // t'=1 not sampled — skip its term
            const bool curDelta = eye[i].delta;
            const bool prevDelta = i >= 1 && eye[i - 1].delta;
            if (!curDelta && !prevDelta) sumRi += ri;
        }
    }

    // Light side: strategies with shorter light subpaths (down to s'=0).
    {
        float ri = 1.0f;
        for (int i = s - 1; i >= 0; --i) {
            float rev = light[i].pdfRev;
            if (i == s - 1 && ov.lightLastRev >= 0.0f) rev = ov.lightLastRev;
            if (i == s - 2 && ov.lightPrevRev >= 0.0f) rev = ov.lightPrevRev;
            ri *= remap0(rev) / remap0(light[i].pdfFwd);
            if (i == 0 && !ov.s0Sampled) break;  // s'=0 handed to light tracing
            const bool curDelta = light[i].delta && i > 0;  // origin delta handled below
            const bool prevDelta = i > 0 ? light[i - 1].delta : ov.lightOriginDelta;
            if (!curDelta && !prevDelta && !(i == 0 && ov.lightOriginDelta)) sumRi += ri;
        }
    }

    const float w = 1.0f / (1.0f + sumRi);
    return srIsFinite(w) ? w : 0.0f;
}

// --------------------------------------------------------------------------
// Full estimator
// --------------------------------------------------------------------------
template <typename Tracer>
SR_INL bool connectionVisible(const SceneView& scene, const Tracer& tracer, Vec3 a, Vec3 na, Vec3 b,
                              int targetLight) {
    Vec3 d = b - a;
    const float dist = length(d);
    if (dist < 1e-5f) return false;
    d = d / dist;
    const Vec3 o = offsetRayOrigin(a, na, d);
    RayHit hit;
    if (!tracer.intersect(o, d, dist * (1.0f - 1e-3f), hit)) return true;
    if (targetLight >= 0) {
        SurfaceInteraction si;
        if (buildSurfaceInteraction(scene, hit, o, d, si) && si.lightIndex == targetLight) return true;
    }
    return false;
}

}  // namespace bdpt

// `splatFb` enables the t=1 light-tracing strategy (thread-safe splats; pass the
// active framebuffer). Splats assume a pinhole / thin-lens-center camera and are
// skipped when the projection is unavailable.
template <typename Tracer>
inline Vec3 traceRadianceBdpt(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                              Rng& rng
#if SOLSTICE_HAVE_OPENPGL
                              ,
                              PathGuiding::ThreadState* guiding = nullptr
#endif
                              ,
                              Framebuffer* splatFb = nullptr, int heroChannel = -1) {
    using namespace bdpt;
    const RenderSettingsData& settings = scene.settings;
    int maxVerts = settings.maxDepth + 1;
    if (maxVerts > kMaxVerts) maxVerts = kMaxVerts;
    if (maxVerts < 2) maxVerts = 2;
    const bool causticsOn = settings.caustics != 0;

    const CameraProj camProj = buildCameraProj(scene);
    const bool doSplats = splatFb != nullptr && camProj.valid;
    if (doSplats) splatFb->addSplatPath();

    Vert eye[kMaxVerts];
    Vert light[kMaxVerts];

    // ---- Light subpath (finite lights only) ----
    int nLight = 0;
    bool lightOriginDelta = false;
    if (scene.lightCount > 0) {
        Vec3 emitDir;
        float pdfDirSa = 0.0f;
        if (startLightPath(scene, rng, light[0], emitDir, pdfDirSa)) {
            nLight = 1;
            lightOriginDelta =
                light[0].lightIndex >= 0 && scene.lights[light[0].lightIndex].type == kLightPoint;
            WalkConfig cfg;
            cfg.eyePath = false;
            cfg.heroChannel = heroChannel;
            const Vec3 o = offsetRayOrigin(light[0].p, light[0].ng, emitDir);
            nLight = randomWalk(scene, tracer, rng, light, nLight, o, emitDir, pdfDirSa, maxVerts, cfg);
        }
    }

    // ---- Eye subpath ----
    eye[0] = Vert{};
    eye[0].type = VType::Camera;
    eye[0].p = origin;
    eye[0].ng = eye[0].ns = direction;
    eye[0].wo = -direction;
    eye[0].beta = Vec3(1.0f);
    eye[0].pdfFwd = 1.0f;
    eye[0].connectable = false;
    WalkConfig eyeCfg;
    eyeCfg.eyePath = true;
    eyeCfg.heroChannel = heroChannel;
#if SOLSTICE_HAVE_OPENPGL
    eyeCfg.guiding = guiding;
#endif
    // The camera ray density matters now that t=1 is a real strategy.
    float camPdfSa = 1.0f;
    if (camProj.valid) {
        const Vec3 dc = transformVector(camProj.worldToCam, direction);
        camPdfSa = cameraPdfOmega(camProj, srMax(1e-4f, -dc.z));
    }
    const int nEye = randomWalk(scene, tracer, rng, eye, 1, origin, direction, camPdfSa, maxVerts, eyeCfg);

    Vec3 L(0.0f);

    // ---- t = 1: splat light-subpath vertices onto the camera (light tracing).
    // This is what carries caustics from small lights: the specular chain is
    // sampled from the light with delta refractions, and the final connection
    // (diffuse vertex → camera) is benign.
    if (doSplats && nLight >= 2) {
        for (int s = 2; s <= nLight; ++s) {
            const Vert& v = light[s - 1];
            if (v.type != VType::Surface || !v.connectable) continue;
            // Caustics off: light→specular-chain→diffuse splats are the caustic
            // family; drop them to match the dark-shadow look.
            bool lightPrefixDelta = false;
            for (int i = 1; i < s - 1; ++i)
                if (light[i].delta) lightPrefixDelta = true;
            if (!causticsOn) {
                if (lightPrefixDelta) continue;
            } else if (lightPrefixDelta && light[0].lightIndex >= 0 &&
                       !lightContributesCaustics(scene.lights[light[0].lightIndex])) {
                // Per-light Contribute to Caustics off.
                continue;
            }
            float px = 0.0f, py = 0.0f, cosTheta = 0.0f, dist2 = 0.0f;
            if (!projectToPixel(camProj, v.p, px, py, cosTheta, dist2)) continue;
            if (dist2 < 1e-8f) continue;
            const Vec3 toCam = normalize(camProj.camPos - v.p);
            const Vec3 f = bsdfF(v.mat, v.ns, v.wo, toCam);
            if (isBlack(f)) continue;
            if (!connectionVisible(scene, tracer, v.p, v.ng, camProj.camPos, -1)) continue;

            const float pdfOmega = cameraPdfOmega(camProj, cosTheta);
            const float cosV = fabsf(dot(v.ns, toCam));
            Vec3 c = v.beta * f * (cosV * pdfOmega / dist2);

            // MIS against the t >= 2 strategies for the same path. When the prefix
            // from the light runs through a delta chain, the s'=0 twin has been
            // handed to this strategy (family partition) — weight it fully.
            MisOverride ov;
            ov.splatStrategy = true;
            ov.s0Sampled = !(causticsOn && lightPrefixDelta);
            ov.lightOriginDelta = lightOriginDelta;
            ov.lightLastRev = toAreaPdf(pdfOmega, camProj.camPos, v.p, v.ns);
            if (s >= 2)
                ov.lightPrevRev = toAreaPdf(bsdfPdfSa(v.mat, v.ns, toCam, normalize(light[s - 2].p - v.p)),
                                            v.p, light[s - 2].p,
                                            light[s - 2].type == VType::Surface ? light[s - 2].ns
                                                                                : light[s - 2].ng);
            Vert camVert = eye[0];
            camVert.p = camProj.camPos;
            const float w = misWeight(&camVert, 1, light, s, ov);
            c = c * w;
            if (s > 2) c = clampContribution(c, settings.clampIndirect);
            if (!isFinite(c)) continue;
            if (heroChannel >= 0) {
                // Hero-channel discipline: deposit only the sampled channel, ×3.
                const float hero =
                    (heroChannel == 0 ? c.x : (heroChannel == 1 ? c.y : c.z)) * 3.0f;
                c = Vec3(0.0f);
                if (heroChannel == 0) c.x = hero;
                else if (heroChannel == 1) c.y = hero;
                else c.z = hero;
            }
            splatFb->addSplat(int(px), int(py), c);
        }
    }

    // ---- s = 0: eye path hit an emitter / the environment ----
    for (int t = 2; t <= nEye; ++t) {
        const Vert& v = eye[t - 1];
        if (v.type == VType::Surface) {
            // Emissive mesh material — not part of the light list, weight 1.
            if (v.mat.emissionStrength > 0.0f && !isBlack(v.mat.emissionColor)) {
                const bool front = dot(v.ns, v.wo) > 0.0f;
                if (front || v.mat.doubleSided) {
                    Vec3 c = v.beta * v.mat.emissionColor * v.mat.emissionStrength;
                    if (t > 2) c = clampContribution(c, settings.clampIndirect);
                    L += c;
                }
            }
            continue;
        }
        if (v.type != VType::Light || v.lightIndex < 0) continue;
        const LightData& l = scene.lights[v.lightIndex];

        // Per-light / global: drop diffuse→specular→light (caustic) hits.
        if (t >= 4) {
            bool sawDiffuseThenSpec = false;
            bool diffuseSeen = false;
            for (int i = 1; i < t - 1; ++i) {
                if (!eye[i].delta) diffuseSeen = true;
                else if (diffuseSeen) sawDiffuseThenSpec = true;
            }
            if (sawDiffuseThenSpec && (!causticsOn || !lightContributesCaustics(l))) break;
        }

        if (l.type == kLightDome) {
            // Environment: MIS against s=1 env NEE (PT-style power heuristic).
            const bool primary = t == 2;
            if (primary && (!settings.envVisibleCamera || !l.visibleCamera)) break;
            const Vec3 dirW = -v.wo;
            Vec3 Le = domeRadiance(scene, l, dirW);
            if (!isBlack(Le)) {
                float w = 1.0f;
                const Vert& prev = eye[t - 2];
                if (t > 2 && !prev.delta && prev.type == VType::Surface) {
                    const float lp = lightPdfDirection(scene, v.lightIndex, prev.p, dirW, prev.p, dirW) *
                                     lightSelectionPdfIndex(scene, v.lightIndex);
                    w = powerHeuristic(1.0f, v.pdfFwd, 1.0f, lp);  // pdfFwd = solid-angle here
                }
                Vec3 c = v.beta * Le * w;
                if (t > 2) c = clampContribution(c, settings.clampIndirect);
                L += c;
#if SOLSTICE_HAVE_OPENPGL
                if (guiding && guiding->active()) guiding->recordBackground(eye[t - 2].p, dirW, Le, w);
#endif
            }
            break;
        }

        // Finite light hit: full BDPT MIS (the t'=1 splat strategy now appears in
        // the eye-side sum, which is what keeps small-light caustic fireflies down).
        const Vec3 lightN = l.type == kLightSphere ? v.ng : areaLightNormal(l);
        const Vec3 wi = -v.wo;  // direction of travel into the light
        Vec3 Le = areaLightEmission(scene, l, wi, lightN);
        if (isBlack(Le)) break;
        // Caustic family partition: diffuse → delta chain → area light cannot be
        // weighted by formal Veach MIS (the delta-chain lens Jacobian is not in
        // the pdf bookkeeping), so hitting a tiny light through glass stays a
        // firefly no matter the sample count. When light tracing is active, that
        // family is delivered exclusively by t=1 splats — drop the noisy s=0 copy.
        // Kept when the camera-adjacent vertex is delta (SDS: splats cannot see it).
        if (causticsOn && doSplats && t >= 4) {
            int j = t - 2;
            int chainLen = 0;
            while (j >= 1 && eye[j].type == VType::Surface && eye[j].delta) {
                ++chainLen;
                --j;
            }
            if (chainLen >= 1 && j >= 1 && !eye[j].delta && !eye[1].delta) break;
        }

        MisOverride ov;
        ov.splatStrategy = doSplats;
        ov.lightOriginDelta = false;
        ov.eyeLastRev = pdfLightOrigin(scene, l, v.lightIndex);
        const Vec3 emitToPrev = normalize(eye[t - 2].p - v.p);
        ov.eyePrevRev = toAreaPdf(pdfLightDirSa(l, lightN, emitToPrev), v.p, eye[t - 2].p,
                                  eye[t - 2].type == VType::Surface ? eye[t - 2].ns : eye[t - 2].ng);
        const float w = misWeight(eye, t, light, 0, ov);
        Vec3 c = v.beta * Le * w;
        if (t > 2) c = clampContribution(c, settings.clampIndirect);
        L += c;
#if SOLSTICE_HAVE_OPENPGL
        if (guiding && guiding->active()) guiding->recordLightHit(v.p, v.wo, Le, w);
#endif
        break;
    }

    // ---- s = 1: resample the light toward each eye vertex (NEE-like) ----
    for (int t = 2; t <= nEye; ++t) {
        const Vert& E = eye[t - 1];
        if (E.type != VType::Surface || !E.connectable) continue;

        float selectPdf = 0.0f;
        const int li = sampleLightIndex(scene, rng.nextFloat(), selectPdf);
        if (li < 0 || selectPdf <= 0.0f) continue;
        const LightData& l = scene.lights[li];

        if (l.type == kLightDome || l.type == kLightDistant) {
            // PT-style env NEE with power heuristic (no deeper strategies exist).
            LightSample ls;
            if (!sampleLight(scene, li, E.p, rng.nextFloat(), rng.nextFloat(), ls)) continue;
            if (ls.pdf <= 0.0f || isBlack(ls.radiance)) continue;
            float visibility = 1.0f;
            if (l.shadowEnable) {
                const Vec3 o = offsetRayOrigin(E.p, E.ng, ls.wi);
                visibility = shadowVisibility(scene, tracer, o, ls.wi, 1.0e8f);
            }
            if (visibility <= 1e-5f) continue;
            const Vec3 f = bsdfF(E.mat, E.ns, E.wo, ls.wi);
            if (isBlack(f)) continue;
            const float bsdfPdf = ls.delta ? 0.0f : bsdfPdfSa(E.mat, E.ns, E.wo, ls.wi);
            const float lightPdf = ls.pdf * selectPdf;
            const float w = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, bsdfPdf);
            Vec3 c = E.beta * f * ls.radiance * (fabsf(dot(E.ns, ls.wi)) * w * visibility / lightPdf);
            if (t > 2) c = clampContribution(c, settings.clampIndirect);
            L += c;
#if SOLSTICE_HAVE_OPENPGL
            if (guiding && guiding->active())
                guiding->addScattered(f * ls.radiance * (fabsf(dot(E.ns, ls.wi)) * w * visibility / lightPdf));
#endif
            continue;
        }
        if (!lightIsFinite(l)) continue;

        // Sample a point on the finite light (area measure).
        Vert Ls{};
        Ls.type = VType::Light;
        Ls.lightIndex = li;
        float pdfPosArea = 0.0f;
        Vec3 lightN;
        if (l.type == kLightPoint) {
            Ls.p = lightOrigin(l);
            Ls.ng = Ls.ns = Vec3(0.0f, 1.0f, 0.0f);
            Ls.delta = true;
            pdfPosArea = 1.0f;  // delta position
            lightN = Ls.ns;
        } else if (l.type == kLightSphere) {
            const Vec3 center = lightOrigin(l);
            const float radius = srMax(1e-5f, sphereLightRadius(l));
            const Vec3 dir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
            Ls.p = center + dir * radius;
            Ls.ng = Ls.ns = dir;
            pdfPosArea = 1.0f / (4.0f * kPi * radius * radius);
            lightN = dir;
        } else {
            if (l.type == kLightRect) {
                const Vec3 pLocal((rng.nextFloat() - 0.5f) * l.width, (rng.nextFloat() - 0.5f) * l.height,
                                  0.0f);
                Ls.p = transformPoint(l.xform, pLocal);
                pdfPosArea = 1.0f / srMax(1e-12f, rectLightArea(l));
            } else {
                const Vec2 d = sampleConcentricDisk(rng.nextFloat(), rng.nextFloat());
                Ls.p = transformPoint(l.xform, Vec3(d.x * l.radius, d.y * l.radius, 0.0f));
                pdfPosArea = 1.0f / srMax(1e-12f, diskLightArea(l));
            }
            Ls.ng = Ls.ns = areaLightNormal(l);
            lightN = Ls.ns;
        }
        Ls.pdfFwd = selectPdf * pdfPosArea;

        Vec3 toL = Ls.p - E.p;
        const float dist2 = lengthSquared(toL);
        if (dist2 < 1e-10f) continue;
        const float dist = sqrtf(dist2);
        const Vec3 wi = toL / dist;

        Vec3 Le(0.0f);
        if (l.type == kLightPoint) {
            Le = l.emittedRadiance() / dist2;  // radiant intensity → irradiance factor
        } else if (l.type == kLightSphere) {
            // Back-facing sphere samples are self-occluded — reject instead of |cos|.
            if (dot(lightN, -wi) <= 1e-6f) continue;
            Le = lightRadiance(l);
        } else {
            Le = areaLightEmission(scene, l, wi, lightN);
        }
        if (isBlack(Le)) continue;

        const Vec3 f = bsdfF(E.mat, E.ns, E.wo, wi);
        if (isBlack(f)) continue;

        if (l.shadowEnable && !connectionVisible(scene, tracer, E.p, E.ng, Ls.p, li)) continue;

        Vec3 c;
        if (l.type == kLightPoint) {
            c = E.beta * f * Le * (fabsf(dot(E.ns, wi)) / srMax(1e-12f, selectPdf));
        } else {
            const float G = geometryTerm(E.p, E.ns, Ls.p, Ls.ns);
            if (G <= 0.0f) continue;
            c = E.beta * f * Le * (G / srMax(1e-12f, Ls.pdfFwd));
        }

        // MIS overrides for this strategy.
        MisOverride ov;
        ov.splatStrategy = doSplats;
        ov.lightOriginDelta = l.type == kLightPoint;
        // Light vertex generated from the eye side: bsdf at E toward L.
        ov.lightLastRev = toAreaPdf(bsdfPdfSa(E.mat, E.ns, E.wo, wi), E.p, Ls.p, Ls.ns);
        // Eye vertex generated from the light: emission dir pdf.
        ov.eyeLastRev = toAreaPdf(pdfLightDirSa(l, lightN, -wi), Ls.p, E.p, E.ns);
        // Eye prev regenerated by bsdf at E arriving from L.
        if (t >= 3)
            ov.eyePrevRev = toAreaPdf(bsdfPdfSa(E.mat, E.ns, wi, normalize(eye[t - 2].p - E.p)), E.p,
                                      eye[t - 2].p,
                                      eye[t - 2].type == VType::Surface ? eye[t - 2].ns : eye[t - 2].ng);
        Vert lightArr[1] = {Ls};
        const float w = misWeight(eye, t, lightArr, 1, ov);
        c = c * w;
        if (t > 2) c = clampContribution(c, settings.clampIndirect);
        if (!isFinite(c)) continue;
        L += c;
#if SOLSTICE_HAVE_OPENPGL
        if (guiding && guiding->active() && !isBlack(E.beta)) {
            const Vec3 local(c.x / srMax(1e-8f, E.beta.x), c.y / srMax(1e-8f, E.beta.y),
                             c.z / srMax(1e-8f, E.beta.z));
            guiding->addScattered(local);
        }
#endif
    }

    // ---- s >= 2: surface ↔ surface connections ----
    for (int t = 2; t <= nEye; ++t) {
        const Vert& E = eye[t - 1];
        if (E.type != VType::Surface || !E.connectable) continue;
        for (int s = 2; s <= nLight; ++s) {
            if (s + t > maxVerts + 1) break;
            const Vert& Lv = light[s - 1];
            if (Lv.type != VType::Surface || !Lv.connectable) continue;

            bool lightPrefixDelta = false;
            for (int i = 1; i < s - 1; ++i)
                if (light[i].delta) lightPrefixDelta = true;
            // Caustics off: skip connections whose eye side has diffuse→specular chains.
            if (!causticsOn) {
                bool diffuseSeen = false, chainAfterDiffuse = false;
                for (int i = 1; i < t; ++i) {
                    if (!eye[i].delta) diffuseSeen = true;
                    else if (diffuseSeen) chainAfterDiffuse = true;
                }
                if (chainAfterDiffuse) continue;
            } else if (lightPrefixDelta && light[0].lightIndex >= 0 &&
                       !lightContributesCaustics(scene.lights[light[0].lightIndex])) {
                continue;
            }

            Vec3 d = Lv.p - E.p;
            const float dist2 = lengthSquared(d);
            if (dist2 < 1e-10f) continue;
            d = d * (1.0f / sqrtf(dist2));

            const Vec3 fE = bsdfF(E.mat, E.ns, E.wo, d);
            if (isBlack(fE)) continue;
            const Vec3 fL = bsdfF(Lv.mat, Lv.ns, Lv.wo, -d);
            if (isBlack(fL)) continue;
            const float G = geometryTerm(E.p, E.ns, Lv.p, Lv.ns);
            if (G <= 0.0f) continue;
            if (!connectionVisible(scene, tracer, E.p, E.ng, Lv.p, -1)) continue;

            MisOverride ov;
            ov.splatStrategy = doSplats;
            ov.s0Sampled = !(causticsOn && doSplats && lightPrefixDelta);
            ov.lightOriginDelta = lightOriginDelta;
            ov.lightLastRev = toAreaPdf(bsdfPdfSa(E.mat, E.ns, E.wo, d), E.p, Lv.p, Lv.ns);
            ov.eyeLastRev = toAreaPdf(bsdfPdfSa(Lv.mat, Lv.ns, Lv.wo, -d), Lv.p, E.p, E.ns);
            if (t >= 3)
                ov.eyePrevRev = toAreaPdf(bsdfPdfSa(E.mat, E.ns, d, normalize(eye[t - 2].p - E.p)), E.p,
                                          eye[t - 2].p,
                                          eye[t - 2].type == VType::Surface ? eye[t - 2].ns : eye[t - 2].ng);
            if (s >= 2)
                ov.lightPrevRev =
                    toAreaPdf(bsdfPdfSa(Lv.mat, Lv.ns, -d, normalize(light[s - 2].p - Lv.p)), Lv.p,
                              light[s - 2].p,
                              light[s - 2].type == VType::Surface ? light[s - 2].ns : light[s - 2].ng);

            const float w = misWeight(eye, t, light, s, ov);
            Vec3 c = E.beta * fE * G * fL * Lv.beta * w;
            c = clampContribution(c, settings.clampIndirect);
            if (!isFinite(c)) continue;
            L += c;
        }
    }

    if (!isFinite(L)) return Vec3(0.0f);
    return vmax(Vec3(0.0f), L);
}

}  // namespace sol
