// Bidirectional path tracing (Veach 1997) with the full multiple-importance
// weighting over all (s,t) strategies (PBRT-v3 style pdfFwd/pdfRev remap).
// CPU / Embree only — included from embree_device.cpp.
//
// Strategies: t = 1 light-tracing splats onto the camera (the workhorse for
// caustics from small lights: light → delta glass chain → diffuse → camera),
// s = 0 (eye path hits an emitter), s = 1 (light resampled toward the eye
// vertex, NEE-like — upgraded to MNEE when the eye arrived through glass and
// the shadow hits a delta dielectric, so caustics under refractive objects
// are visible through refraction), s >= 2 (surface↔surface connections),
// all t >= 2. Dome / distant lights start light subpaths via pbrt SampleLe
// (disk on the scene bounding sphere) so t=1 can carry sun/env caustics;
// s∈{0,1} still handles eye-path env/sun NEE. Optional OpenPGL guiding mixes
// into eye-path BSDF sampling; the mixture pdf is the true forward pdf so MIS
// stays valid.
#pragma once

#include <algorithm>
#include <vector>

#include "core/rng.h"
#include "render/camera_proj.h"
#include "render/framebuffer.h"
#include "render/integrator.h"
#include "render/integrator_mnee.h"
#include "render/lights.h"
#include "render/photon_map.h"
#include "render/shading.h"
#include "render/volume.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPENPGL
#include "render/cpu/path_guiding.h"
#endif

namespace sol {
namespace bdpt {

// Hard cap on eye and light subpath vertices (camera + bounces). Matches the
// UI Max Ray Depth ceiling. Per-thread scratch is sized to the session depth,
// not this compile-time cap.
constexpr int kMaxVerts = 4096;

inline int bdptSessionVerts(int maxDepth) { return std::clamp(maxDepth + 1, 2, kMaxVerts); }

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
    bool connectable = true;  // eyePathNeeConnectable: NEE / s=1 / vertex links
    // Specular or near-specular (low-roughness glass / mirror). Drives the caustic
    // family partition: `delta` alone would drop rough glass back onto the s=0
    // strategy, which cannot find a small light through the chain.
    bool nearSpec = false;
    // Homogeneous medium scatter vertex (null/delta tracking). When set, BSDF
    // eval/pdf use the Henyey–Greenstein phase with mediumG.
    bool mediumScatter = false;
    float mediumG = 0.0f;
    int mediumIndex = -1;
#if SOLSTICE_HAVE_OPENPGL
    // OpenPGL segment opened at this eye vertex — NEE/connection radiance is
    // attributed here (not to the last bounce's currentSegment_).
    void* guideSeg = nullptr;
#endif
};

SR_INL float remap0(float f) { return f > 0.0f ? f : 1.0f; }

// Path Tracer Russian roulette on a BDPT subpath. The vertex already on the
// path stays connectable; failure only stops growing. bounceDepth is scatter
// events from the subpath origin (camera or light), same counting as PT depth.
// Surface: max RGB, floor 0.05. Volume: luminance (volumeRussianRouletteQ).
SR_INL bool bdptRussianRoulette(Vec3& beta, Rng& rng, int bounceDepth, int rrStartDepth,
                                float* qOut = nullptr, bool volume = false) {
    if (bounceDepth < srMax(1, rrStartDepth)) {
        if (qOut) *qOut = 1.0f;
        return true;
    }
    const float q =
        volume ? volumeRussianRouletteQ(beta) : clampf(maxComponent(beta), 0.05f, 1.0f);
    if (qOut) *qOut = q;
    if (rng.nextFloat() > q) return false;
    beta *= 1.0f / q;
    return true;
}

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

// Volume vertices use a direction-less measure: G = 1/r² (no cosine foreshortening).
SR_INL float geometryTerm(const Vert& a, const Vert& b) {
    if (a.mediumScatter || b.mediumScatter) {
        const float dist2 = lengthSquared(b.p - a.p);
        return dist2 > 1e-12f ? 1.0f / dist2 : 0.0f;
    }
    return geometryTerm(a.p, a.ns, b.p, b.ns);
}

// Solid-angle BSDF pdf for wo→wi at a surface vertex (both world-space).
// ng is taken alongside ns so a connection the geometry cannot support evaluates
// to zero instead of leaking through the surface (see shadingNormalConsistent).
SR_INL float bsdfPdfSa(const Material& mat, Vec3 ng, Vec3 ns, Vec3 woW, Vec3 wiW) {
    if (!shadingNormalConsistent(ng, ns, woW, wiW)) return 0.0f;
    const Frame frame(ns);
    const BsdfEval e = bsdfEvalLocal(mat, frame.toLocal(woW), frame.toLocal(wiW));
    return srIsFinite(e.pdf) ? e.pdf : 0.0f;
}

SR_INL float bsdfPdfSa(const Vert& v, Vec3 woW, Vec3 wiW) {
    if (v.mediumScatter) {
        const float cosTheta = clampf(dot(woW, wiW), -1.0f, 1.0f);
        return henyeyGreensteinPdf(cosTheta, v.mediumG);
    }
    return bsdfPdfSa(v.mat, v.ng, v.ns, woW, wiW);
}

SR_INL Vec3 bsdfF(const Material& mat, Vec3 ng, Vec3 ns, Vec3 woW, Vec3 wiW) {
    if (!shadingNormalConsistent(ng, ns, woW, wiW)) return Vec3(0.0f);
    const Frame frame(ns);
    const BsdfEval e = bsdfEvalLocal(mat, frame.toLocal(woW), frame.toLocal(wiW));
    return isFinite(e.f) ? e.f : Vec3(0.0f);
}

SR_INL Vec3 bsdfF(const Vert& v, Vec3 woW, Vec3 wiW) {
    if (v.mediumScatter) {
        const float cosTheta = clampf(dot(woW, wiW), -1.0f, 1.0f);
        const float p = henyeyGreenstein(cosTheta, v.mediumG);
        return Vec3(p);
    }
    return bsdfF(v.mat, v.ng, v.ns, woW, wiW);
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

// Area pdf of sampling the emission origin on this light (flux-only selection).
// Used for light-path-start strategies (no reference point available).
SR_INL float pdfLightOrigin(const SceneView& scene, const LightData& l, int lightIndex) {
    const float select = lightSelectionPdfIndex(scene, lightIndex);
    if (l.type == kLightPoint) return select;  // delta position — only used in ratios
    const float area = lightArea(l);
    return area > 1e-12f ? select / area : 0.0f;
}

// Position-aware overload for BDPT s=0 vs s=1 MIS consistency.
// Use `refP = eye[t-2].p` (the surface vertex the s=1 strategy would connect from).
SR_INL float pdfLightOrigin(const SceneView& scene, const LightData& l, int lightIndex,
                             Vec3 refP) {
    const float select = lightSelectionPdfIndex(scene, refP, lightIndex);
    if (l.type == kLightPoint) return select;
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
// Subpath generation
// --------------------------------------------------------------------------

// pbrt InfiniteLightDensity: Σ PDF_Li(-ray.d) · PMF over infinite lights.
SR_INL float infiniteLightDensity(const SceneView& scene, Vec3 emitDir) {
    const float len2 = lengthSquared(emitDir);
    if (len2 < 1e-20f) return 0.0f;
    const Vec3 wi = emitDir * (-1.0f / sqrtf(len2));  // from the scene toward the env/sun
    float pdf = 0.0f;
    for (int i = 0; i < scene.lightCount; ++i) {
        if (!lightIsInfinite(scene.lights[i])) continue;
        const float sel = lightSelectionPdfIndex(scene, i);
        if (sel <= 0.0f) continue;
        pdf += lightPdfDirection(scene, i, Vec3(0.0f), wi, Vec3(0.0f), wi) * sel;
    }
    return pdf;
}

SR_INL bool lightOriginIsDelta(const SceneView& scene, const Vert& v0) {
    if (v0.lightIndex < 0 || v0.lightIndex >= scene.lightCount) return false;
    const LightData& l = scene.lights[v0.lightIndex];
    if (l.type == kLightPoint) return true;
    if (l.type != kLightDistant) return false;
    return radians(srMax(0.0f, l.angle)) * 0.5f < kDistantDeltaHalfRad;
}

// pbrt GenerateLightSubpath: after the walk, infinite path[0] pdfFwd is the
// direction density and path[1] pdfFwd is the disk area density (× |cos|).
SR_INL void correctInfiniteLightSubpathPdfs(const SceneView& scene, Vert* light, int nLight,
                                            Vec3 emitDir) {
    if (nLight < 1 || light[0].type != VType::Light || light[0].lightIndex < 0) return;
    if (!lightIsInfinite(scene.lights[light[0].lightIndex])) return;
    const float r = sceneRadius(scene);
    const float pdfPos = 1.0f / (kPi * r * r);
    if (nLight >= 2) {
        light[1].pdfFwd = pdfPos;
        if (!light[1].mediumScatter) light[1].pdfFwd *= fabsf(dot(emitDir, light[1].ng));
    }
    const float dens = infiniteLightDensity(scene, emitDir);
    if (dens > 0.0f) light[0].pdfFwd = dens;
}

// Start a light subpath (pbrt SampleLe). Finite lights emit from their surface;
// distant / dome emit from a disk on the scene bounding sphere.
SR_INL bool startLightPath(const SceneView& scene, Rng& rng, Vert& v0, Vec3& emitDir, float& pdfDirSa) {
    float selectPdf = 0.0f;
    const int li = sampleLightIndex(scene, rng.nextFloat(), selectPdf);
    if (li < 0 || selectPdf <= 0.0f) return false;
    const LightData& l = scene.lights[li];

    v0 = Vert{};
    v0.type = VType::Light;
    v0.lightIndex = li;

    if (l.type == kLightDistant || l.type == kLightDome) {
        const float r = sceneRadius(scene);
        const float pdfPos = 1.0f / (kPi * r * r);
        const Vec3 center = scene.worldBounds.valid() ? scene.worldBounds.center() : Vec3(0.0f);
        if (l.type == kLightDistant) {
            const Vec3 axis = normalize(lightAxisZ(l));
            if (lengthSquared(axis) < 1e-12f) return false;
            const float halfAngle = radians(srMax(0.0f, l.angle)) * 0.5f;
            const bool deltaDir = halfAngle < kDistantDeltaHalfRad;
            if (deltaDir) {
                emitDir = -axis;
                pdfDirSa = 1.0f;
            } else {
                const float cosThetaMax = cosf(halfAngle);
                const float omega = kTwoPi * (1.0f - cosThetaMax);
                if (omega <= 1e-20f) return false;
                const Frame frame(axis);
                emitDir = -normalize(frame.toWorld(sampleUniformCone(rng.nextFloat(), rng.nextFloat(),
                                                                    cosThetaMax)));
                pdfDirSa = 1.0f / omega;
            }
            const Frame wFrame(axis);
            const Vec2 cd = sampleConcentricDisk(rng.nextFloat(), rng.nextFloat());
            const Vec3 pDisk = center + wFrame.toWorld(Vec3(cd.x, cd.y, 0.0f)) * r;
            v0.p = pDisk + axis * r;
            v0.ng = v0.ns = -emitDir;
            v0.pdfFwd = selectPdf * pdfPos;
            v0.connectable = false;
            const Vec3 Le = deltaDir ? l.emittedRadiance() : lightRadiance(l);
            if (isBlack(Le) || v0.pdfFwd <= 0.0f || pdfDirSa <= 0.0f) return false;
            v0.beta = Le / srMax(1e-12f, v0.pdfFwd * pdfDirSa);
            return true;
        }
        float pdfDir = 0.0f;
        Vec3 wiLocal;
        if (l.envIndex >= 0 && l.envIndex < scene.envMapCount && scene.envMaps[l.envIndex].sampled()) {
            wiLocal = envSample(scene.envMaps[l.envIndex], rng.nextFloat(), rng.nextFloat(), pdfDir);
        } else {
            wiLocal = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
            pdfDir = kInv4Pi;
        }
        if (pdfDir <= 0.0f) return false;
        const Vec3 wi = normalize(transformVector(l.xform, wiLocal));
        emitDir = -wi;
        pdfDirSa = pdfDir;
        const Frame wFrame(wi);
        const Vec2 cd = sampleConcentricDisk(rng.nextFloat(), rng.nextFloat());
        const Vec3 pDisk = center + wFrame.toWorld(Vec3(cd.x, cd.y, 0.0f)) * r;
        v0.p = pDisk + wi * r;
        v0.ng = v0.ns = -emitDir;
        v0.pdfFwd = selectPdf * pdfPos;
        v0.connectable = false;
        const Vec3 Le = domeRadiance(scene, l, wi, /*nearestTexel=*/true);
        if (isBlack(Le) || v0.pdfFwd <= 0.0f) return false;
        v0.beta = Le / srMax(1e-12f, v0.pdfFwd * pdfDirSa);
        return true;
    }

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
    DispersionContext* dispersion = nullptr;
#if SOLSTICE_HAVE_OPENPGL
    PathGuiding::ThreadState* guiding = nullptr;
#endif
};

// Extend a subpath by BSDF sampling. Russian roulette matches Path Tracer
// (rrStartDepth, same q): vertices already walked stay; only continuation dies.
template <typename Tracer>
SR_INL int randomWalk(const SceneView& scene, const Tracer& tracer, Rng& rng, Vert* path, int count,
                      Vec3 origin, Vec3 dir, float pdfDirSa, int maxVerts, const WalkConfig& cfg) {
    Vec3 beta = path[count - 1].beta;
    float pdfSaFwd = pdfDirSa;
    int passThrough = 0;
    // Eye: Arnold incoming-ray ports. Light: caustic-transport ports (never camera).
    RayShadeKind rayKind = RayShadeKind::Camera;

    int currentMedium = -1;
    while (count < maxVerts) {
        Vert& prev = path[count - 1];
        RayHit hit;
        const bool didHit = tracer.intersect(origin, dir, kFloatMax, hit);

        // Homogeneous null/delta tracking (both eye and light subpaths).
        if (const MediumData* med = getMedium(scene, currentMedium)) {
            const float tMax = didHit ? hit.t : 1.0e6f;
            const MediumSample ms = sampleMediumHomogeneous(*med, tMax, rng, beta);
            if (ms.absorbed || isBlack(beta)) break;
            if (ms.scattered) {
                Vert v{};
                v.type = VType::Surface;
                v.p = origin + dir * ms.t;
                v.ng = v.ns = -dir;
                v.wo = -dir;
                v.beta = beta;
                v.mediumScatter = true;
                v.mediumG = med->g;
                v.mediumIndex = currentMedium;
                v.delta = false;
                v.connectable = true;
                v.nearSpec = false;
                // Distance pdf * direction pdf, stored as a pseudo-area density.
                const float maj = mediumMajorant(*med);
                const float pDist = maj * expf(-maj * ms.t);
                v.pdfFwd = pdfSaFwd * pDist;
                path[count++] = v;
                if (count >= maxVerts) break;
                float phasePdf = 0.0f;
                const Vec3 wi = sampleHenyeyGreenstein(-dir, med->g, rng.nextFloat(), rng.nextFloat(),
                                                      phasePdf);
                if (count >= 2) {
                    Vert& cur = path[count - 1];
                    Vert& prv = path[count - 2];
                    const float revSa = henyeyGreensteinPdf(clampf(dot(wi, -dir), -1.0f, 1.0f), med->g);
                    prv.pdfRev = toAreaPdf(revSa, cur.p, prv.p, prv.ns);
                    (void)prv;
                }
                origin = v.p;
                dir = wi;
                pdfSaFwd = phasePdf;
                if (cfg.eyePath) rayKind = RayShadeKind::Volume;
                float qRr = 1.0f;
                if (!bdptRussianRoulette(beta, rng, count - 1, scene.settings.rrStartDepth, &qRr,
                                         true))
                    break;
#if SOLSTICE_HAVE_OPENPGL
                if (cfg.eyePath && cfg.guiding && cfg.guiding->active())
                    cfg.guiding->setRussianRoulette(qRr);
#endif
                continue;
            }
        }

        if (!didHit) {
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
                // Match the path tracer: camera-invisible light proxies are skipped on
                // the primary ray so the eye can see the scene behind them.
                const InstanceData& inst = scene.instances[si.instanceIndex];
                if (count == 1 && !inst.visibleCamera) {
                    origin = offsetRayOrigin(si.p, si.ng, dir);
                    ++passThrough;
                    continue;
                }
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

        Material mat =
            cfg.eyePath ? materialForRay(scene, si.materialIndex, rayKind)
                        : materialForCausticTransport(scene, si.materialIndex);
        mat = evaluateTexturedMaterial(scene, mat, si.uv, si.ns, si.pObject, si.nObject, si.uvFilterWidth, si.pRef, si.nRef, si.hasPref);
        applyDispersion(mat, cfg.dispersion);

        // Stochastic cutout — pass through without creating a vertex.
        if (mat.opacity <= 1e-6f || (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity)) {
            origin = offsetRayOrigin(si.p, si.ng, dir);
            ++passThrough;
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
        v.mediumIndex = scene.instances[si.instanceIndex].mediumIndex;
        v.pdfFwd = toAreaPdf(pdfSaFwd, prev.p, si.p, si.ns);
        {
            const LobeWeights lw = computeLobes(mat, Frame(si.ns).toLocal(-dir));
            v.delta = lw.delta && lw.diffuse < 1e-4f;
            v.connectable = eyePathNeeConnectable(mat, Frame(si.ns).toLocal(-dir));
            // Include rough refractive casters so Photon/LT family partition matches
            // the photon map (not only α ≤ kCausticAlpha).
            v.nearSpec = v.delta || isNearSpecularLobe(lw) || isPhotonCausticCasterLobe(lw);
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
        // Open a guiding segment before NEE-style training / BSDF sampling so
        // BDPT connections can attribute radiance to this vertex later.
        if (cfg.eyePath && cfg.guiding && cfg.guiding->active()) {
            cfg.guiding->beginSegment(cur.p, cur.wo);
            cur.guideSeg = cfg.guiding->segmentHandle();
        }
#endif

        // Subsurface: same Standard Surface mix lottery as the path tracer.
        // Eye paths run the Chiang volume walk and relocate the vertex to the
        // exit; light paths only keep the specular entry layer (eye owns BSSRDF).
        if (materialSupportsSss(cur.mat) && rng.nextFloat() < saturatef(cur.mat.subsurface)) {
            Material specMat = sssSpecularEntryMaterial(cur.mat);
            const LobeWeights specLw = computeLobes(specMat, woLocal);
            const float pSpec = sssEntrySpecularProb(specMat, woLocal);

            if (pSpec > 0.0f && rng.nextFloat() < pSpec) {
                beta = beta * (1.0f / pSpec);
                cur.mat = specMat;
                cur.delta = specLw.delta && specLw.diffuse < 1e-4f;
                cur.connectable = eyePathNeeConnectable(specMat, woLocal);
                cur.nearSpec = cur.delta || isNearSpecularLobe(specLw) || isPhotonCausticCasterLobe(specLw);
                cur.beta = beta;
                const float uSpec = specLw.diffuse + specLw.specular * rng.nextFloat();
                bs = bsdfSampleLocal(specMat, woLocal, uSpec, rng.nextFloat(), rng.nextFloat(),
                                     rng.nextFloat());
                if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;
                wiWorld = normalize(frame.toWorld(bs.wi));
                haveSample = true;
            } else if (!cfg.eyePath) {
                // Light subpath: no volume walk — stop diffuse transport into SSS.
                break;
            } else {
                if (pSpec > 0.0f && pSpec < 0.999f) beta = beta * (1.0f / (1.0f - pSpec));
                const Material sssBody = sssBodyMaterial(scene, si, mat);
                const SssWalkResult walk = sampleSssRandomWalk(scene, tracer, si, -dir, sssBody, rng);
                if (!walk.escaped || isBlack(walk.pathWeight) || !isFinite(walk.pathWeight)) break;
                cur.p = walk.exitP;
                cur.ng = walk.exitN;
                cur.ns = walk.exitN;
                cur.wo = walk.exitWo;
                cur.mat = sssExitLambertMaterial();
                cur.delta = false;
                cur.connectable = true;
                cur.nearSpec = false;
                beta = beta * walk.pathWeight;
                cur.beta = beta;

                const Frame ssFrame(cur.ns);
                const Vec3 ssWoLocal = ssFrame.toLocal(cur.wo);
                bs = bsdfSampleLocal(cur.mat, ssWoLocal, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                     rng.nextFloat());
                if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;
                wiWorld = normalize(ssFrame.toWorld(bs.wi));
                haveSample = true;
            }
        }

#if SOLSTICE_HAVE_OPENPGL
        // Guide only diffuse-ish eye vertices — near-spec / delta glass is owned
        // by MNEE / light tracing; mixing a guide PDF there fights MIS.
        bool guideReady = false;
        const LobeWeights curLw = computeLobes(cur.mat, woLocal);
        if (!haveSample && cfg.eyePath && cfg.guiding && cfg.guiding->active() && !cur.delta &&
            !cur.nearSpec && curLw.diffuse > 1e-4f) {
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

        // The shading normal may have promised a bounce the geometry cannot carry.
        if (!shadingNormalConsistent(cur.ng, cur.ns, cur.wo, wiWorld)) break;

        // Reverse pdf of the segment we just travelled (for MIS).
        {
            const float revSa = bs.specular ? 0.0f : bsdfPdfSa(cur, wiWorld, cur.wo);
            prev.pdfRev = toAreaPdf(revSa, cur.p, prev.p, prev.type == VType::Surface ? prev.ns : prev.ng);
        }

#if SOLSTICE_HAVE_OPENPGL
        // Record every bounce (delta/near-spec flagged) so caustic radiance
        // propagates back to diffuse receivers. Guide sampling stays diffuse-only.
        if (cfg.eyePath && cfg.guiding && cfg.guiding->active()) {
            const bool deltaSeg = bs.specular || cur.nearSpec;
            cfg.guiding->recordBounce(cur.ns, wiWorld, bs.pdf, bs.weight, deltaSeg, cur.mat.roughness,
                                      computeLobes(cur.mat).eta, 1.0f);
        }
#endif

        beta = beta * bs.weight;
        if (bs.transmitted) beta = applyFakeDispersionThroughput(beta, cur.mat, cfg.dispersion);
        if (!isFinite(beta) || isBlack(beta)) break;
        {
            float qRr = 1.0f;
            if (!bdptRussianRoulette(beta, rng, count - 1, scene.settings.rrStartDepth, &qRr))
                break;
#if SOLSTICE_HAVE_OPENPGL
            if (cfg.eyePath && cfg.guiding && cfg.guiding->active())
                cfg.guiding->setRussianRoulette(qRr);
#endif
        }
        // Delta segments carry pdf 0 → remap0() treats them as unit ratios in MIS
        // and the delta flags keep those strategies out of the sums (PBRT convention).
        pdfSaFwd = bs.specular ? 0.0f : bs.pdf;
        origin = offsetRayOrigin(cur.p, cur.ng, wiWorld);
        dir = wiWorld;
        if (bs.transmitted && cur.mediumIndex >= 0 && mediumIsActive(scene, cur.mediumIndex)) {
            const bool entering = dot(cur.ng, wiWorld) < 0.0f;
            currentMedium = entering ? cur.mediumIndex : -1;
        }
        if (cfg.eyePath) rayKind = nextRayShadeKind(bs, computeLobes(cur.mat, woLocal));
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
// skipped when the projection is unavailable. `photons` enables caustic-only
// photon gather (VCM-style); when non-null, light-tracing caustic splats and the
// through-glass MNEE upgrade are suppressed so the map owns that family.
template <typename Tracer>
inline Vec3 traceRadianceBdpt(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                              Rng& rng
#if SOLSTICE_HAVE_OPENPGL
                              ,
                              PathGuiding::ThreadState* guiding = nullptr
#endif
                              ,
                              Framebuffer* splatFb = nullptr, DispersionContext* dispersion = nullptr,
                              const CausticPhotonMap* photons = nullptr) {
    using namespace bdpt;
    const RenderSettingsData& settings = scene.settings;
    int maxVerts = settings.maxDepth + 1;
    if (maxVerts > kMaxVerts) maxVerts = kMaxVerts;
    if (maxVerts < 2) maxVerts = 2;
    const bool causticsOn = settings.caustics != 0;
    const bool photonEngine = photons != nullptr;
    const bool photonCaustics = photonEngine && !photons->empty();
    const float photonRadius = photonCaustics ? photons->gatherRadius(settings) : 0.0f;

    const CameraProj camProj = buildCameraProj(scene);
    const bool doSplats = splatFb != nullptr && camProj.valid;
    if (doSplats) splatFb->addSplatPath();

    std::vector<Vert> eyeStorage(static_cast<size_t>(maxVerts));
    std::vector<Vert> lightStorage(static_cast<size_t>(maxVerts));
    Vert* eye = eyeStorage.data();
    Vert* light = lightStorage.data();

    // ---- Light subpath (finite lights + pbrt SampleLe for distant/dome) ----
    int nLight = 0;
    bool lightOriginDelta = false;
    if (scene.lightCount > 0) {
        Vec3 emitDir;
        float pdfDirSa = 0.0f;
        if (startLightPath(scene, rng, light[0], emitDir, pdfDirSa)) {
            nLight = 1;
            lightOriginDelta = lightOriginIsDelta(scene, light[0]);
            WalkConfig cfg;
            cfg.eyePath = false;
            cfg.dispersion = dispersion;
            const bool inf = lightIsInfinite(scene.lights[light[0].lightIndex]);
            const Vec3 o = inf ? light[0].p : offsetRayOrigin(light[0].p, light[0].ng, emitDir);
            nLight = randomWalk(scene, tracer, rng, light, nLight, o, emitDir, pdfDirSa, maxVerts, cfg);
            correctInfiniteLightSubpathPdfs(scene, light, nLight, emitDir);
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
    eyeCfg.dispersion = dispersion;
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

    // ---- Caustic photon gather (VCM-style) on eye-path diffuse vertices ----
    if (photonCaustics) {
        for (int t = 2; t <= nEye; ++t) {
            const Vert& E = eye[t - 1];
            if (E.type != VType::Surface || !E.connectable || E.nearSpec) continue;
            Vec3 g = photons->gather(E.p, E.ns, E.wo, E.mat, photonRadius);
            if (isBlack(g) || !isFinite(g)) continue;
            Vec3 c = E.beta * g;
            if (t >= 2) c = clampContribution(c, settings.clampDirect);
            if (!isFinite(c)) continue;
            L += c;
#if SOLSTICE_HAVE_OPENPGL
            if (guiding && guiding->active() && E.guideSeg)
                guiding->addScatteredAt(E.guideSeg, g);
#endif
        }
    }

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
            bool lightPrefixCaustic = false;
            for (int i = 1; i < s - 1; ++i)
                if (light[i].nearSpec && materialContributesCaustics(light[i].mat))
                    lightPrefixCaustic = true;
            if (!causticsOn) {
                if (lightPrefixCaustic) continue;
            } else if (photonEngine && lightPrefixCaustic) {
                // Photon map owns the caustic family — skip LT splats for it.
                continue;
            } else if (lightPrefixCaustic && light[0].lightIndex >= 0 &&
                       !lightContributesCaustics(scene.lights[light[0].lightIndex])) {
                // Per-light Contribute to Caustics off.
                continue;
            }
            float px = 0.0f, py = 0.0f, cosTheta = 0.0f, dist2 = 0.0f;
            if (!projectToPixel(camProj, v.p, px, py, cosTheta, dist2)) continue;
            if (dist2 < 1e-8f) continue;
            const Vec3 toCam = normalize(camProj.camPos - v.p);
            const Vec3 f = bsdfF(v, v.wo, toCam);
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
            ov.s0Sampled = !(causticsOn && lightPrefixCaustic);
            ov.lightOriginDelta = lightOriginDelta;
            ov.lightLastRev = toAreaPdf(pdfOmega, camProj.camPos, v.p, v.ns);
            if (s >= 2)
                ov.lightPrevRev = toAreaPdf(bsdfPdfSa(v, toCam, normalize(light[s - 2].p - v.p)),
                                            v.p, light[s - 2].p,
                                            light[s - 2].type == VType::Surface ? light[s - 2].ns
                                                                                : light[s - 2].ng);
            Vert camVert = eye[0];
            camVert.p = camProj.camPos;
            const float w = misWeight(&camVert, 1, light, s, ov);
            c = c * w;
            // Indirect Clamp on LT in pixel-radiance units (raw splat × camera PDF
            // → threshold scaled by W·H so resolve /N matches Arnold Indirect).
            if (s >= 2) c = clampContribution(c, lightTraceSplatClamp(settings));
            if (!isFinite(c)) continue;
            if (dispersion && dispersion->heroChannel >= 0 && dispersion->used &&
                (dispersion->mode == kDispersionHero || dispersion->mode == kDispersionOptimized ||
                 dispersion->mode == kDispersionSpectral3)) {
                // Hero-channel discipline: deposit only the sampled channel, ×3.
                const int ch = dispersion->heroChannel;
                const float hero = (ch == 0 ? c.x : (ch == 1 ? c.y : c.z)) * 3.0f;
                c = Vec3(0.0f);
                if (ch == 0) c.x = hero;
                else if (ch == 1) c.y = hero;
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
                    if (t > 2) c = clampContribution(c, settings.clampDirect);
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
                if (!eye[i].nearSpec) diffuseSeen = true;
                else if (diffuseSeen) sawDiffuseThenSpec = true;
            }
            if (sawDiffuseThenSpec && (!causticsOn || !lightContributesCaustics(l))) break;
        }

        if (l.type == kLightDome) {
            // Environment: MIS against s=1 env NEE (PT-style power heuristic).
            const bool primary = t == 2;
            if (primary && (!settings.envVisibleCamera || !l.visibleCamera)) break;
            const Vec3 dirW = -v.wo;
            Vec3 Le = domeRadiance(scene, l, dirW, /*nearestTexel=*/t > 2);
            if (!isBlack(Le)) {
                float w = 1.0f;
                const Vert& prev = eye[t - 2];
                if (t > 2 && !prev.delta && prev.type == VType::Surface) {
                    const float lp = lightPdfDirection(scene, v.lightIndex, prev.p, dirW, prev.p, dirW) *
                                     lightSelectionPdfIndex(scene, prev.p, v.lightIndex);
                    w = powerHeuristic(1.0f, v.pdfFwd, 1.0f, lp);  // pdfFwd = solid-angle here
                }
                Vec3 c = v.beta * Le * w;
                if (t > 2) c = clampContribution(c, settings.clampDirect);
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

        // Specular vertices between the camera and the light: either
        //   LDS — camera on diffuse, specular chain, light (light tracing owns this), or
        //   SDS — camera looks through glass/mirror at the light (no splat can land on
        //         that pixel; BSDF sampling a small light through delta glass never
        //         converges — those are the permanent sparkles inside refractive
        //         objects at roughness 0).
        bool specularToLight = false;
        for (int i = 1; i <= t - 2; ++i) {
            if (eye[i].type == VType::Surface && eye[i].nearSpec) {
                specularToLight = true;
                break;
            }
        }
        if (causticsOn && specularToLight && t >= 4) {
            int j = t - 2;
            while (j >= 1 && eye[j].type == VType::Surface && eye[j].nearSpec) --j;
            // LDS: hand the family to light-tracing splats (or the photon map).
            if (j >= 1 && !eye[j].nearSpec && !eye[1].nearSpec) {
                if (doSplats || photonEngine) break;
            }
            // SDS under glass (eye[1] near-spec): MNEE owns when active; with Photon
            // engine drop s=0 so estimators do not stack on the same family.
            if (j >= 1 && !eye[j].nearSpec && eye[1].nearSpec) break;
        }

        MisOverride ov;
        ov.splatStrategy = doSplats;
        ov.lightOriginDelta = false;
        ov.eyeLastRev = pdfLightOrigin(scene, l, v.lightIndex, eye[t - 2].p);
        const Vec3 emitToPrev = normalize(eye[t - 2].p - v.p);
        ov.eyePrevRev = toAreaPdf(pdfLightDirSa(l, lightN, emitToPrev), v.p, eye[t - 2].p,
                                  eye[t - 2].type == VType::Surface ? eye[t - 2].ns : eye[t - 2].ng);
        const float w = misWeight(eye, t, light, 0, ov);
        Vec3 c = v.beta * Le * w;
        if (t > 2) c = clampContribution(c, settings.clampDirect);
        // SDS / specular→light: always capped. causticClamp tightens further; when it
        // is left at 0 we still apply a safety cap so roughness-0 glass does not keep
        // permanent fireflies that more samples will never clean.
        if (specularToLight) {
            c = clampContribution(c, causticFireflyCap(settings));
        }
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
        const int li = sampleLightIndex(scene, E.p, rng.nextFloat(), selectPdf);
        if (li < 0 || selectPdf <= 0.0f) continue;
        const LightData& l = scene.lights[li];

        if (l.type == kLightDome || l.type == kLightDistant) {
            // pbrt: one LightSampler sample, MIS with BSDF (no RIS).
            LightSample ls;
            if (!sampleLight(scene, li, E.p, rng.nextFloat(), rng.nextFloat(), ls) || ls.pdf <= 0.0f ||
                isBlack(ls.radiance))
                continue;
            const Vec3 f = bsdfF(E, E.wo, ls.wi);
            if (isBlack(f)) continue;
            const float lightPdf = ls.pdf * selectPdf;
            const Vec3 unshadowed = f * ls.radiance * (fabsf(dot(E.ns, ls.wi)) / lightPdf);
            float visibility = 1.0f;
            if (scene.lights[li].shadowEnable) {
                const Vec3 o = offsetRayOrigin(E.p, E.ng, ls.wi);
                visibility = shadowVisibility(scene, tracer, o, ls.wi, 1.0e8f, t > 2 ? 1 : 0);
            }
            if (visibility <= 1e-5f) continue;
            const float bsdfPdf = ls.delta ? 0.0f : bsdfPdfSa(E, E.wo, ls.wi);
            const float w = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, bsdfPdf);
            Vec3 local = unshadowed * (w * visibility);
            Vec3 c = E.beta * local;
            if (t >= 2) c = clampContribution(c, settings.clampDirect);
            if (E.nearSpec) c = clampContribution(c, causticFireflyCap(settings));
            L += c;
#if SOLSTICE_HAVE_OPENPGL
            if (guiding && guiding->active() && E.guideSeg && !E.nearSpec)
                guiding->addScatteredAt(E.guideSeg, local);
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

        // Peek the shadow segment: clear, delta-glass (MNEE candidate), or blocked.
        bool clearPath = true;
        bool glassPath = false;
        int blockerInstance = -1;
        if (l.shadowEnable) {
            const Vec3 o = offsetRayOrigin(E.p, E.ng, wi);
            RayHit sh;
            if (tracer.intersect(o, wi, dist * (1.0f - 1e-3f), sh)) {
                SurfaceInteraction bsi;
                clearPath = false;
                if (buildSurfaceInteraction(scene, sh, o, wi, bsi)) {
                    if (bsi.lightIndex == li) {
                        clearPath = true;
                    } else if (bsi.lightIndex < 0 && causticsOn && lightContributesCaustics(l)) {
                        Material bmat = materialForCausticTransport(scene, bsi.materialIndex);
                        bmat = evaluateTexturedMaterial(scene, bmat, bsi.uv, bsi.ns, bsi.pObject, bsi.nObject,
                                                        bsi.uvFilterWidth, bsi.pRef, bsi.nRef, bsi.hasPref);
                        glassPath = mnee::isCausticCaster(bmat);
                        blockerInstance = bsi.instanceIndex;
                    }
                }
            }
        }

        // Eye arrived through near-specular glass/mirror: LT cannot splat the
        // floor under that glass (camera↔floor occluded), so MNEE owns the family.
        bool eyeThroughSpec = false;
        for (int i = 1; i <= t - 2; ++i) {
            if (eye[i].type == VType::Surface && eye[i].nearSpec) {
                eyeThroughSpec = true;
                break;
            }
        }

        if (clearPath) {
            // fall through to the regular BDPT s=1 connection below
        } else {
            // Glass blocks the shadow segment. When the eye arrived through
            // near-specular glass, light-tracing cannot splat these pixels
            // (floor→camera occluded) — MNEE owns that family. On open floor
            // the same glass block is already handled by t=1 splats; skip MNEE
            // there to avoid double-counting.
            if (!causticsUseMnee(settings, &scene) || photonEngine) continue;
            if (!(glassPath && eyeThroughSpec)) continue;
            // Radiance / intensity as expected by manifoldConnect (not /r²).
            const Vec3 LeMnee =
                l.type == kLightPoint ? l.emittedRadiance() : lightRadiance(l);
            if (isBlack(LeMnee)) continue;
            const mnee::MneeResult mr =
                mnee::manifoldConnect(scene, tracer, E.p, E.ns, E.wo, E.mat, li, Ls.p, lightN, LeMnee,
                                      pdfPosArea, selectPdf, blockerInstance, dispersion);
            if (!mr.solved || isBlack(mr.contribution)) continue;
            Vec3 c = E.beta * mr.contribution;
            // Single radiance-scale clamp (same threshold as other eye strategies).
            if (t >= 2) c = clampContribution(c, settings.clampDirect);
            c = clampContribution(c, causticFireflyCap(settings));  // was opt-in; safety floor when 0
            if (!isFinite(c)) continue;
            L += c;
#if SOLSTICE_HAVE_OPENPGL
            // Teach Indirect Guides at the diffuse SDS receiver (floor under
            // glass). Do not sample the guide on near-spec glass itself.
            if (guiding && guiding->active() && E.guideSeg && !E.nearSpec && !isBlack(E.beta)) {
                const Vec3 local(c.x / srMax(1e-8f, E.beta.x), c.y / srMax(1e-8f, E.beta.y),
                                 c.z / srMax(1e-8f, E.beta.z));
                guiding->addScatteredAt(E.guideSeg, local);
            }
#endif
            continue;
        }

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

        const Vec3 f = bsdfF(E, E.wo, wi);
        if (isBlack(f)) continue;

        Vec3 c;
        if (l.type == kLightPoint) {
            c = E.beta * f * Le * (fabsf(dot(E.ns, wi)) / srMax(1e-12f, selectPdf));
        } else {
            const float G = geometryTerm(E, Ls);
            if (G <= 0.0f) continue;
            c = E.beta * f * Le * (G / srMax(1e-12f, Ls.pdfFwd));
        }

        // MIS overrides for this strategy.
        MisOverride ov;
        ov.splatStrategy = doSplats;
        ov.lightOriginDelta = l.type == kLightPoint;
        // Light vertex generated from the eye side: bsdf at E toward L.
        ov.lightLastRev = toAreaPdf(bsdfPdfSa(E, E.wo, wi), E.p, Ls.p, Ls.ns);
        // Eye vertex generated from the light: emission dir pdf.
        ov.eyeLastRev = toAreaPdf(pdfLightDirSa(l, lightN, -wi), Ls.p, E.p, E.ns);
        // Eye prev regenerated by bsdf at E arriving from L.
        if (t >= 3)
            ov.eyePrevRev = toAreaPdf(bsdfPdfSa(E, wi, normalize(eye[t - 2].p - E.p)), E.p,
                                      eye[t - 2].p,
                                      eye[t - 2].type == VType::Surface ? eye[t - 2].ns : eye[t - 2].ng);
        Vert lightArr[1] = {Ls};
        const float w = misWeight(eye, t, lightArr, 1, ov);
        c = c * w;
        if (t >= 2) c = clampContribution(c, settings.clampDirect);
        // Resolving a light through a near-specular lobe by connection: the direction
        // almost never lands in the lobe and the rare hit is enormous.
        if (E.nearSpec) c = clampContribution(c, causticFireflyCap(settings));
        if (!isFinite(c)) continue;
        L += c;
#if SOLSTICE_HAVE_OPENPGL
        if (guiding && guiding->active() && E.guideSeg && !E.nearSpec && !isBlack(E.beta)) {
            const Vec3 local(c.x / srMax(1e-8f, E.beta.x), c.y / srMax(1e-8f, E.beta.y),
                             c.z / srMax(1e-8f, E.beta.z));
            guiding->addScatteredAt(E.guideSeg, local);
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

            bool lightPrefixCaustic = false;
            for (int i = 1; i < s - 1; ++i)
                if (light[i].nearSpec && materialContributesCaustics(light[i].mat))
                    lightPrefixCaustic = true;
            // Caustics off: skip connections whose eye side has diffuse→specular chains.
            if (!causticsOn) {
                bool diffuseSeen = false, chainAfterDiffuse = false;
                for (int i = 1; i < t; ++i) {
                    if (!eye[i].nearSpec) diffuseSeen = true;
                    else if (diffuseSeen && materialContributesCaustics(eye[i].mat))
                        chainAfterDiffuse = true;
                }
                if (chainAfterDiffuse) continue;
            } else if (photonEngine && lightPrefixCaustic) {
                continue;
            } else if (lightPrefixCaustic && light[0].lightIndex >= 0 &&
                       !lightContributesCaustics(scene.lights[light[0].lightIndex])) {
                continue;
            }

            Vec3 d = Lv.p - E.p;
            const float dist2 = lengthSquared(d);
            if (dist2 < 1e-10f) continue;
            d = d * (1.0f / sqrtf(dist2));

            const Vec3 fE = bsdfF(E, E.wo, d);
            if (isBlack(fE)) continue;
            const Vec3 fL = bsdfF(Lv, Lv.wo, -d);
            if (isBlack(fL)) continue;
            const float G = geometryTerm(E, Lv);
            if (G <= 0.0f) continue;
            if (!connectionVisible(scene, tracer, E.p, E.ng, Lv.p, -1)) continue;

            MisOverride ov;
            ov.splatStrategy = doSplats;
            ov.s0Sampled = !(causticsOn && doSplats && lightPrefixCaustic);
            ov.lightOriginDelta = lightOriginDelta;
            ov.lightLastRev = toAreaPdf(bsdfPdfSa(E, E.wo, d), E.p, Lv.p, Lv.ns);
            ov.eyeLastRev = toAreaPdf(bsdfPdfSa(Lv, Lv.wo, -d), Lv.p, E.p, E.ns);
            if (t >= 3)
                ov.eyePrevRev = toAreaPdf(bsdfPdfSa(E, d, normalize(eye[t - 2].p - E.p)), E.p,
                                          eye[t - 2].p,
                                          eye[t - 2].type == VType::Surface ? eye[t - 2].ns : eye[t - 2].ng);
            if (s >= 2)
                ov.lightPrevRev =
                    toAreaPdf(bsdfPdfSa(Lv, -d, normalize(light[s - 2].p - Lv.p)), Lv.p,
                              light[s - 2].p,
                              light[s - 2].type == VType::Surface ? light[s - 2].ns : light[s - 2].ng);

            const float w = misWeight(eye, t, light, s, ov);
            Vec3 c = E.beta * fE * G * fL * Lv.beta * w;
            c = clampContribution(c, settings.clampDirect);
            // Connecting through a near-specular lobe means the connection direction
            // almost never lands in it, and the rare hit is enormous. Those spikes are
            // the sparkle seen inside glass; the light-tracing copy stays uncapped.
            if (lightPrefixCaustic || Lv.nearSpec || E.nearSpec)
                c = clampContribution(c, causticFireflyCap(settings));
            if (!isFinite(c)) continue;
            L += c;
        }
    }

    if (!isFinite(L)) return Vec3(0.0f);
    return vmax(Vec3(0.0f), L);
}

}  // namespace sol
