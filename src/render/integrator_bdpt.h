// Bidirectional path tracing (Veach) with optional OpenPGL guiding on the eye path.
// CPU / Embree only — included from embree_device.cpp.
#pragma once

#include "core/rng.h"
#include "render/integrator.h"
#include "render/lights.h"
#include "render/shading.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPENPGL
#include "render/cpu/path_guiding.h"
#endif

namespace sol {
namespace bdpt {

constexpr int kMaxBdptVerts = 10;

enum class VertKind : uint8_t { Camera = 0, Light = 1, Surface = 2 };

struct Vertex {
    Vec3 p{0.0f};
    Vec3 ng{0.0f, 1.0f, 0.0f};
    Vec3 ns{0.0f, 1.0f, 0.0f};
    Vec3 beta{1.0f};  // path throughput weight at this vertex
    Vec3 wo{0.0f};    // direction toward the previous vertex
    Material mat{};
    float pdfFwd = 0.0f;  // area-measure pdf of arriving here from previous
    int lightIndex = -1;
    int materialIndex = -1;
    VertKind kind = VertKind::Surface;
    bool delta = false;
    bool specular = false;
    bool transmitted = false;
    bool connectable = true;  // false for pure delta (connect via adjacent strategies)
};

SR_INL float geometryTerm(Vec3 a, Vec3 na, Vec3 b, Vec3 nb) {
    Vec3 d = b - a;
    const float dist2 = lengthSquared(d);
    if (dist2 < 1e-12f) return 0.0f;
    d = d * (1.0f / sqrtf(dist2));
    const float cosA = fabsf(dot(na, d));
    const float cosB = fabsf(dot(nb, -d));
    if (cosA < 1e-6f || cosB < 1e-6f) return 0.0f;
    return (cosA * cosB) / dist2;
}

SR_INL float pdfSolidAngleToArea(float pdfSa, Vec3 from, Vec3 to, Vec3 nTo) {
    Vec3 d = to - from;
    const float dist2 = lengthSquared(d);
    if (dist2 < 1e-12f || pdfSa <= 0.0f) return 0.0f;
    d = d * (1.0f / sqrtf(dist2));
    const float cosTo = fabsf(dot(nTo, -d));
    return pdfSa * cosTo / dist2;
}

SR_INL bool allowsReflectiveCaustics(const Material& m) { return m.reflectiveCaustics != 0; }
SR_INL bool allowsRefractiveCaustics(const Material& m) { return m.refractiveCaustics != 0; }

SR_INL bool isConnectableMaterial(const Material& m) {
    const LobeWeights lw = computeLobes(m);
    return lw.diffuse > 1e-4f || (!lw.delta && (lw.specular > 1e-4f || lw.transmission > 1e-4f));
}

// Binary occlusion for BDPT connections (no fake transparent shadows).
template <typename Tracer>
SR_INL bool visibleOpaque(const SceneView& scene, const Tracer& tracer, Vec3 a, Vec3 na, Vec3 b, Vec3 nb) {
    Vec3 d = b - a;
    const float dist = length(d);
    if (dist < 1e-5f) return false;
    d = d / dist;
    if (dot(na, d) <= 1e-5f && dot(nb, -d) <= 1e-5f) return false;
    const Vec3 o = offsetRayOrigin(a, na, d);
    const float tMax = dist * (1.0f - 1e-3f);
    RayHit hit;
    if (!tracer.intersect(o, d, tMax, hit)) return true;
    SurfaceInteraction si;
    if (!buildSurfaceInteraction(scene, hit, o, d, si)) return false;
    // Hitting the destination light proxy is fine.
    if (si.lightIndex >= 0) {
        const float hitDist = length((o + d * hit.t) - b);
        return hitDist < 1e-2f * srMax(1.0f, dist);
    }
    return false;
}

SR_INL bool sampleLightVertex(const SceneView& scene, Rng& rng, Vertex& v, Vec3& emitDir, float& pdfPos,
                              float& pdfDir) {
    float selectPdf = 0.0f;
    const int li = sampleLightIndex(scene, rng.nextFloat(), selectPdf);
    if (li < 0 || selectPdf <= 0.0f) return false;
    const LightData& l = scene.lights[li];
    v = Vertex{};
    v.kind = VertKind::Light;
    v.lightIndex = li;
    v.connectable = true;

    switch (l.type) {
        case kLightRect:
        case kLightDisk: {
            Vec3 pLocal;
            float area = 0.0f;
            if (l.type == kLightRect) {
                pLocal = Vec3((rng.nextFloat() - 0.5f) * l.width, (rng.nextFloat() - 0.5f) * l.height, 0.0f);
                area = rectLightArea(l);
            } else {
                const Vec2 d = sampleConcentricDisk(rng.nextFloat(), rng.nextFloat());
                pLocal = Vec3(d.x * l.radius, d.y * l.radius, 0.0f);
                area = diskLightArea(l);
            }
            if (area <= 1e-12f) return false;
            v.p = transformPoint(l.xform, pLocal);
            v.ng = v.ns = areaLightNormal(l);
            pdfPos = selectPdf / area;
            {
                const Vec3 local = sampleCosineHemisphere(rng.nextFloat(), rng.nextFloat());
                const Frame frame(v.ns);
                emitDir = normalize(frame.toWorld(local));
                pdfDir = fabsf(dot(v.ns, emitDir)) * kInvPi;
                if (pdfDir <= 0.0f) return false;
                v.wo = -emitDir;
                // beta = Le / pdfA  (direction pdf applied when leaving)
                v.beta = lightRadiance(l) / srMax(1e-8f, pdfPos);
                v.pdfFwd = pdfPos;
            }
            return true;
        }
        case kLightSphere: {
            const Vec3 center = lightOrigin(l);
            const float radius = srMax(1e-5f, sphereLightRadius(l));
            const float area = 4.0f * kPi * radius * radius;
            const Vec3 dir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
            v.p = center + dir * radius;
            v.ng = v.ns = dir;
            pdfPos = selectPdf / area;
            const Vec3 local = sampleCosineHemisphere(rng.nextFloat(), rng.nextFloat());
            const Frame frame(v.ns);
            emitDir = normalize(frame.toWorld(local));
            pdfDir = fabsf(dot(v.ns, emitDir)) * kInvPi;
            if (pdfDir <= 0.0f) return false;
            v.wo = -emitDir;
            v.beta = lightRadiance(l) / srMax(1e-8f, pdfPos);
            v.pdfFwd = pdfPos;
            return true;
        }
        case kLightPoint: {
            v.p = lightOrigin(l);
            v.ng = v.ns = Vec3(0.0f, 1.0f, 0.0f);
            pdfPos = selectPdf;
            emitDir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
            pdfDir = kInv4Pi;
            v.wo = -emitDir;
            v.delta = true;
            v.connectable = false;  // connect via s=1 using point-light NEE form
            v.beta = l.emittedRadiance() / srMax(1e-8f, pdfPos);
            v.pdfFwd = pdfPos;
            return true;
        }
        case kLightDistant: {
            const Vec3 axis = normalize(lightAxisZ(l));
            emitDir = axis;
            v.p = -axis * 1.0e5f;
            v.ng = v.ns = -axis;
            pdfPos = selectPdf;
            pdfDir = 1.0f;
            v.wo = -emitDir;
            v.delta = true;
            v.connectable = true;
            v.beta = l.emittedRadiance() / srMax(1e-8f, pdfPos);
            v.pdfFwd = pdfPos;
            return true;
        }
        case kLightDome: {
            float pdf = 0.0f;
            Vec3 dirLocal;
            if (l.envIndex >= 0 && l.envIndex < scene.envMapCount && scene.envMaps[l.envIndex].sampled()) {
                dirLocal = envSample(scene.envMaps[l.envIndex], rng.nextFloat(), rng.nextFloat(), pdf);
            } else {
                dirLocal = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
                pdf = kInv4Pi;
            }
            if (pdf <= 0.0f) return false;
            emitDir = normalize(transformVector(l.xform, dirLocal));
            v.p = -emitDir * 1.0e5f;
            v.ng = v.ns = -emitDir;
            v.wo = -emitDir;
            pdfPos = selectPdf;
            pdfDir = pdf;
            v.beta = domeRadiance(scene, l, emitDir) / srMax(1e-8f, pdfPos * pdfDir);
            v.pdfFwd = pdfPos;
            return true;
        }
        default:
            return false;
    }
}

template <typename Tracer>
SR_INL int extendPath(const SceneView& scene, const Tracer& tracer, Rng& rng, Vertex* path, int count,
                      int maxDepth, bool eyePath
#if SOLSTICE_HAVE_OPENPGL
                      ,
                      PathGuiding::ThreadState* guiding
#endif
) {
    while (count < kMaxBdptVerts && count <= maxDepth) {
        Vertex& prev = path[count - 1];
        if (prev.kind == VertKind::Light && eyePath) break;  // eye path ended on emitter
        if (prev.kind != VertKind::Surface && prev.kind != VertKind::Camera &&
            !(prev.kind == VertKind::Light && !eyePath && count == 1))
            break;

        Vec3 wiWorld;
        float pdfSa = 0.0f;
        Vec3 weight(1.0f);  // f*|cos|/pdf for surface, or cos/pdfDir for light
        bool specular = false;
        bool transmitted = false;
        bool deltaBounce = false;

        if (prev.kind == VertKind::Camera) {
            wiWorld = -prev.wo;
            pdfSa = 1.0f;
            weight = Vec3(1.0f);
        } else if (prev.kind == VertKind::Light && count == 1) {
            wiWorld = -prev.wo;
            // Leaving light: multiply by |cos| / pdfDir. pdfDir was cosine-hemisphere for area.
            const float cosT = fabsf(dot(prev.ns, wiWorld));
            float pdfDir = prev.delta ? (prev.lightIndex >= 0 && scene.lights[prev.lightIndex].type == kLightPoint
                                             ? kInv4Pi
                                             : 1.0f)
                                      : cosT * kInvPi;
            if (scene.lights[prev.lightIndex].type == kLightDome) {
                // beta already includes pdfDir for dome
                pdfDir = 1.0f;
                weight = Vec3(1.0f);
            } else {
                if (pdfDir <= 0.0f) break;
                weight = Vec3(cosT / pdfDir);
            }
            pdfSa = pdfDir;
        } else {
            // Surface BSDF sample (+ optional guiding on eye path).
            const Frame frame(prev.ns);
            const Vec3 woLocal = frame.toLocal(prev.wo);
            bool usedGuide = false;
            BsdfSample bs{};
#if SOLSTICE_HAVE_OPENPGL
            if (eyePath && guiding && guiding->active() && prev.connectable && !prev.delta) {
                if (guiding->prepare(prev.p, prev.ns, rng)) {
                    const float pg = guiding->guideProbability();
                    if (rng.nextFloat() < pg) {
                        float gPdf = 0.0f;
                        if (guiding->sample(rng.nextFloat(), rng.nextFloat(), wiWorld, gPdf) && gPdf > 0.0f) {
                            const Vec3 wiLocal = frame.toLocal(wiWorld);
                            const BsdfEval ev = bsdfEvalLocal(prev.mat, woLocal, wiLocal);
                            if (ev.pdf > 0.0f && !isBlack(ev.f)) {
                                const float mixPdf = pg * gPdf + (1.0f - pg) * ev.pdf;
                                bs.wi = wiLocal;
                                bs.pdf = mixPdf;
                                bs.weight = ev.f * (fabsf(wiLocal.z) / mixPdf);
                                bs.specular = false;
                                bs.transmitted = wiLocal.z < 0.0f;
                                usedGuide = true;
                            }
                        }
                    }
                }
            }
#else
            (void)eyePath;
#endif
            if (!usedGuide) {
                bs = bsdfSampleLocal(prev.mat, woLocal, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                     rng.nextFloat());
                if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;
                wiWorld = normalize(frame.toWorld(bs.wi));
#if SOLSTICE_HAVE_OPENPGL
                if (eyePath && guiding && guiding->active() && guiding->prepared() && !bs.specular) {
                    const float pg = guiding->guideProbability();
                    const float gPdf = guiding->pdf(wiWorld);
                    const float mixPdf = pg * gPdf + (1.0f - pg) * bs.pdf;
                    if (mixPdf > 0.0f) {
                        bs.weight = bs.weight * (bs.pdf / mixPdf);
                        bs.pdf = mixPdf;
                    }
                }
#endif
            }
            // Artist caustic gates: kill disabled specular chains.
            if (bs.transmitted && !allowsRefractiveCaustics(prev.mat)) break;
            if (bs.specular && !bs.transmitted && !allowsReflectiveCaustics(prev.mat) &&
                computeLobes(prev.mat).diffuse < 1e-4f)
                break;

            pdfSa = bs.pdf;
            weight = bs.weight;
            specular = bs.specular;
            transmitted = bs.transmitted;
            deltaBounce = bs.specular && computeLobes(prev.mat).delta;

#if SOLSTICE_HAVE_OPENPGL
            if (eyePath && guiding && guiding->active()) {
                guiding->recordBounce(prev.ns, wiWorld, bs.pdf, weight, bs.specular, prev.mat.roughness,
                                      computeLobes(prev.mat).eta, 1.0f);
            }
#endif
        }

        const Vec3 origin = offsetRayOrigin(prev.p, prev.ng, wiWorld);
        Vertex next{};
        RayHit hit;
        if (!tracer.intersect(origin, wiWorld, kFloatMax, hit)) {
            // Eye path escaped to env — record as infinite light vertex.
            if (eyePath && scene.domeLightIndex >= 0 && scene.settings.envVisibleCamera) {
                next.kind = VertKind::Light;
                next.lightIndex = scene.domeLightIndex;
                next.p = origin + wiWorld * 1.0e5f;
                next.ng = next.ns = -wiWorld;
                next.wo = -wiWorld;
                next.beta = prev.beta * weight;
                next.pdfFwd = pdfSa;
                next.connectable = true;
                path[count++] = next;
#if SOLSTICE_HAVE_OPENPGL
                if (guiding && guiding->active()) {
                    const Vec3 Le = environmentRadiance(scene, wiWorld);
                    guiding->recordBackground(origin, wiWorld, Le, 1.0f);
                }
#endif
            }
            break;
        }
        SurfaceInteraction si;
        if (!buildSurfaceInteraction(scene, hit, origin, wiWorld, si)) break;

        if (si.lightIndex >= 0) {
            next.kind = VertKind::Light;
            next.p = si.p;
            next.ng = next.ns = si.ng;
            next.lightIndex = si.lightIndex;
            next.wo = -wiWorld;
            next.beta = prev.beta * weight;
            next.pdfFwd = pdfSolidAngleToArea(pdfSa, prev.p, next.p, next.ns);
            next.connectable = true;
            path[count++] = next;
#if SOLSTICE_HAVE_OPENPGL
            if (eyePath && guiding && guiding->active()) {
                const LightData& l = scene.lights[si.lightIndex];
                const Vec3 lightN = l.type == kLightSphere ? si.ng : areaLightNormal(l);
                const Vec3 Le = areaLightEmission(scene, l, wiWorld, lightN);
                guiding->recordLightHit(si.p, -wiWorld, Le, 1.0f);
            }
#endif
            if (eyePath) break;
            // Light path hitting another emitter — stop.
            break;
        }

        Material mat = si.materialIndex >= 0 && si.materialIndex < scene.materialCount
                           ? scene.materials[si.materialIndex]
                           : defaultMaterial();
        mat = evaluateTexturedMaterial(scene, mat, si.uv, si.ns, si.pObject, si.nObject, si.uvFilterWidth);
        next.kind = VertKind::Surface;
        next.p = si.p;
        next.ng = si.ng;
        next.ns = si.ns;
        next.mat = mat;
        next.materialIndex = si.materialIndex;
        next.wo = -wiWorld;
        next.beta = prev.beta * weight;
        next.pdfFwd = pdfSolidAngleToArea(pdfSa, prev.p, next.p, next.ns);
        next.specular = specular;
        next.transmitted = transmitted;
        next.delta = deltaBounce || (computeLobes(mat).delta && computeLobes(mat).diffuse < 1e-4f);
        next.connectable = isConnectableMaterial(mat);
        path[count++] = next;

        if (count > scene.settings.rrStartDepth) {
            const float lum = average(vmax(Vec3(0.0f), next.beta));
            const float q = clampf(lum, 0.05f, 0.95f);
            if (rng.nextFloat() > q) break;
            path[count - 1].beta = path[count - 1].beta / q;
#if SOLSTICE_HAVE_OPENPGL
            if (eyePath && guiding && guiding->active()) guiding->setRussianRoulette(q);
#endif
        }
    }
    return count;
}

SR_INL Vec3 emissionOnEyePath(const SceneView& scene, const Vertex* eye, int t) {
    if (t < 2) return Vec3(0.0f);
    const Vertex& v = eye[t - 1];
    if (v.kind != VertKind::Light || v.lightIndex < 0) return Vec3(0.0f);
    const Vertex& prev = eye[t - 2];
    const LightData& l = scene.lights[v.lightIndex];
    Vec3 Le(0.0f);
    if (l.type == kLightDome) {
        Le = environmentRadiance(scene, -v.wo);
    } else {
        const Vec3 wi = normalize(v.p - prev.p);
        const Vec3 lightN = l.type == kLightSphere ? v.ng : areaLightNormal(l);
        Le = areaLightEmission(scene, l, wi, lightN);
    }
    if (isBlack(Le)) return Vec3(0.0f);
    // prev.beta already includes path weight up to prev; for t==2 camera→light, beta=1.
    // For t>2, beta includes BSDF weights; multiply by Le.
    return prev.beta * Le * (t == 2 ? 1.0f : 1.0f);
}

template <typename Tracer>
SR_INL Vec3 connectVertices(const SceneView& scene, const Tracer& tracer, const Vertex* eye, int t,
                            const Vertex* light, int s) {
    if (t < 2 || s < 1) return Vec3(0.0f);
    const Vertex& ev = eye[t - 1];
    const Vertex& lv = light[s - 1];

    // Need a connectable eye surface vertex.
    if (ev.kind != VertKind::Surface || !ev.connectable) return Vec3(0.0f);
    if (ev.delta) return Vec3(0.0f);

    // Caustic gates on the eye endpoint.
    if (ev.specular && ev.transmitted && !allowsRefractiveCaustics(ev.mat)) return Vec3(0.0f);
    if (ev.specular && !ev.transmitted && !allowsReflectiveCaustics(ev.mat) &&
        computeLobes(ev.mat).diffuse < 1e-4f)
        return Vec3(0.0f);

    if (s == 1 && lv.kind == VertKind::Light) {
        // Connect eye surface → light sample (classic NEE / BDPT s=1).
        if (lv.lightIndex < 0) return Vec3(0.0f);
        const LightData& l = scene.lights[lv.lightIndex];
        Vec3 wi;
        float dist = 0.0f;
        float pdf = 0.0f;
        Vec3 radiance(0.0f);
        bool delta = lv.delta;
        if (l.type == kLightDome || l.type == kLightDistant) {
            LightSample ls;
            if (!sampleLight(scene, lv.lightIndex, ev.p, 0.5f, 0.5f, ls)) return Vec3(0.0f);
            wi = ls.wi;
            dist = ls.distance;
            pdf = ls.pdf;
            radiance = ls.radiance;
            delta = ls.delta;
        } else if (l.type == kLightPoint) {
            Vec3 d = lv.p - ev.p;
            const float dist2 = lengthSquared(d);
            if (dist2 < 1e-12f) return Vec3(0.0f);
            dist = sqrtf(dist2);
            wi = d / dist;
            pdf = 1.0f;
            radiance = l.emittedRadiance() / dist2;
            delta = true;
        } else {
            Vec3 d = lv.p - ev.p;
            const float dist2 = lengthSquared(d);
            if (dist2 < 1e-12f) return Vec3(0.0f);
            dist = sqrtf(dist2);
            wi = d / dist;
            float cosL = fabsf(dot(lv.ns, -wi));
            if (cosL < 1e-6f) return Vec3(0.0f);
            float area = 1.0f;
            if (l.type == kLightRect) area = rectLightArea(l);
            else if (l.type == kLightDisk) area = diskLightArea(l);
            else if (l.type == kLightSphere)
                area = 4.0f * kPi * sphereLightRadius(l) * sphereLightRadius(l);
            pdf = dist2 / (cosL * srMax(1e-8f, area));
            radiance = areaLightEmission(scene, l, wi, lv.ns);
        }
        const float selectPdf = lightSelectionPdfIndex(scene, lv.lightIndex);
        if (pdf <= 0.0f || selectPdf <= 0.0f || isBlack(radiance)) return Vec3(0.0f);

        if (l.shadowEnable) {
            const Vec3 o = offsetRayOrigin(ev.p, ev.ng, wi);
            float tMax = 1.0e8f;
            if (dist < 1.0e7f) tMax = dist * (1.0f - 1e-3f);
            if (l.type == kLightDome || l.type == kLightDistant) {
                RayHit hit;
                if (tracer.intersect(o, wi, tMax, hit)) return Vec3(0.0f);
            } else if (!visibleOpaque(scene, tracer, ev.p, ev.ng, lv.p, lv.ns)) {
                return Vec3(0.0f);
            }
        }

        const Frame frame(ev.ns);
        const BsdfEval be = bsdfEvalLocal(ev.mat, frame.toLocal(ev.wo), frame.toLocal(wi));
        if (be.pdf <= 0.0f || isBlack(be.f)) return Vec3(0.0f);
        const float cosTheta = fabsf(dot(ev.ns, wi));
        const float lightPdf = pdf * selectPdf;
        float mis = 1.0f;
        if (!delta) mis = powerHeuristic(1.0f, lightPdf, 1.0f, be.pdf);
        return ev.beta * be.f * (cosTheta * mis / lightPdf) * radiance;
    }

    // General surface ↔ surface (or light path surface) connection.
    if (lv.kind != VertKind::Surface || !lv.connectable || lv.delta) return Vec3(0.0f);
    if (lv.specular && lv.transmitted && !allowsRefractiveCaustics(lv.mat)) return Vec3(0.0f);
    if (lv.specular && !lv.transmitted && !allowsReflectiveCaustics(lv.mat) &&
        computeLobes(lv.mat).diffuse < 1e-4f)
        return Vec3(0.0f);

    if (!visibleOpaque(scene, tracer, ev.p, ev.ng, lv.p, lv.ns)) return Vec3(0.0f);

    Vec3 w = lv.p - ev.p;
    const float dist2 = lengthSquared(w);
    if (dist2 < 1e-12f) return Vec3(0.0f);
    w = w * (1.0f / sqrtf(dist2));
    const Frame fe(ev.ns);
    const Frame fl(lv.ns);
    const BsdfEval be = bsdfEvalLocal(ev.mat, fe.toLocal(ev.wo), fe.toLocal(w));
    const BsdfEval bl = bsdfEvalLocal(lv.mat, fl.toLocal(lv.wo), fl.toLocal(-w));
    if (be.pdf <= 0.0f || isBlack(be.f) || bl.pdf <= 0.0f || isBlack(bl.f)) return Vec3(0.0f);
    const float G = geometryTerm(ev.p, ev.ns, lv.p, lv.ns);
    if (G <= 0.0f) return Vec3(0.0f);
    return ev.beta * be.f * G * bl.f * lv.beta;
}

// Balance heuristic over strategies with the same total vertex count.
SR_INL float misWeight(int s, int t) {
    const int n = s + t - 1;
    if (n <= 1) return 1.0f;
    return 1.0f / float(n);
}

}  // namespace bdpt

template <typename Tracer>
inline Vec3 traceRadianceBdpt(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                              Rng& rng
#if SOLSTICE_HAVE_OPENPGL
                              ,
                              PathGuiding::ThreadState* guiding = nullptr
#endif
) {
    using namespace bdpt;
    const int maxDepth = srMax(1, scene.settings.maxDepth);

    Vertex eye[kMaxBdptVerts];
    Vertex light[kMaxBdptVerts];

    eye[0] = Vertex{};
    eye[0].kind = VertKind::Camera;
    eye[0].p = origin;
    eye[0].ng = eye[0].ns = direction;
    eye[0].wo = -direction;
    eye[0].beta = Vec3(1.0f);
    eye[0].pdfFwd = 1.0f;
    eye[0].connectable = false;

#if SOLSTICE_HAVE_OPENPGL
    if (guiding && guiding->active()) guiding->beginSegment(origin, direction);
#else
    (void)0;
#endif

    int tCount = extendPath(scene, tracer, rng, eye, 1, maxDepth, true
#if SOLSTICE_HAVE_OPENPGL
                            ,
                            guiding
#endif
    );

    int sCount = 0;
    {
        Vec3 emitDir;
        float pdfPos = 0.0f, pdfDir = 0.0f;
        if (sampleLightVertex(scene, rng, light[0], emitDir, pdfPos, pdfDir)) {
            (void)emitDir;
            (void)pdfPos;
            (void)pdfDir;
            sCount = 1;
            // Direction weight (cos / pdfDir) is applied inside extendPath when leaving the light.
            sCount = extendPath(scene, tracer, rng, light, sCount, maxDepth, false
#if SOLSTICE_HAVE_OPENPGL
                                ,
                                nullptr
#endif
            );
        }
    }

    Vec3 luminance(0.0f);

    // (s=0) eye path hit emitter / environment.
    for (int t = 2; t <= tCount; ++t) {
        if (eye[t - 1].kind != VertKind::Light) continue;
        Vec3 c = emissionOnEyePath(scene, eye, t);
        if (isBlack(c)) continue;
        float mis = (t == 2) ? 1.0f : misWeight(0, t);
        c = c * mis;
        if (t > 2) c = clampContribution(c, scene.settings.clampIndirect);
        luminance += c;
#if SOLSTICE_HAVE_OPENPGL
        if (guiding && guiding->active() && t == tCount)
            guiding->recordEmission(c, mis);
#endif
        break;  // only the terminal emitter vertex
    }

    // Connect (s,t) strategies. Skip s=1 when t path already used unidirectional-style
    // emission only — s=1 provides NEE at every eye surface vertex.
    for (int t = 2; t <= tCount; ++t) {
        if (eye[t - 1].kind != VertKind::Surface) continue;
        for (int s = 1; s <= sCount; ++s) {
            if (s + t - 2 > maxDepth) continue;
            Vec3 c = connectVertices(scene, tracer, eye, t, light, s);
            if (isBlack(c)) continue;
            c = c * misWeight(s, t);
            if (t > 2 || s > 1) c = clampContribution(c, scene.settings.clampIndirect);
            luminance += c;
#if SOLSTICE_HAVE_OPENPGL
            if (guiding && guiding->active()) guiding->addScattered(c);
#endif
        }
    }

    if (!isFinite(luminance)) return Vec3(0.0f);
    return vmax(Vec3(0.0f), luminance);
}

}  // namespace sol
