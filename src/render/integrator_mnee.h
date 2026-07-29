// Manifold Next-Event Estimation (MNEE) caustics solver (Hanika et al. 2015 style).
// Unidirectional path tracing + specular-manifold connections through one interface.
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
namespace mnee {

constexpr int kNewtonIters = 5;

SR_INL bool allowsReflectiveCaustics(const Material& m) { return m.reflectiveCaustics != 0; }
SR_INL bool allowsRefractiveCaustics(const Material& m) { return m.refractiveCaustics != 0; }

SR_INL bool isSpecularCausticMaterial(const Material& m) {
    const LobeWeights lw = computeLobes(m);
    if (lw.diffuse > 0.15f) return false;
    const bool refl = lw.specular > 1e-3f && allowsReflectiveCaustics(m);
    const bool refr = lw.transmission > 1e-3f && allowsRefractiveCaustics(m);
    return refl || refr;
}

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

struct SpecVertex {
    Vec3 p{0.0f};
    Vec3 n{0.0f, 1.0f, 0.0f};
    Material mat{};
    bool refractive = false;
};

// Sample a finite light point (area / sphere / point). Returns false for dome/distant.
SR_INL bool sampleLightPoint(const SceneView& scene, Rng& rng, int& lightIndex, Vec3& y, Vec3& yN, Vec3& Le,
                             float& pdfPos) {
    float selectPdf = 0.0f;
    lightIndex = sampleLightIndex(scene, rng.nextFloat(), selectPdf);
    if (lightIndex < 0 || selectPdf <= 0.0f) return false;
    const LightData& l = scene.lights[lightIndex];
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
            y = transformPoint(l.xform, pLocal);
            yN = areaLightNormal(l);
            Le = lightRadiance(l);
            pdfPos = selectPdf / area;
            return true;
        }
        case kLightSphere: {
            const Vec3 center = lightOrigin(l);
            const float radius = srMax(1e-5f, sphereLightRadius(l));
            const float area = 4.0f * kPi * radius * radius;
            yN = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
            y = center + yN * radius;
            Le = lightRadiance(l);
            pdfPos = selectPdf / area;
            return true;
        }
        case kLightPoint: {
            y = lightOrigin(l);
            yN = Vec3(0.0f, 1.0f, 0.0f);
            Le = l.emittedRadiance();
            pdfPos = selectPdf;
            return true;
        }
        default:
            return false;
    }
}

// Constraint: generalized half-vector should align with surface normal (→ 0 when on manifold).
SR_INL Vec3 halfVectorConstraint(Vec3 p, Vec3 y, Vec3 x, Vec3 n, const Material& mat, bool refractive) {
    const Vec3 wi = normalize(p - x);
    const Vec3 wo = normalize(y - x);
    const LobeWeights lw = computeLobes(mat);
    if (!refractive) {
        Vec3 h = wi + wo;
        if (lengthSquared(h) < 1e-12f) return Vec3(1.0f, 0.0f, 0.0f);
        h = normalize(h);
        if (dot(h, n) < 0.0f) h = -h;
        return cross(h, n);
    }
    const float sideP = dot(n, wi);
    const float sideY = dot(n, wo);
    if (sideP * sideY > 0.0f) {
        Vec3 h = wi + wo;
        if (lengthSquared(h) < 1e-12f) return Vec3(1.0f, 0.0f, 0.0f);
        h = normalize(h);
        if (dot(h, n) < 0.0f) h = -h;
        return cross(h, n);
    }
    const float eta = sideP > 0.0f ? lw.eta : 1.0f / lw.eta;
    Vec3 h = normalize(wi + wo * eta);
    if (dot(h, n) < 0.0f) h = -h;
    return cross(h, n);
}

template <typename Tracer>
SR_INL bool reprojectOntoSpecular(const SceneView& scene, const Tracer& tracer, Vec3 p, Vec3 xGuess, Vec3 nGuess,
                                  SpecVertex& sv) {
    RayHit hit;
    SurfaceInteraction si;
    // Prefer a short probe along the guessed normal onto the surface.
    const Vec3 probe = xGuess + nGuess * 0.05f;
    const Vec3 probeDir = normalize(xGuess - probe);
    if (tracer.intersect(probe, probeDir, 0.25f, hit) &&
        buildSurfaceInteraction(scene, hit, probe, probeDir, si)) {
        // ok
    } else {
        const Vec3 d = normalize(xGuess - p);
        const Vec3 o = offsetRayOrigin(p, nGuess, d);
        if (!tracer.intersect(o, d, kFloatMax, hit)) return false;
        if (!buildSurfaceInteraction(scene, hit, o, d, si)) return false;
    }
    Material mat = si.materialIndex >= 0 && si.materialIndex < scene.materialCount
                       ? scene.materials[si.materialIndex]
                       : defaultMaterial();
    mat = evaluateTexturedMaterial(scene, mat, si.uv, si.ns, si.pObject, si.nObject, si.uvFilterWidth);
    if (!isSpecularCausticMaterial(mat)) return false;
    sv.p = si.p;
    sv.n = si.ns;
    sv.mat = mat;
    sv.refractive = computeLobes(mat).transmission > 0.5f;
    return true;
}

template <typename Tracer>
SR_INL bool solveOneBounce(const SceneView& scene, const Tracer& tracer, Vec3 p, Vec3 y, SpecVertex& sv) {
    Vec3 x = sv.p;
    Vec3 n = sv.n;
    for (int iter = 0; iter < kNewtonIters; ++iter) {
        const Vec3 C = halfVectorConstraint(p, y, x, n, sv.mat, sv.refractive);
        if (length(C) < 1e-4f) {
            sv.p = x;
            sv.n = n;
            return true;
        }
        const Frame f(n);
        const Vec3 t1 = f.t;
        const Vec3 t2 = f.b;
        const float h = 5e-4f * srMax(1.0f, length(y - p) * 0.01f);
        const Vec3 Cp = halfVectorConstraint(p, y, x + t1 * h, n, sv.mat, sv.refractive);
        const Vec3 Cq = halfVectorConstraint(p, y, x + t2 * h, n, sv.mat, sv.refractive);
        const float c1 = dot(C, t1);
        const float c2 = dot(C, t2);
        const float j11 = dot(Cp - C, t1) / h;
        const float j12 = dot(Cq - C, t1) / h;
        const float j21 = dot(Cp - C, t2) / h;
        const float j22 = dot(Cq - C, t2) / h;
        const float det = j11 * j22 - j12 * j21;
        float da = 0.0f, db = 0.0f;
        if (fabsf(det) > 1e-10f) {
            da = (j22 * c1 - j12 * c2) / det;
            db = (-j21 * c1 + j11 * c2) / det;
        } else {
            da = c1 * 0.5f;
            db = c2 * 0.5f;
        }
        da = clampf(da, -0.05f, 0.05f);
        db = clampf(db, -0.05f, 0.05f);
        SpecVertex projected = sv;
        if (!reprojectOntoSpecular(scene, tracer, p, x - t1 * da - t2 * db, n, projected)) return false;
        x = projected.p;
        n = projected.n;
        sv = projected;
    }
    if (length(halfVectorConstraint(p, y, x, n, sv.mat, sv.refractive)) < 2.5e-3f) {
        sv.p = x;
        sv.n = n;
        return true;
    }
    return false;
}

template <typename Tracer>
SR_INL bool clearSegment(const SceneView& scene, const Tracer& tracer, Vec3 a, Vec3 na, Vec3 b) {
    Vec3 d = b - a;
    const float dist = length(d);
    if (dist < 1e-5f) return false;
    d = d / dist;
    const Vec3 o = offsetRayOrigin(a, na, d);
    RayHit hit;
    if (!tracer.intersect(o, d, dist * (1.0f - 1e-3f), hit)) return true;
    SurfaceInteraction si;
    if (!buildSurfaceInteraction(scene, hit, o, d, si)) return false;
    return si.lightIndex >= 0;
}

// MNEE from a mostly-diffuse shading point through one specular interface to a light point.
template <typename Tracer>
SR_INL Vec3 manifoldNeeOnce(const SceneView& scene, const Tracer& tracer, Vec3 p, Vec3 n, Vec3 wo,
                            const Material& shadingMat, Rng& rng) {
    // Only useful on surfaces that can receive a caustic (have a non-delta lobe).
    const LobeWeights shadeLw = computeLobes(shadingMat);
    if (shadeLw.diffuse < 1e-4f && shadeLw.delta) return Vec3(0.0f);

    int lightIndex = -1;
    Vec3 y, yN, Le;
    float pdfPos = 0.0f;
    if (!sampleLightPoint(scene, rng, lightIndex, y, yN, Le, pdfPos) || pdfPos <= 0.0f) return Vec3(0.0f);

    Vec3 dir = y - p;
    const float distPy = length(dir);
    if (distPy < 1e-4f) return Vec3(0.0f);
    dir = dir / distPy;

    const Vec3 o0 = offsetRayOrigin(p, n, dir);
    RayHit hit;
    if (!tracer.intersect(o0, dir, distPy * (1.0f - 1e-3f), hit)) return Vec3(0.0f);  // clear → regular NEE
    SurfaceInteraction si;
    if (!buildSurfaceInteraction(scene, hit, o0, dir, si)) return Vec3(0.0f);
    if (si.lightIndex >= 0) return Vec3(0.0f);

    Material mat = si.materialIndex >= 0 && si.materialIndex < scene.materialCount
                       ? scene.materials[si.materialIndex]
                       : defaultMaterial();
    mat = evaluateTexturedMaterial(scene, mat, si.uv, si.ns, si.pObject, si.nObject, si.uvFilterWidth);
    if (!isSpecularCausticMaterial(mat)) return Vec3(0.0f);

    SpecVertex sv;
    sv.p = si.p;
    sv.n = si.ns;
    sv.mat = mat;
    sv.refractive = computeLobes(mat).transmission > 0.5f;
    if (sv.refractive && !allowsRefractiveCaustics(mat)) return Vec3(0.0f);
    if (!sv.refractive && !allowsReflectiveCaustics(mat)) return Vec3(0.0f);

    if (!solveOneBounce(scene, tracer, p, y, sv)) return Vec3(0.0f);
    if (!clearSegment(scene, tracer, p, n, sv.p)) return Vec3(0.0f);
    if (length(halfVectorConstraint(p, y, sv.p, sv.n, sv.mat, sv.refractive)) > 2.5e-3f) return Vec3(0.0f);

    const Vec3 wToLight = normalize(y - sv.p);
    const Vec3 wToSpec = normalize(sv.p - p);
    const LightData& l = scene.lights[lightIndex];
    if (l.shadowEnable) {
        const Vec3 ox = offsetRayOrigin(sv.p, sv.n, wToLight);
        const float dist = length(y - sv.p);
        RayHit h2;
        if (tracer.intersect(ox, wToLight, dist * (1.0f - 1e-3f), h2)) {
            SurfaceInteraction si2;
            if (!buildSurfaceInteraction(scene, h2, ox, wToLight, si2) || si2.lightIndex != lightIndex)
                return Vec3(0.0f);
        }
    }
    if (l.type != kLightPoint && l.type != kLightSphere) {
        if (dot(yN, -wToLight) <= 0.0f && !l.twoSided) return Vec3(0.0f);
    }

    const Frame frame(n);
    const BsdfEval be = bsdfEvalLocal(shadingMat, frame.toLocal(wo), frame.toLocal(wToSpec));
    if (be.pdf <= 0.0f || isBlack(be.f)) return Vec3(0.0f);

    const LobeWeights lw = computeLobes(sv.mat);
    const Vec3 wToP = normalize(p - sv.p);
    Vec3 fs(0.0f);
    if (sv.refractive) {
        const float eta = dot(sv.n, wToP) > 0.0f ? lw.eta : 1.0f / lw.eta;
        const float fr = fresnelDielectric(fabsf(dot(sv.n, wToP)), lw.eta);
        fs = lw.transmissionTint * ((1.0f - fr) / srMax(1e-4f, eta * eta));
    } else {
        fs = fresnelSchlick(lw.f0, fabsf(dot(sv.n, wToP)));
    }
    if (isBlack(fs)) return Vec3(0.0f);

    const float cosP = fabsf(dot(n, wToSpec));
    const float dist2Px = lengthSquared(sv.p - p);
    if (dist2Px < 1e-12f) return Vec3(0.0f);

    Vec3 lightContrib = Le;
    float geomLight = 1.0f;
    if (l.type == kLightPoint) {
        const float dist2 = lengthSquared(y - sv.p);
        lightContrib = Le / srMax(1e-8f, dist2);
        geomLight = fabsf(dot(sv.n, wToLight));
    } else {
        const float G2 = geometryTerm(sv.p, sv.n, y, yN);
        if (G2 <= 0.0f) return Vec3(0.0f);
        geomLight = G2;
    }

    // f_p * cos / |x-p|^2 * |cos at x toward p| * fs * light
    const float cosX = fabsf(dot(sv.n, -wToSpec));
    Vec3 result =
        be.f * (cosP * cosX / dist2Px) * fs * lightContrib * geomLight / srMax(1e-8f, pdfPos);
    // Conservative weight vs unidirectional strategies.
    result = result * 0.5f;
    if (!isFinite(result)) return Vec3(0.0f);
    return vmax(Vec3(0.0f), result);
}

template <typename Tracer>
SR_INL Vec3 manifoldNee(const SceneView& scene, const Tracer& tracer, Vec3 p, Vec3 n, Vec3 wo,
                        const Material& shadingMat, Rng& rng) {
    const int nSamples = srMax(1, scene.settings.lightSamples);
    Vec3 sum(0.0f);
    for (int i = 0; i < nSamples; ++i)
        sum += manifoldNeeOnce(scene, tracer, p, n, wo, shadingMat, rng);
    return sum * (1.0f / float(nSamples));
}

}  // namespace mnee

// Path tracer with MNEE caustic connections at non-specular vertices.
template <typename Tracer, typename Guiding>
SR_INL Vec3 traceRadianceMnee(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                              Rng& rng, Guiding* guiding) {
    Vec3 radiance(0.0f);
    Vec3 throughput(1.0f);
    float bsdfPdf = 0.0f;
    bool specularBounce = true;
    bool suppressCausticLight = false;
    int depth = 0;
    int passThrough = 0;
    const RenderSettingsData& settings = scene.settings;
    const int maxDepth = settings.integrator == kIntegratorDirectLighting ? 1 : srMax(1, settings.maxDepth);

    while (depth <= maxDepth) {
        RayHit hit;
        const bool didHit = tracer.intersect(origin, direction, kFloatMax, hit);

        if (!didHit) {
            if (scene.domeLightIndex >= 0) {
                if (!(suppressCausticLight && !specularBounce)) {
                    const LightData& dome = scene.lights[scene.domeLightIndex];
                    const bool primary = depth == 0 && passThrough == 0;
                    if (!(primary && (!settings.envVisibleCamera || !dome.visibleCamera))) {
                        Vec3 envL = domeRadiance(scene, dome, direction);
                        if (!isBlack(envL)) {
                            float weight = 1.0f;
                            if (!specularBounce) {
                                const float lp = lightPdfDirection(scene, scene.domeLightIndex, origin, direction,
                                                                   origin, direction) *
                                                 lightSelectionPdfIndex(scene, scene.domeLightIndex);
                                weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                            }
                            Vec3 contrib = throughput * envL * weight;
                            if (depth > 0 && !specularBounce)
                                contrib = clampContribution(contrib, settings.clampIndirect);
                            radiance += contrib;
#if !defined(__CUDACC__)
                            if (guiding && guiding->active())
                                guiding->recordBackground(origin, direction, envL, weight);
#endif
                        }
                    }
                }
            }
            break;
        }

        SurfaceInteraction si;
        if (!buildSurfaceInteraction(scene, hit, origin, direction, si)) break;
        const InstanceData& inst = scene.instances[si.instanceIndex];

        if (si.lightIndex >= 0 && depth == 0 && !inst.visibleCamera) {
            origin = offsetRayOrigin(si.p, si.ng, direction);
            ++passThrough;
            if (passThrough > 16) break;
            continue;
        }

        if (si.lightIndex >= 0) {
            if (suppressCausticLight && !specularBounce) break;
            const LightData& light = scene.lights[si.lightIndex];
            const Vec3 lightN = light.type == kLightSphere ? si.ng : areaLightNormal(light);
            Vec3 emitted = areaLightEmission(scene, light, direction, lightN);
            if (!isBlack(emitted)) {
                float weight = 1.0f;
                if (!specularBounce) {
                    const float lp = lightPdfDirection(scene, si.lightIndex, origin, direction, si.p, lightN) *
                                     lightSelectionPdfIndex(scene, si.lightIndex);
                    weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                }
                Vec3 contrib = throughput * emitted * weight;
                if (depth > 0 && !specularBounce)
                    contrib = clampContribution(contrib, settings.clampIndirect);
                radiance += contrib;
#if !defined(__CUDACC__)
                if (guiding && guiding->active())
                    guiding->recordLightHit(si.p, -direction, emitted, weight);
#endif
            }
            break;
        }

        Material baseMat = si.materialIndex >= 0 && si.materialIndex < scene.materialCount
                               ? scene.materials[si.materialIndex]
                               : defaultMaterial();
        Material mat =
            evaluateTexturedMaterial(scene, baseMat, si.uv, si.ns, si.pObject, si.nObject, si.uvFilterWidth);

        if (mat.transmission <= 0.0f && mat.doubleSided && dot(si.ns, -direction) < 0.0f) {
            si.ns = -si.ns;
            si.ng = -si.ng;
        }

        // Emissive surfaces.
        if (mat.emissionStrength > 0.0f && !isBlack(mat.emissionColor)) {
            const bool frontFacing = dot(si.ns, -direction) > 0.0f;
            if (frontFacing || mat.doubleSided)
                radiance += throughput * mat.emissionColor * mat.emissionStrength;
        }

        // Stochastic opacity / cutout.
        if (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity) {
            origin = offsetRayOrigin(si.p, si.ng, direction);
            ++passThrough;
            if (passThrough > 32) break;
            continue;
        }

        if (settings.integrator == kIntegratorAmbientOcclusion) {
            const Frame frame(dot(si.ns, -direction) < 0.0f ? -si.ns : si.ns);
            const Vec3 wi = frame.toWorld(sampleCosineHemisphere(rng.nextFloat(), rng.nextFloat()));
            const Vec3 aoOrigin = offsetRayOrigin(si.p, si.ng, wi);
            const float dist = settings.aoDistance > 0.0f ? settings.aoDistance : kFloatMax;
            const float visibility = tracer.occluded(aoOrigin, wi, dist) ? 0.0f : 1.0f;
            radiance += throughput * Vec3(visibility);
            break;
        }

        if (depth >= maxDepth) break;

        const Frame frame(si.ns);
        const Vec3 wo = -direction;
        const LobeWeights lw = computeLobes(mat);

#if !defined(__CUDACC__)
        bool guideReady = false;
        if (guiding && guiding->active() && !lw.delta) {
            guiding->beginSegment(si.p, wo);
            guideReady = guiding->prepare(si.p, si.ns, rng);
        }
#else
        const bool guideReady = false;
        (void)guiding;
#endif

        // Regular NEE (opaque shadows — glass blocks; caustics via MNEE / BSDF).
        if (!(suppressCausticLight && !specularBounce) && lw.diffuse + (lw.delta ? 0.0f : lw.specular) > 1e-5f) {
            const Vec3 nee = nextEventEstimation(scene, tracer, si, mat, frame, wo, rng, guiding);
            Vec3 contrib = throughput * nee;
            if (depth > 0 && !specularBounce) contrib = clampContribution(contrib, settings.clampIndirect);
            radiance += contrib;
#if !defined(__CUDACC__)
            if (guiding && guiding->active()) guiding->addScattered(nee);
#endif

            // MNEE: manifold connection through specular blockers between this point and lights.
            if (!lw.delta || lw.diffuse > 1e-4f) {
                Vec3 mneeL = mnee::manifoldNee(scene, tracer, si.p, si.ns, wo, mat, rng);
                mneeL = clampContribution(mneeL, settings.clampIndirect > 0 ? settings.clampIndirect : 50.0f);
                radiance += throughput * mneeL;
#if !defined(__CUDACC__)
                if (guiding && guiding->active()) guiding->addScattered(mneeL);
#endif
            }
        }

        const Vec3 woLocal = frame.toLocal(wo);
        BsdfSample bs;
        bool gotSample = false;
#if !defined(__CUDACC__)
        if (guideReady) {
            const float pg = guiding->guideProbability();
            if (rng.nextFloat() < pg) {
                Vec3 wiWorld;
                float gPdf = 0.0f;
                if (guiding->sample(rng.nextFloat(), rng.nextFloat(), wiWorld, gPdf) && gPdf > 0.0f) {
                    const Vec3 wiLocal = frame.toLocal(wiWorld);
                    const BsdfEval be = bsdfEvalLocal(mat, woLocal, wiLocal);
                    if (be.pdf > 0.0f && !isBlack(be.f)) {
                        const float mixPdf = pg * gPdf + (1.0f - pg) * be.pdf;
                        if (mixPdf > 0.0f) {
                            bs.wi = wiLocal;
                            bs.pdf = mixPdf;
                            bs.weight = be.f * (fabsf(wiLocal.z) / mixPdf);
                            bs.specular = false;
                            bs.transmitted = wiLocal.z < 0.0f;
                            gotSample = true;
                        }
                    }
                }
            }
        }
#endif
        if (!gotSample) {
            bs = bsdfSampleLocal(mat, woLocal, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                 rng.nextFloat());
#if !defined(__CUDACC__)
            if (bs.pdf > 0.0f && guideReady) {
                const float pg = guiding->guideProbability();
                const float gPdf = guiding->pdf(normalize(frame.toWorld(bs.wi)));
                const float mixPdf = pg * gPdf + (1.0f - pg) * bs.pdf;
                if (mixPdf > 0.0f) {
                    bs.weight *= bs.pdf / mixPdf;
                    bs.pdf = mixPdf;
                }
            }
#endif
        }
        if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;

        // Caustic gates on specular continuation.
        if (bs.transmitted && mat.refractiveCaustics == 0) {
            suppressCausticLight = true;
        } else if (bs.specular && !bs.transmitted && mat.reflectiveCaustics == 0) {
            suppressCausticLight = true;
        }

        Vec3 weight = bs.weight;
        if (settings.clampIndirect > 0.0f) {
            const float m = maxComponent(weight);
            if (m > settings.clampIndirect) weight *= settings.clampIndirect / m;
        }

        const Vec3 wiWorld = normalize(frame.toWorld(bs.wi));
#if !defined(__CUDACC__)
        if (guiding && guiding->active())
            guiding->recordBounce(si.ns, wiWorld, bs.pdf, weight, bs.specular, mat.roughness, lw.eta, 1.0f);
#endif

        throughput *= weight;
        if (!isFinite(throughput) || isBlack(throughput)) break;

        origin = offsetRayOrigin(si.p, si.ng, wiWorld);
        direction = wiWorld;
        bsdfPdf = bs.pdf;
        specularBounce = bs.specular;
        ++depth;

        if (depth >= srMax(1, settings.rrStartDepth)) {
            const float q = clampf(maxComponent(throughput), 0.05f, 1.0f);
            if (rng.nextFloat() > q) break;
#if !defined(__CUDACC__)
            if (guiding && guiding->active()) guiding->setRussianRoulette(q);
#endif
            throughput /= q;
        }
    }

    if (!isFinite(radiance)) return Vec3(0.0f);
    return radiance;
}

template <typename Tracer>
SR_INL Vec3 traceRadianceMnee(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                              Rng& rng) {
    return traceRadianceMnee<Tracer, NullGuiding>(scene, tracer, origin, direction, rng, nullptr);
}

}  // namespace sol
