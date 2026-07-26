// Backend agnostic path tracing kernel.
//
// The integrator is templated on a Tracer that provides
//     bool intersect(Vec3 origin, Vec3 dir, float tMax, RayHit& hit) const;
//     bool occluded(Vec3 origin, Vec3 dir, float tMax) const;
// so the Embree and OptiX backends run the exact same light transport code.
#pragma once

#include "core/rng.h"
#include "render/lights.h"
#include "render/shading.h"
#include "scene/types.h"
#include "solstice_config.h"

#if !defined(__CUDACC__) && SOLSTICE_HAVE_OPENPGL
#include "render/cpu/path_guiding.h"
#endif

namespace sol {

struct RayHit {
    float t = kFloatMax;
    int instanceIndex = -1;
    uint32_t primIndex = 0;
    float u = 0.0f;
    float v = 0.0f;
};

struct SurfaceInteraction {
    Vec3 p{0.0f, 0.0f, 0.0f};
    Vec3 ng{0.0f, 0.0f, 1.0f};  // geometric normal
    Vec3 ns{0.0f, 0.0f, 1.0f};  // shading normal
    Vec2 uv{0.0f, 0.0f};
    int instanceIndex = -1;
    int materialIndex = -1;
    int lightIndex = -1;
};

SR_INL SR_HD Material defaultMaterial() {
    Material m;
    m.baseColor = Vec3(0.7f, 0.7f, 0.7f);
    m.roughness = 0.5f;
    return m;
}

// Chiang et al. 2016: map artist multiple-scattering albedo A → single-scattering α.
SR_INL SR_HD float chiangSingleScatterAlbedo(float A) {
    A = saturatef(A);
    return 1.0f - expf(-5.09406f * A + 2.61188f * A * A - 4.31805f * A * A * A);
}

SR_INL SR_HD Vec3 chiangSingleScatterAlbedo(Vec3 A) {
    return Vec3(chiangSingleScatterAlbedo(A.x), chiangSingleScatterAlbedo(A.y),
                chiangSingleScatterAlbedo(A.z));
}

SR_INL SR_HD float vecChannel(Vec3 v, int ch) {
    return ch == 0 ? v.x : (ch == 1 ? v.y : v.z);
}

SR_INL SR_HD Vec3 channelMask(int ch, float value) {
    return ch == 0 ? Vec3(value, 0.0f, 0.0f) : (ch == 1 ? Vec3(0.0f, value, 0.0f) : Vec3(0.0f, 0.0f, value));
}

// Reconstruct shading attributes from a hit record.
SR_INL SR_HD bool buildSurfaceInteraction(const SceneView& scene, const RayHit& hit, Vec3 origin, Vec3 dir,
                                          SurfaceInteraction& si) {
    if (hit.instanceIndex < 0 || hit.instanceIndex >= scene.instanceCount) return false;
    const InstanceData& inst = scene.instances[hit.instanceIndex];
    if (inst.meshIndex < 0 || inst.meshIndex >= scene.meshCount) return false;
    const MeshView& mesh = scene.meshes[inst.meshIndex];
    if (hit.primIndex >= mesh.triangleCount) return false;

    const uint32_t i0 = mesh.indices[hit.primIndex * 3 + 0];
    const uint32_t i1 = mesh.indices[hit.primIndex * 3 + 1];
    const uint32_t i2 = mesh.indices[hit.primIndex * 3 + 2];
    const Vec3 p0 = mesh.positions[i0];
    const Vec3 p1 = mesh.positions[i1];
    const Vec3 p2 = mesh.positions[i2];
    const float w = 1.0f - hit.u - hit.v;

    const Vec3 pLocal = p0 * w + p1 * hit.u + p2 * hit.v;
    si.p = transformPoint(inst.xform, pLocal);
    // The hit distance is authoritative for ray offsets.
    si.p = origin + dir * hit.t;

    Vec3 ngLocal = cross(p1 - p0, p2 - p0);
    si.ng = normalize(transformNormalWithInverse(inst.xformInv, ngLocal));

    if (mesh.normals) {
        const Vec3 nLocal = mesh.normals[i0] * w + mesh.normals[i1] * hit.u + mesh.normals[i2] * hit.v;
        Vec3 ns = transformNormalWithInverse(inst.xformInv, nLocal);
        si.ns = lengthSquared(ns) > 0.0f ? normalize(ns) : si.ng;
    } else {
        si.ns = si.ng;
    }
    if (mesh.uvs) {
        const Vec2 uv0 = mesh.uvs[i0], uv1 = mesh.uvs[i1], uv2 = mesh.uvs[i2];
        si.uv = Vec2(uv0.x * w + uv1.x * hit.u + uv2.x * hit.v, uv0.y * w + uv1.y * hit.u + uv2.y * hit.v);
    }
    si.instanceIndex = hit.instanceIndex;
    si.materialIndex = inst.materialIndex;
    si.lightIndex = inst.lightIndex;

    // Keep the shading normal on the same side as the geometric normal.
    if (dot(si.ns, si.ng) < 0.0f) si.ng = -si.ng;
    return true;
}

SR_INL SR_HD Vec3 offsetRayOrigin(Vec3 p, Vec3 n, Vec3 dir) {
    const float scale = 1.0f + srMax(fabsf(p.x), srMax(fabsf(p.y), fabsf(p.z)));
    const Vec3 offset = n * (kRayEpsilon * scale);
    return dot(dir, n) > 0.0f ? p + offset : p - offset;
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
SR_INL SR_HD void generateCameraRay(const SceneView& scene, float pixelX, float pixelY, float lensU, float lensV,
                                    Vec3& origin, Vec3& direction) {
    const CameraData& cam = scene.camera;
    const float resX = float(srMax(1, scene.settings.resolutionX));
    const float resY = float(srMax(1, scene.settings.resolutionY));
    const float sensorHeight = cam.sensorWidth * (resY / resX);
    const float sx = (pixelX / resX - 0.5f) * cam.sensorWidth;
    const float sy = (0.5f - pixelY / resY) * sensorHeight;

    Vec3 dirCam = normalize(Vec3(sx, sy, -srMax(1e-3f, cam.focalLength)));
    Vec3 originCam(0.0f, 0.0f, 0.0f);

    if (cam.fStop > 0.0f) {
        // Scene units are assumed to be metres, focal length is in millimetres.
        const float lensRadius = (cam.focalLength * 0.001f) / (2.0f * cam.fStop);
        const Vec2 lens = sampleConcentricDisk(lensU, lensV) * lensRadius;
        const float ft = srMax(1e-4f, cam.focusDistance) / srMax(1e-6f, -dirCam.z);
        const Vec3 focusPoint = dirCam * ft;
        originCam = Vec3(lens.x, lens.y, 0.0f);
        dirCam = normalize(focusPoint - originCam);
    }

    origin = transformPoint(cam.cameraToWorld, originCam);
    direction = normalize(transformVector(cam.cameraToWorld, dirCam));
}

// ---------------------------------------------------------------------------
// Path tracing
// ---------------------------------------------------------------------------

// Default no-op guiding hooks (OptiX / builds without OpenPGL).
struct NullGuiding {
    SR_INL SR_HD bool active() const { return false; }
    SR_INL SR_HD float guideProbability() const { return 0.0f; }
    SR_INL SR_HD bool prepared() const { return false; }
    SR_INL SR_HD bool prepare(Vec3, Vec3, Rng&) { return false; }
    SR_INL SR_HD float pdf(Vec3) const { return 0.0f; }
    SR_INL SR_HD bool sample(float, float, Vec3&, float&) const { return false; }
    SR_INL SR_HD void beginSegment(Vec3, Vec3) {}
    SR_INL SR_HD void recordEmission(Vec3, float) {}
    SR_INL SR_HD void addScattered(Vec3) {}
    SR_INL SR_HD void recordBounce(Vec3, Vec3, float, Vec3, bool, float, float, float) {}
    SR_INL SR_HD void setRussianRoulette(float) {}
    SR_INL SR_HD void recordBackground(Vec3, Vec3, Vec3, float) {}
    SR_INL SR_HD void recordLightHit(Vec3, Vec3, Vec3, float) {}
};

template <typename Tracer, typename Guiding>
SR_INL SR_HD Vec3 nextEventEstimation(const SceneView& scene, const Tracer& tracer, const SurfaceInteraction& si,
                                      const Material& mat, const Frame& frame, Vec3 wo, Rng& rng,
                                      Guiding* guiding) {
    Vec3 result(0.0f);
    if (scene.lightCount <= 0) return result;

    const float selectPdf = lightSelectionPdf(scene);
    int lightIndex = int(rng.nextFloat() * float(scene.lightCount));
    if (lightIndex >= scene.lightCount) lightIndex = scene.lightCount - 1;

    LightSample ls;
    if (!sampleLight(scene, lightIndex, si.p, rng.nextFloat(), rng.nextFloat(), ls)) return result;
    if (ls.pdf <= 0.0f || isBlack(ls.radiance)) return result;
    if (scene.lights[lightIndex].shadowEnable) {
        const Vec3 shadowOrigin = offsetRayOrigin(si.p, si.ng, ls.wi);
        // Use a large finite tMax for distant/dome lights — Embree is more stable
        // with that than with FLT_MAX, and it still reaches any scene geometry.
        float tMax = 1.0e8f;
        if (ls.distance < 1.0e7f) tMax = ls.distance * (1.0f - 1e-3f);
        if (tracer.occluded(shadowOrigin, ls.wi, tMax)) return result;
    }

    const Vec3 woLocal = frame.toLocal(wo);
    const Vec3 wiLocal = frame.toLocal(ls.wi);
    const BsdfEval be = bsdfEvalLocal(mat, woLocal, wiLocal);
    if (be.pdf <= 0.0f || isBlack(be.f)) return result;

    float scatterPdf = be.pdf;
#if !defined(__CUDACC__)
    if (guiding && guiding->active() && guiding->prepared()) {
        const float pg = guiding->guideProbability();
        const float gPdf = guiding->pdf(ls.wi);
        scatterPdf = pg * gPdf + (1.0f - pg) * be.pdf;
    }
#else
    (void)guiding;
#endif

    const float lightPdf = ls.pdf * selectPdf;
    const float misWeight = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, scatterPdf);
    result = ls.radiance * be.f * (fabsf(wiLocal.z) * misWeight / lightPdf);
    return result;
}

template <typename Tracer>
SR_INL SR_HD Vec3 nextEventEstimation(const SceneView& scene, const Tracer& tracer, const SurfaceInteraction& si,
                                      const Material& mat, const Frame& frame, Vec3 wo, Rng& rng) {
    return nextEventEstimation<Tracer, NullGuiding>(scene, tracer, si, mat, frame, wo, rng, nullptr);
}

template <typename Tracer, typename Guiding>
SR_INL SR_HD Vec3 traceRadiance(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                                Rng& rng, Guiding* guiding) {
    Vec3 radiance(0.0f);
    Vec3 throughput(1.0f);
    float bsdfPdf = 0.0f;
    bool specularBounce = true;   // primary rays behave like a specular bounce for MIS
    int depth = 0;
    int passThrough = 0;
    const RenderSettingsData& settings = scene.settings;
    const int maxDepth = settings.integrator == kIntegratorDirectLighting ? 1 : srMax(1, settings.maxDepth);

    while (depth <= maxDepth) {
        RayHit hit;
        const bool didHit = tracer.intersect(origin, direction, kFloatMax, hit);

        if (!didHit) {
            if (scene.domeLightIndex >= 0) {
                const LightData& dome = scene.lights[scene.domeLightIndex];
                const bool primary = depth == 0 && passThrough == 0;
                if (!(primary && (!settings.envVisibleCamera || !dome.visibleCamera))) {
                    Vec3 envL = domeRadiance(scene, dome, direction);
                    if (!isBlack(envL)) {
                        float weight = 1.0f;
                        if (!specularBounce) {
                            const float lp = lightPdfDirection(scene, scene.domeLightIndex, origin, direction,
                                                               origin, direction) *
                                             lightSelectionPdf(scene);
                            weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                        }
                        radiance += throughput * envL * weight;
#if !defined(__CUDACC__)
                        if (guiding && guiding->active())
                            guiding->recordBackground(origin, direction, envL, weight);
#else
                        (void)guiding;
#endif
                    }
                }
            }
            break;
        }

        SurfaceInteraction si;
        if (!buildSurfaceInteraction(scene, hit, origin, direction, si)) break;

        const InstanceData& inst = scene.instances[si.instanceIndex];

        // Lights that are hidden from the camera let primary rays pass through.
        if (si.lightIndex >= 0 && depth == 0 && !inst.visibleCamera) {
            origin = offsetRayOrigin(si.p, si.ng, direction);
            ++passThrough;
            if (passThrough > 16) break;
            continue;
        }

        // Emission from area light geometry.
        if (si.lightIndex >= 0) {
            const LightData& light = scene.lights[si.lightIndex];
            const Vec3 lightN = light.type == kLightSphere ? si.ng : areaLightNormal(light);
            Vec3 emitted = areaLightEmission(scene, light, direction, lightN);
            if (!isBlack(emitted)) {
                float weight = 1.0f;
                if (!specularBounce) {
                    const float lp = lightPdfDirection(scene, si.lightIndex, origin, direction, si.p, lightN) *
                                     lightSelectionPdf(scene);
                    weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                }
                radiance += throughput * emitted * weight;
#if !defined(__CUDACC__)
                if (guiding && guiding->active())
                    guiding->recordLightHit(si.p, -direction, emitted, weight);
#endif
            }
            // Light geometry is opaque and is not shaded further.
            break;
        }

        Material baseMat = si.materialIndex >= 0 && si.materialIndex < scene.materialCount
                               ? scene.materials[si.materialIndex]
                               : defaultMaterial();
        Material mat = evaluateTexturedMaterial(scene, baseMat, si.uv, si.ns);

        // Two sided shading for opaque surfaces. Winding order varies between
        // DCCs, so back faces are shaded as if their normals pointed at us.
        if (mat.transmission <= 0.0f && mat.doubleSided && dot(si.ns, -direction) < 0.0f) {
            si.ns = -si.ns;
            si.ng = -si.ng;
        }

        // Emissive surfaces (evaluated before opacity cutouts so glowing cutouts work).
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

        const Vec3 wo = -direction;
        const Frame frame(si.ns);
#if !defined(__CUDACC__)
        if (guiding && guiding->active()) {
            guiding->beginSegment(si.p, wo);
            if (mat.emissionStrength > 0.0f && !isBlack(mat.emissionColor)) {
                const bool frontFacing = dot(si.ns, wo) > 0.0f;
                if (frontFacing || mat.doubleSided)
                    guiding->recordEmission(mat.emissionColor * mat.emissionStrength, 1.0f);
            }
        }
#endif

        // Arnold-style random-walk SSS with spectral (RGB) radius.
        //
        // Layering (Standard Surface): specular sits ON TOP of the SSS body.
        // With subsurface=1 the old code skipped specular entirely — and delta
        // mirrors (roughness≈0) also get zero from NEE — so reflections vanished.
        // Fix: Fresnel RR at entry chooses specular bounce vs SSS body; rough
        // specular also gets direct NEE at the entry point.
        //
        // Spectral walk: hero RGB channel, MFP = scale * radius[ch], Chiang α.
        const float sssWeight = saturatef(mat.subsurface);
        const bool canSss =
            sssWeight > 1e-4f && mat.transmission <= 1e-4f && mat.metallic < 0.999f;
        if (canSss && rng.nextFloat() < sssWeight) {
            throughput /= sssWeight;

            // Specular-only material at the ENTRY (dielectric F0 / metal base).
            Material specMat = mat;
            specMat.subsurface = 0.0f;
            specMat.transmission = 0.0f;
            if (specMat.metallic < 0.999f) specMat.baseColor = Vec3(0.0f);

            const Vec3 woLocalEntry = frame.toLocal(wo);
            const LobeWeights specLw = computeLobes(specMat);
            const float cosWo = srMax(0.0f, woLocalEntry.z);
            const float fresnelEst = average(fresnelSchlick(specLw.f0, cosWo));
            // Probability of taking the specular bounce this path (layer over SSS).
            float pSpec = 0.0f;
            if (specLw.specular > 1e-5f && saturatef(mat.specular) > 1e-5f)
                pSpec = clampf(srMax(fresnelEst, specLw.specular * fresnelEst), 0.0f, 0.98f);

            // Direct specular lighting at entry (works for rough GGX; delta → 0).
            if (pSpec > 0.0f) {
                const Vec3 nee =
                    nextEventEstimation(scene, tracer, si, specMat, frame, wo, rng, guiding);
                radiance += throughput * nee;
#if !defined(__CUDACC__)
                if (guiding && guiding->active()) guiding->addScattered(nee);
#endif
            }

            // Fresnel lottery: reflect at entry OR enter the SSS body.
            if (pSpec > 0.0f && rng.nextFloat() < pSpec) {
                throughput /= pSpec;
                // Force the specular lobe (uLobe in specular range).
                const float uSpec = specLw.diffuse + specLw.specular * rng.nextFloat();
                const BsdfSample specBs =
                    bsdfSampleLocal(specMat, woLocalEntry, uSpec, rng.nextFloat(), rng.nextFloat(),
                                    rng.nextFloat());
                if (specBs.pdf > 0.0f && !isBlack(specBs.weight)) {
                    const Vec3 wiWorld = normalize(frame.toWorld(specBs.wi));
#if !defined(__CUDACC__)
                    if (guiding && guiding->active())
                        guiding->recordBounce(si.ns, wiWorld, specBs.pdf, specBs.weight, true,
                                              mat.roughness, computeLobes(specMat).eta, 1.0f);
#endif
                    throughput *= specBs.weight;
                    origin = offsetRayOrigin(si.p, si.ng, wiWorld);
                    direction = wiWorld;
                    bsdfPdf = specBs.pdf;
                    specularBounce = specBs.specular;
                    ++depth;
                    continue;
                }
                break;
            }
            if (pSpec > 0.0f && pSpec < 0.999f) throughput /= (1.0f - pSpec);

            const Vec3 mfpRGB = vmax(Vec3(0.0f), mat.subsurfaceRadius) * srMax(0.0f, mat.subsurfaceScale);
            const Vec3 multiAlbedo = vmax(Vec3(0.0f), mat.subsurfaceColor);
            const Vec3 singleAlbedo = chiangSingleScatterAlbedo(multiAlbedo);

            // White Lambert at exit — spectral colour is carried by pathWeight.
            Material lambert = defaultMaterial();
            lambert.baseColor = Vec3(1.0f);
            lambert.specular = 0.0f;
            lambert.metallic = 0.0f;
            lambert.transmission = 0.0f;
            lambert.subsurface = 0.0f;
            lambert.roughness = 1.0f;

            Vec3 exitP = si.p;
            Vec3 exitN = si.ns;
            Vec3 exitWo = wo;
            Vec3 pathWeight = multiAlbedo;  // fallback: full RGB Lambert look
            bool useEntryFallback = maxComponent(mfpRGB) < 1e-8f;

            if (!useEntryFallback) {
                // Hero channel for this entire walk (equal prior; MIS via 1/pCh).
                int ch = int(rng.nextFloat() * 3.0f);
                if (ch > 2) ch = 2;
                constexpr float kPCh = 1.0f / 3.0f;
                const float mfp = srMax(1e-5f, vecChannel(mfpRGB, ch));
                const float alphaCh = srMax(0.0f, vecChannel(singleAlbedo, ch));

                Vec3 pWalk = si.p - si.ns * (kRayEpsilon * (1.0f + length(si.p)));
                Vec3 ssThroughput = channelMask(ch, 1.0f / kPCh);
                bool escaped = false;

                constexpr int kMaxWalkSteps = 16;
                for (int step = 0; step < kMaxWalkSteps; ++step) {
                    const float stepLen = -logf(srMax(1e-6f, 1.0f - rng.nextFloat())) * mfp;

                    Vec3 walkDir;
                    if (step == 0) {
                        const Frame inFrame(-si.ns);
                        walkDir = inFrame.toWorld(sampleCosineHemisphere(rng.nextFloat(), rng.nextFloat()));
                    } else {
                        walkDir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
                    }
                    if (lengthSquared(walkDir) < 1e-12f) continue;
                    walkDir = normalize(walkDir);

                    const float rayEps = kRayEpsilon * (1.0f + length(pWalk));
                    const Vec3 walkOrigin = pWalk + walkDir * rayEps;
                    RayHit walkHit;
                    if (!tracer.intersect(walkOrigin, walkDir, stepLen, walkHit)) {
                        pWalk = walkOrigin + walkDir * stepLen;
                        float w = vecChannel(ssThroughput, ch) * alphaCh;
                        if (w < 1e-5f) {
                            ssThroughput = Vec3(0.0f);
                            break;
                        }
                        if (step >= 4) {
                            const float q = clampf(w, 0.1f, 1.0f);
                            if (rng.nextFloat() > q) {
                                ssThroughput = Vec3(0.0f);
                                break;
                            }
                            w /= q;
                        }
                        ssThroughput = channelMask(ch, w);
                        continue;
                    }

                    SurfaceInteraction walkSi;
                    if (!buildSurfaceInteraction(scene, walkHit, walkOrigin, walkDir, walkSi)) break;

                    escaped = true;
                    exitP = walkSi.p;
                    exitN = walkSi.ns;
                    if (dot(exitN, walkDir) < 0.0f) exitN = -exitN;
                    if (lengthSquared(exitN) < 1e-12f) exitN = walkDir;
                    else exitN = normalize(exitN);
                    pathWeight = ssThroughput;
                    const Vec3 toEntry = si.p - exitP;
                    const float woLen2 = lengthSquared(toEntry);
                    exitWo = woLen2 > 1e-12f ? normalize(toEntry) : exitN;
                    break;
                }

                if (!escaped || isBlack(pathWeight)) useEntryFallback = true;
            }

            if (useEntryFallback) {
                exitP = si.p;
                exitN = si.ns;
                exitWo = wo;
                pathWeight = multiAlbedo;
                lambert.baseColor = Vec3(1.0f);
            }

            if (dot(exitN, exitWo) < 0.0f) exitWo = exitN;
            SurfaceInteraction ssSi = si;
            ssSi.p = exitP;
            ssSi.ns = exitN;
            ssSi.ng = exitN;
            const Frame ssFrame(exitN);
            const Vec3 nee =
                nextEventEstimation(scene, tracer, ssSi, lambert, ssFrame, exitWo, rng, guiding);
            radiance += throughput * pathWeight * nee;
#if !defined(__CUDACC__)
            if (guiding && guiding->active()) guiding->addScattered(pathWeight * nee);
#endif
            const BsdfSample ssBs =
                bsdfSampleLocal(lambert, ssFrame.toLocal(exitWo), rng.nextFloat(), rng.nextFloat(),
                                rng.nextFloat(), rng.nextFloat());
            if (ssBs.pdf > 0.0f && !isBlack(ssBs.weight)) {
                const Vec3 wiWorld = normalize(ssFrame.toWorld(ssBs.wi));
#if !defined(__CUDACC__)
                if (guiding && guiding->active())
                    guiding->recordBounce(exitN, wiWorld, ssBs.pdf, pathWeight * ssBs.weight, false,
                                          1.0f, 1.0f, 1.0f);
#endif
                throughput *= pathWeight * ssBs.weight;
                origin = offsetRayOrigin(exitP, exitN, wiWorld);
                direction = wiWorld;
                bsdfPdf = ssBs.pdf;
                specularBounce = false;
                ++depth;
                continue;
            }
            break;
        }

        // Complementary BRDF path when SSS lottery was available but not chosen.
        if (canSss) throughput /= srMax(1e-4f, 1.0f - sssWeight);

        const LobeWeights lw = computeLobes(mat);
#if !defined(__CUDACC__)
        const bool guideReady =
            guiding && guiding->active() && !lw.delta && guiding->prepare(si.p, si.ns, rng);
#else
        const bool guideReady = false;
#endif

        const Vec3 nee = nextEventEstimation(scene, tracer, si, mat, frame, wo, rng, guiding);
        radiance += throughput * nee;
#if !defined(__CUDACC__)
        if (guiding && guiding->active()) guiding->addScattered(nee);
#endif

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

        Vec3 weight = bs.weight;
        if (settings.clampIndirect > 0.0f && depth > 0) {
            const float m = maxComponent(weight);
            if (m > settings.clampIndirect) weight *= settings.clampIndirect / m;
        }

        const Vec3 wiWorld = normalize(frame.toWorld(bs.wi));
#if !defined(__CUDACC__)
        if (guiding && guiding->active())
            guiding->recordBounce(si.ns, wiWorld, bs.pdf, weight, bs.specular, mat.roughness, lw.eta,
                                  1.0f);
#endif

        throughput *= weight;
        if (!isFinite(throughput) || isBlack(throughput)) break;

        origin = offsetRayOrigin(si.p, si.ng, wiWorld);
        direction = wiWorld;
        bsdfPdf = bs.pdf;
        specularBounce = bs.specular;
        ++depth;

        // Russian roulette.
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
SR_INL SR_HD Vec3 traceRadiance(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                                Rng& rng) {
    return traceRadiance<Tracer, NullGuiding>(scene, tracer, origin, direction, rng, nullptr);
}

}  // namespace sol
