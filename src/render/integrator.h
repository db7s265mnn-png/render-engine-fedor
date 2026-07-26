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
template <typename Tracer>
SR_INL SR_HD Vec3 nextEventEstimation(const SceneView& scene, const Tracer& tracer, const SurfaceInteraction& si,
                                      const Material& mat, const Frame& frame, Vec3 wo, Rng& rng) {
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

    const float lightPdf = ls.pdf * selectPdf;
    const float misWeight = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, be.pdf);
    result = ls.radiance * be.f * (fabsf(wiLocal.z) * misWeight / lightPdf);
    return result;
}

template <typename Tracer>
SR_INL SR_HD Vec3 traceRadiance(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                                Rng& rng) {
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

        // Arnold-style random-walk SSS (Standard Surface default).
        // Specular is a separate surface lobe (roughness only affects that).
        // SSS replaces the diffuse body with weight = subsurface:
        //   MFP (scene units / metres) = subsurface_scale * subsurface_radius
        const float sssWeight = saturatef(mat.subsurface);
        const bool canSss =
            sssWeight > 1e-4f && mat.transmission <= 1e-4f && mat.metallic < 0.999f;
        if (canSss && rng.nextFloat() < sssWeight) {
            throughput /= sssWeight;

            const Vec3 albedo = vmax(Vec3(0.0f), mat.subsurfaceColor);
            const Vec3 mfpRGB = vmax(Vec3(0.0f), mat.subsurfaceRadius) * srMax(0.0f, mat.subsurfaceScale);

            Material lambert = defaultMaterial();
            lambert.baseColor = albedo;
            lambert.specular = 0.0f;
            lambert.metallic = 0.0f;
            lambert.transmission = 0.0f;
            lambert.subsurface = 0.0f;
            lambert.roughness = 1.0f;

            Vec3 exitP = si.p;
            Vec3 exitN = si.ns;
            Vec3 exitWo = wo;
            Vec3 pathWeight(1.0f);
            bool useEntryFallback = maxComponent(mfpRGB) < 1e-8f;

            if (!useEntryFallback) {
                // Start just under the surface along -Ns (into the medium).
                Vec3 pWalk = si.p - si.ns * (kRayEpsilon * (1.0f + length(si.p)));
                Vec3 ssThroughput(1.0f);
                bool escaped = false;

                // Keep walks short: long chains of albedo^N go black and are not useful.
                constexpr int kMaxWalkSteps = 12;
                for (int step = 0; step < kMaxWalkSteps; ++step) {
                    const float channel = rng.nextFloat();
                    float mfp = mfpRGB.y;
                    if (channel < 0.333f) mfp = mfpRGB.x;
                    else if (channel > 0.666f) mfp = mfpRGB.z;
                    mfp = srMax(1e-5f, mfp);

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
                        // No boundary within the free path — scatter inside the volume.
                        pWalk = walkOrigin + walkDir * stepLen;
                        ssThroughput = ssThroughput * albedo;
                        if (maxComponent(ssThroughput) < 1e-4f) break;
                        if (step >= 3) {
                            const float q = clampf(maxComponent(ssThroughput), 0.1f, 1.0f);
                            if (rng.nextFloat() > q) {
                                ssThroughput = Vec3(0.0f);
                                break;
                            }
                            ssThroughput /= q;
                        }
                        continue;
                    }

                    SurfaceInteraction walkSi;
                    if (!buildSurfaceInteraction(scene, walkHit, walkOrigin, walkDir, walkSi)) break;

                    // ANY interface hit ends the walk. Mesh winding varies (DCC exports
                    // often invert normals); a strict back-face-only test left paths
                    // trapped inside → albedo^N → solid black silhouettes.
                    // Orient the exit normal into air (same hemisphere as walkDir).
                    escaped = true;
                    exitP = walkSi.p;
                    exitN = walkSi.ns;
                    if (dot(exitN, walkDir) < 0.0f) exitN = -exitN;
                    if (lengthSquared(exitN) < 1e-12f) exitN = walkDir;
                    else exitN = normalize(exitN);
                    pathWeight = ssThroughput;
                    Vec3 toEntry = si.p - exitP;
                    const float woLen2 = lengthSquared(toEntry);
                    exitWo = woLen2 > 1e-12f ? normalize(toEntry) : exitN;
                    break;
                }

                if (!escaped || isBlack(pathWeight)) {
                    // Failed / absorbed walk: Lambert at entry (never a black hole).
                    useEntryFallback = true;
                }
            }

            if (useEntryFallback) {
                exitP = si.p;
                exitN = si.ns;
                exitWo = wo;
                pathWeight = Vec3(1.0f);
            }

            if (dot(exitN, exitWo) < 0.0f) exitWo = exitN;
            SurfaceInteraction ssSi = si;
            ssSi.p = exitP;
            ssSi.ns = exitN;
            ssSi.ng = exitN;
            const Frame ssFrame(exitN);
            radiance += throughput * pathWeight *
                        nextEventEstimation(scene, tracer, ssSi, lambert, ssFrame, exitWo, rng);
            const BsdfSample ssBs =
                bsdfSampleLocal(lambert, ssFrame.toLocal(exitWo), rng.nextFloat(), rng.nextFloat(),
                                rng.nextFloat(), rng.nextFloat());
            if (ssBs.pdf > 0.0f && !isBlack(ssBs.weight)) {
                throughput *= pathWeight * ssBs.weight;
                origin = offsetRayOrigin(exitP, exitN, ssFrame.toWorld(ssBs.wi));
                direction = normalize(ssFrame.toWorld(ssBs.wi));
                bsdfPdf = ssBs.pdf;
                specularBounce = false;
                ++depth;
                continue;
            }
            break;
        }

        // Complementary BRDF path when SSS lottery was available but not chosen.
        if (canSss) throughput /= srMax(1e-4f, 1.0f - sssWeight);

        radiance += throughput * nextEventEstimation(scene, tracer, si, mat, frame, wo, rng);

        const Vec3 woLocal = frame.toLocal(wo);
        const BsdfSample bs = bsdfSampleLocal(mat, woLocal, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                              rng.nextFloat());
        if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;

        Vec3 weight = bs.weight;
        if (settings.clampIndirect > 0.0f && depth > 0) {
            const float m = maxComponent(weight);
            if (m > settings.clampIndirect) weight *= settings.clampIndirect / m;
        }
        throughput *= weight;
        if (!isFinite(throughput) || isBlack(throughput)) break;

        const Vec3 wiWorld = normalize(frame.toWorld(bs.wi));
        origin = offsetRayOrigin(si.p, si.ng, wiWorld);
        direction = wiWorld;
        bsdfPdf = bs.pdf;
        specularBounce = bs.specular;
        ++depth;

        // Russian roulette.
        if (depth >= srMax(1, settings.rrStartDepth)) {
            const float q = clampf(maxComponent(throughput), 0.05f, 1.0f);
            if (rng.nextFloat() > q) break;
            throughput /= q;
        }
    }

    if (!isFinite(radiance)) return Vec3(0.0f);
    return radiance;
}

}  // namespace sol
