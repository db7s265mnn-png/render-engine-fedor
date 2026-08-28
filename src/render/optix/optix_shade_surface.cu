// Cycles analogue: integrator_shade_surface.
// No optixTrace — NEE writes a shadow ray for intersect_shadow.
// Opaque BSDF weights: pbrt RGBAlbedoSpectrum (CPU liftBsdfWeight). Dielectric stays 1/η².
// NEE bakes throughput at the vertex (pbrt SampleLd) before the BSDF/RR step.
// Light-trace: splat on connectable vertices after a spec prefix, then continue
// (do not splat from the caster, do not kill the SDS path).
//
// Do not include optix_mnee.cuh here. Eye-path Newton lives in the dedicated
// MNEE pipeline (optix_mnee.cu). Shade only arms a GpuMneeJob; intersect_shadow
// peeks the glass blocker; __raygen__mnee runs the solver.
#include "render/lights.h"
#include "render/optix/optix_geom.cuh"
#include "render/optix/optix_light_trace.cuh"
#include "render/optix/optix_spawn.cuh"
#include "render/optix/optix_spectral.cuh"
#include "render/optix/optix_volume.cuh"

namespace sol {

__device__ inline void shadeSurfacePixel(int pixel) {
    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];

    const SceneView& scene = params.scene;
    const GpuHit& hit = params.hits[pixel];
    GpuShadow& shadow = params.shadows[pixel];
    shadow.queue = kShadowIdle;
    shadow.splatPixel = -1;
    shadow.specContrib = 0;
    shadow.eyeBounceNee = 0;

    Surf si;
    if (!buildSurf(scene, hit, path.origin, path.direction, si)) {
        terminatePath(pixel, path);
        return;
    }

    const InstanceData& volInst = scene.instances[si.instanceIndex];
    if (volInst.volumeIndex >= 0 && volInst.volumeIndex < params.volumeCount && params.volumes) {
        const GpuVolumeGrid& vol = params.volumes[volInst.volumeIndex];
        if (!vol.density) {
            path.origin = offsetRay(si.p, si.ng, path.direction);
            ++path.hops;
            path.queue = kQueueIntersectClosest;
            return;
        }
        if (vol.kind == 1) {
            if (path.mediumIndex == volInst.mediumIndex) {
                path.mediumIndex = -1;
                path.origin = offsetRay(si.p, si.ng, path.direction);
            } else {
                path.mediumIndex = volInst.mediumIndex;
                path.origin = offsetRay(si.p, si.ng * -1.0f, path.direction);
            }
            ++path.hops;
            path.queue = kQueueIntersectClosest;
            return;
        }
        if (vol.kind == 0) {
            float tSdf = hit.t;
            Vec3 nSdf;
            const float tNear = srMax(0.0f, hit.t - 0.05f);
            if (sphereTraceGpuSdf(vol, path.origin, path.direction, tNear, hit.t + 1.0e6f, tSdf, nSdf)) {
                si.p = path.origin + path.direction * tSdf;
                si.ng = nSdf;
                si.ns = nSdf;
            } else {
                path.origin = offsetRay(si.p, si.ng, path.direction);
                ++path.hops;
                path.queue = kQueueIntersectClosest;
                return;
            }
        }
    }

    if (si.lightIndex >= 0 && path.depth == 0 && !path.lightPath) {
        const InstanceData& inst = scene.instances[si.instanceIndex];
        if (!inst.visibleCamera) {
            path.origin = offsetRay(si.p, si.ng, path.direction);
            ++path.hops;
            path.queue = kQueueIntersectClosest;
            return;
        }
    }

    if (si.lightIndex >= 0) {
        if (path.lightPath) {
            terminatePath(pixel, path);
            return;
        }
        const bool suppressCausticLight =
            scene.settings.caustics == 0 && path.causticSuffix;
        if ((params.splatInvLightPaths > 0.0f && path.causticSuffix) || suppressCausticLight) {
            terminatePath(pixel, path);
            return;
        }
        const LightData& light = scene.lights[si.lightIndex];
        if (path.causticSuffix && !lightContributesCaustics(light)) {
            terminatePath(pixel, path);
            return;
        }
        const Vec3 lightN = light.type == kLightSphere ? si.ng : areaLightNormal(light);
        Vec3 emitted = areaLightEmission(scene, light, path.direction, lightN);
        if (!isBlack(emitted)) {
            float weight = 1.0f;
            if (!path.specularBounce) {
                const float lp =
                    lightPdfDirection(scene, si.lightIndex, path.origin, path.direction, si.p, lightN) *
                    lightSelectionPdfIndex(scene, path.origin, si.lightIndex);
                weight = powerHeuristic(1.0f, path.bsdfPdf, 1.0f, lp);
            }
            float Le[kMaxSpectrumSamples];
            specLightEmission(light, path, Le);
            const float rgbScale = length(emitted) / srMax(1e-6f, length(light.emittedRadiance()));
            addPathRadianceS(path, Le, weight * rgbScale,
                             pathContributionClamp(scene.settings, path.depth, path.specularBounce != 0,
                                                    path.causticSuffix != 0));
        }
        terminatePath(pixel, path);
        return;
    }

    if (si.materialIndex < 0 || si.materialIndex >= scene.materialCount || !scene.materials) {
        terminatePath(pixel, path);
        return;
    }

    Material matSrc = scene.materials[si.materialIndex];
    if (path.lightPath) matSrc = gpuMaterialForCausticTransport(scene, si.materialIndex);
    Material mat = optixpt::evaluateMaps(scene, matSrc, si.uv, si.ns);
    if (mat.transmission <= 0.0f && mat.doubleSided && dot(si.ns, -path.direction) < 0.0f) {
        si.ns = -si.ns;
        si.ng = -si.ng;
    }
    if (mat.emissionStrength > 0.0f && !isBlack(mat.emissionColor) && !path.lightPath) {
        const bool frontFacing = dot(si.ns, -path.direction) > 0.0f;
        if (frontFacing || mat.doubleSided)
            addPathEmissionRgb(path, mat.emissionColor * mat.emissionStrength, 1.0f, 0.0f);
    }
    if (mat.opacity <= 1e-6f || (mat.opacity < 0.999f && path.rng.nextFloat() > mat.opacity)) {
        path.origin = offsetRay(si.p, si.ng, path.direction);
        ++path.hops;
        path.queue = kQueueIntersectClosest;
        return;
    }

    const Vec3 wo = -path.direction;
    const Frame frame(si.ns);
    if (path.lightPath) {
        const Vec3 woLocal = frame.toLocal(wo);
        if (gpuLightVertexConnectable(mat, woLocal)) {
            tryEnqueueCausticSplat(pixel, path, shadow, si, mat, frame, wo);
            path.specPrefix = 0;
        }
    }

    const int maxDepth = srMax(1, scene.settings.maxDepth);
    if (path.depth >= maxDepth) {
        terminatePath(pixel, path);
        return;
    }

    const bool suppressCausticLight =
        !path.lightPath && scene.settings.caustics == 0 && path.causticSuffix;
    const bool skipCameraSds =
        !path.lightPath && params.splatInvLightPaths > 0.0f && path.causticSuffix;
    if (params.mneeJobs && !path.lightPath) {
        params.mneeJobs[pixel].armed = 0;
        params.mneeJobs[pixel].pending = 0;
        params.mneeJobs[pixel].cameraSplat = 0;
    }
    if (!path.lightPath && !skipCameraSds && !(suppressCausticLight && !path.specularBounce) &&
        scene.lightCount > 0) {
        float selectPdf = 0.0f;
        const int lightIndex = sampleLightIndex(scene, si.p, path.rng.nextFloat(), selectPdf);
        LightSample ls;
        if (lightIndex >= 0 && selectPdf > 0.0f &&
            sampleLight(scene, lightIndex, si.p, path.rng.nextFloat(), path.rng.nextFloat(), ls) &&
            ls.pdf > 0.0f && !isBlack(ls.radiance) &&
            shadingNormalConsistent(si.ng, si.ns, wo, ls.wi)) {
            const Vec3 woLocal = frame.toLocal(wo);
            const Vec3 wiLocal = frame.toLocal(ls.wi);
            const BsdfEval be = bsdfEvalLocal(mat, woLocal, wiLocal);
            if (be.pdf > 0.0f && !isBlack(be.f)) {
                const float lightPdf = ls.pdf * selectPdf;
                const float mis = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, be.pdf);
                const float scale = (fabsf(wiLocal.z) / lightPdf) * mis;
                float neeS[kMaxSpectrumSamples];
                evalSurfaceNeeSpectral(scene.lights[lightIndex], ls.radiance, mat, woLocal, wiLocal,
                                       scale, path, mat.ior, neeS);
                const Vec3 shadowOrigin = offsetRay(si.p, si.ng, ls.wi);
                float tMax = 1.0e8f;
                if (ls.distance < 1.0e7f) tMax = ls.distance * (1.0f - 1e-3f);
                enqueueOrAddVertexNeeS(path, shadow, shadowOrigin, ls.wi, tMax, neeS,
                                       path.mediumIndex, scene.lights[lightIndex].shadowEnable,
                                       pathContributionClamp(scene.settings, path.depth,
                                                             path.specularBounce != 0,
                                                             path.causticSuffix != 0),
                                       path.depth > 0 ? 1 : 0);
                // Iray: depth>0 NEE Fresnel-continues through glass (intersect_shadow),
                // so MNEE peeks only when the interface still fully blocks (TIR).
                // Arming stays depth>0 && specularBounce so the floor SDS is LT-only.
                if (params.mneeJobs && gpuEyePathMneeEnabled(scene.settings) &&
                    shadow.queue == kShadowTrace &&
                    (path.depth > 0 && path.specularBounce != 0)) {
                    GpuMneeJob& job = params.mneeJobs[pixel];
                    job.p = si.p;
                    job.ns = si.ns;
                    job.ng = si.ng;
                    job.wo = wo;
                    job.uv = si.uv;
                    job.wi = ls.wi;
                    job.distance = ls.distance;
                    job.y = si.p + ls.wi * ls.distance;
                    const LightData& light = scene.lights[lightIndex];
                    job.yN = areaLightNormal(light);
                    if (light.type == kLightSphere) job.yN = normalize(job.y - lightOrigin(light));
                    if (light.type == kLightPoint) job.yN = Vec3(0.0f, 1.0f, 0.0f);
                    job.LeRgb = ls.radiance;
                    job.pdfArea = ls.pdf;
                    job.selectPdf = selectPdf;
                    if (light.type == kLightPoint) {
                        job.pdfArea = 1.0f;
                        job.LeRgb = light.emittedRadiance();
                    } else if (light.type == kLightRect) {
                        const float area = rectLightArea(light);
                        job.pdfArea = area > 1e-12f ? 1.0f / area : 0.0f;
                        job.LeRgb = lightRadiance(light);
                    } else if (light.type == kLightDisk) {
                        const float area = diskLightArea(light);
                        job.pdfArea = area > 1e-12f ? 1.0f / area : 0.0f;
                        job.LeRgb = lightRadiance(light);
                    } else if (light.type == kLightSphere) {
                        const float radius = srMax(1e-5f, sphereLightRadius(light));
                        job.pdfArea = 1.0f / (4.0f * kPi * radius * radius);
                        job.LeRgb = lightRadiance(light);
                    }
                    job.distant = light.type == kLightDistant ? 1 : 0;
                    job.materialIndex = si.materialIndex;
                    job.lightIndex = lightIndex;
                    job.casterInstance = -1;
                    job.clampDepth = path.depth;
                    job.clampSpec = path.specularBounce;
                    job.clampCaustic = path.causticSuffix;
                    job.pending = 0;
                    job.armed = 1;
                    job.cameraSplat = 0;
                    for (int i = 0; i < path.nLambda && i < kMaxSpectrumSamples; ++i)
                        job.throughputS[i] = path.throughputS[i];
                }
            }
        }
    }

    const GpuBsdfSampleS bs =
        bsdfSampleSpectralGpu(mat, frame.toLocal(wo), path.rng.nextFloat(), path.rng.nextFloat(),
                              path.rng.nextFloat(), path.rng.nextFloat(), path, mat.ior);
    if (!bs.valid || bs.pdf <= 0.0f) {
        terminatePath(pixel, path);
        return;
    }
    const Vec3 wiWorld = normalize(frame.toWorld(bs.wi));
    if (!shadingNormalConsistent(si.ng, si.ns, wo, wiWorld)) {
        terminatePath(pixel, path);
        return;
    }

    specMul(path.throughputS, bs.weight, path.nLambda);
    if (!specIsFinite(path.throughputS, path.nLambda) || specIsBlack(path.throughputS, path.nLambda)) {
        terminatePath(pixel, path);
        return;
    }
    path.origin = offsetRay(si.p, si.ng, wiWorld);
    path.direction = wiWorld;
    path.bsdfPdf = bs.pdf;
    path.specularBounce = bs.specular ? 1 : 0;
    BsdfSample rgbBs;
    rgbBs.wi = bs.wi;
    rgbBs.pdf = bs.pdf;
    rgbBs.specular = bs.specular;
    rgbBs.transmitted = bs.transmitted;
    const LobeWeights lw = computeLobes(mat, frame.toLocal(wo));
    if (shouldTerminateSecondaryGpu(rgbBs, lw, mat) && !specSecondaryTerminated(path.pdf, path.nLambda))
        specTerminateSecondary(path.pdf, path.nLambda);
    if (path.lightPath) {
        const bool nearSpec = rgbBs.specular || lw.delta || isNearSpecularLobe(lw);
        if (nearSpec && materialContributesCaustics(mat)) path.specPrefix = 1;
    } else {
        const bool causticBounce = rgbBs.specular || lw.delta || isNearSpecularLobe(lw);
        if (causticBounce && path.sawNonSpecular) path.causticSuffix = 1;
        if (!causticBounce) {
            path.sawNonSpecular = 1;
            path.causticSuffix = 0;
        }
    }
    if (bs.transmitted && volInst.mediumIndex >= 0) {
        const MediumData* med = getMedium(scene, volInst.mediumIndex);
        if (med && med->type == 1) {
            const bool entering = dot(si.ng, wiWorld) < 0.0f;
            path.mediumIndex = entering ? volInst.mediumIndex : -1;
        }
    }
    ++path.depth;

    if (path.depth >= srMax(1, scene.settings.rrStartDepth)) {
        const float q = clampf(specMax(path.throughputS, path.nLambda), 0.05f, 1.0f);
        if (path.rng.nextFloat() > q) {
            terminatePath(pixel, path);
            return;
        }
        specMulS(path.throughputS, 1.0f / q, path.nLambda);
    }

    path.queue = kQueueIntersectClosest;
}

#ifndef SOLSTICE_OPTIX_OPS_ONLY
extern "C" __global__ void __raygen__shade_surface() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;
    GpuPath& path = launchParams().paths[pixel];
    if (!launchParams().compactLaunch && path.queue != kQueueShadeSurface) return;
    shadeSurfacePixel(pixel);
    if (launchParams().compactLaunch) enqueuePathContinuation(pixel);
}
#endif

}  // namespace sol
