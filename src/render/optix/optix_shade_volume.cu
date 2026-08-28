// Cycles analogue: integrator_shade_volume.
// Homogeneous media + VDB fog (hero-λ Woodcock on GPU). No optixTrace.
#include "render/lights.h"
#include "render/optix/optix_spawn.cuh"
#include "render/optix/optix_volume.cuh"

namespace sol {

__device__ inline void shadeVolumePixel(int pixel) {
    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    const SceneView& scene = params.scene;
    GpuHit& hit = params.hits[pixel];
    GpuShadow& shadow = params.shadows[pixel];
    shadow.queue = kShadowIdle;
    shadow.splatPixel = -1;
    shadow.specContrib = 0;
    shadow.eyeBounceNee = 0;

    const MediumData* med = getMedium(scene, path.mediumIndex);
    if (!med) {
        path.queue = hit.didHit ? kQueueShadeSurface : kQueueShadeBackground;
        return;
    }

    const float tMax = hit.didHit ? srMax(0.0f, hit.t) : 1.0e6f;
    MediumData walk = *med;
    if (scene.settings.volumeSimilarity != 0) walk = mediumWithVolumeSimilarity(*med, path.volumeScatters);

    MediumSample ms;
    if (walk.type == 2) {
        if (walk.volumeIndex >= 0 && walk.volumeIndex < params.volumeCount && params.volumes &&
            params.volumes[walk.volumeIndex].density && params.volumes[walk.volumeIndex].kind == 1) {
            ms = sampleGpuFogWl(params.volumes[walk.volumeIndex], walk, path.origin, path.direction, tMax,
                                path.rng, path.throughputS, path.lambda, path.nLambda);
        } else {
            path.queue = hit.didHit ? kQueueShadeSurface : kQueueShadeBackground;
            return;
        }
    } else {
        ms = sampleMediumHomogeneousWl(walk, tMax, path.rng, path.throughputS, path.lambda, path.nLambda);
    }

    if (ms.absorbed || specIsBlack(path.throughputS, path.nLambda) ||
        !specIsFinite(path.throughputS, path.nLambda) ||
        specMax(path.throughputS, path.nLambda) < 1e-20f) {
        terminatePath(pixel, path);
        return;
    }

    if (ms.scattered) {
        const Vec3 p = path.origin + path.direction * ms.t;
        const Vec3 wo = -path.direction;
        if (!isBlack(med->emission) && !path.lightPath)
            addPathEmissionRgb(path, med->emission, 1.0f, 0.0f);
        const bool skipCameraSds =
            !path.lightPath && params.splatInvLightPaths > 0.0f && path.causticSuffix;
        if (!path.lightPath && !skipCameraSds && scene.lightCount > 0) {
            float selectPdf = 0.0f;
            const int lightIndex = sampleLightIndex(scene, p, path.rng.nextFloat(), selectPdf);
            LightSample ls;
            if (lightIndex >= 0 && selectPdf > 0.0f &&
                sampleLight(scene, lightIndex, p, path.rng.nextFloat(), path.rng.nextFloat(), ls) &&
                ls.pdf > 0.0f && !isBlack(ls.radiance)) {
                const float phase = henyeyGreenstein(dot(wo, ls.wi), walk.g);
                const float lightPdf = ls.pdf * selectPdf;
                const float mis = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, phase);
                float neeS[kMaxSpectrumSamples];
                specAuthoredRadiance(scene.lights[lightIndex], ls.radiance, path, neeS);
                specMulS(neeS, (phase / srMax(1e-8f, lightPdf)) * mis, path.nLambda);
                float tSh = 1.0e8f;
                if (ls.distance < 1.0e7f) tSh = ls.distance * (1.0f - 1e-3f);
                enqueueOrAddVertexNeeS(path, shadow, p, ls.wi, tSh, neeS, path.mediumIndex,
                                       scene.lights[lightIndex].shadowEnable,
                                       pathContributionClamp(scene.settings, path.depth, false, false),
                                       path.depth > 0 && !gpuRefractionMneeEnabled(scene.settings)
                                           ? 1
                                           : 0);
            }
        }

        float phasePdf = 0.0f;
        const Vec3 wi = sampleHenyeyGreenstein(wo, walk.g, path.rng.nextFloat(), path.rng.nextFloat(), phasePdf);
        path.origin = p;
        path.direction = wi;
        path.bsdfPdf = phasePdf;
        path.specularBounce = 0;
        path.transmittedBounce = 0;
        path.sawNonSpecular = 1;
        path.causticSuffix = 0;
        ++path.depth;
        ++path.volumeScatters;
        if (path.depth >= srMax(1, scene.settings.maxDepth)) {
            terminatePath(pixel, path);
            return;
        }
        if (path.depth >= srMax(1, scene.settings.rrStartDepth)) {
            const float q = clampf(specMax(path.throughputS, path.nLambda), 0.05f, 1.0f);
            if (path.rng.nextFloat() > q) {
                terminatePath(pixel, path);
                return;
            }
            specMulS(path.throughputS, 1.0f / q, path.nLambda);
        }
        path.queue = kQueueIntersectClosest;
        hit = GpuHit{};
        return;
    }

    path.origin = path.origin + path.direction * ms.t;
    if (walk.type == 2 && (!hit.didHit || ms.t + 1e-4f < hit.t)) {
        path.mediumIndex = -1;
        ++path.hops;
        path.queue = kQueueIntersectClosest;
        hit = GpuHit{};
        return;
    }
    path.queue = hit.didHit ? kQueueShadeSurface : kQueueShadeBackground;
}

#ifndef SOLSTICE_OPTIX_OPS_ONLY
extern "C" __global__ void __raygen__shade_volume() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;
    GpuPath& path = launchParams().paths[pixel];
    if (!launchParams().compactLaunch && path.queue != kQueueShadeVolume) return;
    shadeVolumePixel(pixel);
    if (launchParams().compactLaunch) enqueuePathContinuation(pixel);
}
#endif

}  // namespace sol
