// Camera-path spawn / regen for OptiX PT. No optixTrace.
#pragma once

#include "render/camera_sample.h"
#include "render/optix/optix_geom.cuh"
#include "render/optix/optix_light_emit.cuh"
#include "render/optix/optix_spectral_film.cuh"
#include "render/optix/optix_volume.cuh"
#include "render/optix/optix_work.cuh"

namespace sol {

__device__ inline void spawnCameraPath(int pixel, int x, int y, int sampleOffset) {
    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuHit& hit = params.hits[pixel];
    GpuShadow& shadow = params.shadows[pixel];

    path.sampleRgb = Vec3(0.0f);
    path.nLambda = 0;
    path.filmOpen = 0;
    specZero(path.radianceS, kMaxSpectrumSamples);
    specZero(path.throughputS, kMaxSpectrumSamples);

    const int sampleIndex = params.sampleIndex + sampleOffset;
    const unsigned frameSeed =
        unsigned(params.scene.settings.seed) * 9781u + unsigned(sampleIndex) * 6271u;
    path.rng = makePixelRng(x, y, sampleIndex, frameSeed);
    // OptiX: PCG. Embree keeps Owen-Sobol. Device Joe–Kuo rebuild was a tax.
    const float jitterX = path.rng.nextFloat();
    const float jitterY = path.rng.nextFloat();
    const float lensU = path.rng.nextFloat();
    const float lensV = path.rng.nextFloat();

    cameraRay(params.scene, float(x) + jitterX, float(y) + jitterY, lensU, lensV, path.origin,
              path.direction);
    samplePathWavelengths(path, params.spec);
    path.bsdfPdf = 0.0f;
    path.depth = 0;
    path.hops = 0;
    path.queue = kQueueIntersectClosest;
    path.specularBounce = 1;
    path.mediumIndex = -1;
    path.volumeScatters = 0;
    path.localSample = sampleOffset;
    path.lightPath = 0;
    path.specPrefix = 0;
    path.lightIndex = -1;
    path.sawNonSpecular = 0;
    path.causticSuffix = 0;
    if (params.volumes && params.volumeCount > 0) {
        for (int i = 0; i < params.volumeCount; ++i) {
            const GpuVolumeGrid& g = params.volumes[i];
            if (!g.density || g.kind != 1) continue;
            if (!gpuPointInAabb(path.origin, g.bmin, g.bmax)) continue;
            const int med = gpuMediumIndexForVolume(params.scene, i);
            if (med >= 0 && mediumIsActive(params.scene, med)) {
                path.mediumIndex = med;
                break;
            }
        }
    }

    hit = GpuHit{};
    shadow = GpuShadow{};
    shadow.splatPixel = -1;
    params.accumBuffer[pixel].w += 1.0f;
}

constexpr unsigned kGpuLightPathRngSalt = 0xA7C41E55u;

__device__ inline void applySpawnMedium(const LaunchParams& params, GpuPath& path) {
    if (!params.volumes || params.volumeCount <= 0) return;
    for (int i = 0; i < params.volumeCount; ++i) {
        const GpuVolumeGrid& g = params.volumes[i];
        if (!g.density || g.kind != 1) continue;
        if (!gpuPointInAabb(path.origin, g.bmin, g.bmax)) continue;
        const int med = gpuMediumIndexForVolume(params.scene, i);
        if (med >= 0 && mediumIsActive(params.scene, med)) {
            path.mediumIndex = med;
            break;
        }
    }
}

// One light path per pixel slot. Does not increment accum.w — camera spp owns
// the divisor; splats atomicAdd RGB onto whichever pixel they project to.
__device__ inline bool spawnLightPath(int pixel, int x, int y, int sampleOffset) {
    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuHit& hit = params.hits[pixel];
    GpuShadow& shadow = params.shadows[pixel];

    path.sampleRgb = Vec3(0.0f);
    path.nLambda = 0;
    path.filmOpen = 0;
    specZero(path.radianceS, kMaxSpectrumSamples);
    specZero(path.throughputS, kMaxSpectrumSamples);
    path.lightPath = 1;
    path.specPrefix = 0;
    path.lightIndex = -1;
    path.sawNonSpecular = 0;
    path.causticSuffix = 0;
    path.queue = kQueueDead;
    path.localSample = sampleOffset;
    hit = GpuHit{};
    shadow = GpuShadow{};
    shadow.splatPixel = -1;

    if (params.splatInvLightPaths <= 0.0f) return false;

    const int sampleIndex = params.sampleIndex + sampleOffset;
    const unsigned frameSeed =
        unsigned(params.scene.settings.seed) * 9781u + unsigned(sampleIndex) * 6271u;
    path.rng = makePixelRng(x, y, sampleIndex, frameSeed, kGpuLightPathRngSalt);

    const GpuLightEmit emit = gpuSampleLe(params.scene, path.rng);
    if (!emit.ok) return false;

    samplePathWavelengths(path, params.spec);
    specZero(path.radianceS, path.nLambda);
    path.filmOpen = 0;
    if (emit.lightIndex >= 0 && emit.lightIndex < params.scene.lightCount)
        specAuthoredRadiance(params.scene.lights[emit.lightIndex], emit.betaRgb, path,
                             path.throughputS);
    else
        specUpsampleEmission(gpuSpec(), emit.betaRgb, path.lambda, path.nLambda, path.throughputS);
    if (!specIsFinite(path.throughputS, path.nLambda) || specIsBlack(path.throughputS, path.nLambda))
        return false;

    path.origin = emit.infinite ? emit.origin : offsetRay(emit.origin, emit.n, emit.dir);
    path.direction = emit.dir;
    path.bsdfPdf = 0.0f;
    path.depth = 0;
    path.hops = 0;
    path.specularBounce = 1;
    path.mediumIndex = -1;
    path.volumeScatters = 0;
    path.lightIndex = emit.lightIndex;
    applySpawnMedium(params, path);
    path.queue = kQueueIntersectClosest;
    return true;
}

// Next spp in this wavefront batch. Never spawn while a shadow ray still
// owns GpuShadow / pending NEE — that would wipe radiance and the shadow slot.
__device__ inline bool maybeRegeneratePath(int pixel, GpuPath& path) {
    if (path.queue != kQueueDead) return false;
    const LaunchParams& params = launchParams();
    if (params.shadows && params.shadows[pixel].queue != kShadowIdle) return false;
    const int batch = params.batchSamples > 1 ? params.batchSamples : 1;
    if (path.localSample + 1 >= batch) return false;
    if (params.width <= 0) return false;
    const int x = pixel % params.width;
    const int y = pixel / params.width;
    if (path.lightPath) {
        spawnLightPath(pixel, x, y, path.localSample + 1);
        return path.queue == kQueueIntersectClosest;
    }
    if (params.skipMask && params.skipMask[pixel]) return false;
    spawnCameraPath(pixel, x, y, path.localSample + 1);
    return true;
}

__device__ inline void terminatePath(int pixel, GpuPath& path) {
    killPath(pixel, path);
    maybeRegeneratePath(pixel, path);
}

}  // namespace sol
