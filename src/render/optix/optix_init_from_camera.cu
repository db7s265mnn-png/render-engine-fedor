// Cycles analogue: integrator_init_from_camera.
// Pinhole + thin-lens DoF. Polynomial optics stay on Embree.
// Samples hero wavelengths the same way as SpectralPathIntegrator.
#include "render/camera_sample.h"
#include "render/optix/optix_geom.cuh"
#include "render/optix/optix_spectral_film.cuh"
#include "render/optix/optix_volume.cuh"

namespace sol {

extern "C" __global__ void __raygen__init_from_camera() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuHit& hit = params.hits[pixel];
    GpuShadow& shadow = params.shadows[pixel];

    path.sampleRgb = Vec3(0.0f);
    path.nLambda = 0;
    path.filmOpen = 0;
    specZero(path.radianceS, kMaxSpectrumSamples);
    specZero(path.throughputS, kMaxSpectrumSamples);
    if (params.skipMask && params.skipMask[pixel]) {
        path.queue = kQueueDead;
        hit = GpuHit{};
        shadow = GpuShadow{};
        return;
    }

    path.rng = makePixelRng(x, y, params.sampleIndex, params.frameSeed);
    attachPathSobol(path.rng, x, y, params.sampleIndex);
    // Same stream as Embree: camera consumes dims 0–3, path continues at 4+.
    const float jitterX = path.rng.nextFloat();
    const float jitterY = path.rng.nextFloat();
    const float lensU = path.rng.nextFloat();
    const float lensV = path.rng.nextFloat();

    cameraRay(params.scene, float(x) + jitterX, float(y) + jitterY, lensU, lensV, path.origin, path.direction);
    samplePathWavelengths(path, params.spec);
    path.bsdfPdf = 0.0f;
    path.depth = 0;
    path.hops = 0;
    path.queue = kQueueIntersectClosest;
    path.specularBounce = 1;
    path.mediumIndex = -1;
    path.volumeScatters = 0;
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

    params.accumBuffer[pixel].w += 1.0f;
}

}  // namespace sol
