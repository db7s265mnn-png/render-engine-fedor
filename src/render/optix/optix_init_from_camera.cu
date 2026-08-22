// Cycles analogue: integrator_init_from_camera.
// Pinhole + thin-lens DoF. Polynomial optics stay on Embree.
#include "render/blue_noise.h"
#include "render/optix/optix_geom.cuh"
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

    path.rng = makePixelRng(x, y, params.sampleIndex, params.frameSeed);
    float jitterX = 0.5f, jitterY = 0.5f;
    if (params.pixelSampler == kPixelSamplerBlueNoise) {
        blueNoisePixelJitter(x, y, params.sampleIndex, jitterX, jitterY);
    } else {
        jitterX = path.rng.nextFloat();
        jitterY = path.rng.nextFloat();
    }
    const float lensU = path.rng.nextFloat();
    const float lensV = path.rng.nextFloat();

    cameraRay(params.scene, float(x) + jitterX, float(y) + jitterY, lensU, lensV, path.origin, path.direction);
    path.throughput = Vec3(1.0f);
    path.bsdfPdf = 0.0f;
    path.depth = 0;
    path.hops = 0;
    path.queue = kQueueIntersectClosest;
    path.specularBounce = 1;
    path.mediumIndex = -1;
    path.volumeScatters = 0;
    path.realVolumeScatter = 0;
    if (params.volumes && params.volumeCount > 0) {
        for (int i = 0; i < params.volumeCount; ++i) {
            const GpuVolumeGrid& g = params.volumes[i];
            if (!g.density || g.kind != 1) continue;
            if (!gpuPointInAabb(path.origin, g.bmin, g.bmax)) continue;
            if (!volumeOccupancyIsDense(sampleGpuVolume(g, path.origin), g.majorant)) continue;
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
