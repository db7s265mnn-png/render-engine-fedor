// Cycles analogue: integrator_init_from_camera.
// Pinhole + thin-lens DoF. Polynomial optics stay on Embree.
#include "render/blue_noise.h"
#include "render/camera_sample.h"
#include "render/optix/optix_geom.cuh"
#include "render/optix/optix_volume.cuh"
#include "render/xpu_split.h"

namespace sol {

extern "C" __global__ void __raygen__init_from_camera() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuHit& hit = params.hits[pixel];
    GpuShadow& shadow = params.shadows[pixel];

    if (params.xpuSplit && !xpuGpuOwnsPixel(x, y, params.xpuTileSize, params.xpuGpuParity)) {
        path.queue = kQueueDead;
        path.throughput = Vec3(0.0f);
        hit = GpuHit{};
        shadow = GpuShadow{};
        return;
    }

    path.rng = makePixelRng(x, y, params.sampleIndex, params.frameSeed);
    float jitterX = 0.5f, jitterY = 0.5f, lensU = 0.5f, lensV = 0.5f;
    sampleCameraPixelLens(params.pixelSampler, x, y, params.sampleIndex, params.width, params.frameSeed,
                          params.manualTestMult, jitterX, jitterY, lensU, lensV);

    cameraRay(params.scene, float(x) + jitterX, float(y) + jitterY, lensU, lensV, path.origin, path.direction);
    path.throughput = Vec3(1.0f);
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
