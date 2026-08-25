// Cycles analogue: integrator_init_from_camera.
// Pinhole + thin-lens DoF. Polynomial optics stay on Embree.
#include "render/optix/optix_spawn.cuh"

namespace sol {

extern "C" __global__ void __raygen__init_from_camera() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuHit& hit = params.hits[pixel];
    GpuShadow& shadow = params.shadows[pixel];

    if (params.skipMask && params.skipMask[pixel]) {
        path.queue = kQueueDead;
        path.localSample = 0;
        hit = GpuHit{};
        shadow = GpuShadow{};
        return;
    }

    spawnCameraPath(pixel, x, y, 0);
    enqueueSlot(kSlotIntersect, pixel);
}

}  // namespace sol
