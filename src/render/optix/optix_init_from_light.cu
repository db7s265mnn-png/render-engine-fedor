// Iray analogue: integrator_init_from_light — one SampleLe path per pixel slot.
#include "render/optix/optix_spawn.cuh"

namespace sol {

extern "C" __global__ void __raygen__init_from_light() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuHit& hit = params.hits[pixel];
    GpuShadow& shadow = params.shadows[pixel];

    if (params.splatInvLightPaths <= 0.0f) {
        path.queue = kQueueDead;
        path.lightPath = 0;
        path.localSample = 0;
        hit = GpuHit{};
        shadow = GpuShadow{};
        shadow.splatPixel = -1;
        return;
    }

    spawnLightPath(pixel, x, y, 0);
    if (params.compactLaunch && path.queue == kQueueIntersectClosest) enqueueSlot(kSlotIntersect, pixel);
}

}  // namespace sol
