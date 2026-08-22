// Cycles analogue: __raygen__kernel_optix_integrator_intersect_closest
#include "render/optix/optix_trace.cuh"

namespace sol {

extern "C" __global__ void __raygen__intersect_closest() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    if (path.queue != kQueueIntersectClosest) return;

    GpuHit& hit = params.hits[pixel];
    traceClosest(path.origin, path.direction, kFloatMax, hit);
    path.queue = hit.didHit ? kQueueShadeSurface : kQueueShadeBackground;
}

}  // namespace sol
