// Cycles analogue: __raygen__kernel_optix_integrator_intersect_closest
#include "render/optix/optix_trace.cuh"
#include "render/optix/optix_work.cuh"

namespace sol {

__device__ inline void intersectClosestPixel(int pixel) {
    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    traceClosest(pixel, path.origin, path.direction, kFloatMax);
    const int didHit = params.hits[pixel].didHit;
    if (path.mediumIndex >= 0) {
        path.queue = kQueueShadeVolume;
    } else {
        path.queue = didHit ? kQueueShadeSurface : kQueueShadeBackground;
    }
}

#ifndef SOLSTICE_OPTIX_OPS_ONLY
extern "C" __global__ void __raygen__intersect_closest() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    GpuPath& path = launchParams().paths[pixel];
    if (!launchParams().compactLaunch && path.queue != kQueueIntersectClosest) return;
    intersectClosestPixel(pixel);
    if (launchParams().compactLaunch) enqueuePathContinuation(pixel);
}
#endif

}  // namespace sol
