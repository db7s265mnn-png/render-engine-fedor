// Cycles analogue: __raygen__kernel_optix_integrator_intersect_shadow
#include "render/optix/optix_trace.cuh"
#include "render/optix/optix_work.cuh"

namespace sol {

__device__ inline void intersectShadowPixel(int pixel) {
    GpuShadow& shadow = launchParams().shadows[pixel];
    traceShadow(pixel, shadow.origin, shadow.direction, shadow.tMax);
    shadow.queue = kShadowShade;
}

#ifndef SOLSTICE_OPTIX_OPS_ONLY
extern "C" __global__ void __raygen__intersect_shadow() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;
    GpuShadow& shadow = launchParams().shadows[pixel];
    if (!launchParams().compactLaunch && shadow.queue != kShadowTrace) return;
    intersectShadowPixel(pixel);
}
#endif

}  // namespace sol
