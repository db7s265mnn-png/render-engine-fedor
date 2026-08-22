// Cycles analogue: __raygen__kernel_optix_integrator_intersect_shadow
#include "render/optix/optix_trace.cuh"

namespace sol {

extern "C" __global__ void __raygen__intersect_shadow() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    GpuShadow& shadow = launchParams().shadows[pixel];
    if (shadow.queue != kShadowTrace) return;
    shadow.occluded = traceShadow(shadow.origin, shadow.direction, shadow.tMax);
    shadow.queue = kShadowShade;
}

}  // namespace sol
