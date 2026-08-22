// Cycles analogue: integrator_shade_shadow.
#include "render/optix/optix_wavefront.cuh"

namespace sol {

extern "C" __global__ void __raygen__shade_shadow() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    GpuShadow& shadow = launchParams().shadows[pixel];
    if (shadow.queue != kShadowShade) return;
    if (!shadow.occluded) addRadiance(pixel, shadow.contrib);
    shadow.queue = kShadowIdle;
}

}  // namespace sol
