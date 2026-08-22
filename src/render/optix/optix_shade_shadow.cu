// Cycles analogue: integrator_shade_shadow.
#include "render/optix/optix_volume.cuh"

namespace sol {

extern "C" __global__ void __raygen__shade_shadow() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuShadow& shadow = params.shadows[pixel];
    if (shadow.queue != kShadowShade) return;
    if (!shadow.occluded) {
        Vec3 contrib = shadow.contrib;
        if (shadow.volumeTr)
            contrib = contrib * gpuVolumeShadowTr(params, shadow.origin, shadow.direction, shadow.tMax,
                                                  shadow.mediumIndex);
        addRadiance(pixel, contrib);
    }
    shadow.queue = kShadowIdle;
}

}  // namespace sol
