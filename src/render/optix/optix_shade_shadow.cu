// Cycles analogue: integrator_shade_shadow.
// RGB residual-ratio Tr (same as Embree NEE), then linear upsample × path throughput.
#include "render/optix/optix_spawn.cuh"

namespace sol {

__device__ inline bool shadeShadowPixel(int pixel) {
    const LaunchParams& params = launchParams();
    GpuShadow& shadow = params.shadows[pixel];
    GpuPath& path = params.paths[pixel];
    if (!shadow.occluded) {
        Vec3 contrib = shadow.contrib;
        if (shadow.volumeTr)
            contrib = contrib * gpuVolumeShadowTr(params, shadow.origin, shadow.direction, shadow.tMax,
                                                  shadow.mediumIndex, path.rng);
        if (shadow.splatPixel >= 0)
            addSplatRadiance(shadow.splatPixel, contrib);
        else
            path.filmRgb += contrib;
    }
    shadow.queue = kShadowIdle;
    shadow.splatPixel = -1;
    flushPathFilm(pixel);
    return maybeRegeneratePath(pixel, path);
}

#ifndef SOLSTICE_OPTIX_OPS_ONLY
extern "C" __global__ void __raygen__shade_shadow() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;
    GpuShadow& shadow = launchParams().shadows[pixel];
    if (!launchParams().compactLaunch && shadow.queue != kShadowShade) return;
    if (shadeShadowPixel(pixel)) enqueuePathContinuation(pixel);
}
#endif

}  // namespace sol
