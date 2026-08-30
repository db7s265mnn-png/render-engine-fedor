// Iray-style tail megakernel: remaining live paths finish in one launch.
// Includes optixTrace + shade (split wavefront kernels stay separate TUs).
// Exit to Diffuse walks live in optix_etd.cu — do not add them here
// (optixTrace + BSDF in this TU already hangs cicc / optixModuleCreate).
#define SOLSTICE_OPTIX_OPS_ONLY
#include "render/optix/optix_intersect_closest.cu"
#include "render/optix/optix_intersect_shadow.cu"
#include "render/optix/optix_shade_surface.cu"
#include "render/optix/optix_shade_volume.cu"
#include "render/optix/optix_shade_background.cu"
#include "render/optix/optix_shade_shadow.cu"

namespace sol {

__device__ inline void tailAdvance(int pixel) {
    GpuPath& path = launchParams().paths[pixel];
    GpuShadow& shadow = launchParams().shadows[pixel];
    if (path.queue == kQueueIntersectClosest) intersectClosestPixel(pixel);
    if (path.queue == kQueueShadeVolume) shadeVolumePixel(pixel);
    if (path.queue == kQueueShadeBackground) shadeBackgroundPixel(pixel);
    if (path.queue == kQueueShadeSurface) shadeSurfacePixel(pixel);
    if (shadow.queue == kShadowTrace) intersectShadowPixel(pixel);
    if (shadow.queue == kShadowShade) shadeShadowPixel(pixel);
}

extern "C" __global__ void __raygen__path_tail() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuShadow& shadow = params.shadows[pixel];
    if (path.queue == kQueueExitToDiffuse) return;
    const int maxDepth = params.scene.settings.maxDepth > 0 ? params.scene.settings.maxDepth : 1;
    const int batch = params.batchSamples > 1 ? params.batchSamples : 1;
    const int mcmcK = params.mcmcMutations > 0 ? params.mcmcMutations : 0;
    const int cap = batch * (maxDepth + 18) * (1 + mcmcK);
    for (int i = 0; i < cap; ++i) {
        if (path.queue == kQueueExitToDiffuse) return;
        if (path.queue == kQueueDead && shadow.queue == kShadowIdle) break;
        tailAdvance(pixel);
    }
}

}  // namespace sol
