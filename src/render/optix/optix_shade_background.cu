// Cycles analogue: integrator_shade_background (CUDA in Cycles; OptiX raygen here).
#include "render/lights.h"
#include "render/optix/optix_wavefront.cuh"

namespace sol {

extern "C" __global__ void __raygen__shade_background() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    if (path.queue != kQueueShadeBackground) return;
    path.queue = kQueueDead;

    const SceneView& scene = params.scene;
    const bool primary = path.depth == 0;
    const bool skipInfiniteAfterHalo = path.volumeScatters > 0 && path.realVolumeScatter == 0;

    if (scene.domeLightIndex >= 0 && scene.lights && !skipInfiniteAfterHalo) {
        const LightData& dome = scene.lights[scene.domeLightIndex];
        const bool hidePrimary = primary && (!scene.settings.envVisibleCamera || !dome.visibleCamera);
        if (!hidePrimary) {
            Vec3 envL = domeRadiance(scene, dome, path.direction, /*nearestTexel=*/path.depth > 0);
            if (!isBlack(envL)) {
                float weight = 1.0f;
                if (!path.specularBounce) {
                    const float lp = lightPdfDirection(scene, scene.domeLightIndex, path.origin, path.direction,
                                                       path.origin, path.direction) *
                                     lightSelectionPdfIndex(scene, path.origin, scene.domeLightIndex);
                    weight = powerHeuristic(1.0f, path.bsdfPdf, 1.0f, lp);
                }
                Vec3 contrib = path.throughput * envL * weight;
                if (path.depth > 0) contrib = clampFirefly(contrib, scene.settings.clampDirect);
                addRadiance(pixel, contrib);
            }
        }
    }

    if (!skipInfiniteAfterHalo) {
        const Vec3 sunL = cameraSunDiscRadiance(scene, path.origin, path.direction, path.bsdfPdf,
                                               path.specularBounce != 0, primary, false);
        if (!isBlack(sunL)) addRadiance(pixel, path.throughput * sunL);
    }
}

}  // namespace sol
