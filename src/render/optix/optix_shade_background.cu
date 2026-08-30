// Cycles analogue: integrator_shade_background (CUDA in Cycles; OptiX raygen here).
#include "render/lights.h"
#include "render/optix/optix_spawn.cuh"

namespace sol {

// Exit to Diffuse miss: always take the dome / sun (CPU walk), even when the
// environment is hidden from the camera. Does not terminate the path.
__device__ inline void addExitToDiffuseMissRadiance(GpuPath& path) {
    const LaunchParams& params = launchParams();
    const SceneView& scene = params.scene;
    const float clampValue = pathContributionClamp(scene.settings, path.depth, path.specularBounce != 0,
                                                   path.causticSuffix != 0);
    if (scene.domeLightIndex >= 0 && scene.lights) {
        const LightData& dome = scene.lights[scene.domeLightIndex];
        const Vec3 envL = domeRadiance(scene, dome, path.direction, /*nearestTexel=*/true);
        if (!isBlack(envL)) {
            float envS[kMaxSpectrumSamples];
            if (dome.colorTemperatureK > 50.0f) {
                specLightEmission(dome, path, envS);
                const float rgbScale = length(envL) / srMax(1e-6f, length(dome.emittedRadiance()));
                specMulS(envS, rgbScale, path.nLambda);
            } else {
                specUpsampleEmission(gpuSpec(), envL, path.lambda, path.nLambda, envS);
            }
            addPathRadianceS(path, envS, 1.0f, clampValue);
        }
    }
    const Vec3 sunL =
        cameraSunDiscRadiance(scene, path.origin, path.direction, 0.0f, true, false, false);
    if (!isBlack(sunL)) addPathEmissionRgb(path, sunL, 1.0f, clampValue);
}

__device__ inline void shadeBackgroundPixel(int pixel) {
    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    const SceneView& scene = params.scene;
    const bool primary = path.depth == 0;
    const bool suppressCausticLight = scene.settings.caustics == 0 && path.causticSuffix;

    if (path.lightPath) {
        terminatePath(pixel, path);
        return;
    }
    if (scene.settings.integrator == kIntegratorWireframe) {
        terminatePath(pixel, path);
        return;
    }
    const bool exitEscaping = path.exitEscapeMat >= 0;
    if (!exitEscaping &&
        (gpuSkipCameraSds(scene.settings, path.lightPath, path.causticSuffix, path.throughGlass,
                          params.splatInvLightPaths) ||
         suppressCausticLight)) {
        terminatePath(pixel, path);
        return;
    }

    if (scene.domeLightIndex >= 0 && scene.lights) {
        const LightData& dome = scene.lights[scene.domeLightIndex];
        const bool hidePrimary =
            !exitEscaping && primary && (!scene.settings.envVisibleCamera || !dome.visibleCamera);
        if (!hidePrimary && !(path.causticSuffix && !lightContributesCaustics(dome))) {
            Vec3 envL = domeRadiance(scene, dome, path.direction, /*nearestTexel=*/path.depth > 0);
            if (!isBlack(envL)) {
                float weight = 1.0f;
                if (!path.specularBounce) {
                    const float lp =
                        lightPdfDirection(scene, scene.domeLightIndex, path.origin, path.direction,
                                          path.origin, path.direction) *
                        lightSelectionPdfIndex(scene, path.origin, scene.domeLightIndex);
                    weight = powerHeuristic(1.0f, path.bsdfPdf, 1.0f, lp);
                }
                float envS[kMaxSpectrumSamples];
                if (dome.colorTemperatureK > 50.0f) {
                    specLightEmission(dome, path, envS);
                    const float rgbScale =
                        length(envL) / srMax(1e-6f, length(dome.emittedRadiance()));
                    specMulS(envS, rgbScale, path.nLambda);
                } else {
                    specUpsampleEmission(gpuSpec(), envL, path.lambda, path.nLambda, envS);
                }
                addPathRadianceS(path, envS, weight,
                                 pathContributionClamp(scene.settings, path.depth,
                                                       path.specularBounce != 0,
                                                       path.causticSuffix != 0));
            }
        }
    }

    const Vec3 sunL =
        cameraSunDiscRadiance(scene, path.origin, path.direction, path.bsdfPdf,
                              path.specularBounce != 0, primary, path.causticSuffix != 0);
    if (!isBlack(sunL))
        addPathEmissionRgb(path, sunL, 1.0f,
                           pathContributionClamp(scene.settings, path.depth, path.specularBounce != 0,
                                                 path.causticSuffix != 0));
    terminatePath(pixel, path);
}

#ifndef SOLSTICE_OPTIX_OPS_ONLY
extern "C" __global__ void __raygen__shade_background() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;
    GpuPath& path = launchParams().paths[pixel];
    if (!launchParams().compactLaunch && path.queue != kQueueShadeBackground) return;
    shadeBackgroundPixel(pixel);
    if (launchParams().compactLaunch) enqueuePathContinuation(pixel);
}
#endif

}  // namespace sol
