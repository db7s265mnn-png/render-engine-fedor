// Iray-style tail megakernel: remaining live paths finish in one launch.
// Includes optixTrace + shade (split wavefront kernels stay separate TUs).
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

__device__ inline void settleExitToDiffuseShadow(int pixel) {
    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuShadow& shadow = params.shadows[pixel];
    if (shadow.queue == kShadowTrace) intersectShadowPixel(pixel);
    if (shadow.queue == kShadowShade && !shadow.occluded && shadow.specContrib) {
        float s[kMaxSpectrumSamples];
        const int n = path.nLambda;
        for (int i = 0; i < n; ++i) s[i] = shadow.contribS[i];
        if (shadow.volumeTr) {
            const Vec3 tr = gpuVolumeShadowTr(params, shadow.origin, shadow.direction, shadow.tMax,
                                              shadow.mediumIndex, path.rng);
            float trS[kMaxSpectrumSamples];
            specUpsampleLinear(tr, path.lambda, n, trS);
            specMul(s, trS, n);
        }
        addBakedRadianceS(path, s);
    }
    shadow.queue = kShadowIdle;
    shadow.splatPixel = -1;
    shadow.specContrib = 0;
    shadow.mneeCaster = -1;
    shadow.eyeBounceNee = 0;
    if (params.mneeJobs) {
        params.mneeJobs[pixel].armed = 0;
        params.mneeJobs[pixel].pending = 0;
    }
}

__device__ inline void addExitToDiffuseAreaLight(GpuPath& path, const Surf& si) {
    const SceneView& scene = launchParams().scene;
    if (si.lightIndex < 0 || si.lightIndex >= scene.lightCount || !scene.lights) return;
    const LightData& light = scene.lights[si.lightIndex];
    const Vec3 lightN = light.type == kLightSphere ? si.ng : areaLightNormal(light);
    const Vec3 emitted = areaLightEmission(scene, light, path.direction, lightN);
    if (isBlack(emitted)) return;
    float Le[kMaxSpectrumSamples];
    specLightEmission(light, path, Le);
    const float rgbScale = length(emitted) / srMax(1e-6f, length(light.emittedRadiance()));
    addPathRadianceS(path, Le, rgbScale,
                     pathContributionClamp(scene.settings, path.depth, path.specularBounce != 0,
                                           path.causticSuffix != 0));
}

// One CPU-style opacity walk. Does not terminate — both walks share the path.
__device__ inline void exitToDiffuseWalkGpu(int pixel, Vec3 origin, Vec3 direction, int escapeMat) {
    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuShadow& shadow = params.shadows[pixel];
    const SceneView& scene = params.scene;
    for (int skip = 0; skip < kExitToDiffuseMaxSkips; ++skip) {
        path.origin = origin;
        path.direction = direction;
        intersectClosestPixel(pixel);
        if (!params.hits[pixel].didHit) {
            addExitToDiffuseMissRadiance(path);
            return;
        }
        Surf si;
        if (!buildSurf(scene, params.hits[pixel], origin, direction, si)) return;
        if (si.lightIndex >= 0) {
            addExitToDiffuseAreaLight(path, si);
            return;
        }
        if (exitToDiffuseSkipSelf(escapeMat, si.materialIndex, skip)) {
            origin = offsetRay(si.p, si.ng, direction);
            continue;
        }
        if (si.materialIndex < 0 || si.materialIndex >= scene.materialCount || !scene.materials) return;
        Material matSrc = materialForRay(scene, si.materialIndex, RayShadeKind(path.rayKind));
        Material mat = optixpt::evaluateMaps(scene, matSrc, si.uv, si.ns);
        if (mat.transmission <= 0.0f && mat.doubleSided && dot(si.ns, -direction) < 0.0f) {
            si.ns = -si.ns;
            si.ng = -si.ng;
        }
        if (mat.opacity <= 1e-6f || (mat.opacity < 0.999f && path.rng.nextFloat() > mat.opacity)) {
            origin = offsetRay(si.p, si.ng, direction);
            continue;
        }
        const Vec3 wo = -direction;
        const Frame frame(si.ns);
        tryEnqueueExitToDiffuseNee(path, shadow, scene, si, mat, frame, wo);
        settleExitToDiffuseShadow(pixel);
        return;
    }
}

__device__ inline void finishExitToDiffuseBothWalks(int pixel, bool regenerate) {
    GpuPath& path = launchParams().paths[pixel];
    const int escapeMat = path.exitEscapeMat;
    const Vec3 p = path.exitP;
    const Vec3 ng = path.exitNg;
    const Vec3 incoming = path.direction;
    const int wantsRefract = path.exitWantsRefract;
    if (escapeMat < 0) {
        path.queue = kQueueDead;
        flushPathFilm(pixel);
        if (regenerate) maybeRegeneratePath(pixel, path);
        return;
    }
    const Vec3 refl = exitToDiffuseReflectDirection(incoming, ng);
    exitToDiffuseWalkGpu(pixel, offsetRay(p, ng, refl), refl, escapeMat);
    if (wantsRefract) exitToDiffuseWalkGpu(pixel, offsetRay(p, ng, incoming), incoming, escapeMat);
    path.exitEscapeMat = -1;
    path.exitWantsRefract = 0;
    path.queue = kQueueDead;
    flushPathFilm(pixel);
    if (regenerate) maybeRegeneratePath(pixel, path);
}

extern "C" __global__ void __raygen__path_tail() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuShadow& shadow = params.shadows[pixel];
    const bool etdOnly = params.pathTailExitOnly != 0;
    const int maxDepth = params.scene.settings.maxDepth > 0 ? params.scene.settings.maxDepth : 1;
    const int batch = params.batchSamples > 1 ? params.batchSamples : 1;
    const int mcmcK = params.mcmcMutations > 0 ? params.mcmcMutations : 0;
    const int cap = batch * (maxDepth + 18) * (1 + mcmcK);
    for (int i = 0; i < cap; ++i) {
        if (path.queue == kQueueExitToDiffuse) {
            finishExitToDiffuseBothWalks(pixel, !etdOnly);
            if (etdOnly) break;
            continue;
        }
        if (etdOnly) break;
        if (path.queue == kQueueDead && shadow.queue == kShadowIdle) break;
        tailAdvance(pixel);
    }
}

}  // namespace sol
