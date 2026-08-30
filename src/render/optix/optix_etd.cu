// Dedicated Exit to Diffuse pipeline. Both CPU walks + Lambert NEE live here.
// shade_surface only arms GpuPath (exitP / exitNg / incoming / escapeMat).
// Do not include shade_surface.cu or optix_mnee.cuh — that mix hangs cicc.
#define SOLSTICE_OPTIX_ETD_KERNEL
#include "render/exit_to_diffuse.h"
#include "render/lights.h"
#include "render/optix/optix_geom.cuh"
#include "render/optix/optix_spectral.cuh"
#include "render/optix/optix_trace.cuh"
#include "render/optix/optix_volume.cuh"
#define SOLSTICE_OPTIX_OPS_ONLY
#include "render/optix/optix_intersect_shadow.cu"

namespace sol {

__device__ inline void etdAddMiss(GpuPath& path) {
    const SceneView& scene = launchParams().scene;
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

__device__ inline void etdAddAreaLight(GpuPath& path, const Surf& si) {
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

__device__ inline void etdEnqueueDestNee(GpuPath& path, GpuShadow& shadow, const Surf& si,
                                         const Material& mat, const Frame& frame, Vec3 wo) {
    const SceneView& scene = launchParams().scene;
    if (path.lightPath || scene.lightCount <= 0) return;
    const Material lambert = exitToDiffuseLambert(mat);
    const Vec3 woLocal = frame.toLocal(wo);
    if (!eyePathNeeConnectable(lambert, woLocal)) return;
    float selectPdf = 0.0f;
    const int lightIndex = sampleLightIndex(scene, si.p, path.rng.nextFloat(), selectPdf);
    LightSample ls;
    if (lightIndex < 0 || selectPdf <= 0.0f ||
        !sampleLight(scene, lightIndex, si.p, path.rng.nextFloat(), path.rng.nextFloat(), ls) ||
        ls.pdf <= 0.0f || isBlack(ls.radiance) ||
        !shadingNormalConsistent(si.ng, si.ns, wo, ls.wi))
        return;
    const Vec3 wiLocal = frame.toLocal(ls.wi);
    const BsdfEval be = bsdfEvalLocal(lambert, woLocal, wiLocal);
    if (be.pdf <= 0.0f || isBlack(be.f)) return;
    const float lightPdf = ls.pdf * selectPdf;
    const float mis = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, be.pdf);
    const float scale = (fabsf(wiLocal.z) / lightPdf) * mis;
    float neeS[kMaxSpectrumSamples];
    evalSurfaceNeeSpectral(scene.lights[lightIndex], ls.radiance, lambert, woLocal, wiLocal, scale, path,
                           lambert.ior, neeS);
    const Vec3 shadowOrigin = offsetRay(si.p, si.ng, ls.wi);
    float tMax = 1.0e8f;
    if (ls.distance < 1.0e7f) tMax = ls.distance * (1.0f - 1e-3f);
    const LightData& lightNee = scene.lights[lightIndex];
    enqueueOrAddVertexNeeS(path, shadow, shadowOrigin, ls.wi, tMax, neeS, path.mediumIndex,
                           lightNee.shadowEnable,
                           pathContributionClamp(scene.settings, path.depth, path.specularBounce != 0,
                                                 path.causticSuffix != 0),
                           exitToDiffuseEyeBounceNee());
}

__device__ inline void etdSettleShadow(int pixel) {
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
}

__device__ inline void etdWalk(int pixel, Vec3 origin, Vec3 direction, int escapeMat) {
    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuShadow& shadow = params.shadows[pixel];
    const SceneView& scene = params.scene;
    for (int skip = 0; skip < kExitToDiffuseMaxSkips; ++skip) {
        path.origin = origin;
        path.direction = direction;
        traceClosest(pixel, origin, direction, kFloatMax);
        if (!params.hits[pixel].didHit) {
            etdAddMiss(path);
            return;
        }
        Surf si;
        if (!buildSurf(scene, params.hits[pixel], origin, direction, si)) return;
        if (si.lightIndex >= 0) {
            etdAddAreaLight(path, si);
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
        etdEnqueueDestNee(path, shadow, si, mat, Frame(si.ns), -direction);
        etdSettleShadow(pixel);
        return;
    }
}

extern "C" __global__ void __raygen__etd() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    GpuPath& path = launchParams().paths[pixel];
    if (path.queue != kQueueExitToDiffuse || path.exitEscapeMat < 0) return;

    const int escapeMat = path.exitEscapeMat;
    const Vec3 p = path.exitP;
    const Vec3 ng = path.exitNg;
    const Vec3 incoming = path.direction;
    const int wantsRefract = path.exitWantsRefract;
    const Vec3 refl = exitToDiffuseReflectDirection(incoming, ng);
    etdWalk(pixel, offsetRay(p, ng, refl), refl, escapeMat);
    if (wantsRefract) etdWalk(pixel, offsetRay(p, ng, incoming), incoming, escapeMat);
    path.exitEscapeMat = -1;
    path.exitWantsRefract = 0;
    path.queue = kQueueDead;
    flushPathFilm(pixel);
}

}  // namespace sol
