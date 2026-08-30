// Dedicated Exit to Diffuse pipeline. Both CPU walks + Lambert NEE live here.
// shade_surface only arms GpuPath (exitP / exitNg / incoming / escapeMat).
// Walk control flow is render/exit_to_diffuse_walk.h (same skip/opacity/maps
// as Embree). Do not include shade_surface.cu or optix_mnee.cuh — that mix
// hangs cicc.
#define SOLSTICE_OPTIX_ETD_KERNEL
#include "render/exit_to_diffuse.h"
#include "render/exit_to_diffuse_walk.h"
#include "render/lights.h"
#include "render/optix/optix_geom.cuh"
#include "render/optix/optix_spectral.cuh"
#include "render/optix/optix_trace.cuh"
#include "render/optix/optix_volume.cuh"
#include "render/surface_maps.h"
#define SOLSTICE_OPTIX_OPS_ONLY
#include "render/optix/optix_intersect_shadow.cu"

namespace sol {

struct EtdGpuCtx {
    int pixel = -1;
    Surf last{};
    bool invalid = false;

    SR_INL SR_HD bool intersect(Vec3 o, Vec3 d, float tMax, EtdHit& h) {
        const LaunchParams& params = launchParams();
        GpuPath& path = params.paths[pixel];
        path.origin = o;
        path.direction = d;
        traceClosest(pixel, o, d, tMax);
        if (!params.hits[pixel].didHit) return false;
        if (!buildSurf(params.scene, params.hits[pixel], o, d, last)) {
            invalid = true;
            return false;
        }
        etdHitFromSurfLike(h, last);
        return true;
    }
    SR_INL SR_HD Material evalDestMaps(EtdHit& h) {
        const SceneView& scene = launchParams().scene;
        Material dest = materialForRay(scene, h.materialIndex, RayShadeKind::Camera);
        dest = evaluateSurfaceMaps(scene, dest, h.uv, h.ns);
        last.ns = h.ns;
        return dest;
    }
    SR_INL SR_HD bool skipOpacity(const Material& dest) {
        return exitToDiffuseSkipOpacity(dest, launchParams().paths[pixel].rng.nextFloat());
    }
};

__device__ inline void etdAddMiss(GpuPath& path, float clampValue) {
    const SceneView& scene = launchParams().scene;
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

__device__ inline void etdAddAreaLight(GpuPath& path, const Surf& si, float clampValue) {
    const SceneView& scene = launchParams().scene;
    if (si.lightIndex < 0 || si.lightIndex >= scene.lightCount || !scene.lights) return;
    const LightData& light = scene.lights[si.lightIndex];
    const Vec3 lightN = light.type == kLightSphere ? si.ng : areaLightNormal(light);
    const Vec3 emitted = areaLightEmission(scene, light, path.direction, lightN);
    if (isBlack(emitted)) return;
    float Le[kMaxSpectrumSamples];
    specLightEmission(light, path, Le);
    const float rgbScale = length(emitted) / srMax(1e-6f, length(light.emittedRadiance()));
    addPathRadianceS(path, Le, rgbScale, clampValue);
}

__device__ inline void etdEnqueueDestNee(GpuPath& path, GpuShadow& shadow, const Surf& si,
                                         const Material& mat, const Frame& frame, Vec3 wo, int destMedium,
                                         float clampValue) {
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
    enqueueOrAddVertexNeeS(path, shadow, shadowOrigin, ls.wi, tMax, neeS, destMedium,
                           lightNee.shadowEnable, clampValue, exitToDiffuseEyeBounceNee());
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
    EtdGpuCtx ctx;
    ctx.pixel = pixel;
    EtdHit destHit;
    Material destMat;
    const EtdWalkKind kind = exitToDiffuseWalkFind(ctx, origin, direction, escapeMat, destHit, destMat);
    // Walk pieces stay unclamped; __raygen__etd applies clampDirect to the sum.
    if (kind == EtdWalkKind::Miss) {
        etdAddMiss(path, 0.0f);
        return;
    }
    if (kind == EtdWalkKind::AreaLight) {
        etdAddAreaLight(path, ctx.last, 0.0f);
        return;
    }
    if (kind != EtdWalkKind::Dest) return;
    const int destMedium = exitToDiffuseDestMedium(scene, path.mediumIndex, destHit.instanceIndex);
    etdEnqueueDestNee(path, shadow, ctx.last, destMat, Frame(ctx.last.ns), -direction, destMedium, 0.0f);
    etdSettleShadow(pixel);
}

__device__ inline void etdClampWalkExtra(GpuPath& path, const float* snap, float clampDirect) {
    const int n = path.nLambda;
    float extra[kMaxSpectrumSamples];
    for (int i = 0; i < n; ++i) extra[i] = path.radianceS[i] - snap[i];
    specClampIndirect(extra, n, clampDirect);
    for (int i = 0; i < n; ++i) path.radianceS[i] = snap[i] + extra[i];
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

    float snap[kMaxSpectrumSamples];
    const int n = path.nLambda;
    for (int i = 0; i < n; ++i) snap[i] = path.radianceS[i];

    etdWalk(pixel, offsetRayOrigin(p, ng, refl), refl, escapeMat);
    if (wantsRefract) etdWalk(pixel, offsetRayOrigin(p, ng, incoming), incoming, escapeMat);
    etdClampWalkExtra(path, snap, launchParams().scene.settings.clampDirect);

    path.exitEscapeMat = -1;
    path.exitWantsRefract = 0;
    path.queue = kQueueDead;
    flushPathFilm(pixel);
}

}  // namespace sol
