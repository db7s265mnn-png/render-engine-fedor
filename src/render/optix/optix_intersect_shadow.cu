// Cycles analogue: __raygen__kernel_optix_integrator_intersect_shadow
// Walks closest hits like Embree shadowVisibility. Iray Photoreal: eye NEE after
// a bounce Fresnel-continues through contributing glass; LT camera splats and
// primary NEE keep that glass opaque (SDS on the directly visible floor).
#include "render/optix/optix_geom.cuh"
#include "render/optix/optix_trace.cuh"
#include "render/optix/optix_work.cuh"
#include "render/shading_bsdf.h"
#include "render/spectrum_device.h"

namespace sol {

__device__ inline void intersectShadowPixel(int pixel) {
    const LaunchParams& params = launchParams();
    GpuShadow& shadow = params.shadows[pixel];
    GpuPath& path = params.paths[pixel];
    const SceneView& scene = params.scene;
    shadow.mneeCaster = -1;

    float vis = 1.0f;
    Vec3 o = shadow.origin;
    const Vec3 dir = shadow.direction;
    float remaining = shadow.tMax;
    for (int hop = 0; hop < 24 && vis > 1e-4f && remaining > 1e-4f; ++hop) {
        traceShadowClosest(pixel, o, dir, remaining);
        const GpuHit& hit = params.hits[pixel];
        if (!hit.didHit) break;
        if (!(hit.t > 1e-5f) || hit.t >= remaining) break;

        Surf si;
        if (!buildSurf(scene, hit, o, dir, si)) {
            vis = 0.0f;
            break;
        }
        if (si.lightIndex >= 0) break;

        if (si.instanceIndex >= 0 && si.instanceIndex < scene.instanceCount) {
            const InstanceData& hitInst = scene.instances[si.instanceIndex];
            if (hitInst.volumeIndex >= 0) {
                o = offsetRay(si.p, si.ng, dir);
                remaining -= hit.t;
                continue;
            }
        }

        const int eyeBounceNee =
            (!path.lightPath && shadow.splatPixel < 0 && shadow.eyeBounceNee) ? 1 : 0;
        const float nDotWo = -dot(si.ns, dir);
        const Material mat = gpuMaterialForShadow(scene, si.materialIndex);
        const Material matCau = gpuMaterialForCausticSlot(scene, si.materialIndex);
        const float block =
            shadowBlockFraction(mat, matCau, scene.settings.caustics, eyeBounceNee, nDotWo);
        vis *= (1.0f - block);
        if (block >= 0.999f || vis <= 1e-5f) {
            vis = 0.0f;
            const bool eyePeek = !path.lightPath && shadow.splatPixel < 0;
            const bool sdsPeek =
                path.lightPath && shadow.splatPixel >= 0 && gpuSdsRefractionEnabled(scene.settings);
            if (eyePeek || sdsPeek) {
                const Material caster = gpuMaterialForCausticSlot(scene, si.materialIndex);
                if (isDeltaCausticCaster(caster)) shadow.mneeCaster = si.instanceIndex;
            }
            break;
        }
        o = offsetRay(si.p, si.ng, dir);
        remaining -= hit.t;
    }

    shadow.occluded = vis <= 1e-5f ? 1 : 0;
    if (vis > 1e-5f && vis < 1.0f) {
        shadow.contrib = shadow.contrib * vis;
        if (shadow.specContrib && path.nLambda > 0) specMulS(shadow.contribS, vis, path.nLambda);
    }
    if (!shadow.occluded) shadow.mneeCaster = -1;
    shadow.queue = kShadowShade;
}

#ifndef SOLSTICE_OPTIX_OPS_ONLY
extern "C" __global__ void __raygen__intersect_shadow() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;
    GpuShadow& shadow = launchParams().shadows[pixel];
    if (!launchParams().compactLaunch && shadow.queue != kShadowTrace) return;
    intersectShadowPixel(pixel);
}
#endif

}  // namespace sol
