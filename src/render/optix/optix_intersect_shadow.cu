// Cycles analogue: __raygen__kernel_optix_integrator_intersect_shadow
// Walks closest hits like Embree shadowVisibility: volume proxies skip, lights
// pass, transmissive surfaces use shadowOpacity when caustics are off.
#include "render/optix/optix_geom.cuh"
#include "render/optix/optix_trace.cuh"
#include "render/optix/optix_work.cuh"
#include "render/spectrum_device.h"

namespace sol {

__device__ inline void intersectShadowPixel(int pixel) {
    const LaunchParams& params = launchParams();
    GpuShadow& shadow = params.shadows[pixel];
    GpuPath& path = params.paths[pixel];
    const SceneView& scene = params.scene;

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

        const float block = gpuShadowBlock(scene, si.materialIndex);
        vis *= (1.0f - block);
        if (block >= 0.999f || vis <= 1e-5f) {
            vis = 0.0f;
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
