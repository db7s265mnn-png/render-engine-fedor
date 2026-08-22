// The only place optixTrace is called. Cycles keeps OptiX programs as thin
// intersect wrappers; shade lives in other kernels.
#pragma once

#include "render/optix/optix_wavefront.cuh"

namespace sol {

__device__ inline float3 toFloat3(Vec3 v) { return make_float3(v.x, v.y, v.z); }

__device__ inline void traceClosest(Vec3 origin, Vec3 direction, float tMax, GpuHit& hit) {
    const LaunchParams& params = launchParams();
    unsigned int didHit = 0;
    unsigned int tBits = 0;
    unsigned int instance = 0;
    unsigned int primitive = 0;
    unsigned int uBits = 0;
    unsigned int vBits = 0;
    optixTrace(params.traversable, toFloat3(origin), toFloat3(direction), 0.0f, tMax, 0.0f,
               OptixVisibilityMask(kVisAll), OPTIX_RAY_FLAG_NONE, kRayTypeRadiance, kRayTypeCount,
               kRayTypeRadiance, didHit, tBits, instance, primitive, uBits, vBits);
    hit.didHit = didHit ? 1 : 0;
    hit.t = __uint_as_float(tBits);
    hit.instanceIndex = int(instance);
    hit.primIndex = primitive;
    hit.u = __uint_as_float(uBits);
    hit.v = __uint_as_float(vBits);
}

__device__ inline int traceShadow(Vec3 origin, Vec3 direction, float tMax) {
    const LaunchParams& params = launchParams();
    unsigned int blocked = 1;
    optixTrace(params.traversable, toFloat3(origin), toFloat3(direction), 0.0f, tMax, 0.0f,
               OptixVisibilityMask(kVisShadow),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               kRayTypeShadow, kRayTypeCount, kRayTypeShadow, blocked);
    return blocked != 0 ? 1 : 0;
}

}  // namespace sol
