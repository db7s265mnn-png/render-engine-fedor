// The only place optixTrace is called. Iray on OptiX 7: one optixTrace, no
// branching, closest-hit writes wavefront state (no payload). Shade lives in
// other kernels. SER stays off — shade is not in this raygen (Ampere has no SER
// HW; Ada SER only pays off with reorder→shade in the same launch).
#pragma once

#include "render/optix/optix_wavefront.cuh"

namespace sol {

__device__ inline float3 toFloat3(Vec3 v) { return make_float3(v.x, v.y, v.z); }

__device__ inline void traceClosest(int pixel, Vec3 origin, Vec3 direction, float tMax) {
    const LaunchParams& params = launchParams();
    GpuHit& hit = params.hits[pixel];
    hit = GpuHit{};
    if (!params.traversable) return;
    optixTrace(static_cast<OptixTraversableHandle>(params.traversable), toFloat3(origin),
               toFloat3(direction), 0.0f, tMax, 0.0f, OptixVisibilityMask(kVisAll), OPTIX_RAY_FLAG_NONE,
               kRayTypeRadiance, kRayTypeCount, kRayTypeRadiance);
}

// Closest-hit shadow (no terminate-on-first): Iray-style NEE may continue through
// glass; LT splats still see contributing glass as opaque. Closest-hit writes GpuHit.
__device__ inline void traceShadowClosest(int pixel, Vec3 origin, Vec3 direction, float tMax) {
    const LaunchParams& params = launchParams();
    GpuHit& hit = params.hits[pixel];
    hit = GpuHit{};
    if (!params.traversable) return;
    optixTrace(static_cast<OptixTraversableHandle>(params.traversable), toFloat3(origin),
               toFloat3(direction), 0.0f, tMax, 0.0f, OptixVisibilityMask(kVisShadow),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, kRayTypeShadow, kRayTypeCount, kRayTypeShadow);
}

}  // namespace sol
