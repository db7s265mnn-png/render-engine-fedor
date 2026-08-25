// Iray: closest-hit / miss write wavefront state directly. No payload.
//
// Every module in an OptiX pipeline must declare the launch-parameter blob
// (pipelineLaunchParamsVariableName), even if the program never reads it.
#include "render/optix/optix_wavefront.cuh"

namespace sol {

__device__ inline void writeMissRadiance() {
    const int pixel = wavefrontPixelIndex();
    if (pixel < 0) return;
    launchParamsMutable().hits[pixel] = GpuHit{};
}

__device__ inline void writeMissShadow() {
    const int pixel = wavefrontPixelIndex();
    if (pixel < 0) return;
    launchParamsMutable().shadows[pixel].occluded = 0;
}

__device__ inline void writeClosestHitRadiance() {
    const int pixel = wavefrontPixelIndex();
    if (pixel < 0) return;
    GpuHit& hit = launchParamsMutable().hits[pixel];
    const float2 barycentrics = optixGetTriangleBarycentrics();
    hit.didHit = 1;
    hit.t = optixGetRayTmax();
    hit.instanceIndex = int(optixGetInstanceId());
    hit.primIndex = optixGetPrimitiveIndex();
    hit.u = barycentrics.x;
    hit.v = barycentrics.y;
}

__device__ inline void writeClosestHitShadow() {
    const int pixel = wavefrontPixelIndex();
    if (pixel < 0) return;
    launchParamsMutable().shadows[pixel].occluded = 1;
}

}  // namespace sol

extern "C" __global__ void __miss__radiance() { sol::writeMissRadiance(); }

extern "C" __global__ void __miss__shadow() { sol::writeMissShadow(); }

extern "C" __global__ void __closesthit__radiance() { sol::writeClosestHitRadiance(); }

extern "C" __global__ void __closesthit__shadow() { sol::writeClosestHitShadow(); }
