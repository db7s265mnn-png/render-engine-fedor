// Iray: closest-hit / miss write wavefront state directly. No payload.
//
// Every module in an OptiX pipeline must declare the launch-parameter blob
// (pipelineLaunchParamsVariableName), even if the program never reads it.
#include "render/optix/optix_wavefront.cuh"

extern "C" __global__ void __miss__radiance() {
    const int pixel = sol::wavefrontPixelIndex();
    if (pixel < 0) return;
    sol::launchParams().hits[pixel] = sol::GpuHit{};
}

extern "C" __global__ void __miss__shadow() {
    const int pixel = sol::wavefrontPixelIndex();
    if (pixel < 0) return;
    sol::launchParams().shadows[pixel].occluded = 0;
}

extern "C" __global__ void __closesthit__radiance() {
    const int pixel = sol::wavefrontPixelIndex();
    if (pixel < 0) return;
    sol::GpuHit& hit = sol::launchParams().hits[pixel];
    const float2 barycentrics = optixGetTriangleBarycentrics();
    hit.didHit = 1;
    hit.t = optixGetRayTmax();
    hit.instanceIndex = int(optixGetInstanceId());
    hit.primIndex = optixGetPrimitiveIndex();
    hit.u = barycentrics.x;
    hit.v = barycentrics.y;
}

extern "C" __global__ void __closesthit__shadow() {
    const int pixel = sol::wavefrontPixelIndex();
    if (pixel < 0) return;
    sol::launchParams().shadows[pixel].occluded = 1;
}
