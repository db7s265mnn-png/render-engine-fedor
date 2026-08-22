// OptiX miss / closest-hit programs. Tiny payload writers — a separate nvcc
// job from init / intersect / shade, like Cycles keeping CH small.
//
// Every module in an OptiX pipeline must declare the launch-parameter blob
// (pipelineLaunchParamsVariableName), even if the program never reads it.
#include "render/optix/optix_common.cuh"

extern "C" __global__ void __miss__radiance() { optixSetPayload_0(0u); }

extern "C" __global__ void __miss__shadow() { optixSetPayload_0(0u); }

extern "C" __global__ void __closesthit__radiance() {
    const float2 barycentrics = optixGetTriangleBarycentrics();
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(optixGetInstanceId());
    optixSetPayload_3(optixGetPrimitiveIndex());
    optixSetPayload_4(__float_as_uint(barycentrics.x));
    optixSetPayload_5(__float_as_uint(barycentrics.y));
}

// Present so the shadow ray type has a valid hit group entry; shadow rays
// terminate on the first hit and never invoke it.
extern "C" __global__ void __closesthit__shadow() { optixSetPayload_0(1u); }
