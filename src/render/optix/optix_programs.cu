// OptiX device programs. The light transport itself lives in the shared
// integrator header, so the GPU and CPU backends stay in lockstep.
#include <optix.h>

#include "render/blue_noise.h"
#include "render/integrator.h"
#include "render/optix/launch_params.h"

// nvcc rejects dynamic initialisation of __constant__ variables, and our
// parameter structs carry default member initialisers, so the block is
// declared as raw storage and reinterpreted on access.
extern "C" {
__constant__ __align__(16) unsigned char solsticeLaunchParams[sizeof(sol::LaunchParams)];
}

__device__ __forceinline__ const sol::LaunchParams& launchParams() {
    return *reinterpret_cast<const sol::LaunchParams*>(solsticeLaunchParams);
}

namespace sol {
namespace {

__device__ __forceinline__ float3 toFloat3(Vec3 v) { return make_float3(v.x, v.y, v.z); }

// Adapter that gives the shared integrator its ray casting primitives.
struct OptixTracer {
    __device__ bool intersect(Vec3 origin, Vec3 direction, float tMax, RayHit& hit) const {
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
        if (!didHit) return false;
        hit.t = __uint_as_float(tBits);
        hit.instanceIndex = int(instance);
        hit.primIndex = primitive;
        hit.u = __uint_as_float(uBits);
        hit.v = __uint_as_float(vBits);
        return true;
    }

    __device__ bool occluded(Vec3 origin, Vec3 direction, float tMax) const {
        const LaunchParams& params = launchParams();
        unsigned int blocked = 1;  // the shadow miss program clears this
        optixTrace(params.traversable, toFloat3(origin), toFloat3(direction), 0.0f, tMax, 0.0f,
                   OptixVisibilityMask(kVisShadow),
                   OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                       OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                   kRayTypeShadow, kRayTypeCount, kRayTypeShadow, blocked);
        return blocked != 0;
    }
};

}  // namespace
}  // namespace sol

extern "C" __global__ void __raygen__path() {
    using namespace sol;
    const LaunchParams& params = launchParams();
    const uint3 launchIndex = optixGetLaunchIndex();
    const int x = int(launchIndex.x);
    const int y = int(launchIndex.y);
    if (x >= params.width || y >= params.height) return;

    const unsigned int pixelIndex = unsigned(y) * unsigned(params.width) + unsigned(x);
    Rng rng(hashCombine(pixelIndex, params.frameSeed),
            hashUint(pixelIndex ^ (params.frameSeed * 2654435761u)));

    // Pixel Sampler (camera AA / DoF). Sobol tables are host-heavy — treat as White on GPU.
    float jitterX = 0.5f, jitterY = 0.5f;
    float lensU = 0.5f, lensV = 0.5f;
    if (params.pixelSampler == kPixelSamplerBlueNoise) {
        blueNoisePixelJitter(x, y, params.sampleIndex, jitterX, jitterY);
        blueNoiseLensSample(x, y, params.sampleIndex, lensU, lensV);
    } else {
        jitterX = rng.nextFloat();
        jitterY = rng.nextFloat();
        lensU = rng.nextFloat();
        lensV = rng.nextFloat();
    }

    Vec3 origin, direction;
    generateCameraRay(params.scene, float(x) + jitterX, float(y) + jitterY, lensU, lensV, origin, direction);

    OptixTracer tracer;
    const Vec3 radiance = traceRadiance(params.scene, tracer, origin, direction, rng);

    Vec4& pixel = params.accumBuffer[pixelIndex];
    pixel.x += radiance.x;
    pixel.y += radiance.y;
    pixel.z += radiance.z;
    pixel.w += 1.0f;
}

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
