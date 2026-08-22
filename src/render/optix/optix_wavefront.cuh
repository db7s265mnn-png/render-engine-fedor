// Shared helpers for wavefront OptiX raygens. No optixTrace, no BSDF.
#pragma once

#include "render/optix/optix_common.cuh"

namespace sol {

__device__ inline int wavefrontPixel(int& x, int& y) {
    const LaunchParams& p = launchParams();
    const uint3 li = optixGetLaunchIndex();
    x = int(li.x);
    y = int(li.y);
    if (x >= p.width || y >= p.height || !p.paths || !p.hits || !p.shadows || !p.accumBuffer)
        return -1;
    return y * p.width + x;
}

__device__ inline Vec3 clampFirefly(Vec3 contrib, float clampValue) {
    if (clampValue <= 0.0f || !isFinite(contrib)) return isFinite(contrib) ? contrib : Vec3(0.0f);
    const float m = maxComponent(contrib);
    if (m > clampValue) contrib *= clampValue / m;
    return contrib;
}

__device__ inline void addRadiance(int pixel, Vec3 c) {
    if (!isFinite(c)) return;
    Vec4& a = launchParams().accumBuffer[pixel];
    a.x += c.x;
    a.y += c.y;
    a.z += c.z;
}

__device__ inline Vec3 offsetRay(Vec3 p, Vec3 n, Vec3 dir) {
    const float scale = 1.0f + srMax(fabsf(p.x), srMax(fabsf(p.y), fabsf(p.z)));
    const Vec3 offset = n * (kRayEpsilon * scale);
    return dot(dir, n) > 0.0f ? p + offset : p - offset;
}

}  // namespace sol
