// Shared helpers for wavefront OptiX raygens. No optixTrace, no BSDF.
#pragma once

#include "render/optix/optix_common.cuh"

namespace sol {

__device__ inline int wavefrontPixel(int& x, int& y) {
    const LaunchParams& p = launchParams();
    const uint3 li = optixGetLaunchIndex();
    x = int(li.x) + p.pixelOffsetX;
    y = int(li.y) + p.pixelOffsetY;
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

__device__ inline float wavefrontLuminance(Vec3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ inline void addRadiance(int pixel, Vec3 c) {
    if (!isFinite(c)) return;
    const LaunchParams& p = launchParams();
    Vec4& a = p.accumBuffer[pixel];
    a.x += c.x;
    a.y += c.y;
    a.z += c.z;
    if (!p.paths) return;
    GpuPath& path = p.paths[pixel];
    if (p.lumSq) {
        const float oldL = wavefrontLuminance(path.sampleRgb);
        path.sampleRgb += c;
        const float newL = wavefrontLuminance(path.sampleRgb);
        p.lumSq[pixel] += newL * newL - oldL * oldL;
    } else {
        path.sampleRgb += c;
    }
}

__device__ inline Vec3 offsetRay(Vec3 p, Vec3 n, Vec3 dir) {
    const float scale = 1.0f + srMax(fabsf(p.x), srMax(fabsf(p.y), fabsf(p.z)));
    const Vec3 offset = n * (kRayEpsilon * scale);
    return dot(dir, n) > 0.0f ? p + offset : p - offset;
}

// Snapshot throughput here: shade_surface / shade_volume continue the path
// (BSDF weight, RR) before shade_shadow runs. Multiplying NEE by the later
// throughput makes single-pixel chromatic fireflies.
__device__ inline void enqueueShadow(GpuShadow& shadow, Vec3 origin, Vec3 dir, float tMax, Vec3 contrib,
                                     int mediumIndex, const GpuPath& path, float clampValue) {
    if (!isFinite(contrib) || isBlack(contrib) || path.nLambda <= 0) {
        shadow.queue = kShadowIdle;
        return;
    }
    shadow.origin = origin;
    shadow.direction = dir;
    shadow.tMax = tMax;
    shadow.contrib = contrib;
    shadow.nLambda = path.nLambda;
    shadow.clampValue = clampValue;
    for (int i = 0; i < path.nLambda; ++i) shadow.throughputS[i] = path.throughputS[i];
    for (int i = path.nLambda; i < kMaxSpectrumSamples; ++i) shadow.throughputS[i] = 0.0f;
    shadow.occluded = 0;
    shadow.volumeTr = 1;
    shadow.mediumIndex = mediumIndex;
    shadow.queue = kShadowTrace;
}

}  // namespace sol
