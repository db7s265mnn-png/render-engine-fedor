// Shared helpers for wavefront OptiX raygens. No optixTrace, no BSDF.
#pragma once

#include "render/optix/optix_common.cuh"

namespace sol {

__device__ inline int wavefrontPixel(int& x, int& y) {
    const LaunchParams& p = launchParams();
    const uint3 li = optixGetLaunchIndex();
    if (p.compactLaunch && p.workItems) {
        unsigned n = p.workCount > 0 ? (unsigned)p.workCount : 0u;
        if (p.workSlot >= 0 && p.workSlot < kSlotCount && p.workCounts)
            n = p.workCounts[p.workSlot];
        if (li.x >= n) return -1;
        const int pixel = p.workItems[li.x];
        if (pixel < 0 || !p.paths || !p.hits || !p.shadows || !p.accumBuffer) return -1;
        if (p.width <= 0 || pixel >= p.width * p.height) return -1;
        x = pixel % p.width;
        y = pixel / p.width;
        return pixel;
    }
    x = int(li.x) + p.pixelOffsetX;
    y = int(li.y) + p.pixelOffsetY;
    if (x >= p.width || y >= p.height || !p.paths || !p.hits || !p.shadows || !p.accumBuffer)
        return -1;
    return y * p.width + x;
}

// Closest-hit / miss: same launch index as the tracing raygen (Iray: no payload).
__device__ inline int wavefrontPixelIndex() {
    int x = 0, y = 0;
    return wavefrontPixel(x, y);
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

// Light-trace splat onto another pixel. Does not bump accum.w (camera spp owns
// the divisor). Concurrent wavefront slots can land on the same dest.
__device__ inline void addSplatRadiance(int destPixel, Vec3 c) {
    if (!isFinite(c)) return;
    const LaunchParams& p = launchParams();
    if (!p.accumBuffer || destPixel < 0 || p.width <= 0 || p.height <= 0) return;
    const int n = p.width * p.height;
    if (destPixel >= n) return;
    Vec4& a = p.accumBuffer[destPixel];
    atomicAdd(&a.x, c.x);
    atomicAdd(&a.y, c.y);
    atomicAdd(&a.z, c.z);
}

__device__ inline Vec3 offsetRay(Vec3 p, Vec3 n, Vec3 dir) {
    const float scale = 1.0f + srMax(fabsf(p.x), srMax(fabsf(p.y), fabsf(p.z)));
    const Vec3 offset = n * (kRayEpsilon * scale);
    return dot(dir, n) > 0.0f ? p + offset : p - offset;
}

__device__ inline void enqueueShadow(GpuShadow& shadow, Vec3 origin, Vec3 dir, float tMax, Vec3 contrib,
                                     int mediumIndex, int splatPixel = -1) {
    if (!isFinite(contrib) || isBlack(contrib)) {
        shadow.queue = kShadowIdle;
        shadow.splatPixel = -1;
        return;
    }
    shadow.origin = origin;
    shadow.direction = dir;
    shadow.tMax = tMax;
    shadow.contrib = contrib;
    shadow.occluded = 0;
    shadow.volumeTr = 1;
    shadow.mediumIndex = mediumIndex;
    shadow.splatPixel = splatPixel;
    shadow.queue = kShadowTrace;
}

}  // namespace sol
