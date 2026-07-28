// Blue-noise dither for screen-space sample decorrelation
// (Arnold-style Cranley–Patterson offsets; Georgiev & Fajardo 2016).
//
// Uses interleaved-gradient noise + dimensional scramble instead of a baked
 // mask so the same code is valid on host (Embree) and device (OptiX/CUDA).
#pragma once

#include "core/math.h"
#include "core/rng.h"

namespace sol {

// Jimenez 2014 interleaved gradient noise — bluish screen-space dither.
SR_INL SR_HD float interleavedGradientNoise(int x, int y) {
    const float fx = float(x) + 0.5f;
    const float fy = float(y) + 0.5f;
    const float n = 52.9829189f * (0.06711056f * fx + 0.00583715f * fy);
    return n - floorf(n);
}

SR_INL SR_HD float blueNoiseSample(int x, int y, int dimension) {
    // Pseudo-randomly shift the lookup per sampling dimension (Arnold TOG 2018).
    const uint32_t h = hashUint(uint32_t(dimension) * 0x9e3779b9u + 0x85ebca6bu);
    const int ox = int(h & 255u);
    const int oy = int((h >> 8) & 255u);
    float v = interleavedGradientNoise(x + ox, y + oy);
    // Extra scramble so neighbouring dimensions stay decorrelated.
    const float scramble = float((h >> 16) & 255u) * (1.0f / 255.0f);
    v = v + scramble;
    return v - floorf(v);
}

// Toroidal (Cranley–Patterson) shift of u by blue-noise offset in [0,1).
SR_INL SR_HD float blueNoiseOffset(float u, int x, int y, int dimension) {
    float v = u + blueNoiseSample(x, y, dimension);
    v = v - floorf(v);
    return v;
}

// Pixel jitter for sampleIndex. Sample 0 stays centered; later samples are
// blue-noise dithered so residual error looks higher-frequency (less blotchy).
SR_INL SR_HD void blueNoisePixelJitter(int x, int y, int sampleIndex, float& jx, float& jy) {
    if (sampleIndex <= 0) {
        jx = 0.5f;
        jy = 0.5f;
        return;
    }
    // Rank-1 lattice + blue-noise CP rotation (cheap stand-in for full CMJ).
    const float phiX = 0.7548776662466927f;
    const float phiY = 0.5698402909980532f;
    const float sx = float(sampleIndex) * phiX;
    const float sy = float(sampleIndex) * phiY;
    jx = blueNoiseOffset(sx - floorf(sx), x, y, 0);
    jy = blueNoiseOffset(sy - floorf(sy), x, y, 1);
}

}  // namespace sol
