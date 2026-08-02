// R-sequences (Kronecker / Weyl): golden-ratio 1D + plastic-number R2.
// Used as an optional Pixel Sampler for camera AA / DoF / shutter — path RNG
// stays PCG / Xorshift / Owen (see Path Sampler).
#pragma once

#include "core/math.h"
#include "core/rng.h"

#include <cmath>

namespace sol {

// How the integer index `n` is formed for R2 / golden samples.
enum R2IndexMode : int {
    kR2IndexSpp = 0,           // n = sampleIndex
    kR2IndexSppPixelSalt = 1,  // n = sampleIndex + per-pixel salt
    kR2IndexLinearSpp = 2,     // n = y * width + x + sampleIndex
};

SR_INL SR_HD double fractDouble(double x) {
#if defined(__CUDA_ARCH__)
    return x - floor(x);
#else
    return x - std::floor(x);
#endif
}

// GenPnt1D — golden ratio quasirandom (shutter / 1D). Always double math.
SR_INL SR_HD float genPnt1D(int n) {
    constexpr double g = 1.6180339887498948482;
    constexpr double a1 = 1.0 / g;
    return float(fractDouble(a1 * double(n)));
}

// GenPnt2D — plastic-number R2 (Roberts). Pixel jitter dims.
SR_INL SR_HD void genPnt2D(int n, float& x, float& y) {
    constexpr double g = 1.32471795724474602596;
    constexpr double a1 = 1.0 / g;        // ≈ 0.7548776662466927
    constexpr double a2 = 1.0 / (g * g);  // ≈ 0.5698402909980532
    x = float(fractDouble(a1 * double(n)));
    y = float(fractDouble(a2 * double(n)));
}

// Same plastic base, dims 3–4 for lens / DoF (keeps AA and aperture stratified together).
SR_INL SR_HD void genPnt2DLens(int n, float& u, float& v) {
    constexpr double g = 1.32471795724474602596;
    constexpr double a3 = 1.0 / (g * g * g);
    constexpr double a4 = 1.0 / (g * g * g * g);
    u = float(fractDouble(a3 * double(n)));
    v = float(fractDouble(a4 * double(n)));
}

SR_INL SR_HD float wrap01Add(float u, float offset) {
    float v = u + offset;
#if defined(__CUDA_ARCH__)
    return v - floorf(v);
#else
    return v - std::floor(v);
#endif
}

SR_INL SR_HD float unitFloatFromU32(uint32_t h) {
    return float(h >> 8) * (1.0f / 16777216.0f);
}

// Per-pixel Cranley–Patterson phase so spp0 is not a flat (0,0) wallpaper.
SR_INL SR_HD float r2PixelPhase(int x, int y, uint32_t dimension) {
    const uint32_t h =
        hashUint(uint32_t(x) * 0xA511E9B3u ^ uint32_t(y) * 0xC2B2AE35u ^
                 (dimension * 0x9E3779B9u + 0x85EBCA6Bu));
    return unitFloatFromU32(h);
}

SR_INL SR_HD int r2SampleIndex(int x, int y, int sampleIndex, int width, int mode) {
    const int si = sampleIndex < 0 ? 0 : sampleIndex;
    if (mode == kR2IndexSppPixelSalt) {
        const uint32_t salt =
            uint32_t(hashPixelSample(x, y, 0u, 0u, 0x52A17u)) & 0xffffu;
        return si + int(salt);
    }
    if (mode == kR2IndexLinearSpp) {
        const int w = width > 0 ? width : 1;
        return y * w + x + si;
    }
    return si;  // kR2IndexSpp
}

SR_INL SR_HD void r2PixelJitter(int x, int y, int sampleIndex, int width, int mode, float& jx,
                                float& jy) {
    const int n = r2SampleIndex(x, y, sampleIndex, width, mode);
    float px = 0.0f, py = 0.0f;
    genPnt2D(n, px, py);
    jx = wrap01Add(px, r2PixelPhase(x, y, 0u));
    jy = wrap01Add(py, r2PixelPhase(x, y, 1u));
}

SR_INL SR_HD void r2LensSample(int x, int y, int sampleIndex, int width, int mode, float& u,
                               float& v) {
    const int n = r2SampleIndex(x, y, sampleIndex, width, mode);
    float lu = 0.0f, lv = 0.0f;
    genPnt2DLens(n, lu, lv);
    u = wrap01Add(lu, r2PixelPhase(x, y, 2u));
    v = wrap01Add(lv, r2PixelPhase(x, y, 3u));
}

// Motion-blur shutter in [0,1) — golden 1D + per-pixel phase (same mode as R2 pixel).
SR_INL SR_HD float r2ShutterSample(int x, int y, int sampleIndex, int width, int mode) {
    const int n = r2SampleIndex(x, y, sampleIndex, width, mode);
    return wrap01Add(genPnt1D(n), r2PixelPhase(x, y, 4u));
}

SR_INL SR_HD int r2IndexModeFromPixelSampler(int /*pixelSampler*/) {
    return kR2IndexSpp;  // GenPnt2D uses n = sampleIndex only
}

SR_INL SR_HD bool isR2PixelSampler(int pixelSampler) {
    return pixelSampler == 3;  // kPixelSamplerGenPnt2D
}

}  // namespace sol
