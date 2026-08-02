// Small, fast RNG usable on host and device.
#pragma once

#include "core/math.h"

namespace sol {

// Optional host-side QMC stream (Owen-scrambled Sobol). When set, nextFloat()
// pulls dimension sampleDim, sampleDim+1, … instead of PCG. Device/CUDA keeps
// sampleFn null and uses PCG only.
using RngQmcFn = float (*)(void* ctx, uint32_t dimension);

// PCG32 - O'Neill 2014. Deterministic per pixel/sample which keeps CPU and GPU
// renders reproducible for a given seed.
struct Rng {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc = 0xda3e39cb94b95bdbULL;

    // Host QMC (Sobol path dims). Null on GPU / when unused.
    void* qmcCtx = nullptr;
    RngQmcFn qmcFn = nullptr;
    uint32_t sampleDim = 4u;  // dims 0-3 reserved for pixel/lens

    Rng() = default;

    SR_HD Rng(uint64_t seq, uint64_t seed) { init(seq, seed); }

    SR_HD void init(uint64_t seq, uint64_t seed) {
        state = 0u;
        inc = (seq << 1u) | 1u;
        nextUint();
        state += 0x853c49e6748fea9bULL + seed;
        nextUint();
        qmcCtx = nullptr;
        qmcFn = nullptr;
        sampleDim = 4u;
    }

    SR_HD uint32_t nextUint() {
        const uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        const uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        const uint32_t rot = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    SR_HD float nextFloat() {
#if !defined(__CUDACC__)
        if (qmcFn) return qmcFn(qmcCtx, sampleDim++);
#endif
        // 24 bits of mantissa gives values in [0,1).
        return static_cast<float>(nextUint() >> 8) * 0x1.0p-24f;
    }

    SR_HD Vec2 next2D() {
        const float a = nextFloat();
        const float b = nextFloat();
        return Vec2(a, b);
    }
};

// Hash used to decorrelate pixel/sample/bounce dimensions.
SR_INL SR_HD uint32_t hashCombine(uint32_t a, uint32_t b) {
    a ^= b + 0x9e3779b9u + (a << 6) + (a >> 2);
    return a;
}

SR_INL SR_HD uint32_t hashUint(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Strong 64-bit finalizer (SplitMix64 / PBRT MixBits family). Avoids the
// spatial correlation that weak hashCombine(y*width+x, …) leaves in low bits —
// neighboring pixels used to share related PCG streams and print faint seams.
SR_INL SR_HD uint64_t mixBits64(uint64_t v) {
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    return v ^ (v >> 31);
}

// Decorrelated hash of pixel + progressive sample + frame seed (+ optional salt).
// Uses (x,y) directly — never a linear pixelIndex — so row width cannot imprint
// a lattice on the RNG streams.
SR_INL SR_HD uint64_t hashPixelSample(int x, int y, uint32_t sampleIndex, uint32_t frameSeed,
                                      uint32_t salt = 0u) {
    uint64_t h = uint64_t(uint32_t(x)) * 0x9e3779b97f4a7c15ULL;
    h ^= uint64_t(uint32_t(y)) * 0xbf58476d1ce4e5b9ULL;
    h ^= uint64_t(sampleIndex) * 0x94d049bb133111ebULL;
    h ^= uint64_t(frameSeed) * 0x85ebca77c2b2ae63ULL;
    h ^= uint64_t(salt) * 0xc2b2ae3d27d4eb4fULL;
    return mixBits64(h);
}

// Path / White-camera PCG seeded per pixel. `salt` separates spectral hero channels etc.
SR_INL SR_HD Rng makePixelRng(int x, int y, int sampleIndex, uint32_t frameSeed, uint32_t salt = 0u) {
    const uint64_t stream = hashPixelSample(x, y, uint32_t(sampleIndex < 0 ? 0 : sampleIndex), frameSeed, salt);
    const uint64_t seed =
        hashPixelSample(x, y, uint32_t(sampleIndex < 0 ? 0 : sampleIndex), frameSeed, salt ^ 0xa5a5a5a5u);
    return Rng(stream, seed);
}

}  // namespace sol
