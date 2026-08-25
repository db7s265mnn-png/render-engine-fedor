// Small, fast RNG usable on host and device.
#pragma once

#include "core/math.h"

namespace sol {

// Optional QMC stream (Owen-scrambled Sobol). When useSobol is set, nextFloat()
// pulls dimension sampleDim, sampleDim+1, … from the Sobol index/scramble stored
// on this Rng — host and OptiX device share that path. qmcFn is a host-only
// fallback for older call sites.
using RngQmcFn = float (*)(void* ctx, uint32_t dimension);

#if !defined(SOL_RNG_NO_SOBOL)
// Defined in render/sobol.h (included below). Host table and device on-the-fly
// direction numbers produce the same Owen-scrambled values.
SR_HD float rngOwenSobolSample(uint32_t scramble, uint32_t index, uint32_t dimension);
#endif

enum RngBackend : uint8_t {
    kRngBackendPcg = 0,         // PCG32 (default)
    kRngBackendXorshift32 = 1,  // Marsaglia xorshift32 — optional Path Sampler
};

// PCG32 (O'Neill 2014) or optional xorshift32. Deterministic per pixel/sample.
struct Rng {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc = 0xda3e39cb94b95bdbULL;
    uint32_t xsState = 1u;  // xorshift32 — never 0 after seed()
    uint8_t backend = kRngBackendPcg;

    // Host QMC (Sobol path dims). Null on GPU / when unused.
    void* qmcCtx = nullptr;
    RngQmcFn qmcFn = nullptr;
    uint32_t sampleDim = 0u;  // pbrt: camera+path share one stream from dim 0
    uint32_t sobolScramble = 0u;
    uint32_t sobolIndex = 0u;
    uint8_t useSobol = 0;

    Rng() = default;

    SR_HD Rng(uint64_t seq, uint64_t seed) { init(seq, seed); }

    SR_HD void init(uint64_t seq, uint64_t seed) {
        backend = kRngBackendPcg;
        state = 0u;
        inc = (seq << 1u) | 1u;
        nextUintPcg();
        state += 0x853c49e6748fea9bULL + seed;
        nextUintPcg();
        qmcCtx = nullptr;
        qmcFn = nullptr;
        sampleDim = 0u;
        sobolScramble = 0u;
        sobolIndex = 0u;
        useSobol = 0;
        xsState = 1u;
    }

    // Marsaglia xorshift32: 4 bytes, very fast, period 2^32-1, never emits 0
    // if seeded non-zero. Optional Path Sampler alternative to PCG.
    // Caller should pass an already-mixed non-zero seed (see makePixelRngXorshift32).
    SR_HD void initXorshift32(uint32_t inSeed) {
        backend = kRngBackendXorshift32;
        qmcCtx = nullptr;
        qmcFn = nullptr;
        sampleDim = 0u;
        sobolScramble = 0u;
        sobolIndex = 0u;
        useSobol = 0;
        xsState = inSeed ? inSeed : 1u;
    }

    SR_HD uint32_t nextUintPcg() {
        const uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        const uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        const uint32_t rot = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    SR_HD uint32_t nextUintXorshift32() {
        // state ^= state<<13; state ^= state>>17; state ^= state<<5;
        uint32_t s = xsState;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        xsState = s;
        return s;
    }

    SR_HD uint32_t nextUint() {
        if (backend == kRngBackendXorshift32) return nextUintXorshift32();
        return nextUintPcg();
    }

    SR_HD float nextFloat() {
#if !defined(SOL_RNG_NO_SOBOL)
        if (useSobol) return rngOwenSobolSample(sobolScramble, sobolIndex, sampleDim++);
#endif
#if !defined(__CUDACC__)
        if (qmcFn) return qmcFn(qmcCtx, sampleDim++);
#endif
        if (backend == kRngBackendXorshift32) {
            // Full 32-bit → [0,1); clamp off 1.0 (matches reference f01).
            const float u = float(nextUintXorshift32()) * 2.3283064e-10f;
            return u < 1.0f - 1.192092896e-07f ? u : (1.0f - 1.192092896e-07f);
        }
        // PCG: 24 bits of mantissa → [0,1).
        return static_cast<float>(nextUintPcg() >> 8) * 0x1.0p-24f;
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
//
// Extra avalanche passes: a previous multiply-xor mix left measurable neighbor
// correlation in the low bits of PCG `inc` on some scenes (tile_test cell quilt).
SR_INL SR_HD uint64_t hashPixelSample(int x, int y, uint32_t sampleIndex, uint32_t frameSeed,
                                      uint32_t salt = 0u) {
    // Morton-ish interleave of low bits so (x+1) is not a tiny delta in the key.
    const uint32_t ux = uint32_t(x);
    const uint32_t uy = uint32_t(y);
    uint64_t h = 0x243f6a8885a308d3ULL;  // nothing-up-my-sleeve
    h ^= uint64_t(ux) * 0x9e3779b97f4a7c15ULL;
    h = mixBits64(h);
    h ^= uint64_t(uy) * 0xbf58476d1ce4e5b9ULL;
    h = mixBits64(h);
    h ^= uint64_t(sampleIndex) * 0x94d049bb133111ebULL;
    h = mixBits64(h);
    h ^= uint64_t(frameSeed) * 0x85ebca77c2b2ae63ULL;
    h = mixBits64(h);
    h ^= uint64_t(salt) * 0xc2b2ae3d27d4eb4fULL;
    // Fold x⊕y again so axis-aligned neighbors cannot share a near-linear key.
    h ^= (uint64_t(ux) << 32) | uint64_t(uy);
    return mixBits64(h);
}

// Path / White-camera PCG seeded per pixel. `salt` separates spectral hero channels etc.
SR_INL SR_HD Rng makePixelRng(int x, int y, int sampleIndex, uint32_t frameSeed, uint32_t salt = 0u) {
    const uint32_t si = uint32_t(sampleIndex < 0 ? 0 : sampleIndex);
    // Independent stream + seed — never derive one from the other with a weak xor.
    const uint64_t stream = hashPixelSample(x, y, si, frameSeed, salt);
    const uint64_t seed = hashPixelSample(x, y, si, frameSeed ^ 0xA511E9B3u, salt ^ 0xC2B2AE35u);
    return Rng(stream, seed);
}

// Same pixel keying as makePixelRng, but Marsaglia xorshift32 backend.
SR_INL SR_HD Rng makePixelRngXorshift32(int x, int y, int sampleIndex, uint32_t frameSeed,
                                         uint32_t salt = 0u) {
    const uint32_t si = uint32_t(sampleIndex < 0 ? 0 : sampleIndex);
    uint32_t s = uint32_t(hashPixelSample(x, y, si, frameSeed, salt ^ 0x51F15Fu));
    s = hashUint(s ? s : 2938653863u);
    Rng rng;
    rng.initXorshift32(s ? s : 1u);
    return rng;
}

}  // namespace sol

#ifndef SOL_RNG_NO_SOBOL
#include "render/sobol.h"
#endif
