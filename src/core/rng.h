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

}  // namespace sol
