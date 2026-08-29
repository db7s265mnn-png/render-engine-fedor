// Per-thread BDPT subpath storage (pbrt ScratchBuffer).
// CPU spectral BDPT used to HeapAlloc six std::vectors per pixel, sized
// maxDepth+1. At maxDepth 30 that crosses the Windows ~16 KiB LFH line
// (31 × 528 B Vert with OpenPGL) and the allocator dominates the pass.
// One buffer per ThreadPool slot (N workers + caller = N+1), grown to the
// session Max Ray Depth (not the 4096 compile cap) and reused for every pixel.
#pragma once

#include <algorithm>
#include <vector>

#include "render/integrator_bdpt.h"
#include "render/spectrum.h"

namespace sol {

struct BdptScratch {
    std::vector<bdpt::Vert> eye;
    std::vector<bdpt::Vert> light;
    std::vector<SampledSpectrum> eyeBeta;
    std::vector<SampledSpectrum> lightBeta;
    std::vector<SampledWavelengths> eyeWave;
    std::vector<SampledWavelengths> lightWave;

    void ensure(int verts) {
        const size_t n = size_t(std::max(2, verts));
        if (eye.size() >= n) return;
        resizeTo(int(n));
    }

    // Grow or shrink. Call at pass start when Max Ray Depth changes — never
    // per pixel (that reintroduces the Windows LFH cliff).
    void resizeTo(int verts) {
        const size_t n = size_t(std::max(2, verts));
        if (eye.size() == n) return;
        eye.resize(n);
        light.resize(n);
        eyeBeta.resize(n);
        lightBeta.resize(n);
        eyeWave.resize(n);
        lightWave.resize(n);
    }
};

inline size_t bdptScratchBytes(int verts) {
    const size_t n = size_t(std::max(2, verts));
    return 2 * n * sizeof(bdpt::Vert) + 2 * n * sizeof(SampledSpectrum) +
           2 * n * sizeof(SampledWavelengths);
}

class BdptScratchPool {
public:
    void ensureThreads(int threadSlots, int verts, bool shrink = false) {
        if (threadSlots < 1) threadSlots = 1;
        if (int(slots_.size()) < threadSlots) slots_.resize(size_t(threadSlots));
        for (BdptScratch& s : slots_) {
            if (shrink) s.resizeTo(verts);
            else s.ensure(verts);
        }
    }

    BdptScratch* get(int threadId) {
        if (slots_.empty()) return nullptr;
        size_t i = size_t(threadId < 0 ? 0 : threadId);
        if (i >= slots_.size()) i = slots_.size() - 1;
        return &slots_[i];
    }

    int threadSlots() const { return int(slots_.size()); }

private:
    std::vector<BdptScratch> slots_;
};

// Fallback when Li() is called without a pool (unit tests, missed call sites).
BdptScratch& bdptThreadScratch();

}  // namespace sol
