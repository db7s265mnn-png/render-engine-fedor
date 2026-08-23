// XPU work split: Overlap (even GPU / odd CPU spp) and Mixture films.
#pragma once

#include "scene/types.h"

#include <thread>

namespace sol {

// Overlap: even sample indices on GPU, odd on CPU. GPU may consume many even
// spp while Embree finishes one odd spp.
SR_INL SR_HD bool xpuGpuOwnsSample(int sampleIndex) { return (sampleIndex & 1) == 0; }

// Leave one host thread for OptiX/CUDA submit so Embree cannot starve the GPU.
inline int xpuEmbreeThreadCount(int requested) {
    int n = requested > 0 ? requested : int(std::thread::hardware_concurrency());
    if (n <= 0) n = 4;
    return n <= 1 ? 1 : n - 1;
}

// Independent estimators: CPU Sobol/path RNG lives in a disjoint sample-index band.
inline int xpuCpuSampleIndex(int cpuSpp) { return 0x40000000 + cpuSpp; }

}  // namespace sol
