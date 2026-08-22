// XPU work split: even sample indices on GPU, odd on CPU.
// GPU may consume many even spp while Embree finishes one odd spp.
#pragma once

#include "scene/types.h"

#include <algorithm>
#include <thread>

namespace sol {

SR_INL SR_HD bool xpuGpuOwnsSample(int sampleIndex) { return (sampleIndex & 1) == 0; }

// Leave one host thread for OptiX/CUDA submit so Embree cannot starve the GPU.
inline int xpuEmbreeThreadCount(int requested) {
    int n = requested > 0 ? requested : int(std::thread::hardware_concurrency());
    if (n <= 0) n = 4;
    return n <= 1 ? 1 : n - 1;
}

}  // namespace sol
