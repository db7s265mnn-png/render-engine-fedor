// XPU work split: Overlap (GPU Iray batches vs one Embree spp) and Mixture films.
#pragma once

#include "scene/types.h"

#include <thread>

namespace sol {

// Leave one host thread for OptiX/CUDA submit so Embree cannot starve the GPU.
inline int xpuEmbreeThreadCount(int requested) {
    int n = requested > 0 ? requested : int(std::thread::hardware_concurrency());
    if (n <= 0) n = 4;
    return n <= 1 ? 1 : n - 1;
}

// Independent estimators: CPU Owen-Sobol lives in a disjoint sample-index band.
// GPU Iray regen needs consecutive indices (PCG) starting at 0 on its own film.
inline int xpuCpuSampleIndex(int cpuSpp) { return 0x40000000 + cpuSpp; }

// spp the GPU may still fold into the Iray wavefront (target is the session cap).
inline int xpuGpuRemaining(int target, int gpuDone, int cpuDone) {
    if (target <= 0) return 0;
    const int left = target - gpuDone - cpuDone;
    return left > 0 ? left : 0;
}

}  // namespace sol
