// Compacted pixel queues for the OptiX wavefront (Iray / Cycles work queues).
// Shade kernels never call optixTrace. Host launches 1D on live count.
#pragma once

#include "render/optix/optix_wavefront.cuh"

namespace sol {

__device__ inline void enqueueSlot(int slot, int pixel) {
    const LaunchParams& p = launchParams();
    if (!p.workCounts || pixel < 0 || p.width <= 0 || p.height <= 0) return;
    int* queue = nullptr;
    switch (slot) {
        case kSlotIntersect:
            queue = p.qIntersect;
            break;
        case kSlotIntersectNext:
            queue = p.qIntersectNext;
            break;
        case kSlotVolume:
            queue = p.qVolume;
            break;
        case kSlotSurface:
            queue = p.qSurface;
            break;
        case kSlotBackground:
            queue = p.qBackground;
            break;
        case kSlotShadow:
            queue = p.qShadow;
            break;
        default:
            return;
    }
    if (!queue) return;
    const unsigned cap = unsigned(p.width) * unsigned(p.height);
    const unsigned i = atomicAdd(p.workCounts + slot, 1u);
    if (i < cap) queue[int(i)] = pixel;
}

__device__ inline void enqueuePathContinuation(int pixel) {
    const LaunchParams& p = launchParams();
    if (!p.paths || !p.shadows) return;
    const GpuPath& path = p.paths[pixel];
    const GpuShadow& shadow = p.shadows[pixel];
    if (shadow.queue == kShadowTrace) enqueueSlot(kSlotShadow, pixel);
    if (path.queue == kQueueIntersectClosest) enqueueSlot(kSlotIntersectNext, pixel);
    else if (path.queue == kQueueShadeVolume) enqueueSlot(kSlotVolume, pixel);
    else if (path.queue == kQueueShadeSurface) enqueueSlot(kSlotSurface, pixel);
    else if (path.queue == kQueueShadeBackground) enqueueSlot(kSlotBackground, pixel);
}

}  // namespace sol
