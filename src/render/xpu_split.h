// Shared XPU (Embree+OptiX) work split: Karma-style checkerboard buckets.
#pragma once

#include "scene/types.h"

namespace sol {

SR_INL SR_HD int xpuTileSizeOrDefault(int tileSize) {
    if (tileSize <= 0) return 32;
    if (tileSize < 8) return 8;
    if (tileSize > 256) return 256;
    return tileSize;
}

// GPU owns tiles where (tx + ty) is even when gpuParity is 0 (default).
SR_INL SR_HD bool xpuGpuOwnsTile(int tx, int ty, int gpuParity) {
    return ((tx + ty) & 1) == (gpuParity & 1);
}

SR_INL SR_HD bool xpuGpuOwnsPixel(int x, int y, int tileSize, int gpuParity) {
    const int ts = xpuTileSizeOrDefault(tileSize);
    const int tx = x / ts;
    const int ty = y / ts;
    return xpuGpuOwnsTile(tx, ty, gpuParity);
}

}  // namespace sol
