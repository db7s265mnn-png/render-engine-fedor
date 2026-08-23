// XPU tile geometry (RenderMan / Cycles) and host-thread split.
// Mixture mode does not use these rects — CPU and GPU each own a full-frame film.
#pragma once

#include "scene/types.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace sol {

// RenderMan CPU bucket: 1024 px → 32×32. Cycles CPU tiles are the same order.
inline constexpr int kXpuCpuTilePx = 32;
// RenderMan GPU working set ~500k px. 22×32 = 704 → 704² = 495616.
inline constexpr int kXpuGpuPackPx = 704;

struct XpuWorkRect {
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    int width() const { return x1 - x0; }
    int height() const { return y1 - y0; }
    int area() const { return width() * height(); }
};

struct XpuWorkLists {
    std::vector<XpuWorkRect> gpuPacks;
    std::vector<XpuWorkRect> cpuTiles;
};

// Leave one host thread for OptiX/CUDA submit so Embree cannot starve the GPU.
inline int xpuEmbreeThreadCount(int requested) {
    int n = requested > 0 ? requested : int(std::thread::hardware_concurrency());
    if (n <= 0) n = 4;
    return n <= 1 ? 1 : n - 1;
}

// Independent estimators: CPU Sobol/path RNG lives in a disjoint sample-index band.
inline int xpuCpuSampleIndex(int cpuSpp) { return 0x40000000 + cpuSpp; }

// Non-overlapping partition: large GPU packs where they fit, else 32×32 CPU tiles.
inline XpuWorkLists xpuBuildWorkLists(int width, int height) {
    XpuWorkLists out;
    const int w = std::max(0, width);
    const int h = std::max(0, height);
    if (w <= 0 || h <= 0) return out;
    int y = 0;
    while (y < h) {
        const int packH = std::min(kXpuGpuPackPx, h - y);
        int x = 0;
        while (x < w) {
            const int packW = std::min(kXpuGpuPackPx, w - x);
            if (packW >= 64 && packH >= 64) {
                out.gpuPacks.push_back({x, y, x + packW, y + packH});
            } else {
                for (int ty = y; ty < y + packH; ty += kXpuCpuTilePx) {
                    for (int tx = x; tx < x + packW; tx += kXpuCpuTilePx) {
                        out.cpuTiles.push_back({tx, ty, std::min(tx + kXpuCpuTilePx, x + packW),
                                                std::min(ty + kXpuCpuTilePx, y + packH)});
                    }
                }
            }
            x += packW;
        }
        y += packH;
    }
    return out;
}

}  // namespace sol
