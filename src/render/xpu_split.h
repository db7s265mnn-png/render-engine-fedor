// XPU work split: Overlap (even GPU / odd CPU spp), Mixture films, Tile rects.
#pragma once

#include "scene/types.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace sol {

// Overlap: even sample indices on GPU, odd on CPU. GPU may consume many even
// spp while Embree finishes one odd spp.
SR_INL SR_HD bool xpuGpuOwnsSample(int sampleIndex) { return (sampleIndex & 1) == 0; }

// RenderMan CPU bucket: 1024 px → 32×32. Cycles CPU tiles are the same order.
inline constexpr int kXpuCpuTilePx = 32;
// RenderMan GPU working set ~500k px. 22×32 = 704 → 704² = 495616.
// Kept as documentation; this engine already allocates full-frame wavefront
// buffers, so Tile uses one (or two) large GPU rects instead of a 704 grid.
inline constexpr int kXpuGpuPackPx = 704;

// Below this, a GPU launch is all overhead — give the whole frame to Embree.
inline constexpr int kXpuTileMinGpuPx = 128 * 128;
// Cap Embree's exclusive share so Tile wall time stays near GPU-only.
// A full-width 32-px strip at 1080p is 61k px (~3%); at 4K/8K the strip
// would be larger than this cap, so Tile becomes one GPU launch.
inline constexpr int kXpuTileCpuAreaCap = 64 * 1024;

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

inline int xpuAreaSum(const std::vector<XpuWorkRect>& rects) {
    int a = 0;
    for (const XpuWorkRect& r : rects) a += r.area();
    return a;
}

inline XpuWorkRect xpuBounds(const std::vector<XpuWorkRect>& rects) {
    if (rects.empty()) return {};
    XpuWorkRect u = rects[0];
    for (const XpuWorkRect& r : rects) {
        u.x0 = std::min(u.x0, r.x0);
        u.y0 = std::min(u.y0, r.y0);
        u.x1 = std::max(u.x1, r.x1);
        u.y1 = std::max(u.y1, r.y1);
    }
    return u;
}

inline void xpuPushCpuTiles(XpuWorkLists& out, int x0, int y0, int x1, int y1) {
    for (int y = y0; y < y1; y += kXpuCpuTilePx) {
        for (int x = x0; x < x1; x += kXpuCpuTilePx) {
            out.cpuTiles.push_back(
                {x, y, std::min(x + kXpuCpuTilePx, x1), std::min(y + kXpuCpuTilePx, y1)});
        }
    }
}

inline void xpuPushGpuPack(XpuWorkLists& out, int x0, int y0, int x1, int y1) {
    if (x1 > x0 && y1 > y0) out.gpuPacks.push_back({x0, y0, x1, y1});
}

// Leave one host thread for OptiX/CUDA submit so Embree cannot starve the GPU.
inline int xpuEmbreeThreadCount(int requested) {
    int n = requested > 0 ? requested : int(std::thread::hardware_concurrency());
    if (n <= 0) n = 4;
    return n <= 1 ? 1 : n - 1;
}

// Independent estimators: CPU Sobol/path RNG lives in a disjoint sample-index band.
inline int xpuCpuSampleIndex(int cpuSpp) { return 0x40000000 + cpuSpp; }

// Tile: GPU owns one large exclusive rect (almost the full frame — one OptiX
// launch, not a 704² grid). CPU owns a 32-px exclusive strip of 32×32 buckets
// only when that strip fits the Embree budget. No stealing: GPU 32×32 launches
// cost a full bounce loop each, and Embree on a 704² pack is catastrophic.
inline XpuWorkLists xpuBuildWorkLists(int width, int height) {
    XpuWorkLists out;
    const int w = std::max(0, width);
    const int h = std::max(0, height);
    if (w <= 0 || h <= 0) return out;

    if (w * h < kXpuTileMinGpuPx || w < kXpuCpuTilePx || h < kXpuCpuTilePx) {
        xpuPushCpuTiles(out, 0, 0, w, h);
        return out;
    }

    const int stripArea = w * kXpuCpuTilePx;
    const int gpuH = h - kXpuCpuTilePx;
    if (gpuH > 0 && stripArea <= kXpuTileCpuAreaCap && w * gpuH >= kXpuTileMinGpuPx) {
        xpuPushGpuPack(out, 0, 0, w, gpuH);
        xpuPushCpuTiles(out, 0, gpuH, w, h);
        return out;
    }

    xpuPushGpuPack(out, 0, 0, w, h);
    return out;
}

}  // namespace sol
