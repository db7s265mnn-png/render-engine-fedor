// XPU work split: Overlap (even GPU / odd CPU spp), Mixture films, Tile rects.
#pragma once

#include "scene/types.h"

#include <algorithm>
#include <cmath>
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
// First spp: Embree gets ~1/8 of the frame (32×32 buckets). Then Tile
// resizes that share so CPU ms ≈ GPU ms — both stay busy like RenderMan.
inline constexpr int kXpuTileCpuShare = 8;
inline constexpr int kXpuTileMaxCpuShare = 3;

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

inline int xpuTileMaxCpuArea(int width, int height) {
    const int w = std::max(0, width);
    const int h = std::max(0, height);
    const long long total = 1LL * w * h;
    const int minA = kXpuCpuTilePx * kXpuCpuTilePx;
    if (total <= minA) return int(total);
    const long long capShare = total / kXpuTileMaxCpuShare;
    const long long capGpu = total - kXpuTileMinGpuPx;
    const long long cap = std::max(1LL * minA, std::min(capShare, capGpu));
    return int(std::min(cap, 0x7fffffffLL));
}

inline int xpuDefaultTileCpuArea(int width, int height) {
    const int w = std::max(0, width);
    const int h = std::max(0, height);
    const long long total = 1LL * w * h;
    const int minA = kXpuCpuTilePx * kXpuCpuTilePx;
    if (total <= minA) return int(total);
    const int maxA = xpuTileMaxCpuArea(w, h);
    const int want = int(std::max(1LL * minA, total / kXpuTileCpuShare));
    return std::clamp(want, minA, maxA);
}

// After each Tile spp: grow/shrink the Embree exclusive block so CPU wall ≈ GPU wall.
inline int xpuAdaptTileCpuArea(int curArea, int width, int height, double cpuMs, double gpuMs) {
    const int minA = kXpuCpuTilePx * kXpuCpuTilePx;
    const int maxA = xpuTileMaxCpuArea(width, height);
    if (curArea <= 0) return xpuDefaultTileCpuArea(width, height);
    if (cpuMs < 1.0 || gpuMs < 1.0) return std::clamp(curArea, minA, maxA);
    const int target = int(std::llround(double(curArea) * (gpuMs / cpuMs)));
    const int blended = (curArea + std::clamp(target, minA, maxA)) / 2;
    return std::clamp(blended, minA, maxA);
}

// Tile: GPU traces one or two large exclusive rects (few OptiX launches, not a
// 704² grid). CPU traces a 32×32 exclusive block (~1/8 of the frame, then
// adapted so Embree ms ≈ GPU ms). No steal — GPU 32×32 launches each cost a
// full bounce loop, and Embree on a 704² pack is catastrophic.
inline XpuWorkLists xpuBuildWorkLists(int width, int height, int cpuAreaHint = 0) {
    XpuWorkLists out;
    const int w = std::max(0, width);
    const int h = std::max(0, height);
    if (w <= 0 || h <= 0) return out;

    const int tile = kXpuCpuTilePx;
    const long long total = 1LL * w * h;
    if (total < kXpuTileMinGpuPx || w < tile || h < tile) {
        xpuPushCpuTiles(out, 0, 0, w, h);
        return out;
    }

    int cpuArea = cpuAreaHint > 0 ? cpuAreaHint : xpuDefaultTileCpuArea(w, h);
    cpuArea = std::clamp(cpuArea, tile * tile, xpuTileMaxCpuArea(w, h));

    int cpuW = w;
    int cpuH = tile;
    const int fullRow = w * tile;
    if (fullRow > 0 && cpuArea >= fullRow) {
        cpuH = std::min(h - tile, std::max(tile, (cpuArea / fullRow) * tile));
        if (cpuH < tile) cpuH = tile;
        if (h - cpuH < tile && h > tile) cpuH = h - tile;
        cpuW = w;
    } else {
        const int nTiles = std::max(1, (cpuArea + tile * tile - 1) / (tile * tile));
        const int maxTilesX = std::max(1, w / tile);
        int tilesX = std::min(maxTilesX, nTiles);
        int tilesY = 1;
        if (tilesX * tile * tile < cpuArea) {
            tilesY = std::min(std::max(1, (h / tile) - 1), (nTiles + tilesX - 1) / tilesX);
            if (tilesY < 1) tilesY = 1;
        }
        cpuW = std::min(w, tilesX * tile);
        cpuH = std::min(h, tilesY * tile);
        if (cpuW >= w && cpuH >= h) {
            cpuH = std::max(tile, h - tile);
            cpuW = w;
        }
    }

    if (total - 1LL * cpuW * cpuH < kXpuTileMinGpuPx) {
        cpuW = tile;
        cpuH = tile;
    }

    const int cx0 = std::max(0, w - cpuW);
    const int cy0 = std::max(0, h - cpuH);
    xpuPushCpuTiles(out, cx0, cy0, w, h);
    xpuPushGpuPack(out, 0, 0, w, cy0);
    xpuPushGpuPack(out, 0, cy0, cx0, h);
    if (out.gpuPacks.empty() && out.cpuTiles.empty()) xpuPushCpuTiles(out, 0, 0, w, h);
    return out;
}

}  // namespace sol
