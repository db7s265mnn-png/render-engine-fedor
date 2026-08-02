// PBRT-style FilmTile: thread-local bucket accumulation, then merge into Film.
// Matches ImageTileIntegrator / GetFilmTile+MergeFilmTile (PBRT 3/4 tile loop):
// each worker owns a Bounds2i of pixels, writes samples only into that tile,
// then merges once — no cross-thread pixel contention on the beauty plane.
//
// With a non-box reconstruction filter the tile stores a border of
// ceil(radius) pixels so filtered splats stay local (PBRT FilmTile extent).
#pragma once

#include "core/math.h"
#include "render/pixel_filter.h"

#include <cmath>
#include <algorithm>
#include <vector>

namespace sol {

struct FilmTile {
    int x0 = 0;  // exclusive sample / owned pixel range (no border)
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    int border = 0;                // storage padding around [x0,x1)×[y0,y1)
    int storX0 = 0, storY0 = 0;    // inclusive storage origin
    int storW = 0, storH = 0;
    std::vector<Vec4> pixels;  // RGB + filter weight, row-major in storage space

    FilmTile() = default;

    FilmTile(int x0In, int y0In, int x1In, int y1In, int borderIn = 0)
        : x0(x0In), y0(y0In), x1(std::max(x0In, x1In)), y1(std::max(y0In, y1In)),
          border(std::max(0, borderIn)) {
        storX0 = x0 - border;
        storY0 = y0 - border;
        storW = (x1 + border) - storX0;
        storH = (y1 + border) - storY0;
        pixels.assign(size_t(std::max(0, storW)) * size_t(std::max(0, storH)),
                      Vec4(0.0f, 0.0f, 0.0f, 0.0f));
    }

    int width() const { return x1 - x0; }
    int height() const { return y1 - y0; }
    bool empty() const { return width() <= 0 || height() <= 0 || storW <= 0 || storH <= 0; }

    bool contains(int x, int y) const { return x >= x0 && x < x1 && y >= y0 && y < y1; }

    bool inStorage(int x, int y) const {
        return x >= storX0 && y >= storY0 && x < storX0 + storW && y < storY0 + storH;
    }

    void addWeighted(int x, int y, Vec3 weightedRadiance, float weight) {
        if (!inStorage(x, y) || pixels.empty() || !(weight > 0.0f)) return;
        Vec4& px = pixels[size_t(y - storY0) * size_t(storW) + size_t(x - storX0)];
        px.x += weightedRadiance.x;
        px.y += weightedRadiance.y;
        px.z += weightedRadiance.z;
        px.w += weight;
    }

    // Legacy: discrete pixel, weight 1 (trivial box).
    void addSample(int x, int y, Vec3 radiance) { addWeighted(x, y, radiance, 1.0f); }

    // Continuous film sample at (fx, fy) through the reconstruction filter.
    void addFilteredSample(float fx, float fy, Vec3 radiance, int filterType, float radius,
                           int imgW, int imgH) {
        splatFilteredSample(fx, fy, radiance, filterType, radius, imgW, imgH,
                            [&](int px, int py, Vec3 Lw, float w) { addWeighted(px, py, Lw, w); });
    }
};

// PBRT ParallelFor2D-style tile size: enough tiles for load balance, not so many
// that scheduling overhead dominates. Clamped to [8, 64] like pbrt's practical range.
inline int chooseFilmTileSize(int width, int height, int threadCount) {
    const int w = std::max(1, width);
    const int h = std::max(1, height);
    const int threads = std::max(1, threadCount);
    const int targetTiles = std::max(threads * 8, threads + 1);
    const int nPixels = w * h;
    const int tileArea = std::max(1, (nPixels + targetTiles - 1) / targetTiles);
    int side = int(std::sqrt(float(tileArea)) + 0.5f);
    side = std::clamp(side, 8, 64);
    // Prefer power-of-two-ish sizes for nicer IPR grids when close.
    if (side > 48) side = 64;
    else if (side > 24) side = 32;
    else if (side > 12) side = 16;
    else side = 8;
    return side;
}

}  // namespace sol
