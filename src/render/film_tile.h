// PBRT-style FilmTile: thread-local bucket accumulation, then merge into Film.
// Matches ImageTileIntegrator / GetFilmTile+MergeFilmTile (PBRT 3/4 tile loop):
// each worker owns a Bounds2i of pixels, writes samples only into that tile,
// then merges once — no cross-thread pixel contention on the beauty plane.
#pragma once

#include "core/math.h"

#include <cmath>
#include <algorithm>
#include <vector>

namespace sol {

struct FilmTile {
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
    std::vector<Vec4> pixels;  // RGB + sample weight, row-major in tile space

    FilmTile() = default;

    FilmTile(int x0In, int y0In, int x1In, int y1In)
        : x0(x0In), y0(y0In), x1(std::max(x0In, x1In)), y1(std::max(y0In, y1In)) {
        const int w = width();
        const int h = height();
        pixels.assign(size_t(std::max(0, w)) * size_t(std::max(0, h)), Vec4(0.0f, 0.0f, 0.0f, 0.0f));
    }

    int width() const { return x1 - x0; }
    int height() const { return y1 - y0; }
    bool empty() const { return width() <= 0 || height() <= 0; }

    bool contains(int x, int y) const { return x >= x0 && x < x1 && y >= y0 && y < y1; }

    void addSample(int x, int y, Vec3 radiance) {
        if (!contains(x, y) || pixels.empty()) return;
        Vec4& px = pixels[size_t(y - y0) * size_t(width()) + size_t(x - x0)];
        px.x += radiance.x;
        px.y += radiance.y;
        px.z += radiance.z;
        px.w += 1.0f;
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
