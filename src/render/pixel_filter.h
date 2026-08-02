// Film reconstruction filters (PBRT-v4 PixelFilter).
// A continuous sample at film position (fx, fy) is weighted into discrete pixels.
#pragma once

#include "core/math.h"
#include "scene/types.h"

#include <cmath>

namespace sol {

// Default radii = pbrt-v4 PixelFilter defaults (filters.cpp Create).
SR_INL float defaultFilterRadius(int filterType) {
    switch (filterType) {
        case kPixelFilterTriangle: return 2.0f;
        case kPixelFilterGaussian: return 1.5f;
        case kPixelFilterMitchell: return 2.0f;
        case kPixelFilterBox:
        default: return 0.5f;
    }
}

// Integer pixel border a FilmTile must pad so filtered splats stay local.
SR_INL int filterPixelBorder(float radius) {
    if (!(radius > 0.5f)) return 0;
    return int(std::ceil(radius - 1e-4f));
}

// Separable 1D kernel weight for distance |t| in pixels from the pixel center.
SR_INL float filterWeight1D(int filterType, float t, float radius) {
    const float x = std::fabs(t);
    if (x >= radius) return 0.0f;
    switch (filterType) {
        case kPixelFilterTriangle: {
            // PBRT TriangleFilter is (r-|x|)(r-|y|); we store the unit tent
            // (1-|x|/r). Scale cancels after weight normalization.
            return 1.0f - x / radius;
        }
        case kPixelFilterGaussian: {
            // PBRT-v4 GaussianFilter: sigma=0.5 ≡ legacy α=2 (filters.cpp note).
            constexpr float alpha = 2.0f;
            const float expR = std::exp(-alpha * radius * radius);
            return std::max(0.0f, std::exp(-alpha * x * x) - expR);
        }
        case kPixelFilterMitchell: {
            // Mitchell–Netravali B=C=1/3 (PBRT MitchellFilter), support [-radius,radius]
            // with parametric x mapped so |x|<=2 in the classic polynomial.
            const float xx = x * (2.0f / radius);
            constexpr float B = 1.0f / 3.0f;
            constexpr float C = 1.0f / 3.0f;
            const float ax = std::fabs(xx);
            if (ax < 1.0f) {
                return ((12.0f - 9.0f * B - 6.0f * C) * ax * ax * ax +
                        (-18.0f + 12.0f * B + 6.0f * C) * ax * ax + (6.0f - 2.0f * B)) *
                       (1.0f / 6.0f);
            }
            if (ax < 2.0f) {
                return ((-B - 6.0f * C) * ax * ax * ax + (6.0f * B + 30.0f * C) * ax * ax +
                        (-12.0f * B - 48.0f * C) * ax + (8.0f * B + 24.0f * C)) *
                       (1.0f / 6.0f);
            }
            return 0.0f;
        }
        case kPixelFilterBox:
        default:
            return 1.0f;
    }
}

SR_INL float filterWeight2D(int filterType, float dx, float dy, float radius) {
    return filterWeight1D(filterType, dx, radius) * filterWeight1D(filterType, dy, radius);
}

// True box (one pixel, weight 1) — matches historical addSample(x,y,L) behaviour.
SR_INL bool isTrivialBoxFilter(int filterType, float radius) {
    return filterType == kPixelFilterBox && !(radius > 0.5f);
}

// Splat continuous film sample (fx,fy) with radiance L into discrete pixels via `add`.
// `add(px, py, weightedRadiance, weight)` must accumulate RGB += L*w and W += w.
template <typename AddFn>
SR_INL void splatFilteredSample(float fx, float fy, Vec3 L, int filterType, float radius, int imgW,
                                int imgH, AddFn&& add) {
    if (!(radius > 0.0f) || imgW <= 0 || imgH <= 0) return;
    if (isTrivialBoxFilter(filterType, radius)) {
        const int px = int(std::floor(fx));
        const int py = int(std::floor(fy));
        if (px >= 0 && py >= 0 && px < imgW && py < imgH) add(px, py, L, 1.0f);
        return;
    }
    // Pixel centers at (i+0.5, j+0.5); contribute where |fx-(i+0.5)| < radius.
    const int x0 = std::max(0, int(std::ceil(fx - radius - 0.5f)));
    const int x1 = std::min(imgW, int(std::floor(fx + radius - 0.5f)) + 1);
    const int y0 = std::max(0, int(std::ceil(fy - radius - 0.5f)));
    const int y1 = std::min(imgH, int(std::floor(fy + radius - 0.5f)) + 1);
    for (int py = y0; py < y1; ++py) {
        const float dy = fy - (float(py) + 0.5f);
        for (int px = x0; px < x1; ++px) {
            const float dx = fx - (float(px) + 0.5f);
            const float w = filterWeight2D(filterType, dx, dy, radius);
            if (w > 0.0f) add(px, py, L * w, w);
        }
    }
}

}  // namespace sol
