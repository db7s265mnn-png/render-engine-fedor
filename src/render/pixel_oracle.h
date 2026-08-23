// Karma XPU-style variance pixel oracle.
// After a few primary samples, stop pixels whose relative luminance error
// (and that of their 4-neighbours) is below Noise Threshold. 0 disables.
#pragma once

#include <algorithm>
#include <cmath>

namespace sol {

inline float pixelLuminance(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Karma fires a handful of camera rays before the variance oracle starts.
inline int noiseOracleMinSamples(int maxSpp) {
    if (maxSpp <= 1) return 1;
    return std::max(4, maxSpp / 16);
}

// Relative standard error of per-sample luminance vs threshold.
// lumSqSum is Σ L(sample)². mean* is the reconstructed pixel (Σ rgb / n).
inline bool noiseOraclePixelQuiet(float meanR, float meanG, float meanB, float lumSqSum, int n,
                                  float threshold) {
    if (!(threshold > 0.0f) || n <= 0) return false;
    const float meanL = pixelLuminance(meanR, meanG, meanB);
    // Missing L²: stay open unless the pixel is already black (background / empty).
    // A non-zero mean with no L² means a backend failed to track per-sample energy.
    if (!(lumSqSum > 0.0f)) return meanL <= 1.0e-3f;
    const float invN = 1.0f / float(n);
    const float var = std::max(0.0f, lumSqSum * invN - meanL * meanL);
    const float stderr = std::sqrt(var * invN);
    const float rel = stderr / std::max(meanL, 1.0e-3f);
    return rel <= threshold;
}

}  // namespace sol
