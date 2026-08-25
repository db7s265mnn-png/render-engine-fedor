// Shared spectral limits. Host (Embree) and device (OptiX) must agree.
#pragma once

namespace sol {

// pbrt-v4 NSpectrumSamples: compile-time 4. Not a UI slider.
constexpr int kMaxSpectrumSamples = 4;
constexpr float kSpectrumLambdaMin = 360.0f;
constexpr float kSpectrumLambdaMax = 830.0f;

constexpr int kJakobTableRes = 16;
constexpr int kJakobCoeffCount = 36864;  // 3 × 16³ × 3

}  // namespace sol
