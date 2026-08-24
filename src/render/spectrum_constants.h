// Shared spectral limits. Host (Embree) and device (OptiX) must agree.
#pragma once

namespace sol {

constexpr int kMaxSpectrumSamples = 16;
constexpr float kSpectrumLambdaMin = 360.0f;
constexpr float kSpectrumLambdaMax = 830.0f;

constexpr int kJakobTableRes = 16;
constexpr int kJakobCoeffCount = 36864;  // 3 × 16³ × 3

}  // namespace sol
