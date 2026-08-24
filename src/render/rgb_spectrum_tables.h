// Jakob & Hanika (2019) RGB→spectrum upsampling tables for Solstice PT Spectral.
// Sigmoid of a quadratic polynomial; coefficients looked up from compact 3D tables
// (albedo for reflectance, illuminant for emission). See rgb_spectrum_tables.cpp.
#pragma once

#include <cmath>

#include "render/spectrum.h"

namespace sol {
namespace rgb_spec {

// f(λ) = σ(c0 λ² + c1 λ + c2), σ(x) = ½ + x / (2 √(1+x²)) ∈ [0,1].
struct RgbSigmoidPolynomial {
    float c0 = 0.0f;
    float c1 = 0.0f;
    float c2 = 0.0f;

    float operator()(float lambdaNm) const {
        const float x = (c0 * lambdaNm + c1) * lambdaNm + c2;
        if (!std::isfinite(x))
            return x > 0.0f ? 1.0f : 0.0f;
        return 0.5f + 0.5f * x / sqrtf(1.0f + x * x);
    }
};

// Look up coefficients for linear sRGB in [0,1] (albedo / reflectance gamut).
RgbSigmoidPolynomial fetchAlbedo(Vec3 rgb);

// Look up coefficients for linear sRGB in [0,1] (illuminant / emission gamut).
RgbSigmoidPolynomial fetchIlluminant(Vec3 rgb);

// ACEScg (AP1) tables fitted under CIE D60 (see rgb_spectrum_tables_aces.cpp).
RgbSigmoidPolynomial fetchAlbedoAces(Vec3 rgb);
RgbSigmoidPolynomial fetchIlluminantAces(Vec3 rgb);

// Evaluate a fetched polynomial at hero wavelengths.
inline SampledSpectrum evalPolynomial(const RgbSigmoidPolynomial& p, const SampledWavelengths& w,
                                      float scale = 1.0f) {
    SampledSpectrum s(w.n);
    for (int i = 0; i < w.n; ++i)
        s.values[i] = srMax(0.0f, scale * p(w.lambda[i]));
    return s;
}

}  // namespace rgb_spec
}  // namespace sol
