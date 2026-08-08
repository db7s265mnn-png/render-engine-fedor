// RGB→spectrum upsampling for materials/lights authored in linear sRGB.
//
// Two paths:
//   1) Linear RGB lobes — energy-safe under multiply. MUST be used for Monte Carlo
//      path weights (f·cos/pdf) and aggregate NEE contribs. No round-trip "align":
//      that fits RGB via spectrumToRgb and breaks under TerminateSecondary / glass.
//   2) Jakob & Hanika (2019) tables — authored reflectance / emission. Honest
//      spectra; neutrality comes from CIE→RGB with fixed illuminant-E white, not
//      from post-hoc RGB matching.
#pragma once

#include "render/rgb_spectrum_tables.h"
#include "render/spectrum.h"

namespace sol {

namespace {

// Smooth RGB lobe weights vs wavelength (R≈650, G≈550, B≈450 nm).
inline void rgbLobeWeights(float lambdaNm, float& wr, float& wg, float& wb) {
    auto lobe = [](float lam, float mu, float width) {
        const float d = (lam - mu) / width;
        return expf(-0.5f * d * d);
    };
    wr = lobe(lambdaNm, 650.0f, 55.0f);
    wg = lobe(lambdaNm, 550.0f, 45.0f);
    wb = lobe(lambdaNm, 450.0f, 40.0f);
    const float sum = wr + wg + wb;
    if (sum > 1e-8f) {
        wr /= sum;
        wg /= sum;
        wb /= sum;
    } else {
        wr = wg = wb = 1.0f / 3.0f;
    }
}

}  // namespace

// Linear RGB→spectrum (energy-conserving under multiply). Use for PT/BDPT Spectral
// path throughput weights and NEE/env aggregates that are already MC weights.
inline SampledSpectrum rgbToSpectrumLinear(Vec3 rgb, const SampledWavelengths& w) {
    SampledSpectrum s(w.n);
    const float r = srMax(0.0f, rgb.x);
    const float g = srMax(0.0f, rgb.y);
    const float b = srMax(0.0f, rgb.z);
    for (int i = 0; i < w.n; ++i) {
        float wr, wg, wb;
        rgbLobeWeights(w.lambda[i], wr, wg, wb);
        s.values[i] = wr * r + wg * g + wb * b;
    }
    return s;
}

// Reflectance / albedo: Jakob albedo table (spectra in [0,1]). HDR albedo uses
// the PBRT unbounded path (scale = 2·max, lookup rgb/scale).
inline SampledSpectrum rgbToSpectrumReflectance(Vec3 rgb, const SampledWavelengths& w) {
    const float r = srMax(0.0f, rgb.x);
    const float g = srMax(0.0f, rgb.y);
    const float b = srMax(0.0f, rgb.z);
    const Vec3 c(r, g, b);
    if (r <= 0.0f && g <= 0.0f && b <= 0.0f) return SampledSpectrum::zero(w.n);

    const float m = srMax(r, srMax(g, b));
    if (m <= 1.0f) return rgb_spec::evalPolynomial(rgb_spec::fetchAlbedo(c), w);
    const float scale = 2.0f * m;
    return rgb_spec::evalPolynomial(rgb_spec::fetchAlbedo(c / scale), w, scale);
}

// Emission / illuminant: Jakob illuminant table + PBRT unbounded scaling for HDR.
// Only for authored emission / light colours — not MC path weights.
inline SampledSpectrum rgbToSpectrumEmission(Vec3 rgb, const SampledWavelengths& w) {
    const float r = srMax(0.0f, rgb.x);
    const float g = srMax(0.0f, rgb.y);
    const float b = srMax(0.0f, rgb.z);
    const Vec3 c(r, g, b);
    if (r <= 0.0f && g <= 0.0f && b <= 0.0f) return SampledSpectrum::zero(w.n);

    const float m = srMax(r, srMax(g, b));
    if (m <= 1.0f) return rgb_spec::evalPolynomial(rgb_spec::fetchIlluminant(c), w);
    const float scale = 2.0f * m;
    return rgb_spec::evalPolynomial(rgb_spec::fetchIlluminant(c / scale), w, scale);
}

}  // namespace sol
