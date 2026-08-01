// RGB→spectrum upsampling for materials/lights authored in linear sRGB.
//
// Two paths:
//   1) Linear RGB lobes — energy-safe under multiply. MUST be used for Monte Carlo
//      path weights (f·cos/pdf) and aggregate NEE contribs in the hybrid RGB-BSDF
//      spectral integrator. Jakob sigmoid is nonlinear: upsample(a)*upsample(b)
//      ≠ upsample(a*b), which blows energy on glass / multi-bounce paths.
//   2) Jakob & Hanika (2019) tables — for authored reflectance / emission colours
//      when those are lifted on their own (not as MC weights).
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

inline void applyRgbScaleToSpectrum(SampledSpectrum& s, const SampledWavelengths& w, Vec3 scale) {
    for (int i = 0; i < s.n && i < w.n; ++i) {
        float wr, wg, wb;
        rgbLobeWeights(w.lambda[i], wr, wg, wb);
        const float m = wr * scale.x + wg * scale.y + wb * scale.z;
        s.values[i] = srMax(0.0f, s.values[i] * m);
    }
}

// Round-trip polish so spectrumToRgb(s) ≈ rgb. Tight clamp — large corrections
// were a second energy-amplification path on HDRI / gold glass.
inline void alignSpectrumToRgb(SampledSpectrum& s, const SampledWavelengths& w, Vec3 rgb,
                               float maxBoost = 4.0f) {
    const Vec3 got = spectrumToRgb(s, w);
    Vec3 scale(1.0f);
    if (got.x > 1e-8f) scale.x = rgb.x / got.x;
    if (got.y > 1e-8f) scale.y = rgb.y / got.y;
    if (got.z > 1e-8f) scale.z = rgb.z / got.z;
    const float lo = 1.0f / srMax(maxBoost, 1.0f);
    scale.x = clampf(scale.x, lo, maxBoost);
    scale.y = clampf(scale.y, lo, maxBoost);
    scale.z = clampf(scale.z, lo, maxBoost);
    if (fabsf(scale.x - 1.0f) < 1e-3f && fabsf(scale.y - 1.0f) < 1e-3f &&
        fabsf(scale.z - 1.0f) < 1e-3f)
        return;
    applyRgbScaleToSpectrum(s, w, scale);
}

}  // namespace

// Linear RGB→spectrum (energy-conserving under multiply). Use for PT Spectral
// path throughput weights and NEE/env aggregates.
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
    // One soft align so flat greys stay neutral under spectrumToRgb white-point.
    alignSpectrumToRgb(s, w, Vec3(r, g, b), 2.0f);
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

    SampledSpectrum s;
    const float m = srMax(r, srMax(g, b));
    if (m <= 1.0f) {
        s = rgb_spec::evalPolynomial(rgb_spec::fetchAlbedo(c), w);
    } else {
        const float scale = 2.0f * m;
        s = rgb_spec::evalPolynomial(rgb_spec::fetchAlbedo(c / scale), w, scale);
    }
    alignSpectrumToRgb(s, w, c, 4.0f);
    return s;
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
    SampledSpectrum s;
    if (m <= 1.0f) {
        // Unit-range emission: no 2·m inflate (that was amplifying dim lights).
        s = rgb_spec::evalPolynomial(rgb_spec::fetchIlluminant(c), w);
    } else {
        const float scale = 2.0f * m;
        s = rgb_spec::evalPolynomial(rgb_spec::fetchIlluminant(c / scale), w, scale);
    }
    alignSpectrumToRgb(s, w, c, 4.0f);
    return s;
}

}  // namespace sol
