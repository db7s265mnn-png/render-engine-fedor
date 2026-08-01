// RGB→spectrum upsampling for materials/lights authored in linear sRGB.
// Jakob & Hanika (2019) tabulated sigmoid polynomials + one round-trip align pass
// so spectrumToRgb(upsample(c)) ≈ c (including HDR emissions / HDRI).
#pragma once

#include "render/rgb_spectrum_tables.h"
#include "render/spectrum.h"

namespace sol {

namespace {

// Smooth RGB lobe weights vs wavelength — used only for the round-trip polish scale.
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

// One align pass so spectrumToRgb(s) ≈ rgb (kills pink HDRI / white drift).
inline void alignSpectrumToRgb(SampledSpectrum& s, const SampledWavelengths& w, Vec3 rgb) {
    const Vec3 got = spectrumToRgb(s, w);
    Vec3 scale(1.0f);
    if (got.x > 1e-8f) scale.x = rgb.x / got.x;
    if (got.y > 1e-8f) scale.y = rgb.y / got.y;
    if (got.z > 1e-8f) scale.z = rgb.z / got.z;
    scale.x = clampf(scale.x, 0.05f, 20.0f);
    scale.y = clampf(scale.y, 0.05f, 20.0f);
    scale.z = clampf(scale.z, 0.05f, 20.0f);
    if (fabsf(scale.x - 1.0f) < 1e-3f && fabsf(scale.y - 1.0f) < 1e-3f &&
        fabsf(scale.z - 1.0f) < 1e-3f)
        return;
    applyRgbScaleToSpectrum(s, w, scale);
}

}  // namespace

// Reflectance / albedo: Jakob albedo table (spectra in [0,1]). HDR albedo uses
// the PBRT unbounded path (scale = 2·max, lookup rgb/scale).
inline SampledSpectrum rgbToSpectrumReflectance(Vec3 rgb, const SampledWavelengths& w) {
    const float r = srMax(0.0f, rgb.x);
    const float g = srMax(0.0f, rgb.y);
    const float b = srMax(0.0f, rgb.z);
    const Vec3 c(r, g, b);
    if (r <= 0.0f && g <= 0.0f && b <= 0.0f)
        return SampledSpectrum::zero(w.n);

    SampledSpectrum s;
    const float m = srMax(r, srMax(g, b));
    if (m <= 1.0f) {
        s = rgb_spec::evalPolynomial(rgb_spec::fetchAlbedo(c), w);
    } else {
        // Unbounded reflectance (rare): same scaling as PBRT RGBUnboundedSpectrum.
        const float scale = 2.0f * m;
        s = rgb_spec::evalPolynomial(rgb_spec::fetchAlbedo(c / scale), w, scale);
    }
    alignSpectrumToRgb(s, w, c);
    return s;
}

// Emission / illuminant: Jakob illuminant table + PBRT unbounded scaling for HDR.
inline SampledSpectrum rgbToSpectrumEmission(Vec3 rgb, const SampledWavelengths& w) {
    const float r = srMax(0.0f, rgb.x);
    const float g = srMax(0.0f, rgb.y);
    const float b = srMax(0.0f, rgb.z);
    const Vec3 c(r, g, b);
    if (r <= 0.0f && g <= 0.0f && b <= 0.0f)
        return SampledSpectrum::zero(w.n);

    const float m = srMax(r, srMax(g, b));
    const float scale = 2.0f * m;
    SampledSpectrum s =
        rgb_spec::evalPolynomial(rgb_spec::fetchIlluminant(scale > 0.0f ? c / scale : Vec3(0.0f)), w,
                                 scale);
    alignSpectrumToRgb(s, w, c);
    return s;
}

}  // namespace sol
