// RGB→spectrum upsampling for materials/lights authored in linear sRGB.
// Built so spectrumToRgb(upsample(c)) ≈ c (including HDR emissions / HDRI).
#pragma once

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

inline SampledSpectrum rgbToSpectrumBasis(Vec3 rgb, const SampledWavelengths& w) {
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

inline void applyRgbScaleToSpectrum(SampledSpectrum& s, const SampledWavelengths& w, Vec3 scale) {
    for (int i = 0; i < s.n && i < w.n; ++i) {
        float wr, wg, wb;
        rgbLobeWeights(w.lambda[i], wr, wg, wb);
        const float m = wr * scale.x + wg * scale.y + wb * scale.z;
        s.values[i] = srMax(0.0f, s.values[i] * m);
    }
}

}  // namespace

// Reflectance in [0,1] (and HDR if needed): basis + round-trip align to spectrumToRgb.
inline SampledSpectrum rgbToSpectrumReflectance(Vec3 rgb, const SampledWavelengths& w) {
    SampledSpectrum s = rgbToSpectrumBasis(rgb, w);
    // 1–2 align passes so spectrumToRgb(s) ≈ rgb (kills pink HDRI / white drift).
    for (int pass = 0; pass < 2; ++pass) {
        const Vec3 got = spectrumToRgb(s, w);
        Vec3 scale(1.0f);
        if (got.x > 1e-8f) scale.x = rgb.x / got.x;
        if (got.y > 1e-8f) scale.y = rgb.y / got.y;
        if (got.z > 1e-8f) scale.z = rgb.z / got.z;
        // Clamp extreme corrections (near-black channels).
        scale.x = clampf(scale.x, 0.05f, 20.0f);
        scale.y = clampf(scale.y, 0.05f, 20.0f);
        scale.z = clampf(scale.z, 0.05f, 20.0f);
        if (fabsf(scale.x - 1.0f) < 1e-3f && fabsf(scale.y - 1.0f) < 1e-3f &&
            fabsf(scale.z - 1.0f) < 1e-3f)
            break;
        applyRgbScaleToSpectrum(s, w, scale);
    }
    return s;
}

inline SampledSpectrum rgbToSpectrumEmission(Vec3 rgb, const SampledWavelengths& w) {
    return rgbToSpectrumReflectance(rgb, w);
}

}  // namespace sol
