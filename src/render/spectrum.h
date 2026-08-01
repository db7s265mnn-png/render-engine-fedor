// Hero-wavelength SampledSpectrum (PBRT-style) for PT Spectral.
// RGB framebuffer conversion uses CIE XYZ → sRGB (D65).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/math.h"

namespace sol {

constexpr int kMaxSpectrumSamples = 16;
constexpr float kSpectrumLambdaMin = 360.0f;
constexpr float kSpectrumLambdaMax = 830.0f;

struct SampledWavelengths {
    float lambda[kMaxSpectrumSamples]{};
    float pdf[kMaxSpectrumSamples]{};
    int n = 0;

    static SampledWavelengths sampleUniform(int count, float uPrimary) {
        SampledWavelengths w;
        w.n = std::clamp(count, 1, kMaxSpectrumSamples);
        const float span = kSpectrumLambdaMax - kSpectrumLambdaMin;
        const float pdf = 1.0f / span;
        // Stratified hero wavelengths across the visible range.
        const float offset = std::clamp(uPrimary, 0.0f, 0.999999f);
        for (int i = 0; i < w.n; ++i) {
            const float t = (float(i) + offset) / float(w.n);
            w.lambda[i] = kSpectrumLambdaMin + t * span;
            w.pdf[i] = pdf;  // uniform density on [λmin, λmax]
        }
        return w;
    }
};

struct SampledSpectrum {
    float values[kMaxSpectrumSamples]{};
    int n = 0;

    SampledSpectrum() = default;
    explicit SampledSpectrum(int n_) : n(std::clamp(n_, 0, kMaxSpectrumSamples)) {}

    static SampledSpectrum zero(int n) { return SampledSpectrum(n); }

    static SampledSpectrum constant(int n, float v) {
        SampledSpectrum s(n);
        for (int i = 0; i < s.n; ++i) s.values[i] = v;
        return s;
    }

    float operator[](int i) const { return values[i]; }
    float& operator[](int i) { return values[i]; }
};

inline SampledSpectrum operator+(SampledSpectrum a, SampledSpectrum b) {
    SampledSpectrum r(std::max(a.n, b.n));
    for (int i = 0; i < r.n; ++i) {
        const float av = i < a.n ? a.values[i] : 0.0f;
        const float bv = i < b.n ? b.values[i] : 0.0f;
        r.values[i] = av + bv;
    }
    return r;
}
inline SampledSpectrum operator*(SampledSpectrum a, SampledSpectrum b) {
    SampledSpectrum r(std::min(a.n, b.n));
    for (int i = 0; i < r.n; ++i) r.values[i] = a.values[i] * b.values[i];
    return r;
}
inline SampledSpectrum operator*(SampledSpectrum a, float s) {
    for (int i = 0; i < a.n; ++i) a.values[i] *= s;
    return a;
}
inline SampledSpectrum operator*(float s, SampledSpectrum a) { return a * s; }
inline SampledSpectrum& operator+=(SampledSpectrum& a, SampledSpectrum b) {
    a = a + b;
    return a;
}
inline SampledSpectrum& operator*=(SampledSpectrum& a, SampledSpectrum b) {
    a = a * b;
    return a;
}
inline SampledSpectrum& operator*=(SampledSpectrum& a, float s) {
    a = a * s;
    return a;
}

inline float spectrumMaxComponent(const SampledSpectrum& s) {
    float m = 0.0f;
    for (int i = 0; i < s.n; ++i) m = srMax(m, s.values[i]);
    return m;
}
inline float spectrumAvg(const SampledSpectrum& s) {
    if (s.n <= 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < s.n; ++i) sum += s.values[i];
    return sum / float(s.n);
}

// CIE 1931 2° XYZ matching (5 nm), 360..830 inclusive → 95 entries.
inline void cieXyzAtLambda(float lambda, float& x, float& y, float& z) {
    // Compact analytic fit (Wyman et al. / multi-lobe Gaussians) — good enough for hero PT.
    auto gauss = [](float lam, float mu, float s1, float s2) {
        const float g = lam < mu ? s1 : s2;
        const float d = (lam - mu) / g;
        return expf(-0.5f * d * d);
    };
    x = 1.065f * gauss(lambda, 595.8f, 33.33f, 37.05f) + 0.366f * gauss(lambda, 446.8f, 16.01f, 22.40f);
    y = 1.014f * gauss(lambda, 556.7f, 46.07f, 40.83f);
    z = 1.839f * gauss(lambda, 449.1f, 19.44f, 28.54f);
}

// SampledSpectrum → linear sRGB (D65), dividing out wavelength pdfs.
inline Vec3 spectrumToRgb(const SampledSpectrum& s, const SampledWavelengths& w) {
    if (s.n <= 0 || w.n <= 0) return Vec3(0.0f);
    const int n = std::min(s.n, w.n);
    float X = 0.0f, Y = 0.0f, Z = 0.0f;
    float Xw = 0.0f, Yw = 0.0f, Zw = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float pdf = srMax(w.pdf[i], 1e-8f);
        float cx, cy, cz;
        cieXyzAtLambda(w.lambda[i], cx, cy, cz);
        const float invPdf = 1.0f / pdf;
        X += s.values[i] * cx * invPdf;
        Y += s.values[i] * cy * invPdf;
        Z += s.values[i] * cz * invPdf;
        // Reference white: unit spectrum under the same samples.
        Xw += cx * invPdf;
        Yw += cy * invPdf;
        Zw += cz * invPdf;
    }
    const float invN = 1.0f / float(n);
    X *= invN;
    Y *= invN;
    Z *= invN;
    Xw *= invN;
    Yw *= invN;
    Zw *= invN;
    // Normalize so a flat spectrum of 1 → RGB ≈ (1,1,1).
    if (Yw > 1e-8f) {
        X /= Yw;
        Y /= Yw;
        Z /= Yw;
        Xw /= Yw;
        Zw /= Yw;
        Yw = 1.0f;
    }
    (void)Xw;
    (void)Zw;
    float r = 3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    float g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    float b = 0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
    return Vec3(srMax(0.0f, r), srMax(0.0f, g), srMax(0.0f, b));
}

// False-color: map a spectral bin / wavelength to a visible debug color.
inline Vec3 wavelengthToFalseColor(float lambdaNm) {
    const float t = saturatef((lambdaNm - kSpectrumLambdaMin) / (kSpectrumLambdaMax - kSpectrumLambdaMin));
    // Smooth hue sweep: violet → blue → cyan → green → yellow → red.
    const float h = (1.0f - t) * 0.75f;  // 270° → 0° in HSV-ish
    const float s = 1.0f, v = 1.0f;
    const float c = v * s;
    const float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    const float m = v - c;
    float r = 0, g = 0, b = 0;
    const int sector = int(h * 6.0f) % 6;
    switch (sector) {
        case 0: r = c; g = x; break;
        case 1: r = x; g = c; break;
        case 2: g = c; b = x; break;
        case 3: g = x; b = c; break;
        case 4: r = x; b = c; break;
        default: r = c; b = x; break;
    }
    return Vec3(r + m, g + m, b + m);
}

}  // namespace sol
