// Hero-wavelength SampledSpectrum (PBRT-v4 style) for spectral integrators.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/math.h"
#include "render/cie_tables.h"
#include "render/color_space.h"

namespace sol {

constexpr int kMaxSpectrumSamples = 16;
constexpr float kSpectrumLambdaMin = 360.0f;
constexpr float kSpectrumLambdaMax = 830.0f;

struct SampledWavelengths {
    float lambda[kMaxSpectrumSamples]{};
    float pdf[kMaxSpectrumSamples]{};
    int n = 0;

    // Stratified uniform wavelengths on [λmin, λmax].
    static SampledWavelengths sampleUniform(int count, float uPrimary) {
        SampledWavelengths w;
        w.n = std::clamp(count, 1, kMaxSpectrumSamples);
        const float span = kSpectrumLambdaMax - kSpectrumLambdaMin;
        const float pdf = 1.0f / span;
        const float offset = std::clamp(uPrimary, 0.0f, 0.999999f);
        for (int i = 0; i < w.n; ++i) {
            const float t = (float(i) + offset) / float(w.n);
            w.lambda[i] = kSpectrumLambdaMin + t * span;
            w.pdf[i] = pdf;
        }
        return w;
    }

    // Importance-sample visible wavelengths (pbrt SampleVisible — CIE Y–like PDF).
    static SampledWavelengths sampleVisible(int count, float uPrimary) {
        SampledWavelengths w;
        w.n = std::clamp(count, 1, kMaxSpectrumSamples);
        for (int i = 0; i < w.n; ++i) {
            float up = uPrimary + float(i) / float(w.n);
            if (up > 1.0f) up -= 1.0f;
            w.lambda[i] = sampleVisibleWavelength(up);
            w.pdf[i] = visibleWavelengthPdf(w.lambda[i]);
        }
        return w;
    }

    // pbrt VisibleWavelengthsPDF / SampleVisibleWavelengths.
    static float visibleWavelengthPdf(float lambdaNm) {
        if (lambdaNm < kSpectrumLambdaMin || lambdaNm > kSpectrumLambdaMax) return 0.0f;
        const float x = 0.0072f * (lambdaNm - 538.0f);
        const float c = coshf(x);
        return 0.0039398042f / (c * c);
    }
    static float sampleVisibleWavelength(float u) {
        const float x = clampf(0.85691062f - 1.82750197f * u, -0.999999f, 0.999999f);
        return 538.0f - 138.888889f * (0.5f * logf((1.0f + x) / (1.0f - x)));
    }

    // Move hero sample to slot 0 (for TerminateSecondary + geometric dispersion).
    void promoteHero(int heroIdx) {
        heroIdx = std::clamp(heroIdx, 0, std::max(0, n - 1));
        if (heroIdx == 0 || n <= 0) return;
        std::swap(lambda[0], lambda[heroIdx]);
        std::swap(pdf[0], pdf[heroIdx]);
    }

    bool secondaryTerminated() const {
        for (int i = 1; i < n; ++i)
            if (pdf[i] != 0.0f) return false;
        return true;
    }

    // After first scattering: keep λ₀ only; scale pdf so ToXYZ stays unbiased (pbrt).
    void terminateSecondary() {
        if (secondaryTerminated()) return;
        for (int i = 1; i < n; ++i) pdf[i] = 0.0f;
        pdf[0] /= float(std::max(1, n));
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

// Safe divide for terminated wavelengths (pdf == 0 → 0).
inline float safeDivSpectrum(float num, float pdf) {
    return (pdf > 0.0f) ? (num / pdf) : 0.0f;
}

// Tabulated CIE (preferred). Kept name cieXyzAtLambda for call-site compat.
inline void cieXyzAtLambda(float lambda, float& x, float& y, float& z) {
    cieXyzAtLambdaTabulated(lambda, x, y, z);
}

inline Xyz spectrumToXyz(const SampledSpectrum& s, const SampledWavelengths& w) {
    if (s.n <= 0 || w.n <= 0) return {};
    const int n = std::min(s.n, w.n);
    float X = 0.0f, Y = 0.0f, Z = 0.0f;
    for (int i = 0; i < n; ++i) {
        float cx, cy, cz;
        cieXyzAtLambda(w.lambda[i], cx, cy, cz);
        const float invPdf = safeDivSpectrum(1.0f, w.pdf[i]);
        X += s.values[i] * cx * invPdf;
        Y += s.values[i] * cy * invPdf;
        Z += s.values[i] * cz * invPdf;
    }
    const float invN = 1.0f / float(n);
    // Absolute XYZ scale (pbrt): divide by ∫ ȳ(λ) dλ.
    const float scale = invN / cie_tab::kCieYIntegral1nm;
    return Xyz(X * scale, Y * scale, Z * scale);
}

// SampledSpectrum → RGB. Tabulated CIE CMFs + RGBColorSpace, with equal-energy
// white-balance so flat spectra stay neutral under Jakob-authored RGB assets.
inline Vec3 spectrumToRgb(const SampledSpectrum& s, const SampledWavelengths& w,
                          const RGBColorSpace& cs = colorSpaceSrgb()) {
    if (s.n <= 0 || w.n <= 0) return Vec3(0.0f);
    const int n = std::min(s.n, w.n);
    float X = 0.0f, Y = 0.0f, Z = 0.0f;
    float Xw = 0.0f, Yw = 0.0f, Zw = 0.0f;
    for (int i = 0; i < n; ++i) {
        float cx, cy, cz;
        cieXyzAtLambda(w.lambda[i], cx, cy, cz);
        const float invPdf = safeDivSpectrum(1.0f, w.pdf[i]);
        X += s.values[i] * cx * invPdf;
        Y += s.values[i] * cy * invPdf;
        Z += s.values[i] * cz * invPdf;
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
    if (Xw > 1e-8f && Yw > 1e-8f && Zw > 1e-8f) {
        X /= Xw;
        Y /= Yw;
        Z /= Zw;
    } else if (Yw > 1e-8f) {
        X /= Yw;
        Y /= Yw;
        Z /= Yw;
    }
    Vec3 rgb = cs.toRgb(Xyz(X, Y, Z));
    const Vec3 whiteRgb = cs.toRgb(Xyz(1.0f, 1.0f, 1.0f));
    rgb.x /= srMax(1e-8f, whiteRgb.x);
    rgb.y /= srMax(1e-8f, whiteRgb.y);
    rgb.z /= srMax(1e-8f, whiteRgb.z);
    return Vec3(srMax(0.0f, rgb.x), srMax(0.0f, rgb.y), srMax(0.0f, rgb.z));
}

inline Vec3 spectrumToRgb(const SampledSpectrum& s, const SampledWavelengths& w, int colorSpaceId) {
    return spectrumToRgb(s, w, colorSpaceById(colorSpaceId));
}

// False-color: map a spectral bin / wavelength to a visible debug color.
inline Vec3 wavelengthToFalseColor(float lambdaNm) {
    const float t = saturatef((lambdaNm - kSpectrumLambdaMin) / (kSpectrumLambdaMax - kSpectrumLambdaMin));
    const float h = (1.0f - t) * 0.75f;
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
