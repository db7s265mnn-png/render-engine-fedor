// Device-safe hero-λ helpers (no STL). Shared by OptiX kernels and CPU tests.
// Matches Embree: visible/uniform sampling, Jakob albedo/illuminant, D60/D65
// illuminant, pbrt ToXYZ film while secondaries live, grey s(λ) after terminate.
#pragma once

#include "core/math.h"
#include "render/spectrum_constants.h"

namespace sol {

constexpr int kCieDevSamples = 95;
constexpr float kCieDevLambdaMin = 360.0f;
constexpr float kCieDevLambdaMax = 830.0f;
constexpr float kCieDevLambdaStep = 5.0f;
constexpr float kCieYIntegral1nm = 106.85691710f;

constexpr int kIllumDevSamples = 107;
constexpr float kIllumDevLambdaMin = 300.0f;
constexpr float kIllumDevLambdaMax = 830.0f;
constexpr float kIllumDevLambdaStep = 5.0f;

struct GpuSpectralTables {
    const float* albedoScale = nullptr;
    const float* albedoCoeffs = nullptr;
    const float* illuminantScale = nullptr;
    const float* illuminantCoeffs = nullptr;
    const float* cieX = nullptr;
    const float* cieY = nullptr;
    const float* cieZ = nullptr;
    const float* illuminantSpd = nullptr;
    float rgbFromXyz[9]{};
    int samples = 4;
    int wavelengthSampling = 0;  // 0 = visible, 1 = uniform
};

SR_INL SR_HD int specClampN(int n) {
    if (n < 1) return 1;
    if (n > kMaxSpectrumSamples) return kMaxSpectrumSamples;
    return n;
}

SR_INL SR_HD void specZero(float* v, int n) {
    for (int i = 0; i < n; ++i) v[i] = 0.0f;
}

SR_INL SR_HD void specFill(float* v, int n, float x) {
    for (int i = 0; i < n; ++i) v[i] = x;
}

SR_INL SR_HD void specAddMul(float* dst, const float* a, const float* b, float s, int n) {
    for (int i = 0; i < n; ++i) dst[i] += a[i] * b[i] * s;
}

SR_INL SR_HD void specMul(float* v, const float* a, int n) {
    for (int i = 0; i < n; ++i) v[i] *= a[i];
}

SR_INL SR_HD void specMulS(float* v, float s, int n) {
    for (int i = 0; i < n; ++i) v[i] *= s;
}

SR_INL SR_HD float specMax(const float* v, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) m = srMax(m, v[i]);
    return m;
}

SR_INL SR_HD bool specIsBlack(const float* v, int n) {
    for (int i = 0; i < n; ++i)
        if (v[i] > 0.0f) return false;
    return true;
}

SR_INL SR_HD bool specIsFinite(const float* v, int n) {
    for (int i = 0; i < n; ++i)
        if (!srIsFinite(v[i])) return false;
    return true;
}

SR_INL SR_HD void specClampIndirect(float* v, int n, float clampValue) {
    if (clampValue <= 0.0f) return;
    const float m = specMax(v, n);
    if (!(m > clampValue)) return;
    specMulS(v, clampValue / m, n);
}

SR_INL SR_HD float specSampleTable(const float* tab, int n, float lambdaMin, float lambdaMax,
                                   float step, float lambdaNm) {
    if (!tab || n <= 0) return 0.0f;
    if (lambdaNm <= lambdaMin) return tab[0];
    if (lambdaNm >= lambdaMax) return tab[n - 1];
    const float t = (lambdaNm - lambdaMin) / step;
    const int i0 = int(t);
    const int i1 = (i0 + 1 < n) ? i0 + 1 : n - 1;
    const float f = t - float(i0);
    return tab[i0] * (1.0f - f) + tab[i1] * f;
}

SR_INL SR_HD void specCieXyz(const GpuSpectralTables& tab, float lambdaNm, float& x, float& y,
                             float& z) {
    x = specSampleTable(tab.cieX, kCieDevSamples, kCieDevLambdaMin, kCieDevLambdaMax, kCieDevLambdaStep,
                        lambdaNm);
    y = specSampleTable(tab.cieY, kCieDevSamples, kCieDevLambdaMin, kCieDevLambdaMax, kCieDevLambdaStep,
                        lambdaNm);
    z = specSampleTable(tab.cieZ, kCieDevSamples, kCieDevLambdaMin, kCieDevLambdaMax, kCieDevLambdaStep,
                        lambdaNm);
}

SR_INL SR_HD float specIlluminant(const GpuSpectralTables& tab, float lambdaNm) {
    return specSampleTable(tab.illuminantSpd, kIllumDevSamples, kIllumDevLambdaMin, kIllumDevLambdaMax,
                           kIllumDevLambdaStep, lambdaNm);
}

SR_INL SR_HD float specVisibleWavelengthPdf(float lambdaNm) {
    if (lambdaNm < kSpectrumLambdaMin || lambdaNm > kSpectrumLambdaMax) return 0.0f;
    const float x = 0.0072f * (lambdaNm - 538.0f);
    const float c = coshf(x);
    return 0.0039398042f / (c * c);
}

SR_INL SR_HD float specSampleVisibleWavelength(float u) {
    const float x = clampf(0.85691062f - 1.82750197f * u, -0.999999f, 0.999999f);
    return 538.0f - 138.888889f * (0.5f * logf((1.0f + x) / (1.0f - x)));
}

SR_INL SR_HD void specSampleUniform(int count, float uPrimary, float* lambda, float* pdf, int& n) {
    n = specClampN(count);
    const float span = kSpectrumLambdaMax - kSpectrumLambdaMin;
    const float p = 1.0f / span;
    const float offset = clampf(uPrimary, 0.0f, 0.999999f);
    for (int i = 0; i < n; ++i) {
        const float t = (float(i) + offset) / float(n);
        lambda[i] = kSpectrumLambdaMin + t * span;
        pdf[i] = p;
    }
}

SR_INL SR_HD void specSampleVisible(int count, float uPrimary, float* lambda, float* pdf, int& n) {
    n = specClampN(count);
    for (int i = 0; i < n; ++i) {
        float up = uPrimary + float(i) / float(n);
        if (up > 1.0f) up -= 1.0f;
        lambda[i] = specSampleVisibleWavelength(up);
        pdf[i] = specVisibleWavelengthPdf(lambda[i]);
    }
}

SR_INL SR_HD void specPromoteHero(float* lambda, float* pdf, int n, int heroIdx) {
    if (n <= 0) return;
    if (heroIdx < 0) heroIdx = 0;
    if (heroIdx > n - 1) heroIdx = n - 1;
    if (heroIdx == 0) return;
    const float lam = lambda[0];
    const float p = pdf[0];
    lambda[0] = lambda[heroIdx];
    pdf[0] = pdf[heroIdx];
    lambda[heroIdx] = lam;
    pdf[heroIdx] = p;
}

SR_INL SR_HD bool specSecondaryTerminated(const float* pdf, int n) {
    for (int i = 1; i < n; ++i)
        if (pdf[i] != 0.0f) return false;
    return true;
}

SR_INL SR_HD void specTerminateSecondary(float* pdf, int n) {
    if (specSecondaryTerminated(pdf, n)) return;
    for (int i = 1; i < n; ++i) pdf[i] = 0.0f;
    pdf[0] /= float(n > 0 ? n : 1);
}

SR_INL SR_HD float specSafeDiv(float num, float pdf) { return (pdf > 0.0f) ? (num / pdf) : 0.0f; }

SR_INL SR_HD Vec3 specXyzToRgb(const GpuSpectralTables& tab, float X, float Y, float Z) {
    const float* m = tab.rgbFromXyz;
    return Vec3(m[0] * X + m[1] * Y + m[2] * Z, m[3] * X + m[4] * Y + m[5] * Z,
                m[6] * X + m[7] * Y + m[8] * Z);
}

// Same film rule as spectrumToRgb: ToXYZ while ≥2 λ live; grey s(λ) after terminate.
SR_INL SR_HD Vec3 specToRgb(const GpuSpectralTables& tab, const float* s, const float* lambda,
                            const float* pdf, int n) {
    if (n <= 0) return Vec3(0.0f);
    int active = 0;
    for (int i = 0; i < n; ++i)
        if (pdf[i] > 0.0f) ++active;

    if (active >= 2 && !specSecondaryTerminated(pdf, n)) {
        float X = 0.0f, Y = 0.0f, Z = 0.0f;
        for (int i = 0; i < n; ++i) {
            float cx, cy, cz;
            specCieXyz(tab, lambda[i], cx, cy, cz);
            const float invPdf = specSafeDiv(1.0f, pdf[i]);
            X += s[i] * cx * invPdf;
            Y += s[i] * cy * invPdf;
            Z += s[i] * cz * invPdf;
        }
        const float scale = (1.0f / float(n)) / kCieYIntegral1nm;
        return specXyzToRgb(tab, X * scale, Y * scale, Z * scale);
    }

    float hero = 0.0f;
    for (int i = 0; i < n; ++i)
        if (pdf[i] > 0.0f) hero = s[i];
    return Vec3(hero, hero, hero);
}

SR_INL SR_HD void specRgbLobes(float lambdaNm, float& wr, float& wg, float& wb) {
    const float dr = (lambdaNm - 650.0f) / 55.0f;
    const float dg = (lambdaNm - 550.0f) / 45.0f;
    const float db = (lambdaNm - 450.0f) / 40.0f;
    wr = expf(-0.5f * dr * dr);
    wg = expf(-0.5f * dg * dg);
    wb = expf(-0.5f * db * db);
    const float sum = wr + wg + wb;
    if (sum > 1e-8f) {
        wr /= sum;
        wg /= sum;
        wb /= sum;
    } else {
        wr = wg = wb = 1.0f / 3.0f;
    }
}

SR_INL SR_HD void specUpsampleLinear(Vec3 rgb, const float* lambda, int n, float* out) {
    const float r = srMax(0.0f, rgb.x);
    const float g = srMax(0.0f, rgb.y);
    const float b = srMax(0.0f, rgb.z);
    for (int i = 0; i < n; ++i) {
        float wr, wg, wb;
        specRgbLobes(lambda[i], wr, wg, wb);
        out[i] = wr * r + wg * g + wb * b;
    }
}

struct GpuSigmoid {
    float c0 = 0.0f;
    float c1 = 0.0f;
    float c2 = 0.0f;

    SR_HD float eval(float lambdaNm) const {
        const float x = (c0 * lambdaNm + c1) * lambdaNm + c2;
        if (!srIsFinite(x)) return x > 0.0f ? 1.0f : 0.0f;
        return 0.5f + 0.5f * x / sqrtf(1.0f + x * x);
    }
};

SR_INL SR_HD int specJakobFindScale(const float* nodes, int n, float z) {
    int left = 0;
    int size = n - 2;
    while (size > 0) {
        const int half = size >> 1;
        const int mid = left + half + 1;
        if (nodes[mid] <= z) {
            left = mid;
            size -= half + 1;
        } else {
            size = half;
        }
    }
    return left < n - 2 ? left : n - 2;
}

SR_INL SR_HD GpuSigmoid specJakobFetch(const float* scale, const float* coeffs, Vec3 rgb) {
    GpuSigmoid p;
    if (!scale || !coeffs) return p;
    float r = clampf(rgb.x, 0.0f, 1.0f);
    float g = clampf(rgb.y, 0.0f, 1.0f);
    float b = clampf(rgb.z, 0.0f, 1.0f);
    if (r == g && g == b) {
        if (r <= 0.0f)
            p.c2 = -8192.0f;
        else if (r >= 1.0f)
            p.c2 = 8192.0f;
        else
            p.c2 = (r - 0.5f) / sqrtf(r * (1.0f - r));
        return p;
    }
    int maxc = (r > g) ? ((r > b) ? 0 : 2) : ((g > b) ? 1 : 2);
    const float rgbv[3] = {r, g, b};
    const float z = rgbv[maxc];
    if (z <= 0.0f) {
        p.c0 = -8192.0f;
        return p;
    }
    const int res = kJakobTableRes;
    const float x = rgbv[(maxc + 1) % 3] * float(res - 1) / z;
    const float y = rgbv[(maxc + 2) % 3] * float(res - 1) / z;
    const int xi = int(x) < res - 2 ? int(x) : res - 2;
    const int yi = int(y) < res - 2 ? int(y) : res - 2;
    const int zi = specJakobFindScale(scale, res, z);
    const float dx = x - float(xi);
    const float dy = y - float(yi);
    const float denom = scale[zi + 1] - scale[zi];
    const float dz = (fabsf(denom) > 1e-20f) ? (z - scale[zi]) / denom : 0.0f;
    float c[3];
    for (int ci = 0; ci < 3; ++ci) {
        float corner[2][2][2];
        for (int dk = 0; dk < 2; ++dk)
            for (int dj = 0; dj < 2; ++dj)
                for (int di = 0; di < 2; ++di) {
                    const int idx =
                        ((((maxc * res + (zi + dk)) * res + (yi + dj)) * res + (xi + di)) * 3) + ci;
                    corner[dk][dj][di] = coeffs[idx];
                }
        const float c00 = corner[0][0][0] * (1.0f - dx) + corner[0][0][1] * dx;
        const float c10 = corner[0][1][0] * (1.0f - dx) + corner[0][1][1] * dx;
        const float c01 = corner[1][0][0] * (1.0f - dx) + corner[1][0][1] * dx;
        const float c11 = corner[1][1][0] * (1.0f - dx) + corner[1][1][1] * dx;
        const float c0 = c00 * (1.0f - dy) + c10 * dy;
        const float c1 = c01 * (1.0f - dy) + c11 * dy;
        c[ci] = c0 * (1.0f - dz) + c1 * dz;
    }
    p.c0 = c[0];
    p.c1 = c[1];
    p.c2 = c[2];
    return p;
}

SR_INL SR_HD void specEvalSigmoid(const GpuSigmoid& p, const float* lambda, int n, float scale,
                                  float* out) {
    for (int i = 0; i < n; ++i) out[i] = srMax(0.0f, scale * p.eval(lambda[i]));
}

SR_INL SR_HD void specUpsampleReflectance(const GpuSpectralTables& tab, Vec3 rgb, const float* lambda,
                                          int n, float* out) {
    specZero(out, n);
    const float r = srMax(0.0f, rgb.x);
    const float g = srMax(0.0f, rgb.y);
    const float b = srMax(0.0f, rgb.z);
    if (r <= 0.0f && g <= 0.0f && b <= 0.0f) return;
    const Vec3 c(r, g, b);
    const float m = srMax(r, srMax(g, b));
    if (m <= 1.0f) {
        specEvalSigmoid(specJakobFetch(tab.albedoScale, tab.albedoCoeffs, c), lambda, n, 1.0f, out);
        return;
    }
    const float scale = 2.0f * m;
    specEvalSigmoid(specJakobFetch(tab.albedoScale, tab.albedoCoeffs, c / scale), lambda, n, scale, out);
}

SR_INL SR_HD void specUpsampleUnbounded(const GpuSpectralTables& tab, Vec3 rgb, const float* lambda,
                                        int n, float* out) {
    specZero(out, n);
    const float r = srMax(0.0f, rgb.x);
    const float g = srMax(0.0f, rgb.y);
    const float b = srMax(0.0f, rgb.z);
    if (r <= 0.0f && g <= 0.0f && b <= 0.0f) return;
    const float m = srMax(r, srMax(g, b));
    const float scale = 2.0f * m;
    specEvalSigmoid(specJakobFetch(tab.illuminantScale, tab.illuminantCoeffs, Vec3(r, g, b) / scale),
                    lambda, n, scale, out);
}

SR_INL SR_HD void specUpsampleEmission(const GpuSpectralTables& tab, Vec3 rgb, const float* lambda,
                                       int n, float* out) {
    specUpsampleUnbounded(tab, rgb, lambda, n, out);
    for (int i = 0; i < n; ++i) out[i] *= specIlluminant(tab, lambda[i]);
}

SR_INL SR_HD float specBlackbody(float lambdaNm, float kelvin) {
    const float lambda = lambdaNm * 1e-9f;
    if (!(lambda > 0.0f) || !(kelvin > 0.0f)) return 0.0f;
    constexpr float c = 299792458.0f;
    constexpr float h = 6.62606957e-34f;
    constexpr float kb = 1.3806488e-23f;
    const float l5 = (lambda * lambda) * (lambda * lambda) * lambda;
    const float exponent = h * c / (lambda * kb * kelvin);
    if (exponent > 80.0f) return 0.0f;
    return (2.0f * h * c * c) / (l5 * (expf(exponent) - 1.0f));
}

SR_INL SR_HD void specBlackbodyNorm(float kelvin, const float* lambda, int n, float* out) {
    const float T = srMax(100.0f, kelvin);
    const float peakNm = clampf(2.897771955e6f / T, kSpectrumLambdaMin, kSpectrumLambdaMax);
    const float norm = 1.0f / srMax(1e-30f, specBlackbody(peakNm, T));
    for (int i = 0; i < n; ++i) out[i] = specBlackbody(lambda[i], T) * norm;
}

SR_INL SR_HD float specDielectricIor(float iorNd, float abbeVd, float lambdaNm) {
    if (abbeVd <= 1e-3f) return iorNd;
    const float lamD = 589.3f;
    const float lamF = 486.1f;
    const float lamC = 656.3f;
    const float invD = 1.0f / (lamD * lamD);
    const float invF = 1.0f / (lamF * lamF);
    const float invC = 1.0f / (lamC * lamC);
    const float B = (iorNd - 1.0f) / (abbeVd * (invF - invC));
    const float A = iorNd - B * invD;
    const float inv = 1.0f / (lambdaNm * lambdaNm);
    return srMax(1.0f, A + B * inv);
}

}  // namespace sol
