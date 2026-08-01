// Shared spectral film buffer + RGB↔spectrum lift helpers (PT / BDPT Spectral).
#pragma once

#include <algorithm>
#include <vector>

#include "render/metal_spectra.h"
#include "render/shading.h"
#include "render/spectrum.h"
#include "render/spectrum_rgb.h"

namespace sol {

struct SpectralBinBuffer {
    int width = 0;
    int height = 0;
    int bins = 0;
    std::vector<float> accum;

    void resize(int w, int h, int b) {
        width = std::max(0, w);
        height = std::max(0, h);
        bins = std::clamp(b, 0, 64);
        accum.assign(size_t(std::max(width, 0)) * size_t(std::max(height, 0)) * size_t(std::max(bins, 1)),
                     0.0f);
    }
    void clear() { std::fill(accum.begin(), accum.end(), 0.0f); }

    void addSample(int x, int y, const SampledSpectrum& s, const SampledWavelengths& w) {
        if (bins <= 0 || width <= 0 || height <= 0 || s.n <= 0) return;
        if (x < 0 || y < 0 || x >= width || y >= height) return;
        const size_t base = (size_t(y) * size_t(width) + size_t(x)) * size_t(bins);
        const float span = kSpectrumLambdaMax - kSpectrumLambdaMin;
        const int n = std::min(s.n, w.n);
        for (int i = 0; i < n; ++i) {
            float t = (w.lambda[i] - kSpectrumLambdaMin) / span;
            int bin = int(t * float(bins));
            bin = std::clamp(bin, 0, bins - 1);
            const float pdf = srMax(w.pdf[i], 1e-8f);
            accum[base + size_t(bin)] += s.values[i] / pdf;
        }
    }

    float binValue(int x, int y, int bin) const {
        if (bins <= 0 || !width || bin < 0 || bin >= bins) return 0.0f;
        if (x < 0 || y < 0 || x >= width || y >= height) return 0.0f;
        return accum[(size_t(y) * size_t(width) + size_t(x)) * size_t(bins) + size_t(bin)];
    }
};

inline SampledSpectrum upsampleRgb(Vec3 rgb, const SampledWavelengths& w) {
    // Hybrid RGB-BSDF spectral PT multiplies these spectra along the path.
    // Linear lobes keep energy: Jakob sigmoid does not (see spectrum_rgb.h).
    return rgbToSpectrumLinear(rgb, w);
}

// Cap spectral path contributions in linear radiance (Arnold-style), matching
// RGB clampContribution: scale all λ bins by clamp/max so chromaticity is kept.
inline SampledSpectrum clampSpectrumIndirect(SampledSpectrum s, float clampValue) {
    if (clampValue <= 0.0f) return s;
    float m = 0.0f;
    for (int i = 0; i < s.n; ++i) m = srMax(m, s.values[i]);
    if (!(m > clampValue)) return s;
    const float scale = clampValue / m;
    for (int i = 0; i < s.n; ++i) s.values[i] *= scale;
    return s;
}

// Lift an RGB BSDF weight; conductors use η/κ, dispersing dielectrics get per-λ Fresnel.
// baseIor = undispersed IOR; heroIdx selects the wavelength that drove refraction sampling.
inline SampledSpectrum liftBsdfWeight(const Material& mat, const Frame& frame, Vec3 wo, Vec3 wi,
                                      Vec3 rgbWeight, const SampledWavelengths& w, float baseIor,
                                      int heroIdx) {
    SampledSpectrum base = rgbToSpectrumLinear(rgbWeight, w);
    const bool useConductor =
        mat.metallic >= 0.5f && (mat.conductorK.x + mat.conductorK.y + mat.conductorK.z) > 1e-4f;
    if (useConductor) {
        const Vec3 wh = normalize(wo + wi);
        if (length(wh) < 1e-6f) return base;
        SampledSpectrum s(w.n);
        for (int i = 0; i < w.n; ++i) {
            const SpectralNk nk = nkFromRgb(mat.conductorEta, mat.conductorK, w.lambda[i]);
            const float F = conductorFresnel(dot(wh, wo), nk.eta, nk.k);
            const float mag = (rgbWeight.x + rgbWeight.y + rgbWeight.z) * (1.0f / 3.0f);
            s.values[i] = mag * (0.25f + 0.75f * F);
        }
        return s;
    }

    // Chromatic dispersion: path bent with hero λ; scale other λ by relative Fresnel.
    if (mat.dispersionAbbe > 1e-3f && mat.transmission > 1e-4f && w.n > 0) {
        const bool transmitted = (dot(wo, frame.n) * dot(wi, frame.n)) < 0.0f;
        heroIdx = std::clamp(heroIdx, 0, w.n - 1);
        const float cosTheta = clampf(dot(wo, frame.n), -1.0f, 1.0f);
        auto lobe = [&](float etaAbs) {
            const float f = fresnelDielectric(cosTheta, etaAbs);
            return transmitted ? srMax(1e-4f, 1.0f - f) : srMax(1e-4f, f);
        };
        const float heroLobe =
            lobe(dielectricIorFromAbbe(baseIor, mat.dispersionAbbe, w.lambda[heroIdx]));
        SampledSpectrum s = base;
        for (int i = 0; i < w.n; ++i) {
            const float eta = dielectricIorFromAbbe(baseIor, mat.dispersionAbbe, w.lambda[i]);
            s.values[i] *= lobe(eta) / heroLobe;
        }
        return s;
    }
    return base;
}

}  // namespace sol
