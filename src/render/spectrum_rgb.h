// RGB→spectrum upsampling (pbrt-v4 ch. 4 / 10 / 12).
//
//   rgbToSpectrumLinear      — energy-safe under multiply. MC path weights
//                              that are already f·cos/pdf aggregates, 1/η²,
//                              and NEE RGB sums. Not a colour appearance model.
//   rgbToSpectrumReflectance — RGBAlbedoSpectrum: Jakob albedo, [0,1]
//                              (unbounded: scale = 2·max, lookup rgb/scale).
//   rgbToSpectrumUnbounded   — RGBUnboundedSpectrum: Jakob illuminant table
//                              with scale = 2·max (HDR / general RGB).
//   rgbToSpectrumEmission    — RGBIlluminantSpectrum: unbounded Jakob × the
//                              working-space illuminant (D65 / D60).
//
// ACEScg working space uses the D60-fitted AP1 tables; sRGB/Rec.2020/P3 keep
// the E-fitted sRGB tables.
#pragma once

#include "render/illuminant_spd.h"
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

inline bool colorSpaceUsesAcesTables(const RGBColorSpace& cs) {
    return cs.whiteIlluminant == kWhiteIlluminantD60;
}

inline rgb_spec::RgbSigmoidPolynomial fetchAlbedoPoly(Vec3 rgb, const RGBColorSpace& cs) {
    return colorSpaceUsesAcesTables(cs) ? rgb_spec::fetchAlbedoAces(rgb) : rgb_spec::fetchAlbedo(rgb);
}

inline rgb_spec::RgbSigmoidPolynomial fetchIlluminantPoly(Vec3 rgb, const RGBColorSpace& cs) {
    return colorSpaceUsesAcesTables(cs) ? rgb_spec::fetchIlluminantAces(rgb)
                                        : rgb_spec::fetchIlluminant(rgb);
}

}  // namespace

// Linear RGB→spectrum (energy-conserving under multiply). Use for Path Tracer / BDPT
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

// Reflectance / albedo: pbrt RGBAlbedoSpectrum. HDR albedo uses the unbounded
// path (scale = 2·max, lookup rgb/scale) on the albedo table.
inline SampledSpectrum rgbToSpectrumReflectance(Vec3 rgb, const SampledWavelengths& w,
                                                const RGBColorSpace& cs = colorSpaceSrgb()) {
    const float r = srMax(0.0f, rgb.x);
    const float g = srMax(0.0f, rgb.y);
    const float b = srMax(0.0f, rgb.z);
    const Vec3 c(r, g, b);
    if (r <= 0.0f && g <= 0.0f && b <= 0.0f) return SampledSpectrum::zero(w.n);

    const float m = srMax(r, srMax(g, b));
    if (m <= 1.0f) return rgb_spec::evalPolynomial(fetchAlbedoPoly(c, cs), w);
    const float scale = 2.0f * m;
    return rgb_spec::evalPolynomial(fetchAlbedoPoly(c / scale, cs), w, scale);
}

// pbrt RGBUnboundedSpectrum: scale = 2·max(rgb), Jakob of rgb/scale.
inline SampledSpectrum rgbToSpectrumUnbounded(Vec3 rgb, const SampledWavelengths& w,
                                              const RGBColorSpace& cs = colorSpaceSrgb()) {
    const float r = srMax(0.0f, rgb.x);
    const float g = srMax(0.0f, rgb.y);
    const float b = srMax(0.0f, rgb.z);
    if (r <= 0.0f && g <= 0.0f && b <= 0.0f) return SampledSpectrum::zero(w.n);
    const float m = srMax(r, srMax(g, b));
    const float scale = 2.0f * m;
    return rgb_spec::evalPolynomial(fetchIlluminantPoly(Vec3(r, g, b) / scale, cs), w, scale);
}

inline SampledSpectrum illuminantSpectrum(const SampledWavelengths& w,
                                          const RGBColorSpace& cs = colorSpaceSrgb()) {
    SampledSpectrum s(w.n);
    for (int i = 0; i < w.n; ++i) s.values[i] = sampleColorSpaceIlluminant(cs, w.lambda[i]);
    return s;
}

// Lights / HDR env / emission textures: pbrt RGBIlluminantSpectrum.
inline SampledSpectrum rgbToSpectrumEmission(Vec3 rgb, const SampledWavelengths& w,
                                             const RGBColorSpace& cs = colorSpaceSrgb()) {
    SampledSpectrum s = rgbToSpectrumUnbounded(rgb, w, cs);
    for (int i = 0; i < s.n && i < w.n; ++i)
        s.values[i] *= sampleColorSpaceIlluminant(cs, w.lambda[i]);
    return s;
}

}  // namespace sol
