// Shared spectral film buffer + spectral BSDF helpers (PT / BDPT Spectral).
#pragma once

#include <algorithm>
#include <vector>

#include "render/metal_spectra.h"
#include "render/shading.h"
#include "render/spectrum.h"
#include "render/spectrum_rgb.h"
#include "render/spectrum_types.h"
#include "render/volume.h"
#include "render/volume_track.h"

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
            if (w.pdf[i] <= 0.0f) continue;
            float t = (w.lambda[i] - kSpectrumLambdaMin) / span;
            int bin = int(t * float(bins));
            bin = std::clamp(bin, 0, bins - 1);
            accum[base + size_t(bin)] += s.values[i] / w.pdf[i];
        }
    }

    float binValue(int x, int y, int bin) const {
        if (bins <= 0 || !width || bin < 0 || bin >= bins) return 0.0f;
        if (x < 0 || y < 0 || x >= width || y >= height) return 0.0f;
        return accum[(size_t(y) * size_t(width) + size_t(x)) * size_t(bins) + size_t(bin)];
    }
};

// MC path weights / NEE aggregates — energy-safe under multiply.
inline SampledSpectrum upsampleRgb(Vec3 rgb, const SampledWavelengths& w) {
    return rgbToSpectrumLinear(rgb, w);
}

// pbrt null-scattering / delta tracking on the sampled wavelengths (hero = slot 0).
inline MediumSample sampleMediumHomogeneousSpectral(const MediumData& m, float tMax, Rng& rng,
                                                    SampledSpectrum& throughput,
                                                    const SampledWavelengths& lambda) {
    MediumSample out;
    const SampledSpectrum sigA = upsampleRgb(mediumSigmaA(m), lambda);
    const SampledSpectrum sigS = upsampleRgb(mediumSigmaS(m), lambda);
    float majorant = 0.0f;
    for (int i = 0; i < lambda.n; ++i) majorant = srMax(majorant, sigA[i] + sigS[i]);
    if (majorant <= 1e-8f || tMax <= 0.0f || lambda.n <= 0) {
        out.t = tMax;
        return out;
    }
    const int hero = 0;
    const float stHero = sigA[hero] + sigS[hero];
    float t = 0.0f;
    bool anyEvent = false;
    constexpr int kNullCollisionMaxIters = 1 << 20;
    for (int iter = 0; iter < kNullCollisionMaxIters; ++iter) {
        const float u = srMax(1e-6f, 1.0f - rng.nextFloat());
        t += -logf(u) / majorant;
        if (t >= tMax) {
            if (!anyEvent) {
                for (int i = 0; i < throughput.n && i < lambda.n; ++i) {
                    const float st = sigA[i] + sigS[i];
                    throughput[i] *= expf(-(st - stHero) * tMax);
                }
            }
            out.t = tMax;
            return out;
        }
        anyEvent = true;
        const float pA = sigA[hero] / majorant;
        const float pS = sigS[hero] / majorant;
        const float uMode = rng.nextFloat();
        if (uMode < pA) {
            for (int i = 0; i < throughput.n; ++i) throughput[i] = 0.0f;
            out.t = t;
            out.absorbed = true;
            return out;
        }
        if (uMode < pA + pS) {
            const float sHero = srMax(sigS[hero], 1e-12f);
            for (int i = 0; i < throughput.n && i < lambda.n; ++i) throughput[i] *= sigS[i] / sHero;
            out.t = t;
            out.scattered = true;
            return out;
        }
        const float nHero = srMax(majorant - stHero, 1e-12f);
        for (int i = 0; i < throughput.n && i < lambda.n; ++i) {
            const float nI = majorant - sigA[i] - sigS[i];
            throughput[i] *= nI / nHero;
        }
    }
    out.t = tMax;
    return out;
}

// Hero-λ absorb / scatter / null at a fog collision (occupancy-scaled σ).
inline bool fogInteractSpectral(const MediumData& medium, float densityScale, float occupancy, float tHit,
                                Rng& rng, SampledSpectrum& throughput, const SampledWavelengths& lambda,
                                MediumSample& out) {
    const float dens = srMax(0.0f, occupancy) * densityScale;
    const SampledSpectrum sigA = upsampleRgb(medium.sigmaA * dens, lambda);
    const SampledSpectrum sigS = upsampleRgb(medium.sigmaS * dens, lambda);
    float majorant = 0.0f;
    for (int i = 0; i < lambda.n; ++i) majorant = srMax(majorant, sigA[i] + sigS[i]);
    if (majorant <= 1e-12f) return false;
    const int hero = 0;
    const float stHero = sigA[hero] + sigS[hero];
    const float pA = sigA[hero] / majorant;
    const float pS = sigS[hero] / majorant;
    const float uMode = rng.nextFloat();
    if (uMode < pA) {
        for (int i = 0; i < throughput.n; ++i) throughput[i] = 0.0f;
        out.t = tHit;
        out.absorbed = true;
        return true;
    }
    if (uMode < pA + pS) {
        const float sHero = srMax(sigS[hero], 1e-12f);
        for (int i = 0; i < throughput.n && i < lambda.n; ++i) throughput[i] *= sigS[i] / sHero;
        out.t = tHit;
        out.scattered = true;
        return true;
    }
    const float nHero = srMax(majorant - stHero, 1e-12f);
    for (int i = 0; i < throughput.n && i < lambda.n; ++i) {
        const float nI = majorant - sigA[i] - sigS[i];
        throughput[i] *= nI / nHero;
    }
    return false;
}

inline bool fogCollideSpectral(const MediumData& medium, float densityScale, float occupancy, float majorant,
                               float tHit, Rng& rng, SampledSpectrum& throughput,
                               const SampledWavelengths& lambda, MediumSample& out) {
    const float dens = srMax(0.0f, occupancy) * densityScale;
    const SampledSpectrum sigT = upsampleRgb((medium.sigmaA + medium.sigmaS) * dens, lambda);
    float stHero = 0.0f;
    for (int i = 0; i < lambda.n; ++i) stHero = srMax(stHero, sigT[i]);
    if (rng.nextFloat() >= stHero / srMax(majorant, 1e-12f)) return false;
    return fogInteractSpectral(medium, densityScale, occupancy, tHit, rng, throughput, lambda, out);
}

// Same piecewise-majorant walk as RGB fog, with spectral null-collision.
template <typename Grid>
inline MediumSample sampleHeterogeneousFogSpectral(const Grid& grid, const MediumData& medium, Vec3 origin,
                                                   Vec3 direction, float tMax, Rng& rng,
                                                   SampledSpectrum& throughput,
                                                   const SampledWavelengths& lambda) {
    MediumSample out;
    if (tMax <= 0.0f) {
        out.t = tMax;
        return out;
    }
    float aabbEnter = 0.0f;
    float aabbExit = tMax;
    if (rayAabbInterval(origin, direction, grid.bmin(), grid.bmax(), aabbEnter, aabbExit))
        tMax = srMin(tMax, srMax(0.0f, aabbExit));
    if (tMax <= 0.0f) {
        out.t = 0.0f;
        return out;
    }
    float t = srMax(0.0f, aabbEnter);
    if (t >= tMax) {
        out.t = tMax;
        return out;
    }

    const float densityScale = srMax(0.0f, medium.density);
    const SampledSpectrum sigT0 = upsampleRgb(medium.sigmaA + medium.sigmaS, lambda);
    float baseMaj = 0.0f;
    for (int i = 0; i < lambda.n; ++i) baseMaj = srMax(baseMaj, sigT0[i]);
    if (baseMaj <= 1e-12f || densityScale <= 0.0f) {
        out.t = tMax;
        return out;
    }
    const float st0 = baseMaj * densityScale;

    for (int iter = 0; iter < kNullCollisionMaxIters && t < tMax; ++iter) {
        const Vec3 pLook = origin + direction * (t + 1e-5f);
        if (grid.hasMajorantBricks() && grid.brickEmpty(pLook)) {
            const float tBr = grid.brickExitT(origin, direction, t, tMax);
            t = (tBr > t) ? tBr : t + 1e-4f;
            continue;
        }
        float minD = 0.0f;
        float maxD = grid.majorant();
        grid.occupancy(pLook, minD, maxD);
        const float tCell = grid.hasMajorantGrid() ? grid.cellExitT(origin, direction, t, tMax) : tMax;
        if (!(tCell > t)) {
            t += 1e-4f;
            continue;
        }
        if (maxD <= 1e-8f) {
            t = tCell;
            continue;
        }
        const float majorant = srMax(1e-8f, baseMaj * maxD * densityScale);
        const bool homog = (maxD - minD) <= (1e-3f * srMax(maxD, 1e-6f) + 1e-4f);
        if (homog) {
            const float occ = 0.5f * (minD + maxD);
            const float u = srMax(1e-6f, 1.0f - rng.nextFloat());
            const float tHit = t + (-logf(u) / majorant);
            if (tHit >= tCell) {
                t = tCell;
                continue;
            }
            if (fogCollideSpectral(medium, densityScale, occ, majorant, tHit, rng, throughput, lambda, out))
                return out;
            t = tHit;
            continue;
        }
        const float muC = st0 * srMax(0.0f, minD);
        const float residualMaj = srMax(0.0f, majorant - muC);
        float tLocal = t;
        const float tCtrl =
            (muC > 1e-8f) ? t + (-logf(srMax(1e-6f, 1.0f - rng.nextFloat())) / muC) : tCell + 1.0f;
        bool leftCell = false;
        while (iter < kNullCollisionMaxIters) {
            float tRes = tCell + 1.0f;
            if (residualMaj > 1e-8f) {
                tLocal += -logf(srMax(1e-6f, 1.0f - rng.nextFloat())) / residualMaj;
                tRes = tLocal;
            }
            const float tEvent = srMin(tCtrl, tRes);
            if (tEvent >= tCell) {
                t = tCell;
                leftCell = true;
                break;
            }
            if (tCtrl <= tRes) {
                fogInteractSpectral(medium, densityScale, minD, tCtrl, rng, throughput, lambda, out);
                return out;
            }
            const float occ = clampf(grid.sampleOcc(origin + direction * tRes), minD, maxD);
            const float pReal = (st0 * occ - muC) / srMax(residualMaj, 1e-12f);
            ++iter;
            if (rng.nextFloat() >= clampf(pReal, 0.0f, 1.0f)) continue;
            fogInteractSpectral(medium, densityScale, occ, tRes, rng, throughput, lambda, out);
            return out;
        }
        if (leftCell) continue;
        break;
    }
    out.t = tMax;
    return out;
}

// Film colour space for spectral PT/BDPT. ACEScg working space is source of
// truth so OptiX RGB (ACEScg) and Embree spectral cannot drift; the spectral
// menu is honoured only in linear-sRGB working space.
inline const RGBColorSpace& pathColorSpace(const RenderSettingsData& settings) {
    if (settings.workingSpace == kWorkingSpaceAcesCg) return colorSpaceAcesCg();
    return colorSpaceById(settings.spectralColorSpace);
}

// Authored reflectance (textures / albedo on hit) — pbrt RGBAlbedoSpectrum.
inline SampledSpectrum upsampleAlbedo(Vec3 rgb, const SampledWavelengths& w,
                                      const RGBColorSpace& cs = colorSpaceSrgb()) {
    return rgbToSpectrumReflectance(rgb, w, cs);
}

// Authored emission / lights / HDR env — pbrt RGBIlluminantSpectrum (× D65/D60).
inline SampledSpectrum upsampleEmission(Vec3 rgb, const SampledWavelengths& w,
                                        const RGBColorSpace& cs = colorSpaceSrgb()) {
    return rgbToSpectrumEmission(rgb, w, cs);
}

// Light spectrum: optional blackbody CCT, else RGBIlluminantSpectrum of emittedRadiance().
inline SampledSpectrum lightEmissionSpectrum(const LightData& light, const SampledWavelengths& w,
                                            const RGBColorSpace& cs = colorSpaceSrgb()) {
    const Vec3 rgb = light.emittedRadiance();
    if (light.colorTemperatureK > 50.0f) {
        BlackbodySpectrum bb(light.colorTemperatureK);
        SampledSpectrum s = bb.sample(w);
        const Vec3 bbRgb = spectrumToRgb(s, w, cs);
        const float bbLum = 0.2126f * bbRgb.x + 0.7152f * bbRgb.y + 0.0722f * bbRgb.z;
        const float wantLum = 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
        s *= wantLum / srMax(bbLum, 1e-8f);
        const Vec3 tint = light.color;
        if (fabsf(tint.x - 1.0f) > 1e-4f || fabsf(tint.y - 1.0f) > 1e-4f ||
            fabsf(tint.z - 1.0f) > 1e-4f) {
            SampledSpectrum t = rgbToSpectrumEmission(tint, w, cs);
            const float tAvg = srMax(spectrumAvg(t), 1e-8f);
            for (int i = 0; i < s.n; ++i) s.values[i] *= t.values[i] / tAvg;
        }
        return s;
    }
    return upsampleEmission(rgb, w, cs);
}

inline SampledSpectrum clampSpectrumIndirect(SampledSpectrum s, float clampValue) {
    if (clampValue <= 0.0f) return s;
    float m = 0.0f;
    for (int i = 0; i < s.n; ++i) m = srMax(m, s.values[i]);
    if (!(m > clampValue)) return s;
    const float scale = clampValue / m;
    for (int i = 0; i < s.n; ++i) s.values[i] *= scale;
    return s;
}

inline bool usesSpectralDielectric(const Material& mat) {
    return saturatef(mat.transmission) > 1e-4f && saturatef(mat.metallic) < 0.5f;
}

inline float spectralAbsoluteIor(float baseIor, float abbeVd, float lambdaNm) {
    return dielectricIorFromAbbe(baseIor, abbeVd, lambdaNm);
}

// pbrt-style: keep multi-λ on specular / near-specular; terminate on rough/diffuse.
inline bool shouldTerminateSecondaryWavelengths(const BsdfSample& bs, const LobeWeights& lw) {
    return !(bs.specular || isNearSpecularLobe(lw));
}

struct BsdfSampleSpectral {
    SampledSpectrum weight{};
    Vec3 wi{0.0f};
    float pdf = 0.0f;
    bool specular = false;
    bool transmitted = false;
    bool valid = false;
};

// Opaque / metal / thin-film lift of an RGB BSDF weight (not for glass dielectrics).
// Textures are filtered in RGB (evaluateTexturedMaterial); the book converts that
// RGB to a spectrum at lookup. The BSDF is still RGB, so the weight already holds
// the filtered colour — RGBUnboundedSpectrum is pbrt's conversion for that RGB.
inline SampledSpectrum liftBsdfWeight(const Material& mat, const Frame& frame, Vec3 wo, Vec3 wi,
                                      Vec3 rgbWeight, const SampledWavelengths& w, float /*baseIor*/,
                                      int /*heroIdx*/, const RGBColorSpace& cs = colorSpaceSrgb()) {
    SampledSpectrum base = rgbToSpectrumUnbounded(rgbWeight, w, cs);
    const bool useConductor =
        mat.metallic >= 0.5f && (mat.conductorK.x + mat.conductorK.y + mat.conductorK.z) > 1e-4f;
    if (useConductor) {
        const Vec3 wh = normalize(wo + wi);
        if (length(wh) < 1e-6f) return base;
        SampledSpectrum s(w.n);
        const float mag = (rgbWeight.x + rgbWeight.y + rgbWeight.z) * (1.0f / 3.0f);
        for (int i = 0; i < w.n; ++i) {
            if (w.pdf[i] <= 0.0f) {
                s.values[i] = 0.0f;
                continue;
            }
            const SpectralNk nk = nkFromRgb(mat.conductorEta, mat.conductorK, w.lambda[i]);
            const float F = conductorFresnel(dot(wh, wo), nk.eta, nk.k);
            s.values[i] = mag * (0.25f + 0.75f * F);
        }
        return s;
    }
    if (mat.thinFilmThickness > 0.5f) {
        const Vec3 wh = normalize(wo + wi);
        const float cosTheta =
            clampf(length(wh) > 1e-6f ? dot(wh, wo) : dot(wo, frame.n), 0.0f, 1.0f);
        const float metallic = saturatef(mat.metallic);
        const Vec3 baseCol = vmax(Vec3(0.0f), mat.baseColor);
        const float dielectricF0 = 0.08f * saturatef(mat.specular);
        const Vec3 f0 =
            lerp(Vec3(dielectricF0) * vmax(Vec3(0.0f), mat.specularColor), baseCol, metallic);
        const float r23 =
            (sqrtf(clampf(f0.x, 0.0f, 1.0f)) + sqrtf(clampf(f0.y, 0.0f, 1.0f)) +
             sqrtf(clampf(f0.z, 0.0f, 1.0f))) *
            (1.0f / 3.0f);
        const Vec3 Frgb = thinFilmFresnel(f0, cosTheta, mat.thinFilmIor, mat.thinFilmThickness);
        const float FrgbAvg = srMax((Frgb.x + Frgb.y + Frgb.z) * (1.0f / 3.0f), 1e-4f);
        for (int i = 0; i < w.n; ++i) {
            if (w.pdf[i] <= 0.0f) {
                base.values[i] = 0.0f;
                continue;
            }
            const float Fλ = airyReflectanceScalar(cosTheta, mat.thinFilmIor, mat.thinFilmThickness,
                                                   w.lambda[i], r23);
            base.values[i] *= Fλ / FrgbAvg;
        }
    }
    return base;
}

// Per-λ dielectric BSDF f (delta lobes return 0 — same as RGB eval).
inline SampledSpectrum bsdfEvalSpectralDielectric(const Material& mat, Vec3 wo, Vec3 wi,
                                                  const SampledWavelengths& w, float baseIor,
                                                  const RGBColorSpace& cs = colorSpaceSrgb()) {
    SampledSpectrum out = SampledSpectrum::zero(w.n);
    if (!usesSpectralDielectric(mat) || fabsf(wo.z) < 1e-6f || fabsf(wi.z) < 1e-6f) return out;

    Material matNd = mat;
    matNd.ior = baseIor;
    const LobeWeights lw = computeLobes(matNd, wo);
    if (lw.transmission <= 1e-5f || lw.delta) return out;

    const float tw = saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
    const bool reflecting = wo.z * wi.z > 0.0f;
    const SampledSpectrum tint = rgbToSpectrumReflectance(lw.transmissionTint, w, cs);

    if (reflecting) {
        Vec3 h = wo + wi;
        if (lengthSquared(h) <= 0.0f) return out;
        h = normalize(h);
        if (h.z < 0.0f) h = -h;
        const bool allowDielectricReflect = wo.z > 0.0f || mat.internalReflections > 0.5f;
        const float d = ggxD(h, lw.alpha);
        const float g = smithG2(wo, wi, lw.alpha);
        for (int i = 0; i < w.n; ++i) {
            if (w.pdf[i] <= 0.0f) continue;
            const float etaAbs = spectralAbsoluteIor(baseIor, mat.dispersionAbbe, w.lambda[i]);
            const float eta = wo.z > 0.0f ? etaAbs : 1.0f / etaAbs;
            if (!allowDielectricReflect) {
                const float cosHI = absDot(wo, h);
                const float sin2 = srMax(0.0f, 1.0f - cosHI * cosHI);
                if ((sin2 / (eta * eta)) < 1.0f) continue;
            }
            const float fr = fresnelDielectric(dot(wo, h), eta);
            out.values[i] = (d * g * fr / (4.0f * fabsf(wo.z) * fabsf(wi.z))) * tw;
        }
        return out;
    }

    for (int i = 0; i < w.n; ++i) {
        if (w.pdf[i] <= 0.0f) continue;
        const float etaAbs = spectralAbsoluteIor(baseIor, mat.dispersionAbbe, w.lambda[i]);
        const float eta = wo.z > 0.0f ? etaAbs : 1.0f / etaAbs;
        Vec3 h = -(wo + wi * eta);
        if (lengthSquared(h) <= 0.0f) continue;
        h = normalize(h);
        if (h.z < 0.0f) h = -h;
        const float dotOH = dot(wo, h);
        const float dotIH = dot(wi, h);
        if (dotOH * wo.z <= 0.0f) continue;
        const float sqrtDenom = dotOH + eta * dotIH;
        if (fabsf(sqrtDenom) <= 1e-6f) continue;
        const float fr = fresnelDielectric(dotOH, eta);
        const float d = ggxD(h, lw.alpha);
        const float g = smithG2(wo, wi, lw.alpha);
        const float factor = fabsf(dotIH * dotOH / (wo.z * wi.z));
        const float ft = (1.0f - fr) * d * g * factor * (eta * eta) / (sqrtDenom * sqrtDenom);
        out.values[i] = tint.values[i] * ft * tw;
    }
    return out;
}

// Spectral BSDF f for BDPT connections / NEE-style evals.
inline SampledSpectrum bsdfEvalSpectral(const Material& mat, Vec3 ng, Vec3 ns, Vec3 woW, Vec3 wiW,
                                        const SampledWavelengths& w, float baseIor,
                                        const RGBColorSpace& cs = colorSpaceSrgb()) {
    if (!shadingNormalConsistent(ng, ns, woW, wiW)) return SampledSpectrum::zero(w.n);
    const Frame frame(ns);
    const Vec3 wo = frame.toLocal(woW);
    const Vec3 wi = frame.toLocal(wiW);

    if (usesSpectralDielectric(mat)) {
        Material matNd = mat;
        matNd.ior = baseIor;
        const LobeWeights lw = computeLobes(matNd, wo);
        if (lw.transmission > 0.5f && lw.diffuse < 1e-3f)
            return bsdfEvalSpectralDielectric(mat, wo, wi, w, baseIor, cs);
        SampledSpectrum f = bsdfEvalSpectralDielectric(mat, wo, wi, w, baseIor, cs);
        Material matOpaque = matNd;
        matOpaque.transmission = 0.0f;
        f += liftBsdfWeight(matOpaque, frame, woW, wiW, bsdfEvalLocal(matOpaque, wo, wi).f, w,
                            baseIor, 0, cs);
        return f;
    }

    const BsdfEval e = bsdfEvalLocal(mat, wo, wi);
    return liftBsdfWeight(mat, frame, woW, wiW, e.f, w, baseIor, 0, cs);
}

// Hero-λ geometry sample; per-λ Fresnel and 1/η(λ)² for dielectric weights.
inline BsdfSampleSpectral bsdfSampleSpectral(const Material& mat, Vec3 woLocal, float uLobe,
                                             float u1, float u2, float uChoice,
                                             const SampledWavelengths& w, float baseIor,
                                             int heroIdx, const RGBColorSpace& cs = colorSpaceSrgb()) {
    BsdfSampleSpectral out;
    out.weight = SampledSpectrum::zero(w.n);
    heroIdx = std::clamp(heroIdx, 0, std::max(0, w.n - 1));

    Material matHero = mat;
    matHero.ior = spectralAbsoluteIor(baseIor, mat.dispersionAbbe, w.lambda[heroIdx]);
    const BsdfSample rgb = bsdfSampleLocal(matHero, woLocal, uLobe, u1, u2, uChoice);
    if (rgb.pdf <= 0.0f || isBlack(rgb.weight)) return out;

    out.wi = rgb.wi;
    out.pdf = rgb.pdf;
    out.specular = rgb.specular;
    out.transmitted = rgb.transmitted;
    out.valid = true;

    const Frame localZ(Vec3(0.0f, 0.0f, 1.0f));
    if (!usesSpectralDielectric(mat)) {
        out.weight =
            liftBsdfWeight(mat, localZ, woLocal, rgb.wi, rgb.weight, w, baseIor, heroIdx, cs);
        return out;
    }

    Material matNd = mat;
    matNd.ior = baseIor;
    const LobeWeights lw = computeLobes(matNd, woLocal);
    const bool fromDielectricLobe = uLobe >= lw.diffuse + lw.specular - 1e-6f || rgb.transmitted;
    if (!fromDielectricLobe) {
        out.weight =
            liftBsdfWeight(mat, localZ, woLocal, rgb.wi, rgb.weight, w, baseIor, heroIdx, cs);
        return out;
    }

    const float tw = saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
    const SampledSpectrum tint = rgbToSpectrumReflectance(lw.transmissionTint, w, cs);
    const float heroEtaAbs =
        spectralAbsoluteIor(baseIor, mat.dispersionAbbe, w.lambda[heroIdx]);
    const float heroEta = woLocal.z > 0.0f ? heroEtaAbs : 1.0f / heroEtaAbs;

    Vec3 h;
    if (lw.delta) {
        h = Vec3(0.0f, 0.0f, woLocal.z > 0.0f ? 1.0f : -1.0f);
    } else if (rgb.transmitted) {
        h = -(woLocal + rgb.wi * heroEta);
        if (lengthSquared(h) > 0.0f) {
            h = normalize(h);
            if (h.z < 0.0f) h = -h;
        } else {
            h = Vec3(0.0f, 0.0f, woLocal.z > 0.0f ? 1.0f : -1.0f);
        }
    } else {
        h = normalize(woLocal + rgb.wi);
        if (h.z < 0.0f) h = -h;
    }
    const float dotOH = dot(woLocal, h);
    const float Fhero = fresnelDielectric(dotOH, heroEta);
    const bool allowInternalReflect = mat.internalReflections > 0.5f || woLocal.z > 0.0f;

    if (lw.delta) {
        for (int i = 0; i < w.n; ++i) {
            if (w.pdf[i] <= 0.0f) {
                out.weight.values[i] = 0.0f;
                continue;
            }
            const float etaAbs = spectralAbsoluteIor(baseIor, mat.dispersionAbbe, w.lambda[i]);
            const float eta = woLocal.z > 0.0f ? etaAbs : 1.0f / etaAbs;
            const float F = fresnelDielectric(dotOH, eta);
            if (rgb.transmitted) {
                float scale = tw / (eta * eta * srMax(1e-4f, lw.transmission));
                if (!allowInternalReflect && woLocal.z < 0.0f) {
                    scale *= (1.0f - F);
                } else {
                    scale *= (1.0f - F) / srMax(1e-4f, 1.0f - Fhero);
                }
                out.weight.values[i] = tint.values[i] * scale;
            } else {
                float scale = tw / srMax(1e-4f, lw.transmission);
                if (Fhero > 1e-4f) scale *= F / Fhero;
                out.weight.values[i] = scale;
            }
        }
        return out;
    }

    SampledSpectrum f = bsdfEvalSpectralDielectric(mat, woLocal, rgb.wi, w, baseIor, cs);
    const float invPdf = 1.0f / srMax(rgb.pdf, 1e-8f);
    for (int i = 0; i < w.n; ++i)
        out.weight.values[i] = f.values[i] * fabsf(rgb.wi.z) * invPdf;
    return out;
}

}  // namespace sol
