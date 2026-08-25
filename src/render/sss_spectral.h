// Spectral Chiang random-walk BSSRDF (path hero-λ, not RGB-hero).
// Artist subsurface_radius / subsurface_color stay RGB; σ_t(λ) and α(λ) are
// lifted per sampled wavelength. RGB PT / MNEE / BDPT keep sampleSssRandomWalk.
#pragma once

#include "render/integrator.h"
#include "render/metal_spectra.h"
#include "render/spectral_common.h"

namespace sol {

struct SssWalkResultSpectral {
    bool escaped = false;
    Vec3 exitP{0.0f};
    Vec3 exitN{0.0f, 1.0f, 0.0f};
    Vec3 exitWo{0.0f, 1.0f, 0.0f};
    SampledSpectrum pathWeight{};
};

inline float sssMfpAtLambda(const Material& mat, float lambdaNm) {
    const Vec3 mfpRgb = vmax(Vec3(0.0f), mat.subsurfaceRadius) * srMax(0.0f, mat.subsurfaceScale);
    return srMax(1e-5f, rgbTripletAtLambda(mfpRgb, lambdaNm));
}

inline float sssAlbedoAtLambda(const Material& mat, float lambdaNm) {
    return saturatef(rgbTripletAtLambda(vmax(Vec3(0.0f), mat.subsurfaceColor), lambdaNm));
}

inline bool sssSpectrumWeightValid(const SampledSpectrum& s) {
    if (s.n <= 0) return false;
    float m = 0.0f;
    for (int i = 0; i < s.n; ++i) {
        if (!srIsFinite(s.values[i]) || s.values[i] < 0.0f) return false;
        m = srMax(m, s.values[i]);
    }
    return m > 1e-8f;
}

template <typename Tracer>
SR_INL SssWalkResultSpectral sampleSssRandomWalkSpectral(const SceneView& scene, const Tracer& tracer,
                                                         const SurfaceInteraction& entrySi, Vec3 wo,
                                                         const Material& mat, Rng& rng,
                                                         const SampledWavelengths& waves) {
    SssWalkResultSpectral out;
    out.exitP = entrySi.p;
    out.exitN = entrySi.ns;
    out.exitWo = wo;
    out.escaped = false;
    out.pathWeight = SampledSpectrum::zero(waves.n);

    const Vec3 mfpRgb = vmax(Vec3(0.0f), mat.subsurfaceRadius) * srMax(0.0f, mat.subsurfaceScale);
    auto fromRgbWalk = [&](const SssWalkResult& rgb) -> SssWalkResultSpectral {
        SssWalkResultSpectral s;
        s.escaped = rgb.escaped;
        s.exitP = rgb.exitP;
        s.exitN = rgb.exitN;
        s.exitWo = rgb.exitWo;
        s.pathWeight = upsampleRgb(rgb.pathWeight, waves);
        if (!s.escaped || !sssSpectrumWeightValid(s.pathWeight)) {
            s.escaped = false;
            s.pathWeight = SampledSpectrum::zero(waves.n);
        }
        return s;
    };

    if (maxComponent(mfpRgb) < 1e-8f || waves.n <= 0)
        return fromRgbWalk(sampleSssChristensenBurley(scene, tracer, entrySi, wo, mat, rng));

    const int n = waves.n < kMaxSpectrumSamples ? waves.n : kMaxSpectrumSamples;
    float sigma[kMaxSpectrumSamples];
    float alpha[kMaxSpectrumSamples];
    float sel[kMaxSpectrumSamples];
    float selSum = 0.0f;
    int nLive = 0;
    for (int i = 0; i < n; ++i) {
        if (waves.pdf[i] <= 0.0f) {
            sigma[i] = 0.0f;
            alpha[i] = 0.0f;
            sel[i] = 0.0f;
            continue;
        }
        sigma[i] = 1.0f / sssMfpAtLambda(mat, waves.lambda[i]);
        alpha[i] = chiangSingleScatterAlbedo(sssAlbedoAtLambda(mat, waves.lambda[i]));
        sel[i] = srMax(1e-3f, alpha[i] / srMax(1e-8f, sigma[i]));
        selSum += sel[i];
        ++nLive;
    }
    if (nLive <= 0 || selSum <= 0.0f)
        return fromRgbWalk(sampleSssChristensenBurley(scene, tracer, entrySi, wo, mat, rng));

    float uSel = rng.nextFloat() * selSum;
    int hero = 0;
    for (int i = 0; i < n; ++i) {
        if (sel[i] <= 0.0f) continue;
        if (uSel < sel[i]) {
            hero = i;
            break;
        }
        uSel -= sel[i];
        hero = i;
    }
    const float sigmaH = sigma[hero];
    if (sigmaH <= 0.0f)
        return fromRgbWalk(sampleSssChristensenBurley(scene, tracer, entrySi, wo, mat, rng));

    float pathPdf[kMaxSpectrumSamples];
    float thr[kMaxSpectrumSamples];
    for (int i = 0; i < n; ++i) {
        pathPdf[i] = sel[i] / selSum;
        thr[i] = waves.pdf[i] > 0.0f ? 1.0f : 0.0f;
    }

    Vec3 pWalk = entrySi.p - entrySi.ns * (kRayEpsilon * (1.0f + length(entrySi.p)));
    bool escaped = false;

    constexpr int kMaxWalkSteps = 24;
    for (int step = 0; step < kMaxWalkSteps; ++step) {
        const float stepLen = -logf(srMax(1e-6f, 1.0f - rng.nextFloat())) / sigmaH;

        Vec3 walkDir;
        if (step == 0) {
            const Frame inFrame(-entrySi.ns);
            walkDir = inFrame.toWorld(sampleCosineHemisphere(rng.nextFloat(), rng.nextFloat()));
        } else {
            walkDir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
        }
        if (lengthSquared(walkDir) < 1e-12f) continue;
        walkDir = normalize(walkDir);

        const float rayEps = kRayEpsilon * (1.0f + length(pWalk));
        const Vec3 walkOrigin = pWalk + walkDir * rayEps;
        RayHit walkHit;
        if (!tracer.intersect(walkOrigin, walkDir, stepLen, walkHit)) {
            pWalk = walkOrigin + walkDir * stepLen;
            for (int i = 0; i < n; ++i) {
                if (waves.pdf[i] <= 0.0f) continue;
                const float tr = expf(-sigma[i] * stepLen);
                const float dens = sigma[i] * tr;
                thr[i] *= alpha[i] * dens;
                pathPdf[i] *= dens;
            }
            const float wHero = thr[hero] / srMax(1e-20f, pathPdf[hero]);
            if (wHero < 1e-5f) {
                for (int i = 0; i < n; ++i) thr[i] = 0.0f;
                break;
            }
            if (step >= 8) {
                const float q = clampf(wHero, 0.25f, 1.0f);
                if (rng.nextFloat() > q) {
                    for (int i = 0; i < n; ++i) thr[i] = 0.0f;
                    break;
                }
                for (int i = 0; i < n; ++i) thr[i] /= q;
            }
            continue;
        }

        SurfaceInteraction walkSi;
        if (!buildSurfaceInteraction(scene, walkHit, walkOrigin, walkDir, walkSi)) break;

        const float tHit = srMax(0.0f, walkHit.t);
        for (int i = 0; i < n; ++i) {
            if (waves.pdf[i] <= 0.0f) continue;
            const float tr = expf(-sigma[i] * tHit);
            thr[i] *= tr;
            pathPdf[i] *= tr;
        }

        escaped = true;
        out.exitP = walkSi.p;
        out.exitN = walkSi.ns;
        if (dot(out.exitN, walkDir) < 0.0f) out.exitN = -out.exitN;
        if (lengthSquared(out.exitN) < 1e-12f) out.exitN = walkDir;
        else out.exitN = normalize(out.exitN);

        float pdfSum = 0.0f;
        for (int i = 0; i < n; ++i) pdfSum += pathPdf[i];
        out.pathWeight = SampledSpectrum(n);
        if (pdfSum > 1e-20f) {
            for (int i = 0; i < n; ++i)
                out.pathWeight.values[i] = waves.pdf[i] > 0.0f ? thr[i] / pdfSum : 0.0f;
        }
        const Vec3 toEntry = entrySi.p - out.exitP;
        const float woLen2 = lengthSquared(toEntry);
        out.exitWo = woLen2 > 1e-12f ? normalize(toEntry) : out.exitN;
        break;
    }

    if (!escaped || !sssSpectrumWeightValid(out.pathWeight))
        return fromRgbWalk(sampleSssChristensenBurley(scene, tracer, entrySi, wo, mat, rng));
    if (dot(out.exitN, out.exitWo) < 0.0f) out.exitWo = out.exitN;
    out.escaped = true;
    return out;
}

}  // namespace sol
