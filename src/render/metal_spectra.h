// Spectral η/κ metal tables + dielectric IOR(λ) from Abbe (Cauchy).
// Material authoring stores RGB triplets (η/κ ≈ samples at 650/550/450 nm);
// tables remain for presets that seed those triplets.
#pragma once

#include "core/math.h"
#include "render/spectrum.h"

namespace sol {

struct SpectralNk {
    float eta;
    float k;
};

// Interpolate an RGB triplet authored at R≈650, G≈550, B≈450 nm.
inline float rgbTripletAtLambda(Vec3 rgb, float lambdaNm) {
    if (lambdaNm <= 450.0f) return rgb.z;
    if (lambdaNm >= 650.0f) return rgb.x;
    if (lambdaNm <= 550.0f) {
        const float t = (lambdaNm - 450.0f) / 100.0f;
        return lerpf(rgb.z, rgb.y, t);
    }
    const float t = (lambdaNm - 550.0f) / 100.0f;
    return lerpf(rgb.y, rgb.x, t);
}

inline SpectralNk nkFromRgb(Vec3 etaRgb, Vec3 kRgb, float lambdaNm) {
    return {rgbTripletAtLambda(etaRgb, lambdaNm), rgbTripletAtLambda(kRgb, lambdaNm)};
}

// Invert conductor F0 = ((n-1)²+k²)/((n+1)²+k²) per RGB channel, holding η fixed.
// Used when metalness is on but conductor_k was never authored (base_color is F0).
inline void conductorNkFromReflectance(Vec3 f0, Vec3 etaHint, Vec3& etaRgb, Vec3& kRgb) {
    auto solveK = [](float r, float n) -> float {
        r = clampf(r, 1e-4f, 0.999f);
        n = srMax(1e-3f, n);
        const float np = n + 1.0f;
        const float nm = n - 1.0f;
        const float k2 = (r * np * np - nm * nm) / srMax(1e-6f, 1.0f - r);
        return sqrtf(srMax(0.0f, k2));
    };
    etaRgb = Vec3(srMax(1e-3f, etaHint.x), srMax(1e-3f, etaHint.y), srMax(1e-3f, etaHint.z));
    kRgb = Vec3(solveK(f0.x, etaRgb.x), solveK(f0.y, etaRgb.y), solveK(f0.z, etaRgb.z));
}

// Very compact tabulated metals (visible range). Linear interpolate in λ.
inline SpectralNk metalNk(const char* preset, float lambdaNm) {
    struct Sample {
        float lam, eta, k;
    };
    // Sparse visible samples (approximate published data).
    auto eval = [&](const Sample* tab, int n) -> SpectralNk {
        if (lambdaNm <= tab[0].lam) return {tab[0].eta, tab[0].k};
        if (lambdaNm >= tab[n - 1].lam) return {tab[n - 1].eta, tab[n - 1].k};
        for (int i = 0; i + 1 < n; ++i) {
            if (lambdaNm <= tab[i + 1].lam) {
                const float t = (lambdaNm - tab[i].lam) / (tab[i + 1].lam - tab[i].lam);
                return {lerpf(tab[i].eta, tab[i + 1].eta, t), lerpf(tab[i].k, tab[i + 1].k, t)};
            }
        }
        return {tab[n - 1].eta, tab[n - 1].k};
    };

    // Gold
    static const Sample au[] = {{400, 1.47f, 1.95f}, {450, 1.40f, 1.88f}, {500, 0.86f, 1.94f},
                                {550, 0.33f, 2.70f}, {600, 0.21f, 3.47f}, {650, 0.17f, 3.80f},
                                {700, 0.17f, 4.30f}};
    // Silver
    static const Sample ag[] = {{400, 0.05f, 2.07f}, {450, 0.04f, 2.52f}, {500, 0.05f, 2.98f},
                                {550, 0.06f, 3.45f}, {600, 0.06f, 3.90f}, {650, 0.07f, 4.30f},
                                {700, 0.08f, 4.70f}};
    // Copper
    static const Sample cu[] = {{400, 1.18f, 2.21f}, {450, 1.17f, 2.36f}, {500, 1.12f, 2.60f},
                                {550, 0.76f, 2.62f}, {600, 0.27f, 3.32f}, {650, 0.21f, 3.86f},
                                {700, 0.21f, 4.34f}};
    // Aluminium
    static const Sample al[] = {{400, 0.49f, 4.80f}, {450, 0.60f, 5.35f}, {500, 0.79f, 6.05f},
                                {550, 1.02f, 6.60f}, {600, 1.26f, 7.15f}, {650, 1.51f, 7.65f},
                                {700, 1.80f, 8.15f}};

    if (preset && preset[0]) {
        if (preset[0] == 'A' && preset[1] == 'u') return eval(au, 7);
        if (preset[0] == 'A' && preset[1] == 'g') return eval(ag, 7);
        if (preset[0] == 'C' && preset[1] == 'u') return eval(cu, 7);
        if (preset[0] == 'A' && preset[1] == 'l') return eval(al, 7);
    }
    // Fallback: treat as dielectric-ish conductor from grey.
    return {0.2f, 3.0f};
}

// Seed MaterialX conductor_eta / conductor_k from a named metal table.
inline void metalNkRgbPreset(const char* preset, Vec3& etaRgb, Vec3& kRgb) {
    const SpectralNk r = metalNk(preset, 650.0f);
    const SpectralNk g = metalNk(preset, 550.0f);
    const SpectralNk b = metalNk(preset, 450.0f);
    etaRgb = Vec3(r.eta, g.eta, b.eta);
    kRgb = Vec3(r.k, g.k, b.k);
}

// Cauchy: n(λ) = A + B/λ² with A,B from nd + Vd (Abbe).
inline float dielectricIorFromAbbe(float iorNd, float abbeVd, float lambdaNm) {
    if (abbeVd <= 1e-3f) return iorNd;
    const float lamD = 589.3f;
    const float lamF = 486.1f;
    const float lamC = 656.3f;
    const float invD = 1.0f / (lamD * lamD);
    const float invF = 1.0f / (lamF * lamF);
    const float invC = 1.0f / (lamC * lamC);
    // nd = A + B/λd², Vd = (nd-1)/(nF-nC) = (nd-1) / (B(invF-invC))
    const float B = (iorNd - 1.0f) / (abbeVd * (invF - invC));
    const float A = iorNd - B * invD;
    const float inv = 1.0f / (lambdaNm * lambdaNm);
    return srMax(1.0f, A + B * inv);
}

// Complex Fresnel (conductor) reflectance for unpolarized light.
inline float conductorFresnel(float cosTheta, float eta, float k) {
    cosTheta = clampf(fabsf(cosTheta), 0.0f, 1.0f);
    const float cos2 = cosTheta * cosTheta;
    const float sin2 = 1.0f - cos2;
    const float eta2 = eta * eta;
    const float k2 = k * k;
    const float t0 = eta2 - k2 - sin2;
    const float a2plusb2 = sqrtf(srMax(0.0f, t0 * t0 + 4.0f * eta2 * k2));
    const float a = sqrtf(srMax(0.0f, 0.5f * (a2plusb2 + t0)));
    const float t1 = a2plusb2 + cos2;
    const float t2 = 2.0f * a * cosTheta;
    const float rs = (t1 - t2) / srMax(1e-8f, t1 + t2);
    const float t3 = cos2 * a2plusb2 + sin2 * sin2;
    const float t4 = t2 * sin2;
    const float rp = rs * (t3 - t4) / srMax(1e-8f, t3 + t4);
    return saturatef(0.5f * (rs + rp));
}

}  // namespace sol
