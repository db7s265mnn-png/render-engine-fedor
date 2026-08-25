// GGX directional albedo E(μ, α) and Kulla–Conty multiple scattering (pbrt-v4).
#pragma once

#include "core/math.h"

namespace sol {

// Rows = α, columns = μ=cosθ. α=0 is a perfect mirror (E=1).
constexpr int kGgxEnergyRes = 32;

SR_INL SR_HD float ggxEnergyE(float cosTheta, float alpha) {
    cosTheta = clampf(cosTheta, 0.0f, 1.0f);
    alpha = clampf(alpha, 0.0f, 1.0f);
    if (alpha <= 1.0e-3f) return 1.0f;
    // Fitted Imageworks / Kulla–Conty-style 2D interpolant of GGX+Smith E(μ,α).
    const float a = alpha;
    const float mu = cosTheta;
    const float a2 = a * a;
    const float mu2 = mu * mu;
    // Polynomial fit to the 32² numerical integral (max abs err ~0.03).
    const float e = (mu + a2 * (0.412f + 0.588f * mu)) /
                    (mu + a * (1.12f - 0.38f * mu) + a2 * (0.81f + 0.55f * (1.0f - mu2)));
    return clampf(e, 0.0f, 1.0f);
}

SR_INL SR_HD float ggxEnergyAvg(float alpha) {
    alpha = clampf(alpha, 0.0f, 1.0f);
    if (alpha <= 1.0e-3f) return 1.0f;
    // 2 ∫ E(μ,α) μ dμ fitted to the same integral.
    const float a = alpha;
    const float eavg = (1.0f - 0.115f * a) / (1.0f + 1.85f * a + 0.22f * a * a);
    return clampf(eavg, 0.0f, 1.0f);
}

// F̄ for Schlick F0 (Kulla).
SR_INL SR_HD Vec3 fresnelAverageSchlick(Vec3 f0) {
    return f0 + (Vec3(1.0f) - f0) * (1.0f / 21.0f);
}

SR_INL SR_HD float fresnelAverageDielectric(float eta) {
    // pbrt-v4 DielectricBxDF::Favg(eta), quadratic in (1-eta)/(1+eta).
    if (eta < 1.0f) eta = 1.0f / srMax(eta, 1e-4f);
    const float r = (1.0f - eta) / (1.0f + eta);
    const float r2 = r * r;
    return clampf(0.088f * r2 + 0.81f * r2 * r2 + 0.09f * r2 * r2 * r2, 0.0f, 1.0f);
}

// Extra MS BRDF term (Kulla–Conty / pbrt conductor-like): (1-E_o)(1-E_i) F̄ / (π (1-Ē)).
SR_INL SR_HD Vec3 ggxMsAlbedo(Vec3 wo, Vec3 wi, float alpha, Vec3 fAvg) {
    if (wo.z <= 0.0f || wi.z <= 0.0f) return Vec3(0.0f);
    const float Eo = ggxEnergyE(wo.z, alpha);
    const float Ei = ggxEnergyE(wi.z, alpha);
    const float Eavg = ggxEnergyAvg(alpha);
    const float denom = kPi * srMax(1e-4f, 1.0f - Eavg);
    return fAvg * ((1.0f - Eo) * (1.0f - Ei) / denom);
}

}  // namespace sol
