// Shared physically based BSDF used by Embree and OptiX.
//
// The model is a trimmed down "principled" surface: Lambert / Oren–Nayar diffuse,
// anisotropic GGX microfacet reflection (Smith + Heitz VNDF), Charlie sheen
// (Estevez–Kulla), and a rough dielectric transmission lobe (Walter et al. 2007 /
// pbrt-v4 DielectricBxDF). Perfectly smooth lobes degrade to delta distributions.
//
// Dielectric convention (pbrt): the microfacet normal wm is FaceForward'ed to +z.
// Fresnel always takes the stored outside relative IOR; a negative wo·wm flips
// the interface (glass→air, including TIR). Sample, RGB eval, and spectral eval
// share evalDielectricGgx so they cannot drift.
//
// This header must NOT include render/procedural.h. OptiX shade kernels that
// pull the MaterialX graph interpreter compile as megakernels; maps/procedurals
// stay in shading.h (CPU) and optix_bsdf.cuh (GPU bilinear maps).
#pragma once

#include "core/math.h"
#include "render/ggx_energy.h"
#include "scene/types.h"

namespace sol {

constexpr float kMinAlpha = 1.0e-3f;
constexpr float kDeltaAlpha = 2.0e-3f;
// Vertices at or below this GGX alpha (roughness ≈ 0.22) still focus light tightly
// enough that a caustic behaves like a specular chain: BSDF-sampling the tiny light
// from the eye side is hopeless, so those chains are routed to light tracing.
constexpr float kCausticAlpha = 5.0e-2f;

struct BsdfSample {
    Vec3 weight{0.0f, 0.0f, 0.0f};  // f * cos / pdf
    Vec3 wi{0.0f, 0.0f, 1.0f};      // world space
    float pdf = 0.0f;
    bool specular = false;
    bool transmitted = false;
};

struct BsdfEval {
    Vec3 f{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
};

SR_INL SR_HD float roughnessToAlpha(float roughness) {
    const float r = clampf(roughness, 0.0f, 1.0f);
    return srMax(kMinAlpha, r * r);
}

SR_INL SR_HD Vec3 fresnelSchlick(Vec3 f0, float cosTheta) {
    const float m = clampf(1.0f - cosTheta, 0.0f, 1.0f);
    const float m2 = m * m;
    const float m5 = m2 * m2 * m;
    return f0 + (Vec3(1.0f) - f0) * m5;
}

// ---------------------------------------------------------------------------
// Thin-film iridescence: Airy interference in a single film (air / film / base)
// evaluated at the three RGB wavelengths. Base reflectivity comes from the
// material's f0 (|r23| ≈ sqrt(f0) — exact for dielectrics, a good approximation
// for conductors), so metals keep their tint under the film.
// ---------------------------------------------------------------------------
SR_INL SR_HD float airyReflectanceScalar(float cosTheta1, float filmIor, float thicknessNm,
                                         float lambdaNm, float r23mag) {
    const float n1 = 1.0f;
    const float n2 = srMax(1.01f, filmIor);
    const float c1 = clampf(cosTheta1, 0.0f, 1.0f);
    const float s1 = sqrtf(srMax(0.0f, 1.0f - c1 * c1));
    const float s2 = n1 * s1 / n2;
    if (s2 >= 1.0f) return 1.0f;  // TIR at the film entry
    const float c2 = sqrtf(srMax(0.0f, 1.0f - s2 * s2));

    // Air→film amplitude coefficients, s/p averaged.
    const float r12s = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2);
    const float r12p = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2);

    // Interference phase for one round trip inside the film.
    const float phi = (4.0f * kPi * n2 * thicknessNm * c2) / srMax(1.0f, lambdaNm);
    const float cphi = cosf(phi);

    float sum = 0.0f;
    for (int pol = 0; pol < 2; ++pol) {
        const float r12 = pol == 0 ? r12s : r12p;
        const float r23 = r12 < 0.0f ? -r23mag : r23mag;  // keep the phase relation
        // |r12 + r23 e^{iφ}|² / |1 + r12 r23 e^{iφ}|²
        const float num = r12 * r12 + r23 * r23 + 2.0f * r12 * r23 * cphi;
        const float den = 1.0f + r12 * r12 * r23 * r23 + 2.0f * r12 * r23 * cphi;
        sum += clampf(num / srMax(1e-6f, den), 0.0f, 1.0f);
    }
    return 0.5f * sum;
}

SR_INL SR_HD Vec3 thinFilmFresnel(Vec3 f0, float cosTheta, float filmIor, float thicknessNm) {
    const Vec3 r23(sqrtf(clampf(f0.x, 0.0f, 1.0f)), sqrtf(clampf(f0.y, 0.0f, 1.0f)),
                   sqrtf(clampf(f0.z, 0.0f, 1.0f)));
    return Vec3(airyReflectanceScalar(cosTheta, filmIor, thicknessNm, 630.0f, r23.x),
                airyReflectanceScalar(cosTheta, filmIor, thicknessNm, 532.0f, r23.y),
                airyReflectanceScalar(cosTheta, filmIor, thicknessNm, 465.0f, r23.z));
}

// Fresnel for the opaque specular lobe: Schlick, or Airy when a film is present.
SR_INL SR_HD Vec3 specularFresnel(const Material& mat, Vec3 f0, float cosTheta) {
    if (mat.thinFilmThickness > 0.5f)
        return thinFilmFresnel(f0, cosTheta, mat.thinFilmIor, mat.thinFilmThickness);
    return fresnelSchlick(f0, cosTheta);
}

// pbrt FrDielectric: eta is the stored outside relative IOR (η_t/η_i on the
// material, typically > 1). A negative cosThetaI flips the interface (glass→air).
SR_INL SR_HD float fresnelDielectric(float cosThetaI, float eta) {
    cosThetaI = clampf(cosThetaI, -1.0f, 1.0f);
    if (cosThetaI < 0.0f) {
        eta = 1.0f / eta;
        cosThetaI = -cosThetaI;
    }
    const float sin2ThetaI = srMax(0.0f, 1.0f - cosThetaI * cosThetaI);
    const float sin2ThetaT = sin2ThetaI / (eta * eta);
    if (sin2ThetaT >= 1.0f) return 1.0f;  // total internal reflection
    const float cosThetaT = sqrtf(srMax(0.0f, 1.0f - sin2ThetaT));
    const float rParl = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    const float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return 0.5f * (rParl * rParl + rPerp * rPerp);
}

// Iray Photoreal (Keller et al. 2017, arXiv:1705.01263): NEE visibility may
// partially evaluate transmissive materials instead of treating delta glass as a
// hard occluder. Light-tracing camera connections and primary (depth-0) NEE stay
// opaque so SDS lands only on directly visible receivers — NVIDIA: "The caustic
// sampler only improves on directly visible caustics. Caustics seen through
// mirrors or windows are currently not improved."
//
// Returns occlusion in [0,1] (1 = fully blocked). η² is a radiance measure
// conversion, not opacity — it is not applied here.
//
// eyeBounceNee: 1 = camera/BDPT NEE after a bounce (Fresnel continue along the
// straight shadow ray, biased vs Snell). 0 = primary NEE or LT splat / BDPT t=1.
SR_INL SR_HD float shadowBlockFraction(const Material& matShadow, const Material& matCaustic,
                                       int causticsOn, int eyeBounceNee, float nDotWo) {
    if (matShadow.transmission <= 1e-3f) return 1.0f;
    if (causticsOn == 0 || matCaustic.contributeCaustics == 0)
        return saturatef(matShadow.shadowOpacity);
    if (eyeBounceNee == 0) return 1.0f;

    const float F = fresnelDielectric(nDotWo, srMax(1.0001f, matShadow.ior));
    const float tint = srMax(0.0f, luminance(matShadow.transmissionColor));
    const float T = (1.0f - F) * saturatef(matShadow.transmission) *
                    (1.0f - saturatef(matShadow.metallic)) * tint;
    return saturatef(1.0f - T);
}

// Rotate wo/wi in the tangent plane for specular_rotation (turns in [0,1] = 0–360°).
SR_INL SR_HD Vec3 rotateForAnisotropy(Vec3 w, float rotationTurns) {
    if (fabsf(rotationTurns) < 1e-8f) return w;
    const float a = rotationTurns * kTwoPi;
    const float c = cosf(a);
    const float s = sinf(a);
    return Vec3(c * w.x - s * w.y, s * w.x + c * w.y, w.z);
}

SR_INL SR_HD Vec3 unrotateForAnisotropy(Vec3 w, float rotationTurns) {
    return rotateForAnisotropy(w, -rotationTurns);
}

// Disney/pbrt aspect: αx = α/√(1-0.9A), αy = α√(1-0.9A). Geometric mean stays α.
SR_INL SR_HD void specularAlphas(const Material& mat, float& ax, float& ay) {
    const float alpha = roughnessToAlpha(mat.roughness);
    const float aniso = saturatef(mat.specularAnisotropy);
    if (aniso <= 1e-5f) {
        ax = ay = alpha;
        return;
    }
    const float aspect = sqrtf(srMax(1e-4f, 1.0f - 0.9f * aniso));
    ax = srMax(kMinAlpha, alpha / aspect);
    ay = srMax(kMinAlpha, alpha * aspect);
}

// GGX / Trowbridge–Reitz NDF (anisotropic; ax==ay matches the old isotropic formula).
SR_INL SR_HD float ggxD(Vec3 h, float ax, float ay) {
    const float cos2 = h.z * h.z;
    if (cos2 <= 1e-16f) return 0.0f;
    const float ax2 = ax * ax;
    const float ay2 = ay * ay;
    const float a = (h.x * h.x) / ax2 + (h.y * h.y) / ay2 + cos2;
    if (a <= 0.0f) return 0.0f;
    return 1.0f / (kPi * ax * ay * a * a);
}

SR_INL SR_HD float ggxD(Vec3 h, float alpha) { return ggxD(h, alpha, alpha); }

SR_INL SR_HD float smithLambda(Vec3 w, float ax, float ay) {
    const float cos2 = w.z * w.z;
    if (cos2 >= 1.0f) return 0.0f;
    const float sin2 = srMax(0.0f, 1.0f - cos2);
    const float alpha2 = (ax * ax * w.x * w.x + ay * ay * w.y * w.y) / srMax(1e-12f, sin2);
    const float tan2 = sin2 / srMax(1e-8f, cos2);
    return 0.5f * (sqrtf(1.0f + alpha2 * tan2) - 1.0f);
}

SR_INL SR_HD float smithLambda(Vec3 w, float alpha) { return smithLambda(w, alpha, alpha); }

SR_INL SR_HD float smithG1(Vec3 w, float ax, float ay) { return 1.0f / (1.0f + smithLambda(w, ax, ay)); }

SR_INL SR_HD float smithG1(Vec3 w, float alpha) { return smithG1(w, alpha, alpha); }

SR_INL SR_HD float smithG2(Vec3 wo, Vec3 wi, float ax, float ay) {
    return 1.0f / (1.0f + smithLambda(wo, ax, ay) + smithLambda(wi, ax, ay));
}

SR_INL SR_HD float smithG2(Vec3 wo, Vec3 wi, float alpha) { return smithG2(wo, wi, alpha, alpha); }

// Heitz 2018: sampling the GGX distribution of visible normals (anisotropic stretch).
SR_INL SR_HD Vec3 sampleGgxVndf(Vec3 wo, float ax, float ay, float u1, float u2) {
    const Vec3 vh = normalize(Vec3(ax * wo.x, ay * wo.y, wo.z));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const Vec3 t1 = lensq > 0.0f ? Vec3(-vh.y, vh.x, 0.0f) * (1.0f / sqrtf(lensq)) : Vec3(1.0f, 0.0f, 0.0f);
    const Vec3 t2 = cross(vh, t1);
    const float r = sqrtf(u1);
    const float phi = kTwoPi * u2;
    const float p1 = r * cosf(phi);
    float p2 = r * sinf(phi);
    const float s = 0.5f * (1.0f + vh.z);
    p2 = (1.0f - s) * sqrtf(srMax(0.0f, 1.0f - p1 * p1)) + s * p2;
    const Vec3 nh = t1 * p1 + t2 * p2 + vh * sqrtf(srMax(0.0f, 1.0f - p1 * p1 - p2 * p2));
    return normalize(Vec3(ax * nh.x, ay * nh.y, srMax(1e-6f, nh.z)));
}

SR_INL SR_HD Vec3 sampleGgxVndf(Vec3 wo, float alpha, float u1, float u2) {
    return sampleGgxVndf(wo, alpha, alpha, u1, u2);
}

SR_INL SR_HD float ggxVndfPdf(Vec3 wo, Vec3 h, float ax, float ay) {
    const float cosWo = fabsf(wo.z);
    if (cosWo <= 0.0f) return 0.0f;
    return smithG1(wo, ax, ay) * ggxD(h, ax, ay) * absDot(wo, h) / cosWo;
}

SR_INL SR_HD float ggxVndfPdf(Vec3 wo, Vec3 h, float alpha) { return ggxVndfPdf(wo, h, alpha, alpha); }

// Charlie NDF (Estevez–Kulla 2017) + Neubelt visibility for sheen.
SR_INL SR_HD float charlieSheenD(Vec3 h, float roughness) {
    const float r = srMax(kMinAlpha, saturatef(roughness));
    const float invR = 1.0f / r;
    const float sin2h = srMax(1.0f - h.z * h.z, 0.0078125f);
    return (2.0f + invR) * powf(sin2h, 0.5f * invR) / kTwoPi;
}

SR_INL SR_HD float neubeltVisibility(float noV, float noL) {
    noV = srMax(1e-6f, noV);
    noL = srMax(1e-6f, noL);
    return 1.0f / (4.0f * (noL + noV - noL * noV));
}

SR_INL SR_HD float charlieSheenPdf(Vec3 wo, Vec3 h, float roughness) {
    return charlieSheenD(h, roughness) * srMax(0.0f, h.z) / (4.0f * srMax(1e-6f, absDot(wo, h)));
}

SR_INL SR_HD Vec3 sampleCharlieH(float roughness, float u1, float u2) {
    const float r = srMax(kMinAlpha, saturatef(roughness));
    const float sinTheta = powf(srMax(1e-6f, u1), r / (2.0f * r + 1.0f));
    const float cosTheta = sqrtf(srMax(0.0f, 1.0f - sinTheta * sinTheta));
    const float phi = kTwoPi * u2;
    return Vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}

// Oren–Nayar qualitative. roughness=0 is Lambert (albedo/π).
SR_INL SR_HD Vec3 evalOrenNayar(Vec3 albedo, float roughness, Vec3 wo, Vec3 wi) {
    if (wo.z <= 0.0f || wi.z <= 0.0f) return Vec3(0.0f);
    const float sigma = saturatef(roughness);
    if (sigma <= 1e-5f) return albedo * kInvPi;
    const float sigma2 = sigma * sigma;
    const float A = 1.0f - 0.5f * sigma2 / (sigma2 + 0.33f);
    const float B = 0.45f * sigma2 / (sigma2 + 0.09f);
    const float sinI = sqrtf(srMax(0.0f, 1.0f - wi.z * wi.z));
    const float sinO = sqrtf(srMax(0.0f, 1.0f - wo.z * wo.z));
    float cosPhi = 0.0f;
    if (sinI > 1e-4f && sinO > 1e-4f)
        cosPhi = clampf((wi.x * wo.x + wi.y * wo.y) / (sinI * sinO), -1.0f, 1.0f);
    const float tanI = sinI / srMax(1e-6f, wi.z);
    const float tanO = sinO / srMax(1e-6f, wo.z);
    const float sinAlpha = sinI > sinO ? sinI : sinO;
    const float tanBeta = sinI > sinO ? tanO : tanI;
    return albedo * (kInvPi * (A + B * srMax(0.0f, cosPhi) * sinAlpha * tanBeta));
}

// Lobe weights, derived once and reused by eval and sample.
struct LobeWeights {
    float diffuse = 0.0f;
    float specular = 0.0f;
    float transmission = 0.0f;
    float sheen = 0.0f;
    Vec3 f0{0.04f, 0.04f, 0.04f};
    Vec3 diffuseAlbedo{0.0f, 0.0f, 0.0f};
    Vec3 transmissionTint{1.0f, 1.0f, 1.0f};
    Vec3 sheenColor{0.0f, 0.0f, 0.0f};
    float alpha = 0.1f;
    float ax = 0.1f;
    float ay = 0.1f;
    float eta = 1.5f;
    bool delta = false;
};

SR_INL SR_HD LobeWeights computeLobes(const Material& mat, Vec3 woLocal) {
    LobeWeights lw;
    const float metallic = saturatef(mat.metallic);
    const float transmission = saturatef(mat.transmission);
    const float specularControl = saturatef(mat.specular);
    lw.alpha = roughnessToAlpha(mat.roughness);
    specularAlphas(mat, lw.ax, lw.ay);
    lw.delta = lw.alpha <= kDeltaAlpha;
    lw.eta = srMax(1.01f, mat.ior);
    const Vec3 base = vmax(Vec3(0.0f), mat.baseColor);
    const Vec3 specularColor = vmax(Vec3(0.0f), mat.specularColor);
    const Vec3 transmissionColor = vmax(Vec3(0.0f), mat.transmissionColor);
    // Standard Surface: diffuse = base * base_color (SSS is mixed separately in the integrator).
    const float baseWeight = srMax(0.0f, mat.baseWeight);
    // Specular = 0 must fully kill dielectric reflections (artist expectation).
    // Arnold: dielectric F0 = 0.08 * specular * specular_color; metals use base_color.
    const float dielectricF0 = 0.08f * specularControl;
    lw.f0 = lerp(Vec3(dielectricF0) * specularColor, base, metallic);
    lw.diffuseAlbedo = base * (baseWeight * (1.0f - metallic) * (1.0f - transmission));
    // Arnold transmission_color tints refraction — not base_color.
    lw.transmissionTint = transmissionColor;
    lw.sheenColor = vmax(Vec3(0.0f), mat.sheenColor) * saturatef(mat.sheen);

    // The transmission lobe already contains its own Fresnel reflection, so the
    // opaque specular lobe is faded out as transmission increases.
    const float opaqueSpec = 1.0f - transmission * (1.0f - metallic);
    // Lobe lottery = energy estimate (pbrt): diffuse × albedo, spec × E(μ,α) F,
    // trans × (1−F). μ = |wo·n| so the discrete PDF tracks directional albedo.
    const float mu = srMax(1e-4f, fabsf(woLocal.z));
    const float Eo = ggxEnergyE(mu, lw.alpha);
    const float Fnorm = fresnelDielectric(mu, lw.eta);
    const float diffuseWeight = luminance(lw.diffuseAlbedo);
    float specWeight = 0.0f;
    if (metallic > 1e-4f) {
        specWeight = opaqueSpec * luminance(lw.f0) * Eo;
    } else if (specularControl > 1e-4f) {
        specWeight = opaqueSpec * Fnorm * Eo;
    }
    const float transWeight =
        (1.0f - metallic) * transmission * (1.0f - Fnorm) * average(lw.transmissionTint);
    const float sheenWeight = luminance(lw.sheenColor);
    const float total = diffuseWeight + specWeight + transWeight + sheenWeight;
    if (total <= 0.0f) {
        // Pure black / invalid material: fall back to a tiny diffuse lobe.
        lw.diffuse = 1.0f;
    } else {
        lw.diffuse = diffuseWeight / total;
        lw.specular = specWeight / total;
        lw.transmission = transWeight / total;
        lw.sheen = sheenWeight / total;
    }
    return lw;
}

SR_INL SR_HD LobeWeights computeLobes(const Material& mat) {
    return computeLobes(mat, Vec3(0.0f, 0.0f, 1.0f));
}

// Purely specular / transmissive vertex whose lobe is still tight enough that the
// caustics it casts have to be delivered by light tracing rather than by the eye
// path stumbling onto the light. True delta lobes are a subset of this.
SR_INL SR_HD bool isNearSpecularLobe(const LobeWeights& lw) {
    return lw.alpha <= kCausticAlpha && lw.diffuse < 1e-3f && lw.sheen < 1e-3f;
}

// Casters owned by the photon / VCM map: delta + near-spec mirrors/glass, and also
// rough refractive glass (any α). MNEE cannot solve rough refraction; when the
// Photon engine is active these lobes must continue the light→…→diffuse chain and
// eye-path BSDF hits of the same family must be suppressed (no double-count).
SR_INL SR_HD bool isPhotonCausticCasterLobe(const LobeWeights& lw) {
    if (lw.diffuse >= 1e-3f || lw.sheen >= 1e-3f) return false;
    if (lw.delta || isNearSpecularLobe(lw)) return true;
    return lw.transmission > 0.25f;
}

// Delta transmissive caster (CPU MNEE / GPU eye-path MNEE). Rough glass is not
// a Newton manifold — light tracing continues through it instead.
SR_INL SR_HD bool isDeltaCausticCaster(const Material& m) {
    if (m.contributeCaustics == 0) return false;
    const LobeWeights lw = computeLobes(m);
    return lw.delta && lw.transmission > 0.25f && lw.diffuse < 1e-3f;
}

// Light-trace vertex that may connect to the camera. Casters (delta, near-spec,
// or transmissive glass including roughness 0.1) are not connectable: do not
// splat from them and do not kill the path — continue the SDS chain.
SR_INL SR_HD bool lightTraceConnectable(const Material& mat, Vec3 woLocal) {
    const LobeWeights lw = computeLobes(mat, woLocal);
    if (lw.delta && lw.diffuse < 1e-4f) return false;
    if (isNearSpecularLobe(lw)) return false;
    if (lw.transmission > 0.25f && lw.diffuse < 1e-3f) return false;
    return true;
}

// Veach adjoint BSDF / pbrt: evaluate in the shading frame and spawn with ng.
// Do not kill transport when ns and ng disagree about reflection vs transmission.
SR_INL SR_HD bool shadingNormalConsistent(Vec3 ng, Vec3 ns, Vec3 wo, Vec3 wi) {
    (void)ng;
    (void)ns;
    (void)wo;
    (void)wi;
    return true;
}

// Separate dielectric coat (pbrt overlay): Beer–Lambert optical depth + GGX interface.
SR_INL SR_HD Vec3 coatBeer(const Material& mat, float cosTheta) {
    const float tau = srMax(0.0f, mat.coatThickness);
    if (tau <= 1e-8f) return Vec3(1.0f);
    const Vec3 sigmaA = (Vec3(1.0f) - vmax(Vec3(0.0f), mat.coatColor)) * tau;
    const float mu = srMax(1e-3f, fabsf(cosTheta));
    return Vec3(expf(-sigmaA.x / mu), expf(-sigmaA.y / mu), expf(-sigmaA.z / mu));
}

SR_INL SR_HD float coatPickProb(const Material& mat, Vec3 wo) {
    if (mat.coat <= 1e-4f || wo.z <= 0.0f) return 0.0f;
    const float alpha = roughnessToAlpha(mat.coatRoughness);
    const float Eo = ggxEnergyE(wo.z, alpha);
    const float F = fresnelDielectric(wo.z, srMax(1.01f, mat.coatIor));
    return clampf(saturatef(mat.coat) * Eo * F, 0.0f, 0.95f);
}

SR_INL SR_HD BsdfEval evalDielectricCoat(Vec3 wo, Vec3 wi, float alpha, float eta) {
    BsdfEval o;
    if (wo.z <= 0.0f || wi.z <= 0.0f) return o;
    if (alpha <= kDeltaAlpha) return o;
    Vec3 h = wo + wi;
    if (lengthSquared(h) <= 0.0f) return o;
    h = normalize(h);
    if (h.z < 0.0f) h = -h;
    const float D = ggxD(h, alpha);
    const float G = smithG2(wo, wi, alpha);
    const float F = fresnelDielectric(dot(wo, h), eta);
    o.f = Vec3(D * G * F / (4.0f * wo.z * wi.z));
    o.pdf = ggxVndfPdf(wo, h, alpha) / (4.0f * srMax(1e-6f, absDot(wo, h)));
    return o;
}

// pbrt Refract(): etaAbs is the stored outside IOR. Negative wo·n flips n and η.
SR_INL SR_HD bool refractDielectric(Vec3 wo, Vec3 n, float etaAbs, float& etap, Vec3& wt) {
    float cosThetaI = dot(n, wo);
    float eta = etaAbs;
    Vec3 nn = n;
    if (cosThetaI < 0.0f) {
        eta = 1.0f / eta;
        cosThetaI = -cosThetaI;
        nn = -n;
    }
    etap = eta;
    const float sin2ThetaI = srMax(0.0f, 1.0f - cosThetaI * cosThetaI);
    const float sin2ThetaT = sin2ThetaI / (eta * eta);
    if (sin2ThetaT >= 1.0f) return false;
    const float cosThetaT = sqrtf(srMax(0.0f, 1.0f - sin2ThetaT));
    wt = normalize(wo * (-1.0f / eta) + nn * (cosThetaI / eta - cosThetaT));
    return true;
}

// pbrt DielectricBxDF GGX f + pdf (one lobe, no mixture weight).
// wm is FaceForward'ed to +z. Fresnel always takes etaAbs; signed wo·wm flips
// air→glass vs glass→air (and TIR). f is un-tinted; pdf is P(wi | dielectric).
struct DielectricGgxEval {
    float f = 0.0f;
    float pdf = 0.0f;
};

SR_INL SR_HD DielectricGgxEval evalDielectricGgx(Vec3 wo, Vec3 wi, float ax, float ay, float etaAbs,
                                                bool allowInternalReflections) {
    DielectricGgxEval out;
    if (fabsf(wo.z) < 1e-6f || fabsf(wi.z) < 1e-6f) return out;
    const bool reflecting = wo.z * wi.z > 0.0f;
    const float etap = reflecting ? 1.0f : (wo.z > 0.0f ? etaAbs : 1.0f / etaAbs);
    Vec3 h = reflecting ? (wo + wi) : -(wo + wi * etap);
    if (lengthSquared(h) <= 0.0f) return out;
    h = normalize(h);
    if (h.z < 0.0f) h = -h;
    const float dotOH = dot(wo, h);
    const float dotIH = dot(wi, h);
    // pbrt: discard back-facing microfacets (wm must face both wo and wi).
    if (dotOH * wo.z < 0.0f || dotIH * wi.z < 0.0f) return out;

    const float fr = fresnelDielectric(dotOH, etaAbs);
    const float d = ggxD(h, ax, ay);
    const float g = smithG2(wo, wi, ax, ay);
    if (reflecting) {
        if (!allowInternalReflections && wo.z < 0.0f && fr < 1.0f - 1e-5f) return out;
        out.f = d * g * fr / (4.0f * fabsf(wo.z) * fabsf(wi.z));
        out.pdf = fr * ggxVndfPdf(wo, h, ax, ay) / (4.0f * srMax(1e-6f, absDot(wo, h)));
        return out;
    }
    const float sqrtDenom = dotOH + etap * dotIH;
    if (fabsf(sqrtDenom) <= 1e-6f) return out;
    const float factor = fabsf(dotIH * dotOH / (wo.z * wi.z));
    out.f = (1.0f - fr) * d * g * factor * (etap * etap) / (sqrtDenom * sqrtDenom);
    const float dwhDwi = fabsf(etap * etap * dotIH) / (sqrtDenom * sqrtDenom);
    const float reflectProb = (!allowInternalReflections && wo.z < 0.0f) ? 0.0f : fr;
    out.pdf = (1.0f - reflectProb) * ggxVndfPdf(wo, h, ax, ay) * dwhDwi;
    return out;
}

// Evaluate the BSDF for a pair of directions expressed in the local shading
// frame (z = shading normal). Delta lobes return zero.
SR_INL SR_HD BsdfEval bsdfEvalLocal(const Material& mat, Vec3 wo, Vec3 wi) {
    BsdfEval out;
    const LobeWeights lw = computeLobes(mat, wo);
    const bool reflecting = wo.z * wi.z > 0.0f;
    if (fabsf(wo.z) < 1e-6f || fabsf(wi.z) < 1e-6f) return out;
    const float tw = saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
    const float opaqueSpec = 1.0f - tw;
    const Vec3 woS = rotateForAnisotropy(wo, mat.specularRotation);
    const Vec3 wiS = rotateForAnisotropy(wi, mat.specularRotation);

    if (reflecting) {
        if (wo.z > 0.0f && wi.z > 0.0f && !isBlack(lw.diffuseAlbedo)) {
            out.f += evalOrenNayar(lw.diffuseAlbedo, mat.diffuseRoughness, wo, wi);
            out.pdf += lw.diffuse * (wi.z * kInvPi);
        }
        if (wo.z > 0.0f && wi.z > 0.0f && lw.sheen > 1e-6f && !isBlack(lw.sheenColor)) {
            Vec3 hSheen = wo + wi;
            if (lengthSquared(hSheen) > 0.0f) {
                hSheen = normalize(hSheen);
                if (hSheen.z < 0.0f) hSheen = -hSheen;
                const float D = charlieSheenD(hSheen, mat.sheenRoughness);
                const float V = neubeltVisibility(wo.z, wi.z);
                out.f += lw.sheenColor * (D * V);
                out.pdf += lw.sheen * charlieSheenPdf(wo, hSheen, mat.sheenRoughness);
            }
        }
        if (!lw.delta) {
            const bool allowIR = wo.z > 0.0f || mat.internalReflections > 0.5f;
            if (tw > 0.0f) {
                const DielectricGgxEval mf =
                    evalDielectricGgx(woS, wiS, lw.ax, lw.ay, lw.eta, allowIR);
                out.f += Vec3(mf.f * tw);
                out.pdf += lw.transmission * mf.pdf;
            }
            Vec3 h = woS + wiS;
            if (lengthSquared(h) > 0.0f) {
                h = normalize(h);
                if (h.z < 0.0f) h = -h;
                const float d = ggxD(h, lw.ax, lw.ay);
                const float g = smithG2(woS, wiS, lw.ax, lw.ay);
                const float cosOH = absDot(woS, h);
                // Skip the opaque specular lobe entirely when the artist set Specular to 0
                // (and the surface is not metal) — including grazing-angle Fresnel.
                const bool hasOpaqueSpec =
                    lw.specular > 0.0f && (saturatef(mat.metallic) > 1e-4f || saturatef(mat.specular) > 1e-4f);
                if (opaqueSpec > 0.0f && hasOpaqueSpec && wo.z > 0.0f && wi.z > 0.0f) {
                    const Vec3 fr = specularFresnel(mat, lw.f0, cosOH);
                    out.f += fr * (d * g / (4.0f * wo.z * wi.z)) * opaqueSpec;
                    out.pdf += lw.specular * ggxVndfPdf(woS, h, lw.ax, lw.ay) / (4.0f * srMax(1e-6f, cosOH));
                }
            }
        }
        // Kulla–Conty / pbrt multiple-scattering compensation on the opaque specular lobe.
        if (opaqueSpec > 0.0f && wo.z > 0.0f && wi.z > 0.0f && !lw.delta) {
            const bool msOpaque =
                lw.specular > 0.0f && (saturatef(mat.metallic) > 1e-4f || saturatef(mat.specular) > 1e-4f);
            if (msOpaque) {
                const Vec3 fAvg = saturatef(mat.metallic) > 1e-4f ? fresnelAverageSchlick(lw.f0)
                                                                  : Vec3(fresnelAverageDielectric(lw.eta));
                out.f += ggxMsAlbedo(wo, wi, lw.alpha, fAvg) * opaqueSpec;
            }
        }
    } else if (!lw.delta && lw.transmission > 0.0f) {
        const bool allowIR = wo.z > 0.0f || mat.internalReflections > 0.5f;
        const DielectricGgxEval mf = evalDielectricGgx(woS, wiS, lw.ax, lw.ay, lw.eta, allowIR);
        out.f += lw.transmissionTint * (mf.f * tw);
        out.pdf += lw.transmission * mf.pdf;
    }
    const float pCoat = coatPickProb(mat, wo);
    if (pCoat > 0.0f && wo.z > 0.0f) {
        const Vec3 beer = coatBeer(mat, wo.z) * (wi.z > 0.0f ? coatBeer(mat, wi.z) : Vec3(1.0f));
        const float Fwo = fresnelDielectric(wo.z, srMax(1.01f, mat.coatIor));
        out.f = out.f * ((1.0f - saturatef(mat.coat) * Fwo) * beer);
        out.pdf *= (1.0f - pCoat);
        if (wi.z > 0.0f) {
            const BsdfEval coat = evalDielectricCoat(wo, wi, roughnessToAlpha(mat.coatRoughness),
                                                     srMax(1.01f, mat.coatIor));
            out.f += coat.f * saturatef(mat.coat);
            out.pdf += pCoat * coat.pdf;
        }
    }
    if (!isFinite(out.f) || !srIsFinite(out.pdf)) {
        out.f = Vec3(0.0f);
        out.pdf = 0.0f;
    }
    return out;
}

// uLobe picks the lobe, (u1,u2) samples the direction and uChoice decides
// between reflection and refraction inside the dielectric lobe.
SR_INL SR_HD BsdfSample bsdfSampleLocal(const Material& mat, Vec3 wo, float uLobe, float u1, float u2,
                                        float uChoice) {
    BsdfSample s;
    if (fabsf(wo.z) < 1e-6f) return s;

    const float pCoat = coatPickProb(mat, wo);
    if (pCoat > 1e-5f && uLobe < pCoat && wo.z > 0.0f) {
        const float alphaC = roughnessToAlpha(mat.coatRoughness);
        const float etaC = srMax(1.01f, mat.coatIor);
        if (alphaC <= kDeltaAlpha) {
            const Vec3 wi(-wo.x, -wo.y, wo.z);
            if (wi.z <= 0.0f) return s;
            s.wi = wi;
            s.specular = true;
            s.pdf = 1.0f;
            const float F = fresnelDielectric(wo.z, etaC);
            s.weight = Vec3(saturatef(mat.coat) * F / srMax(1e-4f, pCoat));
            return s;
        }
        const Vec3 h = sampleGgxVndf(wo, alphaC, u1, u2);
        const Vec3 wi = reflect(wo, h);
        if (wi.z <= 0.0f) return s;
        const BsdfEval e = bsdfEvalLocal(mat, wo, wi);
        if (e.pdf <= 0.0f) return s;
        s.wi = wi;
        s.pdf = e.pdf;
        s.weight = e.f * (wi.z / e.pdf);
        return s;
    }
    if (pCoat > 1e-5f) uLobe = (uLobe - pCoat) / srMax(1e-6f, 1.0f - pCoat);

    const LobeWeights lw = computeLobes(mat, wo);

    const float pDiffuse = lw.diffuse;
    const float pSheen = lw.sheen;
    const float pSpecular = lw.specular;

    if (uLobe < pDiffuse && wo.z > 0.0f) {
        const Vec3 wi = sampleCosineHemisphere(u1, u2);
        if (wi.z <= 0.0f) return s;
        s.wi = wi;
        const BsdfEval e = bsdfEvalLocal(mat, wo, wi);
        if (e.pdf <= 0.0f) return s;
        s.pdf = e.pdf;
        s.weight = e.f * (wi.z / e.pdf);
        return s;
    }

    if (pSheen > 1e-5f && uLobe < pDiffuse + pSheen && wo.z > 0.0f) {
        const Vec3 h = sampleCharlieH(mat.sheenRoughness, u1, u2);
        const Vec3 wi = reflect(wo, h);
        if (wi.z <= 0.0f) return s;
        s.wi = wi;
        const BsdfEval e = bsdfEvalLocal(mat, wo, wi);
        if (e.pdf <= 0.0f) return s;
        s.pdf = e.pdf;
        s.weight = e.f * (wi.z / e.pdf);
        return s;
    }

    if (pSpecular > 1e-5f && uLobe < pDiffuse + pSheen + pSpecular) {
        // Opaque specular reflection.
        if (lw.delta) {
            const Vec3 wi(-wo.x, -wo.y, wo.z);
            if (wi.z * wo.z <= 0.0f) return s;
            const float opaqueSpec = 1.0f - saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
            s.wi = wi;
            s.specular = true;
            s.pdf = 1.0f;  // delta lobe: f*cos/pdf collapses to Fresnel over the lobe probability
            s.weight = specularFresnel(mat, lw.f0, fabsf(wo.z)) * (opaqueSpec / srMax(1e-4f, pSpecular));
            return s;
        }
        const Vec3 woUp = wo.z > 0.0f ? wo : -wo;
        const Vec3 woS = rotateForAnisotropy(woUp, mat.specularRotation);
        const Vec3 hS = sampleGgxVndf(woS, lw.ax, lw.ay, u1, u2);
        const Vec3 wiS = reflect(woS, hS);
        const Vec3 wiLocal = unrotateForAnisotropy(wiS, mat.specularRotation);
        if (wiLocal.z <= 0.0f) return s;
        const Vec3 wi = wo.z > 0.0f ? wiLocal : Vec3(wiLocal.x, wiLocal.y, -wiLocal.z);
        const BsdfEval e = bsdfEvalLocal(mat, wo, wi);
        if (e.pdf <= 0.0f) return s;
        s.wi = wi;
        s.pdf = e.pdf;
        s.weight = e.f * (fabsf(wi.z) / e.pdf);
        return s;
    }

    // Dielectric transmission lobe.
    if (lw.transmission <= 1e-5f) return s;
    const Vec3 woS = rotateForAnisotropy(wo, mat.specularRotation);
    Vec3 hS;
    if (lw.delta) {
        hS = Vec3(0.0f, 0.0f, 1.0f);  // pbrt: wm = +z, signed wo·wm flips Fresnel
    } else {
        const Vec3 woUpS = woS.z > 0.0f ? woS : -woS;
        hS = sampleGgxVndf(woUpS, lw.ax, lw.ay, u1, u2);
    }
    const float fr = fresnelDielectric(dot(woS, hS), lw.eta);

    // Inside + Internal Reflections off: skip Fresnel reflection. TIR still
    // reflects — refraction is impossible (Arnold keeps critical-angle TIR).
    const bool allowInternalReflect = mat.internalReflections > 0.5f || wo.z > 0.0f;
    const bool tir = fr >= 1.0f - 1e-5f;
    const bool chooseReflect = tir || (allowInternalReflect && uChoice < fr);

    if (chooseReflect) {
        const Vec3 wi = unrotateForAnisotropy(reflect(woS, hS), mat.specularRotation);
        if (wi.z * wo.z <= 0.0f) return s;
        s.wi = wi;
        s.transmitted = false;
        if (lw.delta) {
            const float tw = saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
            s.specular = true;
            s.pdf = 1.0f;  // the Fresnel term cancels with the reflect/refract choice
            s.weight = Vec3(tw / srMax(1e-4f, lw.transmission));
            return s;
        }
        const BsdfEval e = bsdfEvalLocal(mat, wo, wi);
        if (e.pdf <= 0.0f) return s;
        s.pdf = e.pdf;
        s.weight = e.f * (fabsf(wi.z) / e.pdf);
        return s;
    }

    float etap = 1.0f;
    Vec3 wiS;
    if (!refractDielectric(woS, hS, lw.eta, etap, wiS)) return s;
    const Vec3 wiN = unrotateForAnisotropy(wiS, mat.specularRotation);
    if (wiN.z * wo.z >= 0.0f) return s;
    s.wi = wiN;
    s.transmitted = true;
    if (lw.delta) {
        const float tw = saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
        s.specular = true;
        s.pdf = 1.0f;
        // 1/eta^2 is the radiance compression when crossing the interface.
        // Normally Fresnel cancels with the reflect/refract choice. When Internal
        // Reflections is off from inside we always refract, so multiply by (1-fr).
        float scale = tw / (etap * etap * srMax(1e-4f, lw.transmission));
        if (!allowInternalReflect && wo.z < 0.0f) scale *= (1.0f - fr);
        s.weight = lw.transmissionTint * scale;
        return s;
    }
    const BsdfEval e = bsdfEvalLocal(mat, wo, wiN);
    if (e.pdf <= 0.0f) return s;
    s.pdf = e.pdf;
    s.weight = e.f * (fabsf(wiN.z) / e.pdf);
    return s;
}

}  // namespace sol
