// Spectral BSDF lift for OptiX PT (pbrt RGBAlbedoSpectrum / dielectric 1/η²).
// Dielectric sample/eval use the shared CPU BSDF (shading_bsdf.h).
#pragma once

#include "render/optix/optix_bsdf.cuh"
#include "render/optix/optix_spectral_film.cuh"

namespace sol {

__device__ inline bool usesSpectralDielectricGpu(const Material& mat) {
    return saturatef(mat.transmission) > 1e-4f && saturatef(mat.metallic) < 0.5f;
}

__device__ inline void liftBsdfWeightGpu(Vec3 rgbWeight, const GpuPath& path, float* out) {
    specUpsampleReflectance(gpuSpec(), rgbWeight, path.lambda, path.nLambda, out);
}

// Per-λ dielectric BSDF f — same formulas as bsdfEvalSpectralDielectric (CPU).
__device__ inline void evalDielectricF(const Material& mat, Vec3 wo, Vec3 wi, const GpuPath& path,
                                       float baseIor, float* out) {
    const int n = path.nLambda;
    specZero(out, n);
    if (fabsf(wo.z) < 1e-6f || fabsf(wi.z) < 1e-6f) return;
    Material matNd = mat;
    matNd.ior = baseIor;
    const LobeWeights lw = computeLobes(matNd, wo);
    if (lw.transmission <= 1e-5f || lw.delta) return;
    const float tw = saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
    const bool reflecting = wo.z * wi.z > 0.0f;
    const Vec3 woS = rotateForAnisotropy(wo, mat.specularRotation);
    const Vec3 wiS = rotateForAnisotropy(wi, mat.specularRotation);
    float tint[kMaxSpectrumSamples];
    specUpsampleReflectance(gpuSpec(), lw.transmissionTint, path.lambda, n, tint);
    if (reflecting) {
        Vec3 h = woS + wiS;
        if (lengthSquared(h) <= 0.0f) return;
        h = normalize(h);
        if (h.z < 0.0f) h = -h;
        const bool allowDielectricReflect = wo.z > 0.0f || mat.internalReflections > 0.5f;
        const float d = ggxD(h, lw.ax, lw.ay);
        const float g = smithG2(woS, wiS, lw.ax, lw.ay);
        for (int i = 0; i < n; ++i) {
            if (path.pdf[i] <= 0.0f) continue;
            const float etaAbs = specDielectricIor(baseIor, mat.dispersionAbbe, path.lambda[i]);
            const float eta = wo.z > 0.0f ? etaAbs : 1.0f / etaAbs;
            if (!allowDielectricReflect) {
                const float cosHI = absDot(woS, h);
                const float sin2 = srMax(0.0f, 1.0f - cosHI * cosHI);
                if ((sin2 / (eta * eta)) < 1.0f) continue;
            }
            const float fr = fresnelDielectric(dot(woS, h), eta);
            out[i] = (d * g * fr / (4.0f * fabsf(wo.z) * fabsf(wi.z))) * tw;
        }
        return;
    }
    for (int i = 0; i < n; ++i) {
        if (path.pdf[i] <= 0.0f) continue;
        const float etaAbs = specDielectricIor(baseIor, mat.dispersionAbbe, path.lambda[i]);
        const float eta = wo.z > 0.0f ? etaAbs : 1.0f / etaAbs;
        Vec3 h = -(woS + wiS * eta);
        if (lengthSquared(h) <= 0.0f) continue;
        h = normalize(h);
        if (h.z < 0.0f) h = -h;
        const float dotOH = dot(woS, h);
        const float dotIH = dot(wiS, h);
        if (dotOH * wo.z <= 0.0f) continue;
        const float sqrtDenom = dotOH + eta * dotIH;
        if (fabsf(sqrtDenom) <= 1e-6f) continue;
        const float fr = fresnelDielectric(dotOH, eta);
        const float d = ggxD(h, lw.ax, lw.ay);
        const float g = smithG2(woS, wiS, lw.ax, lw.ay);
        const float factor = fabsf(dotIH * dotOH / (wo.z * wi.z));
        const float ft = (1.0f - fr) * d * g * factor * (eta * eta) / (sqrtDenom * sqrtDenom);
        out[i] = tint[i] * ft * tw;
    }
}

// Spectral BSDF f for NEE / camera connections — matches bsdfEvalSpectral.
__device__ inline void evalBsdfSpectralGpu(const Material& mat, Vec3 wo, Vec3 wi, const GpuPath& path,
                                           float baseIor, float* out) {
    const int n = path.nLambda;
    specZero(out, n);
    if (!usesSpectralDielectricGpu(mat)) {
        const BsdfEval e = bsdfEvalLocal(mat, wo, wi);
        liftBsdfWeightGpu(e.f, path, out);
        return;
    }
    evalDielectricF(mat, wo, wi, path, baseIor, out);
    Material matNd = mat;
    matNd.ior = baseIor;
    const LobeWeights lw = computeLobes(matNd, wo);
    if (lw.transmission > 0.5f && lw.diffuse < 1e-3f) return;
    Material matOpaque = matNd;
    matOpaque.transmission = 0.0f;
    const BsdfEval opaque = bsdfEvalLocal(matOpaque, wo, wi);
    float lift[kMaxSpectrumSamples];
    liftBsdfWeightGpu(opaque.f, path, lift);
    for (int i = 0; i < n; ++i) out[i] += lift[i];
}

__device__ inline void evalSurfaceNeeSpectral(const LightData& light, Vec3 rgbLe, const Material& mat,
                                             Vec3 wo, Vec3 wi, float scale, const GpuPath& path,
                                             float baseIor, float* out) {
    float Le[kMaxSpectrumSamples];
    float fS[kMaxSpectrumSamples];
    specAuthoredRadiance(light, rgbLe, path, Le);
    evalBsdfSpectralGpu(mat, wo, wi, path, baseIor, fS);
    for (int i = 0; i < path.nLambda; ++i) out[i] = Le[i] * fS[i] * scale;
}

struct GpuBsdfSampleS {
    float weight[kMaxSpectrumSamples]{};
    Vec3 wi{0.0f};
    float pdf = 0.0f;
    bool specular = false;
    bool transmitted = false;
    bool valid = false;
};

__device__ inline GpuBsdfSampleS bsdfSampleSpectralGpu(const Material& mat, Vec3 woLocal, float uLobe,
                                                       float u1, float u2, float uChoice,
                                                       const GpuPath& path, float baseIor) {
    GpuBsdfSampleS out;
    const int n = path.nLambda;
    specZero(out.weight, n);
    Material matHero = mat;
    matHero.ior = specDielectricIor(baseIor, mat.dispersionAbbe, path.lambda[0]);
    const BsdfSample rgb = bsdfSampleLocal(matHero, woLocal, uLobe, u1, u2, uChoice);
    if (rgb.pdf <= 0.0f || isBlack(rgb.weight)) return out;
    out.wi = rgb.wi;
    out.pdf = rgb.pdf;
    out.specular = rgb.specular;
    out.transmitted = rgb.transmitted;
    out.valid = true;

    if (!usesSpectralDielectricGpu(mat)) {
        liftBsdfWeightGpu(rgb.weight, path, out.weight);
        return out;
    }

    Material matNd = mat;
    matNd.ior = baseIor;
    const LobeWeights lw = computeLobes(matNd, woLocal);
    const bool fromDielectricLobe = uLobe >= lw.diffuse + lw.specular - 1e-6f || rgb.transmitted;
    if (!fromDielectricLobe) {
        liftBsdfWeightGpu(rgb.weight, path, out.weight);
        return out;
    }

    const float tw = saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
    float tint[kMaxSpectrumSamples];
    specUpsampleReflectance(gpuSpec(), lw.transmissionTint, path.lambda, n, tint);
    const float heroEtaAbs = specDielectricIor(baseIor, mat.dispersionAbbe, path.lambda[0]);
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
        for (int i = 0; i < n; ++i) {
            if (path.pdf[i] <= 0.0f) {
                out.weight[i] = 0.0f;
                continue;
            }
            const float etaAbs = specDielectricIor(baseIor, mat.dispersionAbbe, path.lambda[i]);
            const float eta = woLocal.z > 0.0f ? etaAbs : 1.0f / etaAbs;
            const float F = fresnelDielectric(dotOH, eta);
            if (rgb.transmitted) {
                float scale = tw / (eta * eta * srMax(1e-4f, lw.transmission));
                if (!allowInternalReflect && woLocal.z < 0.0f) {
                    scale *= (1.0f - F);
                } else {
                    scale *= (1.0f - F) / srMax(1e-4f, 1.0f - Fhero);
                }
                out.weight[i] = tint[i] * scale;
            } else {
                float scale = tw / srMax(1e-4f, lw.transmission);
                if (Fhero > 1e-4f) scale *= F / Fhero;
                out.weight[i] = scale;
            }
        }
        return out;
    }

    evalDielectricF(mat, woLocal, rgb.wi, path, baseIor, out.weight);
    const float invPdf = 1.0f / srMax(rgb.pdf, 1e-8f);
    for (int i = 0; i < n; ++i) out.weight[i] *= fabsf(rgb.wi.z) * invPdf;
    return out;
}

__device__ inline bool shouldTerminateSecondaryGpu(const BsdfSample& bs, const LobeWeights& lw) {
    return !(bs.specular || isNearSpecularLobe(lw));
}

__device__ inline bool dielectricEtaVariesGpu(const Material& mat) {
    return usesSpectralDielectricGpu(mat) && mat.dispersionAbbe > 1e-3f;
}

__device__ inline bool shouldTerminateSecondaryGpu(const BsdfSample& bs, const LobeWeights& lw,
                                                   const Material& mat) {
    if (dielectricEtaVariesGpu(mat)) return true;
    return shouldTerminateSecondaryGpu(bs, lw);
}

}  // namespace sol
