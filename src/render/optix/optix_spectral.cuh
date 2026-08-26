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
    const bool allowIR = wo.z > 0.0f || mat.internalReflections > 0.5f;
    for (int i = 0; i < n; ++i) {
        if (path.pdf[i] <= 0.0f) continue;
        const float etaAbs = specDielectricIor(baseIor, mat.dispersionAbbe, path.lambda[i]);
        const DielectricGgxEval mf = evalDielectricGgx(woS, wiS, lw.ax, lw.ay, etaAbs, allowIR);
        out[i] = reflecting ? (mf.f * tw) : (tint[i] * mf.f * tw);
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
    const Vec3 woS = rotateForAnisotropy(woLocal, mat.specularRotation);
    const bool allowInternalReflect = mat.internalReflections > 0.5f || woLocal.z > 0.0f;

    if (lw.delta) {
        const Vec3 h(0.0f, 0.0f, 1.0f);
        const float dotOH = dot(woS, h);
        const float Fhero = fresnelDielectric(dotOH, heroEtaAbs);
        for (int i = 0; i < n; ++i) {
            if (path.pdf[i] <= 0.0f) {
                out.weight[i] = 0.0f;
                continue;
            }
            const float etaAbs = specDielectricIor(baseIor, mat.dispersionAbbe, path.lambda[i]);
            const float etaRel = woLocal.z > 0.0f ? etaAbs : 1.0f / etaAbs;
            const float F = fresnelDielectric(dotOH, etaAbs);
            if (rgb.transmitted) {
                float scale = tw / (etaRel * etaRel * srMax(1e-4f, lw.transmission));
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
    const Vec3 wiS = rotateForAnisotropy(rgb.wi, mat.specularRotation);
    const DielectricGgxEval mf =
        evalDielectricGgx(woS, wiS, lw.ax, lw.ay, heroEtaAbs, allowInternalReflect);
    float pdf = lw.transmission * mf.pdf;
    const float pCoatS = coatPickProb(mat, woLocal);
    if (pCoatS > 0.0f && woLocal.z > 0.0f) pdf *= (1.0f - pCoatS);
    if (pdf <= 0.0f) {
        out.valid = false;
        out.pdf = 0.0f;
        specZero(out.weight, n);
        return out;
    }
    out.pdf = pdf;
    const float invPdf = 1.0f / pdf;
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
