// Spectral BSDF lift for OptiX PT (linear RGB weights / dielectric 1/η²).
#pragma once

#include "render/optix/optix_bsdf.cuh"
#include "render/optix/optix_spectral_film.cuh"

namespace sol {

__device__ inline bool usesSpectralDielectricGpu(const Material& mat) {
    return saturatef(mat.transmission) > 1e-4f && saturatef(mat.metallic) < 0.5f;
}

__device__ inline void liftBsdfWeightGpu(Vec3 rgbWeight, const GpuPath& path, float* out) {
    specUpsampleLinear(rgbWeight, path.lambda, path.nLambda, out);
}

__device__ inline void evalDielectricF(const Material& mat, Vec3 wo, Vec3 wi, const GpuPath& path,
                                       float baseIor, float* out) {
    const int n = path.nLambda;
    specZero(out, n);
    if (fabsf(wo.z) < 1e-6f || fabsf(wi.z) < 1e-6f) return;
    Material matNd = mat;
    matNd.ior = baseIor;
    const optixpt::LobeWeights lw = optixpt::computeLobes(matNd, wo);
    if (lw.transmission <= 1e-5f || lw.delta) return;
    const float tw = saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
    const bool reflecting = wo.z * wi.z > 0.0f;
    float tint[kMaxSpectrumSamples];
    specUpsampleReflectance(gpuSpec(), lw.transmissionTint, path.lambda, n, tint);
    if (reflecting) {
        Vec3 h = wo + wi;
        if (lengthSquared(h) <= 0.0f) return;
        h = normalize(h);
        if (h.z < 0.0f) h = -h;
        const bool allowDielectricReflect = wo.z > 0.0f || mat.internalReflections > 0.5f;
        const float d = optixpt::ggxD(h, lw.alpha);
        const float g = optixpt::smithG2(wo, wi, lw.alpha);
        for (int i = 0; i < n; ++i) {
            if (path.pdf[i] <= 0.0f) continue;
            const float etaAbs = specDielectricIor(baseIor, mat.dispersionAbbe, path.lambda[i]);
            const float eta = wo.z > 0.0f ? etaAbs : 1.0f / etaAbs;
            if (!allowDielectricReflect) {
                const float cosHI = absDot(wo, h);
                const float sin2 = srMax(0.0f, 1.0f - cosHI * cosHI);
                if ((sin2 / (eta * eta)) < 1.0f) continue;
            }
            const float fr = optixpt::fresnelDielectric(dot(wo, h), eta);
            out[i] = (d * g * fr / (4.0f * fabsf(wo.z) * fabsf(wi.z))) * tw;
        }
        return;
    }
    for (int i = 0; i < n; ++i) {
        if (path.pdf[i] <= 0.0f) continue;
        const float etaAbs = specDielectricIor(baseIor, mat.dispersionAbbe, path.lambda[i]);
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
        const float fr = optixpt::fresnelDielectric(dotOH, eta);
        const float d = optixpt::ggxD(h, lw.alpha);
        const float g = optixpt::smithG2(wo, wi, lw.alpha);
        const float factor = fabsf(dotIH * dotOH / (wo.z * wi.z));
        const float ft = (1.0f - fr) * d * g * factor * (eta * eta) / (sqrtDenom * sqrtDenom);
        out[i] = tint[i] * ft * tw;
    }
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
    const optixpt::BsdfSample rgb =
        optixpt::bsdfSampleLocal(matHero, woLocal, uLobe, u1, u2, uChoice);
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
    const optixpt::LobeWeights lw = optixpt::computeLobes(matNd, woLocal);
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
    const float Fhero = optixpt::fresnelDielectric(dotOH, heroEta);
    const bool allowInternalReflect = mat.internalReflections > 0.5f || woLocal.z > 0.0f;

    if (lw.delta) {
        for (int i = 0; i < n; ++i) {
            if (path.pdf[i] <= 0.0f) {
                out.weight[i] = 0.0f;
                continue;
            }
            const float etaAbs = specDielectricIor(baseIor, mat.dispersionAbbe, path.lambda[i]);
            const float eta = woLocal.z > 0.0f ? etaAbs : 1.0f / etaAbs;
            const float F = optixpt::fresnelDielectric(dotOH, eta);
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

__device__ inline bool shouldTerminateSecondaryGpu(const optixpt::BsdfSample& bs,
                                                   const optixpt::LobeWeights& lw) {
    return !(bs.specular || optixpt::isNearSpecularLobe(lw));
}

__device__ inline bool dielectricEtaVariesGpu(const Material& mat) {
    return usesSpectralDielectricGpu(mat) && mat.dispersionAbbe > 1e-3f;
}

__device__ inline bool shouldTerminateSecondaryGpu(const optixpt::BsdfSample& bs,
                                                   const optixpt::LobeWeights& lw, const Material& mat) {
    if (dielectricEtaVariesGpu(mat)) return true;
    return shouldTerminateSecondaryGpu(bs, lw);
}

}  // namespace sol
