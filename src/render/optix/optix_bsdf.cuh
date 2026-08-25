// GPU-only "basic surface" BSDF for the OptiX path tracer.
//
// This is intentionally NOT render/shading.h: that header pulls MaterialX
// procedurals (noise / triplanar / autobump) into whatever TU includes it,
// and cicc then spends hours on one megakernel. Cycles keeps the OptiX
// programs small and interprets shader graphs (SVM) or JITs OSL only when
// needed; Karma XPU compiles user shaders separately from render kernels;
// Iray JITs MDL to PTX callables per material.
//
// Here the GPU shade is USD-Preview-like: constants + 2D image maps +
// Lambert / GGX / dielectric transmission. No procedurals, no thin-film,
// no SSS, no polynomial optics.
#pragma once

#include "core/math.h"
#include "render/ggx_energy.h"
#include "scene/types.h"

namespace sol {
namespace optixpt {

constexpr float kMinAlpha = 1.0e-3f;
constexpr float kDeltaAlpha = 2.0e-3f;
constexpr float kCausticAlpha = 5.0e-2f;

struct BsdfSample {
    Vec3 weight{0.0f, 0.0f, 0.0f};
    Vec3 wi{0.0f, 0.0f, 1.0f};
    float pdf = 0.0f;
    bool specular = false;
    bool transmitted = false;
};

struct BsdfEval {
    Vec3 f{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
};

struct LobeWeights {
    float diffuse = 0.0f;
    float specular = 0.0f;
    float transmission = 0.0f;
    Vec3 f0{0.04f, 0.04f, 0.04f};
    Vec3 diffuseAlbedo{0.0f, 0.0f, 0.0f};
    Vec3 transmissionTint{1.0f, 1.0f, 1.0f};
    float alpha = 0.1f;
    float eta = 1.5f;
    bool delta = false;
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

SR_INL SR_HD float fresnelDielectric(float cosThetaI, float eta) {
    cosThetaI = clampf(cosThetaI, -1.0f, 1.0f);
    if (cosThetaI < 0.0f) {
        eta = 1.0f / eta;
        cosThetaI = -cosThetaI;
    }
    const float sin2ThetaI = srMax(0.0f, 1.0f - cosThetaI * cosThetaI);
    const float sin2ThetaT = sin2ThetaI / (eta * eta);
    if (sin2ThetaT >= 1.0f) return 1.0f;
    const float cosThetaT = sqrtf(srMax(0.0f, 1.0f - sin2ThetaT));
    const float rParl = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    const float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return 0.5f * (rParl * rParl + rPerp * rPerp);
}

SR_INL SR_HD float ggxD(Vec3 h, float alpha) {
    const float a2 = alpha * alpha;
    const float cos2 = h.z * h.z;
    const float t = cos2 * (a2 - 1.0f) + 1.0f;
    if (t <= 0.0f) return 0.0f;
    return a2 / (kPi * t * t);
}

SR_INL SR_HD float smithLambda(Vec3 w, float alpha) {
    const float cos2 = w.z * w.z;
    if (cos2 >= 1.0f) return 0.0f;
    const float tan2 = srMax(0.0f, 1.0f - cos2) / srMax(1e-8f, cos2);
    return 0.5f * (sqrtf(1.0f + alpha * alpha * tan2) - 1.0f);
}

SR_INL SR_HD float smithG1(Vec3 w, float alpha) { return 1.0f / (1.0f + smithLambda(w, alpha)); }

SR_INL SR_HD float smithG2(Vec3 wo, Vec3 wi, float alpha) {
    return 1.0f / (1.0f + smithLambda(wo, alpha) + smithLambda(wi, alpha));
}

SR_INL SR_HD Vec3 sampleGgxVndf(Vec3 wo, float alpha, float u1, float u2) {
    const Vec3 vh = normalize(Vec3(alpha * wo.x, alpha * wo.y, wo.z));
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
    return normalize(Vec3(alpha * nh.x, alpha * nh.y, srMax(1e-6f, nh.z)));
}

SR_INL SR_HD float ggxVndfPdf(Vec3 wo, Vec3 h, float alpha) {
    const float cosWo = fabsf(wo.z);
    if (cosWo <= 0.0f) return 0.0f;
    return smithG1(wo, alpha) * ggxD(h, alpha) * absDot(wo, h) / cosWo;
}

SR_INL SR_HD LobeWeights computeLobes(const Material& mat, Vec3 woLocal) {
    LobeWeights lw;
    const float metallic = saturatef(mat.metallic);
    const float transmission = saturatef(mat.transmission);
    const float specularControl = saturatef(mat.specular);
    lw.alpha = roughnessToAlpha(mat.roughness);
    lw.delta = lw.alpha <= kDeltaAlpha;
    lw.eta = srMax(1.01f, mat.ior);
    const Vec3 base = vmax(Vec3(0.0f), mat.baseColor);
    const Vec3 specularColor = vmax(Vec3(0.0f), mat.specularColor);
    const Vec3 transmissionColor = vmax(Vec3(0.0f), mat.transmissionColor);
    const float baseWeight = srMax(0.0f, mat.baseWeight);
    const float dielectricF0 = 0.08f * specularControl;
    lw.f0 = lerp(Vec3(dielectricF0) * specularColor, base, metallic);
    lw.diffuseAlbedo = base * (baseWeight * (1.0f - metallic) * (1.0f - transmission));
    lw.transmissionTint = transmissionColor;
    const float opaqueSpec = 1.0f - transmission * (1.0f - metallic);
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
    const float total = diffuseWeight + specWeight + transWeight;
    if (total <= 0.0f) {
        lw.diffuse = 1.0f;
    } else {
        lw.diffuse = diffuseWeight / total;
        lw.specular = specWeight / total;
        lw.transmission = transWeight / total;
    }
    return lw;
}

SR_INL SR_HD bool isNearSpecularLobe(const LobeWeights& lw) {
    return lw.alpha <= kCausticAlpha && lw.diffuse < 1e-3f;
}

SR_INL SR_HD LobeWeights computeLobes(const Material& mat) {
    return computeLobes(mat, Vec3(0.0f, 0.0f, 1.0f));
}

SR_INL SR_HD bool shadingNormalConsistent(Vec3 ng, Vec3 ns, Vec3 wo, Vec3 wi) {
    (void)ng;
    (void)ns;
    (void)wo;
    (void)wi;
    return true;
}

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

SR_INL SR_HD BsdfEval bsdfEvalLocal(const Material& mat, Vec3 wo, Vec3 wi) {
    BsdfEval out;
    const LobeWeights lw = computeLobes(mat, wo);
    const bool reflecting = wo.z * wi.z > 0.0f;
    if (fabsf(wo.z) < 1e-6f || fabsf(wi.z) < 1e-6f) return out;
    const float tw = saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
    const float opaqueSpec = 1.0f - tw;

    if (reflecting) {
        if (wo.z > 0.0f && wi.z > 0.0f && !isBlack(lw.diffuseAlbedo)) {
            out.f += lw.diffuseAlbedo * kInvPi;
            out.pdf += lw.diffuse * (wi.z * kInvPi);
        }
        if (!lw.delta) {
            Vec3 h = wo + wi;
            if (lengthSquared(h) > 0.0f) {
                h = normalize(h);
                if (h.z < 0.0f) h = -h;
                const float d = ggxD(h, lw.alpha);
                const float g = smithG2(wo, wi, lw.alpha);
                const float cosOH = absDot(wo, h);
                bool allowDielectricReflect = wo.z > 0.0f || mat.internalReflections > 0.5f;
                if (!allowDielectricReflect && tw > 0.0f) {
                    const float eta = lw.eta > 0.0f ? 1.0f / lw.eta : 1.0f;
                    const float cosHI = absDot(wo, h);
                    const float sin2 = srMax(0.0f, 1.0f - cosHI * cosHI);
                    allowDielectricReflect = (sin2 / (eta * eta)) >= 1.0f;
                }
                if (tw > 0.0f && allowDielectricReflect) {
                    const float fr = fresnelDielectric(dot(wo, h), lw.eta);
                    const float specF = d * g * fr / (4.0f * fabsf(wo.z) * fabsf(wi.z));
                    out.f += Vec3(specF * tw);
                    out.pdf += lw.transmission * fr * ggxVndfPdf(wo, h, lw.alpha) / (4.0f * srMax(1e-6f, cosOH));
                }
                const bool hasOpaqueSpec =
                    lw.specular > 0.0f && (saturatef(mat.metallic) > 1e-4f || saturatef(mat.specular) > 1e-4f);
                if (opaqueSpec > 0.0f && hasOpaqueSpec && wo.z > 0.0f && wi.z > 0.0f) {
                    const Vec3 fr = fresnelSchlick(lw.f0, cosOH);
                    out.f += fr * (d * g / (4.0f * wo.z * wi.z)) * opaqueSpec;
                    out.pdf += lw.specular * ggxVndfPdf(wo, h, lw.alpha) / (4.0f * srMax(1e-6f, cosOH));
                }
            }
        }
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
        const float eta = wo.z > 0.0f ? lw.eta : 1.0f / lw.eta;
        Vec3 h = -(wo + wi * eta);
        if (lengthSquared(h) > 0.0f) {
            h = normalize(h);
            if (h.z < 0.0f) h = -h;
            const float dotOH = dot(wo, h);
            const float dotIH = dot(wi, h);
            if (dotOH * wo.z > 0.0f) {
                const float sqrtDenom = dotOH + eta * dotIH;
                if (fabsf(sqrtDenom) > 1e-6f) {
                    const float fr = fresnelDielectric(dotOH, lw.eta);
                    const float d = ggxD(h, lw.alpha);
                    const float g = smithG2(wo, wi, lw.alpha);
                    const float factor = fabsf(dotIH * dotOH / (wo.z * wi.z));
                    const float ft = (1.0f - fr) * d * g * factor * (eta * eta) / (sqrtDenom * sqrtDenom);
                    out.f += lw.transmissionTint * (ft * tw);
                    const float dwhDwi = fabsf(eta * eta * dotIH) / (sqrtDenom * sqrtDenom);
                    const float reflectProb =
                        (wo.z < 0.0f && mat.internalReflections <= 0.5f) ? 0.0f : fr;
                    out.pdf += lw.transmission * (1.0f - reflectProb) * ggxVndfPdf(wo, h, lw.alpha) * dwhDwi;
                }
            }
        }
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
    const float pSpecular = lw.specular;

    if (uLobe < pDiffuse && wo.z > 0.0f) {
        const Vec3 wi = sampleCosineHemisphere(u1, u2);
        if (wi.z <= 0.0f) return s;
        const BsdfEval e = bsdfEvalLocal(mat, wo, wi);
        if (e.pdf <= 0.0f) return s;
        s.wi = wi;
        s.pdf = e.pdf;
        s.weight = e.f * (wi.z / e.pdf);
        return s;
    }

    if (pSpecular > 1e-5f && uLobe < pDiffuse + pSpecular) {
        if (lw.delta) {
            const Vec3 wi(-wo.x, -wo.y, wo.z);
            if (wi.z * wo.z <= 0.0f) return s;
            const float opaqueSpec = 1.0f - saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
            s.wi = wi;
            s.specular = true;
            s.pdf = 1.0f;
            s.weight = fresnelSchlick(lw.f0, fabsf(wo.z)) * (opaqueSpec / srMax(1e-4f, pSpecular));
            return s;
        }
        const Vec3 woUp = wo.z > 0.0f ? wo : -wo;
        const Vec3 h = sampleGgxVndf(woUp, lw.alpha, u1, u2);
        const Vec3 wiLocal = reflect(woUp, h);
        if (wiLocal.z <= 0.0f) return s;
        const Vec3 wi = wo.z > 0.0f ? wiLocal : Vec3(wiLocal.x, wiLocal.y, -wiLocal.z);
        const BsdfEval e = bsdfEvalLocal(mat, wo, wi);
        if (e.pdf <= 0.0f) return s;
        s.wi = wi;
        s.pdf = e.pdf;
        s.weight = e.f * (fabsf(wi.z) / e.pdf);
        return s;
    }

    if (lw.transmission <= 1e-5f) return s;
    const float eta = wo.z > 0.0f ? lw.eta : 1.0f / lw.eta;
    Vec3 h;
    if (lw.delta) {
        h = Vec3(0.0f, 0.0f, wo.z > 0.0f ? 1.0f : -1.0f);
    } else {
        const Vec3 woUp = wo.z > 0.0f ? wo : -wo;
        h = sampleGgxVndf(woUp, lw.alpha, u1, u2);
        if (wo.z < 0.0f) h = -h;
    }
    const float dotOH = dot(wo, h);
    const float fr = fresnelDielectric(dotOH, eta);
    const float sin2ThetaI = srMax(0.0f, 1.0f - dotOH * dotOH);
    const float sin2ThetaT = sin2ThetaI / (eta * eta);
    const bool tir = sin2ThetaT >= 1.0f;
    const bool allowInternalReflect = mat.internalReflections > 0.5f || wo.z > 0.0f;
    if (tir || (allowInternalReflect && uChoice < fr)) {
        Vec3 wi = reflect(wo, h);
        if (wi.z * wo.z <= 0.0f) wi.z = -wi.z;
        if (lw.delta) {
            s.wi = wi;
            s.specular = true;
            s.pdf = 1.0f;
            s.weight = Vec3(fr / srMax(1e-4f, lw.transmission));
            return s;
        }
        const BsdfEval e = bsdfEvalLocal(mat, wo, wi);
        if (e.pdf <= 0.0f) return s;
        s.wi = wi;
        s.pdf = e.pdf;
        s.weight = e.f * (fabsf(wi.z) / e.pdf);
        return s;
    }
    const float cosThetaT = sqrtf(srMax(0.0f, 1.0f - sin2ThetaT));
    const Vec3 wi = normalize(wo * (-eta) + h * (eta * fabsf(dotOH) - cosThetaT) * (wo.z > 0.0f ? 1.0f : -1.0f));
    if (wi.z * wo.z >= 0.0f) return s;
    s.transmitted = true;
    if (lw.delta) {
        s.wi = wi;
        s.specular = true;
        s.pdf = 1.0f;
        s.weight = lw.transmissionTint * ((1.0f - fr) / srMax(1e-4f, lw.transmission));
        return s;
    }
    const BsdfEval e = bsdfEvalLocal(mat, wo, wi);
    if (e.pdf <= 0.0f) return s;
    s.wi = wi;
    s.pdf = e.pdf;
    s.weight = e.f * (fabsf(wi.z) / e.pdf);
    return s;
}

SR_INL SR_HD Vec4 fetchTexelWrap(const float* pixels, int width, int height, int ix, int iy) {
    ix = ((ix % width) + width) % width;
    iy = iy < 0 ? 0 : (iy >= height ? height - 1 : iy);
    const size_t idx = (size_t(iy) * size_t(width) + size_t(ix)) * 4;
    return Vec4(pixels[idx + 0], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3]);
}

SR_INL SR_HD Vec4 sampleTextureWrap(const float* pixels, int width, int height, float u, float v) {
    if (!pixels || width <= 0 || height <= 0) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    u = u - floorf(u);
    v = clampf(v, 0.0f, 1.0f);
    const float x = u * float(width) - 0.5f;
    const float y = v * float(height) - 0.5f;
    const int x0 = int(floorf(x));
    const int y0 = int(floorf(y));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    const Vec4 c00 = fetchTexelWrap(pixels, width, height, x0, y0);
    const Vec4 c10 = fetchTexelWrap(pixels, width, height, x0 + 1, y0);
    const Vec4 c01 = fetchTexelWrap(pixels, width, height, x0, y0 + 1);
    const Vec4 c11 = fetchTexelWrap(pixels, width, height, x0 + 1, y0 + 1);
    const Vec4 c0 = Vec4(c00.x * (1.0f - fx) + c10.x * fx, c00.y * (1.0f - fx) + c10.y * fx,
                         c00.z * (1.0f - fx) + c10.z * fx, c00.w * (1.0f - fx) + c10.w * fx);
    const Vec4 c1 = Vec4(c01.x * (1.0f - fx) + c11.x * fx, c01.y * (1.0f - fx) + c11.y * fx,
                         c01.z * (1.0f - fx) + c11.z * fx, c01.w * (1.0f - fx) + c11.w * fx);
    return Vec4(c0.x * (1.0f - fy) + c1.x * fy, c0.y * (1.0f - fy) + c1.y * fy, c0.z * (1.0f - fy) + c1.z * fy,
                c0.w * (1.0f - fy) + c1.w * fy);
}

SR_INL SR_HD Vec3 sampleMapRgb(const SceneView& scene, int texIndex, Vec2 uv, Vec3 fallback) {
    if (texIndex < 0 || texIndex >= scene.textureCount || !scene.textures) return fallback;
    const TextureView& tex = scene.textures[texIndex];
    if (!tex.valid()) return fallback;
    const Vec4 c = sampleTextureWrap(tex.pixels, tex.width, tex.height, uv.x, uv.y);
    return vmax(Vec3(0.0f), Vec3(c.x, c.y, c.z));
}

SR_INL SR_HD float sampleMapScalar(const SceneView& scene, int texIndex, Vec2 uv, float fallback) {
    if (texIndex < 0 || texIndex >= scene.textureCount || !scene.textures) return fallback;
    const TextureView& tex = scene.textures[texIndex];
    if (!tex.valid()) return fallback;
    const Vec4 c = sampleTextureWrap(tex.pixels, tex.width, tex.height, uv.x, uv.y);
    return saturatef(c.x);
}

SR_INL SR_HD Material evaluateMaps(const SceneView& scene, const Material& base, Vec2 uv, Vec3& ns) {
    Material mat = base;
    mat.baseColor = sampleMapRgb(scene, base.baseColorTex, uv, base.baseColor);
    mat.roughness = sampleMapScalar(scene, base.roughnessTex, uv, base.roughness);
    mat.metallic = sampleMapScalar(scene, base.metallicTex, uv, base.metallic);
    mat.opacity = sampleMapScalar(scene, base.opacityTex, uv, base.opacity);
    mat.emissionColor = sampleMapRgb(scene, base.emissionTex, uv, base.emissionColor);
    mat.specularColor = sampleMapRgb(scene, base.specularColorTex, uv, base.specularColor);
    mat.transmissionColor = sampleMapRgb(scene, base.transmissionColorTex, uv, base.transmissionColor);

    if (base.normalTex >= 0 && base.normalTex < scene.textureCount && scene.textures) {
        const float nScale = srIsFinite(base.normalScale) ? base.normalScale : 1.0f;
        Vec3 nMap = sampleMapRgb(scene, base.normalTex, uv, Vec3(0.5f, 0.5f, 1.0f));
        nMap = nMap * 2.0f - Vec3(1.0f);
        nMap.x *= nScale;
        nMap.y *= nScale;
        const float xy2 = nMap.x * nMap.x + nMap.y * nMap.y;
        nMap.z = srMax(0.05f, sqrtf(srMax(0.0f, 1.0f - xy2)));
        nMap = normalize(nMap);
        if (!srIsFinite(nMap.x) || !srIsFinite(nMap.y) || !srIsFinite(nMap.z)) nMap = Vec3(0.0f, 0.0f, 1.0f);
        const Frame frame(ns);
        ns = normalize(frame.toWorld(nMap));
        if (!srIsFinite(ns.x) || !srIsFinite(ns.y) || !srIsFinite(ns.z)) ns = frame.n;
    }
    return mat;
}

}  // namespace optixpt
}  // namespace sol
