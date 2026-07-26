// Physically based BSDF used by both backends.
//
// The model is a trimmed down "principled" surface: Lambert diffuse, a GGX
// microfacet reflection lobe with Smith masking-shadowing and VNDF sampling,
// and a rough dielectric transmission lobe (Walter et al. 2007). Perfectly
// smooth lobes degrade to delta distributions.
#pragma once

#include "core/math.h"
#include "core/rng.h"
#include "scene/types.h"

namespace sol {

constexpr float kMinAlpha = 1.0e-3f;
constexpr float kDeltaAlpha = 2.0e-3f;

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

// Bilinear fetch in normalized UV [0,1] with clamp (no wrap).
SR_INL SR_HD Vec4 sampleTextureClampedRGBA(const float* pixels, int width, int height, float u, float v) {
    if (!pixels || width <= 0 || height <= 0) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    u = clampf(u, 0.0f, 1.0f);
    v = clampf(v, 0.0f, 1.0f);
    const float x = u * float(width) - 0.5f;
    const float y = v * float(height) - 0.5f;
    const int x0 = int(floorf(x));
    const int y0 = int(floorf(y));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    auto fetchX = [&](int ix, int iy) -> Vec4 {
        ix = ix < 0 ? 0 : (ix >= width ? width - 1 : ix);
        iy = iy < 0 ? 0 : (iy >= height ? height - 1 : iy);
        const size_t idx = (size_t(iy) * size_t(width) + size_t(ix)) * 4;
        return Vec4(pixels[idx + 0], pixels[idx + 1], pixels[idx + 2], pixels[idx + 3]);
    };
    const Vec4 c00 = fetchX(x0, y0);
    const Vec4 c10 = fetchX(x0 + 1, y0);
    const Vec4 c01 = fetchX(x0, y0 + 1);
    const Vec4 c11 = fetchX(x0 + 1, y0 + 1);
    const Vec4 c0 = Vec4(c00.x * (1.0f - fx) + c10.x * fx, c00.y * (1.0f - fx) + c10.y * fx,
                          c00.z * (1.0f - fx) + c10.z * fx, c00.w * (1.0f - fx) + c10.w * fx);
    const Vec4 c1 = Vec4(c01.x * (1.0f - fx) + c11.x * fx, c01.y * (1.0f - fx) + c11.y * fx,
                          c01.z * (1.0f - fx) + c11.z * fx, c01.w * (1.0f - fx) + c11.w * fx);
    return Vec4(c0.x * (1.0f - fy) + c1.x * fy, c0.y * (1.0f - fy) + c1.y * fy, c0.z * (1.0f - fy) + c1.z * fy,
                c0.w * (1.0f - fy) + c1.w * fy);
}

// Bilinear texture fetch for MaterialX image maps.
// Regular maps wrap U / clamp V.
// UDIM atlases follow MaterialX View: floor(uv) selects the tile (Mari 1001+U+V*10),
// fract(uv) samples inside that tile — equivalent to setUdimString + wrap.
SR_INL SR_HD Vec4 sampleTextureRGBA(const TextureView& tex, Vec2 uv) {
    if (!tex.valid()) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);

    if (tex.isUdimAtlas()) {
        const int gridU = tex.udimGridU;
        const int gridV = tex.udimGridV;
        // Select tile the same way MaterialX Mesh::splitByUdims does.
        int tileU = int(floorf(uv.x));
        int tileV = int(floorf(uv.y));
        float fu = uv.x - float(tileU);
        float fv = uv.y - float(tileV);
        // Keep fractional part in [0,1) even for negative UVs.
        if (fu < 0.0f) fu += 1.0f;
        if (fv < 0.0f) fv += 1.0f;
        if (tileU < 0 || tileV < 0 || tileU >= gridU || tileV >= gridV) {
            // Missing tile (or UV authored slightly past the set) → opaque black.
            return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        // Sample strictly inside the tile cell to avoid cross-tile bilinear bleed.
        // Atlas is baked with OpenGL/MaterialX V (V=0 at bottom of each tile image).
        const float cellW = 1.0f / float(gridU);
        const float cellH = 1.0f / float(gridV);
        const float u = (float(tileU) + clampf(fu, 0.0f, 0.999999f)) * cellW;
        const float v = (float(tileV) + clampf(fv, 0.0f, 0.999999f)) * cellH;
        return sampleTextureClampedRGBA(tex.pixels, tex.width, tex.height, u, v);
    }

    const float u = uv.x - floorf(uv.x);
    const float v = clampf(uv.y, 0.0f, 1.0f);
    const float x = u * float(tex.width) - 0.5f;
    const float y = v * float(tex.height) - 0.5f;
    const int x0 = int(floorf(x));
    const int y0 = int(floorf(y));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    auto fetchX = [&](int ix, int iy) -> Vec4 {
        ix = ((ix % tex.width) + tex.width) % tex.width;
        iy = iy < 0 ? 0 : (iy >= tex.height ? tex.height - 1 : iy);
        const size_t idx = (size_t(iy) * size_t(tex.width) + size_t(ix)) * 4;
        return Vec4(tex.pixels[idx + 0], tex.pixels[idx + 1], tex.pixels[idx + 2], tex.pixels[idx + 3]);
    };
    const Vec4 c00 = fetchX(x0, y0);
    const Vec4 c10 = fetchX(x0 + 1, y0);
    const Vec4 c01 = fetchX(x0, y0 + 1);
    const Vec4 c11 = fetchX(x0 + 1, y0 + 1);
    const Vec4 c0 = Vec4(c00.x * (1.0f - fx) + c10.x * fx, c00.y * (1.0f - fx) + c10.y * fx,
                          c00.z * (1.0f - fx) + c10.z * fx, c00.w * (1.0f - fx) + c10.w * fx);
    const Vec4 c1 = Vec4(c01.x * (1.0f - fx) + c11.x * fx, c01.y * (1.0f - fx) + c11.y * fx,
                          c01.z * (1.0f - fx) + c11.z * fx, c01.w * (1.0f - fx) + c11.w * fx);
    return Vec4(c0.x * (1.0f - fy) + c1.x * fy, c0.y * (1.0f - fy) + c1.y * fy, c0.z * (1.0f - fy) + c1.z * fy,
                c0.w * (1.0f - fy) + c1.w * fy);
}

SR_INL SR_HD Vec3 sampleTextureRGB(const SceneView& scene, int texIndex, Vec2 uv, Vec3 fallback) {
    if (texIndex < 0 || texIndex >= scene.textureCount || !scene.textures) return fallback;
    const Vec4 c = sampleTextureRGBA(scene.textures[texIndex], uv);
    return vmax(Vec3(0.0f), c.xyz());
}

SR_INL SR_HD float sampleTextureScalar(const SceneView& scene, int texIndex, Vec2 uv, float fallback) {
    if (texIndex < 0 || texIndex >= scene.textureCount || !scene.textures) return fallback;
    const Vec4 c = sampleTextureRGBA(scene.textures[texIndex], uv);
    return saturatef(c.x);
}

// Apply MaterialX-style texture maps and normal mapping to a base material.
SR_INL SR_HD Material evaluateTexturedMaterial(const SceneView& scene, const Material& base, Vec2 uv, Vec3& ns) {
    Material mat = base;
    mat.baseColor = sampleTextureRGB(scene, base.baseColorTex, uv, base.baseColor);
    if (base.baseColorTex >= 0) mat.baseColor = mat.baseColor * vmax(Vec3(0.0f), base.baseColor);
    mat.roughness = sampleTextureScalar(scene, base.roughnessTex, uv, base.roughness);
    if (base.roughnessTex >= 0) mat.roughness = saturatef(mat.roughness * base.roughness);
    mat.metallic = sampleTextureScalar(scene, base.metallicTex, uv, base.metallic);
    if (base.metallicTex >= 0) mat.metallic = saturatef(mat.metallic * base.metallic);
    mat.opacity = sampleTextureScalar(scene, base.opacityTex, uv, base.opacity);
    if (base.opacityTex >= 0) mat.opacity = saturatef(mat.opacity * base.opacity);
    mat.emissionColor = sampleTextureRGB(scene, base.emissionTex, uv, base.emissionColor);
    if (base.emissionTex >= 0) mat.emissionColor = mat.emissionColor * vmax(Vec3(0.0f), base.emissionColor);
    mat.subsurfaceColor = sampleTextureRGB(scene, base.subsurfaceTex, uv, base.subsurfaceColor);
    if (base.subsurfaceTex >= 0)
        mat.subsurfaceColor = mat.subsurfaceColor * vmax(Vec3(0.0f), base.subsurfaceColor);

    if (base.normalTex >= 0 && base.normalTex < scene.textureCount && scene.textures) {
        Vec3 nMap = sampleTextureRGB(scene, base.normalTex, uv, Vec3(0.5f, 0.5f, 1.0f));
        nMap = nMap * 2.0f - Vec3(1.0f);
        nMap.z = srMax(0.05f, nMap.z);
        nMap = normalize(nMap);
        const Frame frame(ns);
        ns = normalize(frame.toWorld(nMap));
    }
    return mat;
}

SR_INL SR_HD Vec3 fresnelSchlick(Vec3 f0, float cosTheta) {
    const float m = clampf(1.0f - cosTheta, 0.0f, 1.0f);
    const float m2 = m * m;
    const float m5 = m2 * m2 * m;
    return f0 + (Vec3(1.0f) - f0) * m5;
}

// Exact Fresnel for dielectrics. eta is the relative IOR (transmitted/incident).
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

// GGX / Trowbridge-Reitz normal distribution (isotropic).
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

// Heitz 2018: sampling the GGX distribution of visible normals.
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

// Lobe weights, derived once and reused by eval and sample.
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

SR_INL SR_HD LobeWeights computeLobes(const Material& mat) {
    LobeWeights lw;
    const float metallic = saturatef(mat.metallic);
    const float transmission = saturatef(mat.transmission);
    const float specularControl = saturatef(mat.specular);
    lw.alpha = roughnessToAlpha(mat.roughness);
    lw.delta = lw.alpha <= kDeltaAlpha;
    lw.eta = srMax(1.01f, mat.ior);
    const Vec3 base = vmax(Vec3(0.0f), mat.baseColor);
    // Standard Surface: diffuse = base * base_color (SSS is mixed separately in the integrator).
    const float baseWeight = srMax(0.0f, mat.baseWeight);
    // Specular = 0 must fully kill dielectric reflections (artist expectation).
    const float dielectricF0 = 0.08f * specularControl;
    lw.f0 = lerp(Vec3(dielectricF0), base, metallic);
    lw.diffuseAlbedo = base * (baseWeight * (1.0f - metallic) * (1.0f - transmission));
    lw.transmissionTint = base;

    // The transmission lobe already contains its own Fresnel reflection, so the
    // opaque specular lobe is faded out as transmission increases.
    const float opaqueSpec = 1.0f - transmission * (1.0f - metallic);
    const float diffuseWeight =
        (1.0f - metallic) * (1.0f - transmission) * average(base) * baseWeight;
    float specWeight = 0.0f;
    if (metallic > 1e-4f) {
        specWeight = opaqueSpec;
    } else if (specularControl > 1e-4f) {
        // Scale with the specular control so lowering it actually reduces the lobe.
        specWeight = clampf(average(lw.f0) * 4.0f + 0.15f * specularControl, 0.0f, 1.0f) * opaqueSpec *
                     specularControl;
    }
    const float transWeight = (1.0f - metallic) * transmission;
    const float total = diffuseWeight + specWeight + transWeight;
    if (total <= 0.0f) {
        // Pure black / invalid material: fall back to a tiny diffuse lobe.
        lw.diffuse = 1.0f;
    } else {
        lw.diffuse = diffuseWeight / total;
        lw.specular = specWeight / total;
        lw.transmission = transWeight / total;
    }
    return lw;
}

// Evaluate the BSDF for a pair of directions expressed in the local shading
// frame (z = shading normal). Delta lobes return zero.
SR_INL SR_HD BsdfEval bsdfEvalLocal(const Material& mat, Vec3 wo, Vec3 wi) {
    BsdfEval out;
    const LobeWeights lw = computeLobes(mat);
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
                if (tw > 0.0f) {
                    const float fr = fresnelDielectric(dot(wo, h), lw.eta);
                    const float specF = d * g * fr / (4.0f * fabsf(wo.z) * fabsf(wi.z));
                    out.f += Vec3(specF * tw);
                    out.pdf += lw.transmission * fr * ggxVndfPdf(wo, h, lw.alpha) / (4.0f * srMax(1e-6f, cosOH));
                }
                // Skip the opaque specular lobe entirely when the artist set Specular to 0
                // (and the surface is not metal) — including grazing-angle Fresnel.
                const bool hasOpaqueSpec =
                    lw.specular > 0.0f && (saturatef(mat.metallic) > 1e-4f || saturatef(mat.specular) > 1e-4f);
                if (opaqueSpec > 0.0f && hasOpaqueSpec && wo.z > 0.0f && wi.z > 0.0f) {
                    const Vec3 fr = fresnelSchlick(lw.f0, cosOH);
                    out.f += fr * (d * g / (4.0f * wo.z * wi.z)) * opaqueSpec;
                    out.pdf += lw.specular * ggxVndfPdf(wo, h, lw.alpha) / (4.0f * srMax(1e-6f, cosOH));
                }
            }
        }
    } else if (!lw.delta && lw.transmission > 0.0f) {
        // Refraction: build the generalized half vector.
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
                    const float ft = (1.0f - fr) * d * g * factor / (sqrtDenom * sqrtDenom);
                    out.f += lw.transmissionTint * (ft * tw);
                    const float dwhDwi = fabsf(eta * eta * dotIH) / (sqrtDenom * sqrtDenom);
                    out.pdf += lw.transmission * (1.0f - fr) * ggxVndfPdf(wo, h, lw.alpha) * dwhDwi;
                }
            }
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
    const LobeWeights lw = computeLobes(mat);
    if (fabsf(wo.z) < 1e-6f) return s;

    const float pDiffuse = lw.diffuse;
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

    if (pSpecular > 1e-5f && uLobe < pDiffuse + pSpecular) {
        // Opaque specular reflection.
        if (lw.delta) {
            const Vec3 wi(-wo.x, -wo.y, wo.z);
            if (wi.z * wo.z <= 0.0f) return s;
            const float opaqueSpec = 1.0f - saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
            s.wi = wi;
            s.specular = true;
            s.pdf = 1.0f;  // delta lobe: f*cos/pdf collapses to Fresnel over the lobe probability
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

    // Dielectric transmission lobe.
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
    const float fr = fresnelDielectric(dotOH, lw.eta);

    if (uChoice < fr) {
        const Vec3 wi = reflect(wo, h);
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

    // Refract wo about h (h is always on the same side as wo, so cosThetaI > 0).
    const float cosThetaI = dotOH;
    const float sin2ThetaI = srMax(0.0f, 1.0f - cosThetaI * cosThetaI);
    const float sin2ThetaT = sin2ThetaI / (eta * eta);
    if (sin2ThetaT >= 1.0f) return s;  // total internal reflection, handled by the reflect branch
    const float cosThetaT = sqrtf(srMax(0.0f, 1.0f - sin2ThetaT));
    const Vec3 wiN = normalize(-wo * (1.0f / eta) + h * (cosThetaI / eta - cosThetaT));
    if (wiN.z * wo.z >= 0.0f) return s;
    s.wi = wiN;
    s.transmitted = true;
    if (lw.delta) {
        const float tw = saturatef(mat.transmission) * (1.0f - saturatef(mat.metallic));
        s.specular = true;
        s.pdf = 1.0f;
        // 1/eta^2 is the radiance compression when crossing the interface.
        s.weight = lw.transmissionTint * (tw / (eta * eta * srMax(1e-4f, lw.transmission)));
        return s;
    }
    const BsdfEval e = bsdfEvalLocal(mat, wo, wiN);
    if (e.pdf <= 0.0f) return s;
    s.pdf = e.pdf;
    s.weight = e.f * (fabsf(wiN.z) / e.pdf);
    return s;
}

}  // namespace sol
