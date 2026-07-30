// Physically based BSDF used by both backends.
//
// The model is a trimmed down "principled" surface: Lambert diffuse, a GGX
// microfacet reflection lobe with Smith masking-shadowing and VNDF sampling,
// and a rough dielectric transmission lobe (Walter et al. 2007). Perfectly
// smooth lobes degrade to delta distributions.
#pragma once

#include "core/math.h"
#include "core/rng.h"
#include "render/procedural.h"
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

// ---------------------------------------------------------------------------
// Chromatic dispersion (hero-channel RGB): Cauchy model fitted through the
// material IOR (taken at the d-line) and the Abbe number V = (n_d-1)/(n_F-n_C).
// Channel wavelengths: R 630 nm, G 532 nm, B 465 nm.
// ---------------------------------------------------------------------------
SR_INL SR_HD float dispersedIor(float ior, float abbe, int channel) {
    if (abbe <= 0.0f || channel < 0) return ior;
    constexpr float invLd2 = 1.0f / (0.5876f * 0.5876f);
    constexpr float invLF2 = 1.0f / (0.4861f * 0.4861f);
    constexpr float invLC2 = 1.0f / (0.6563f * 0.6563f);
    const float B = (ior - 1.0f) / (srMax(1.0f, abbe) * (invLF2 - invLC2));
    const float A = ior - B * invLd2;
    const float lam = channel == 0 ? 0.630f : (channel == 1 ? 0.532f : 0.465f);
    return srMax(1.005f, A + B / (lam * lam));
}

// Per-path dispersion state (ray_switch-aware): only the material actually shaded
// for this hit can enable dispersion — camera-port Abbe must not tint shadow rays.
struct DispersionContext {
    int mode = 0;           // DispersionMode
    int heroChannel = -1;   // 0..2 or -1 (full RGB / off)
    int disperseHits = 0;   // dispersing interfaces that changed IOR so far
    int maxHits = 100000;   // Optimized: cap (enter+exit of one glass ≈ 2)
    bool used = false;      // path actually applied dispersing IOR or fake tint
};

SR_INL SR_HD bool materialHasDispersion(const Material& m) {
    return m.dispersionAbbe > 0.0f && m.transmission > 1e-4f;
}

// Artistic tint for Fake mode (no ray bending).
SR_INL SR_HD Vec3 fakeDispersionTint(const Material& m) {
    const float nR = dispersedIor(m.ior, m.dispersionAbbe, 0);
    const float nG = dispersedIor(m.ior, m.dispersionAbbe, 1);
    const float nB = dispersedIor(m.ior, m.dispersionAbbe, 2);
    const float invG = 1.0f / srMax(1e-3f, nG);
    Vec3 t(nR * invG, 1.0f, nB * invG);
    const float strength = clampf(40.0f / srMax(1.0f, m.dispersionAbbe), 0.0f, 2.0f);
    t = lerp(Vec3(1.0f), t, 0.35f * strength);
    return Vec3(srMax(0.05f, t.x), srMax(0.05f, t.y), srMax(0.05f, t.z));
}

// Adjusts transmissive IOR for the path hero channel. No-op for Fake mode,
// missing Abbe, or when Optimized has exhausted its interface budget.
// Legacy: applyDispersion(mat, channel) keeps working.
SR_INL SR_HD void applyDispersion(Material& m, int channel) {
    if (channel >= 0 && materialHasDispersion(m)) m.ior = dispersedIor(m.ior, m.dispersionAbbe, channel);
}

SR_INL SR_HD void applyDispersion(Material& m, DispersionContext* ctx) {
    if (!ctx || ctx->heroChannel < 0) return;
    if (!materialHasDispersion(m)) return;
    if (ctx->mode == kDispersionFake) return;  // tint applied on transmission bounce
    if (ctx->mode == kDispersionOptimized && ctx->disperseHits >= ctx->maxHits) return;
    m.ior = dispersedIor(m.ior, m.dispersionAbbe, ctx->heroChannel);
    ctx->used = true;
    ++ctx->disperseHits;
}

// Call after a transmitted BSDF sample on a dispersing material (Fake mode).
SR_INL SR_HD Vec3 applyFakeDispersionThroughput(Vec3 throughput, const Material& m,
                                                DispersionContext* ctx) {
    if (!ctx || ctx->mode != kDispersionFake || !materialHasDispersion(m)) return throughput;
    ctx->used = true;
    return throughput * fakeDispersionTint(m);
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

SR_INL SR_HD Vec4 sampleTextureWrappedRGBA(const float* pixels, int width, int height, float u, float v) {
    if (!pixels || width <= 0 || height <= 0) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    u = u - floorf(u);
    v = clampf(v, 0.0f, 1.0f);
    const float x = u * float(width) - 0.5f;
    const float y = v * float(height) - 0.5f;
    const int x0 = int(floorf(x));
    const int y0 = int(floorf(y));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    auto fetchX = [&](int ix, int iy) -> Vec4 {
        ix = ((ix % width) + width) % width;
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

SR_INL SR_HD void mipLevelSize(const TextureView& tex, int level, int& width, int& height) {
    width = srMax(1, tex.width >> level);
    height = srMax(1, tex.height >> level);
}

SR_INL SR_HD const float* mipLevelPixels(const TextureView& tex, int level) {
    if (!tex.pixels) return nullptr;
    const float* p = tex.pixels;
    int w = tex.width;
    int h = tex.height;
    const int maxLevel = srMax(0, tex.mipCount - 1);
    level = level < 0 ? 0 : (level > maxLevel ? maxLevel : level);
    for (int i = 0; i < level; ++i) {
        p += size_t(srMax(1, w)) * size_t(srMax(1, h)) * 4;
        w = srMax(1, w >> 1);
        h = srMax(1, h >> 1);
    }
    return p;
}

// Trilinear mip sample. lod = 0 is level 0 (full res).
SR_INL SR_HD Vec4 sampleTextureRGBALod(const TextureView& tex, Vec2 uv, float lod) {
    if (!tex.valid()) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);

    auto sampleLevel = [&](int level, float u, float v) -> Vec4 {
        int w = 0, h = 0;
        mipLevelSize(tex, level, w, h);
        const float* pixels = mipLevelPixels(tex, level);
        if (tex.isUdimAtlas()) return sampleTextureClampedRGBA(pixels, w, h, u, v);
        return sampleTextureWrappedRGBA(pixels, w, h, u, v);
    };

    float sampleU = uv.x;
    float sampleV = uv.y;
    if (tex.isUdimAtlas()) {
        const int gridU = tex.udimGridU;
        const int gridV = tex.udimGridV;
        int tileU = int(floorf(uv.x));
        int tileV = int(floorf(uv.y));
        float fu = uv.x - float(tileU);
        float fv = uv.y - float(tileV);
        if (fu < 0.0f) fu += 1.0f;
        if (fv < 0.0f) fv += 1.0f;
        if (tileU < 0 || tileV < 0 || tileU >= gridU || tileV >= gridV)
            return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
        const float cellW = 1.0f / float(gridU);
        const float cellH = 1.0f / float(gridV);
        sampleU = (float(tileU) + clampf(fu, 0.0f, 0.999999f)) * cellW;
        sampleV = (float(tileV) + clampf(fv, 0.0f, 0.999999f)) * cellH;
    }

    if (!tex.hasMips()) {
        if (tex.isUdimAtlas())
            return sampleTextureClampedRGBA(tex.pixels, tex.width, tex.height, sampleU, sampleV);
        return sampleTextureWrappedRGBA(tex.pixels, tex.width, tex.height, sampleU, sampleV);
    }

    const float maxLod = float(tex.mipCount - 1);
    lod = clampf(lod, 0.0f, maxLod);
    const int level0 = int(floorf(lod));
    const int level1 = level0 < tex.mipCount - 1 ? level0 + 1 : level0;
    const float frac = lod - float(level0);
    const Vec4 c0 = sampleLevel(level0, sampleU, sampleV);
    if (frac < 1e-4f || level0 == level1) return c0;
    const Vec4 c1 = sampleLevel(level1, sampleU, sampleV);
    return Vec4(c0.x * (1.0f - frac) + c1.x * frac, c0.y * (1.0f - frac) + c1.y * frac,
                c0.z * (1.0f - frac) + c1.z * frac, c0.w * (1.0f - frac) + c1.w * frac);
}

// Bilinear texture fetch for MaterialX image maps (LOD 0).
// Regular maps wrap U / clamp V.
// UDIM atlases follow MaterialX View: floor(uv) selects the tile (Mari 1001+U+V*10),
// fract(uv) samples inside that tile — equivalent to setUdimString + wrap.
SR_INL SR_HD Vec4 sampleTextureRGBA(const TextureView& tex, Vec2 uv) {
    return sampleTextureRGBALod(tex, uv, 0.0f);
}

SR_INL SR_HD float textureLodFromFilterWidth(const TextureView& tex, float uvFilterWidth) {
    if (!tex.valid() || uvFilterWidth <= 0.0f || !tex.hasMips()) return 0.0f;
    const float texels = uvFilterWidth * float(srMax(tex.width, tex.height));
    if (texels <= 1.0f) return 0.0f;
    return log2f(texels);
}

SR_INL SR_HD Vec3 sampleTextureRGB(const SceneView& scene, int texIndex, Vec2 uv, Vec3 fallback,
                                   float uvFilterWidth = 0.0f) {
    if (texIndex < 0 || texIndex >= scene.textureCount || !scene.textures) return fallback;
    const TextureView& tex = scene.textures[texIndex];
    const Vec4 c = sampleTextureRGBALod(tex, uv, textureLodFromFilterWidth(tex, uvFilterWidth));
    return vmax(Vec3(0.0f), c.xyz());
}

SR_INL SR_HD float sampleTextureScalar(const SceneView& scene, int texIndex, Vec2 uv, float fallback,
                                       float uvFilterWidth = 0.0f) {
    if (texIndex < 0 || texIndex >= scene.textureCount || !scene.textures) return fallback;
    const TextureView& tex = scene.textures[texIndex];
    const Vec4 c = sampleTextureRGBALod(tex, uv, textureLodFromFilterWidth(tex, uvFilterWidth));
    return saturatef(c.x);
}

// Apply MaterialX-style texture maps, shade-time procedurals, and normal mapping.
SR_INL SR_HD Material evaluateTexturedMaterial(const SceneView& scene, const Material& base, Vec2 uv, Vec3& ns,
                                               Vec3 pObject, Vec3 nObject, float uvFilterWidth = 0.0f) {
    Material mat = base;
    ProceduralCtx ctx;
    ctx.uv = uv;
    ctx.pObject = pObject;
    ctx.nObject = nObject;
    ctx.filterWidth = uvFilterWidth;

    auto sampleRgbSlot = [&](int proc, int tex, Vec3 fallback) -> Vec3 {
        if (proc >= 0) {
            const Vec4 c = evalProceduralRoot(scene, proc, ctx);
            return vmax(Vec3(0.0f), Vec3(c.x, c.y, c.z));
        }
        return sampleTextureRGB(scene, tex, uv, fallback, uvFilterWidth);
    };
    auto sampleScalarSlot = [&](int proc, int tex, float fallback) -> float {
        if (proc >= 0) {
            const Vec4 c = evalProceduralRoot(scene, proc, ctx);
            // Color/vector patterns wired into float ports: use luminance, not just .x.
            const float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
            return saturatef(lum);
        }
        return sampleTextureScalar(scene, tex, uv, fallback, uvFilterWidth);
    };
    auto sampleHeightAt = [&](Vec2 sampleUv, Vec3 sampleP) -> float {
        ProceduralCtx hctx = ctx;
        hctx.uv = sampleUv;
        hctx.pObject = sampleP;
        if (base.bumpProc >= 0) {
            const Vec4 c = evalProceduralRoot(scene, base.bumpProc, hctx);
            return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
        }
        return sampleTextureScalar(scene, base.bumpTex, sampleUv, 0.0f, uvFilterWidth);
    };

    mat.baseColor = sampleRgbSlot(base.baseColorProc, base.baseColorTex, base.baseColor);
    // Texture / procedural fully replaces the authored constant (Arnold map semantics).
    mat.roughness = sampleScalarSlot(base.roughnessProc, base.roughnessTex, base.roughness);
    mat.metallic = sampleScalarSlot(base.metallicProc, base.metallicTex, base.metallic);
    mat.opacity = sampleScalarSlot(base.opacityProc, base.opacityTex, base.opacity);
    mat.emissionColor = sampleRgbSlot(base.emissionProc, base.emissionTex, base.emissionColor);
    mat.subsurfaceColor = sampleRgbSlot(base.subsurfaceProc, base.subsurfaceTex, base.subsurfaceColor);
    mat.specularColor = sampleRgbSlot(base.specularColorProc, base.specularColorTex, base.specularColor);
    mat.transmissionColor =
        sampleRgbSlot(base.transmissionColorProc, base.transmissionColorTex, base.transmissionColor);

    const float nScale = srIsFinite(base.normalScale) ? base.normalScale : 1.0f;
    if (base.bumpProc >= 0 || (base.bumpTex >= 0 && base.bumpTex < scene.textureCount && scene.textures)) {
        // MaterialX <bump>: finite-difference height → tangent-space normal.
        // UV FD covers noise2d/image; object-P FD covers triplanar/noise3d (Karma/Arnold).
        float epsUv = uvFilterWidth > 1e-6f ? uvFilterWidth : 1.0f / 512.0f;
        if (base.bumpTex >= 0 && base.bumpTex < scene.textureCount && scene.textures) {
            const TextureView& tex = scene.textures[base.bumpTex];
            if (tex.width > 1) epsUv = srMax(epsUv, 1.0f / float(tex.width));
        }
        epsUv = srMax(1e-5f, epsUv);
        const float h = sampleHeightAt(uv, pObject);
        float dHu = 0.0f;
        float dHv = 0.0f;
        if (base.bumpProc >= 0) {
            const float hu = sampleHeightAt(Vec2(uv.x + epsUv, uv.y), pObject);
            const float hv = sampleHeightAt(Vec2(uv.x, uv.y + epsUv), pObject);
            dHu += (hu - h) / epsUv;
            dHv += (hv - h) / epsUv;
            // Object-space FD in the shading tangent frame (triplanar / 3d noise).
            const Frame frameP(ns);
            float epsP = uvFilterWidth > 1e-6f ? uvFilterWidth : 1.0e-3f;
            // Prefer a world/object scale tied to filter width; clamp for stability.
            epsP = clampf(epsP, 1.0e-4f, 0.05f);
            const float ht = sampleHeightAt(uv, pObject + frameP.t * epsP);
            const float hb = sampleHeightAt(uv, pObject + frameP.b * epsP);
            dHu += (ht - h) / epsP;
            dHv += (hb - h) / epsP;
        } else {
            const float hu = sampleHeightAt(Vec2(uv.x + epsUv, uv.y), pObject);
            const float hv = sampleHeightAt(Vec2(uv.x, uv.y + epsUv), pObject);
            dHu = (hu - h) / epsUv;
            dHv = (hv - h) / epsUv;
        }
        Vec3 nMap(-dHu * nScale, -dHv * nScale, 1.0f);
        if (!srIsFinite(nMap.x) || !srIsFinite(nMap.y) || !srIsFinite(nMap.z)) nMap = Vec3(0.0f, 0.0f, 1.0f);
        nMap = normalize(nMap);
        const Frame frame(ns);
        ns = normalize(frame.toWorld(nMap));
        if (!srIsFinite(ns.x) || !srIsFinite(ns.y) || !srIsFinite(ns.z)) ns = frame.n;
    } else if (base.normalProc >= 0 ||
               (base.normalTex >= 0 && base.normalTex < scene.textureCount && scene.textures)) {
        // MaterialX <normalmap> / image→normal: tangent-space RGB normal map.
        Vec3 nMap = sampleRgbSlot(base.normalProc, base.normalTex, Vec3(0.5f, 0.5f, 1.0f));
        nMap = nMap * 2.0f - Vec3(1.0f);
        nMap.x *= nScale;
        nMap.y *= nScale;
        // Reconstruct Z so scaled maps stay unit-length and don't fold.
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

// Purely specular / transmissive vertex whose lobe is still tight enough that the
// caustics it casts have to be delivered by light tracing rather than by the eye
// path stumbling onto the light. True delta lobes are a subset of this.
SR_INL SR_HD bool isNearSpecularLobe(const LobeWeights& lw) {
    return lw.alpha <= kCausticAlpha && lw.diffuse < 1e-3f;
}

// Interpolated shading normals disagree with the geometry near silhouettes and on
// intricate meshes. When the two normals disagree about whether a direction pair is
// a reflection or a transmission, the BSDF describes transport the geometry cannot
// carry: the ray starts on the wrong side of the surface, so it can skip a thin
// feature entirely and arrive somewhere it never should. Those samples are dropped.
SR_INL SR_HD bool shadingNormalConsistent(Vec3 ng, Vec3 ns, Vec3 wo, Vec3 wi) {
    const float shading = dot(ns, wo) * dot(ns, wi);
    const float geometric = dot(ng, wo) * dot(ng, wi);
    return (shading > 0.0f) == (geometric > 0.0f);
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
                // Inside the medium: skip dielectric reflection when Internal Reflections
                // is off (Arnold Advanced). TIR still contributes — nowhere else to go.
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
                // Skip the opaque specular lobe entirely when the artist set Specular to 0
                // (and the surface is not metal) — including grazing-angle Fresnel.
                const bool hasOpaqueSpec =
                    lw.specular > 0.0f && (saturatef(mat.metallic) > 1e-4f || saturatef(mat.specular) > 1e-4f);
                if (opaqueSpec > 0.0f && hasOpaqueSpec && wo.z > 0.0f && wi.z > 0.0f) {
                    const Vec3 fr = specularFresnel(mat, lw.f0, cosOH);
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
                    // When Internal Reflections is off from inside we always sample
                    // refraction (except TIR), so drop the (1-fr) from the PDF.
                    const float reflectProb =
                        (wo.z < 0.0f && mat.internalReflections <= 0.5f) ? 0.0f : fr;
                    out.pdf += lw.transmission * (1.0f - reflectProb) * ggxVndfPdf(wo, h, lw.alpha) * dwhDwi;
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
            s.weight = specularFresnel(mat, lw.f0, fabsf(wo.z)) * (opaqueSpec / srMax(1e-4f, pSpecular));
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
    // h is flipped onto wo's side, so cosThetaI is positive and the Fresnel term
    // must use the side-relative eta — passing lw.eta here would evaluate the
    // air→glass interface for rays leaving the medium and never reach TIR.
    const float dotOH = dot(wo, h);
    const float fr = fresnelDielectric(dotOH, eta);

    // Inside + Internal Reflections off: skip Fresnel reflection. TIR still
    // reflects — refraction is impossible (Arnold keeps critical-angle TIR).
    const bool allowInternalReflect = mat.internalReflections > 0.5f || wo.z > 0.0f;
    const float cosThetaI = dotOH;
    const float sin2ThetaI = srMax(0.0f, 1.0f - cosThetaI * cosThetaI);
    const float sin2ThetaT = sin2ThetaI / (eta * eta);
    const bool tir = sin2ThetaT >= 1.0f;
    const bool chooseReflect = tir || (allowInternalReflect && uChoice < fr);

    if (chooseReflect) {
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
        // Normally Fresnel cancels with the reflect/refract choice. When Internal
        // Reflections is off from inside we always refract, so multiply by (1-fr).
        float scale = tw / (eta * eta * srMax(1e-4f, lw.transmission));
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
