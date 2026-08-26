// Physically based BSDF used by both backends.
//
// Lobe math lives in shading_bsdf.h so OptiX can share the CPU dielectric
// (Snell, 1/η², anisotropy) without pulling MaterialX procedurals.
// This file keeps RGB-hero dispersion state plus texture / procedural evaluation.
#pragma once

#include "render/procedural.h"
#include "render/shading_bsdf.h"

namespace sol {

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
// Pref / Nref (Arnold): when hasPref!=0, triplanar / noise3d / autobump sample the
// pre-displace cage so beauty stays locked to geometric displacement.
SR_INL SR_HD Material evaluateTexturedMaterial(const SceneView& scene, const Material& base, Vec2 uv, Vec3& ns,
                                               Vec3 pObject, Vec3 nObject, float uvFilterWidth = 0.0f,
                                               Vec3 pRef = Vec3(0.0f), Vec3 nRef = Vec3(0.0f, 0.0f, 1.0f),
                                               int hasPref = 0) {
    Material mat = base;
    ProceduralCtx ctx;
    ctx.uv = uv;
    ctx.pObject = pObject;
    ctx.nObject = nObject;
    ctx.pRef = hasPref ? pRef : pObject;
    ctx.nRef = hasPref ? nRef : nObject;
    ctx.hasPref = hasPref ? 1 : 0;
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
        // Bump FD in object space: keep Pref locked to the sample point so
        // triplanar blend stays on the cage while probing neighbourhood.
        if (hctx.hasPref) {
            hctx.pRef = sampleP;
        }
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
    const bool hasBump =
        base.bumpProc >= 0 || (base.bumpTex >= 0 && base.bumpTex < scene.textureCount && scene.textures);
    // Autobump only when there is real high-frequency content (map/proc). Constant
    // height has zero FD and must not enable this path. Geometric displace already
    // rebuilt vertex normals — autobump supplies the Pref-space displace normal.
    const bool hasAutobump =
        base.autobump != 0 && !hasBump &&
        (base.displacementProc >= 0 ||
         (base.displacementTex >= 0 && base.displacementTex < scene.textureCount && scene.textures));

    auto sampleDispHeightAt = [&](Vec2 sampleUv, Vec3 sampleP) -> float {
        ProceduralCtx hctx = ctx;
        hctx.uv = sampleUv;
        // Arnold autobump: evaluate the displacement shader at Pref (cage), not
        // at the already-displaced P — otherwise triplanar axes flip with the
        // new relief and compound the height.
        hctx.pObject = sampleP;
        hctx.pRef = sampleP;
        hctx.nObject = ctx.hasPref ? ctx.nRef : ctx.nObject;
        hctx.nRef = hctx.nObject;
        hctx.hasPref = 1;
        hctx.forDisplacement = 1;
        float h = base.displacementHeight;
        if (base.displacementProc >= 0) {
            const Vec4 c = evalProceduralRoot(scene, base.displacementProc, hctx);
            h = c.x;  // mono height in R (see displace.cpp)
        } else if (base.displacementTex >= 0 && base.displacementTex < scene.textureCount && scene.textures) {
            const TextureView& tex = scene.textures[base.displacementTex];
            const Vec4 c = sampleTextureRGBALod(tex, sampleUv, textureLodFromFilterWidth(tex, uvFilterWidth));
            h = c.x;
        }
        return (h - base.displacementZeroValue);
    };

    if (hasBump) {
        // MaterialX <bump>: finite-difference height → tangent-space normal.
        // UV FD covers noise2d/image; object-P FD covers triplanar/noise3d (Karma/Arnold).
        // With Pref, probe the cage so triplanar axes do not flip with displaced N.
        float epsUv = uvFilterWidth > 1e-6f ? uvFilterWidth : 1.0f / 512.0f;
        if (base.bumpTex >= 0 && base.bumpTex < scene.textureCount && scene.textures) {
            const TextureView& tex = scene.textures[base.bumpTex];
            if (tex.width > 1) epsUv = srMax(epsUv, 1.0f / float(tex.width));
        }
        epsUv = srMax(1e-5f, epsUv);
        const Vec3 bumpP = ctx.hasPref ? ctx.pRef : pObject;
        const Vec3 bumpN = ctx.hasPref ? ctx.nRef : nObject;
        const float h = sampleHeightAt(uv, bumpP);
        float dHu = 0.0f;
        float dHv = 0.0f;
        if (base.bumpProc >= 0) {
            const float hu = sampleHeightAt(Vec2(uv.x + epsUv, uv.y), bumpP);
            const float hv = sampleHeightAt(Vec2(uv.x, uv.y + epsUv), bumpP);
            dHu += (hu - h) / epsUv;
            dHv += (hv - h) / epsUv;
            // Object-space FD in the Pref/shading tangent frame (triplanar / 3d noise).
            const Vec3 frameN = lengthSquared(bumpN) > 1e-12f ? normalize(bumpN) : ns;
            const Frame frameP(frameN);
            float epsP = uvFilterWidth > 1e-6f ? uvFilterWidth : 1.0e-3f;
            // Prefer a world/object scale tied to filter width; clamp for stability.
            epsP = clampf(epsP, 1.0e-4f, 0.05f);
            const float ht = sampleHeightAt(uv, bumpP + frameP.t * epsP);
            const float hb = sampleHeightAt(uv, bumpP + frameP.b * epsP);
            dHu += (ht - h) / epsP;
            dHv += (hb - h) / epsP;
        } else {
            const float hu = sampleHeightAt(Vec2(uv.x + epsUv, uv.y), bumpP);
            const float hv = sampleHeightAt(Vec2(uv.x, uv.y + epsUv), bumpP);
            dHu = (hu - h) / epsUv;
            dHv = (hv - h) / epsUv;
        }
        Vec3 nMap(-dHu * nScale, -dHv * nScale, 1.0f);
        if (!srIsFinite(nMap.x) || !srIsFinite(nMap.y) || !srIsFinite(nMap.z)) nMap = Vec3(0.0f, 0.0f, 1.0f);
        nMap = normalize(nMap);
        const Frame frame(ns);
        ns = normalize(frame.toWorld(nMap));
        if (!srIsFinite(ns.x) || !srIsFinite(ns.y) || !srIsFinite(ns.z)) ns = frame.n;
    } else if (hasAutobump) {
        // Arnold autobump: estimate the shading normal as if the mesh were diced
        // far past the geometric tessellation. Evaluate the displacement shader at
        // Pref (pre-displace cage) so triplanar axes stay locked — evaluating on
        // displaced P would flip projections and compound the height.
        // No residual×1/(1+iters): denser geo already matches this Pref normal, so
        // the visual "strength" falls off naturally (Arnold-like). Applies to all
        // ray types (camera / specular / diffuse / transmission).
        const Vec3 geoNs = ns;
        const float dispScale = srIsFinite(base.displacementScale) ? base.displacementScale : 1.0f;
        const Vec3 pref = ctx.hasPref ? ctx.pRef : pObject;
        const Vec3 nref = ctx.hasPref ? ctx.nRef : nObject;
        const Frame frameP(lengthSquared(nref) > 1e-12f ? normalize(nref) : geoNs);
        float epsP = uvFilterWidth > 1e-6f ? uvFilterWidth : 1.0e-3f;
        epsP = clampf(epsP, 1.0e-4f, 0.05f);
        const float h = sampleDispHeightAt(uv, pref) * dispScale;
        float dHt = 0.0f;
        float dHb = 0.0f;
        if (base.displacementProc >= 0) {
            const float ht = sampleDispHeightAt(uv, pref + frameP.t * epsP) * dispScale;
            const float hb = sampleDispHeightAt(uv, pref + frameP.b * epsP) * dispScale;
            dHt = (ht - h) / epsP;
            dHb = (hb - h) / epsP;
            float epsUv = uvFilterWidth > 1e-6f ? uvFilterWidth : 1.0f / 512.0f;
            epsUv = srMax(1e-5f, epsUv);
            const float hu = sampleDispHeightAt(Vec2(uv.x + epsUv, uv.y), pref) * dispScale;
            const float hv = sampleDispHeightAt(Vec2(uv.x, uv.y + epsUv), pref) * dispScale;
            const float uvToWorld = epsP / epsUv;
            dHt += (hu - h) / epsUv / srMax(uvToWorld, 1e-4f);
            dHb += (hv - h) / epsUv / srMax(uvToWorld, 1e-4f);
        } else {
            float epsUv = uvFilterWidth > 1e-6f ? uvFilterWidth : 1.0f / 512.0f;
            if (base.displacementTex >= 0 && base.displacementTex < scene.textureCount && scene.textures) {
                const TextureView& tex = scene.textures[base.displacementTex];
                if (tex.width > 1) epsUv = srMax(epsUv, 1.0f / float(tex.width));
            }
            epsUv = srMax(1e-5f, epsUv);
            const float hu = sampleDispHeightAt(Vec2(uv.x + epsUv, uv.y), pref) * dispScale;
            const float hv = sampleDispHeightAt(Vec2(uv.x, uv.y + epsUv), pref) * dispScale;
            const float worldPerUv = epsP / epsUv;
            dHt = (hu - h) / epsUv / srMax(worldPerUv, 1e-4f);
            dHb = (hv - h) / epsUv / srMax(worldPerUv, 1e-4f);
        }
        float bx = -dHt;
        float by = -dHb;
        // Clamp slope so shading normals stay within ~60° of the geometric normal
        // (avoids black NEE from shadingNormalConsistent / tiny N·L).
        const float xy = sqrtf(bx * bx + by * by);
        const float maxXy = 1.7320508f;  // tan(60°)
        if (xy > maxXy) {
            bx *= maxXy / xy;
            by *= maxXy / xy;
        }
        Vec3 nMap(bx, by, 1.0f);
        if (!srIsFinite(nMap.x) || !srIsFinite(nMap.y) || !srIsFinite(nMap.z)) nMap = Vec3(0.0f, 0.0f, 1.0f);
        nMap = normalize(nMap);
        // Pref/Nref frame = "insanely subdivided" displace normal (Arnold).
        ns = normalize(frameP.toWorld(nMap));
        if (!srIsFinite(ns.x) || !srIsFinite(ns.y) || !srIsFinite(ns.z)) ns = geoNs;
        if (dot(ns, geoNs) < 0.0f) ns = -ns;
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

}  // namespace sol
