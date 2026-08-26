// GPU surface maps for OptiX. BSDF lobes come from render/shading_bsdf.h —
// the same dielectric as Embree (Snell, 1/η², anisotropy, sheen, coat).
//
// This file still must NOT include render/shading.h: that pulls MaterialX
// procedurals into the shade megakernel. Maps here are bilinear 2D images.
#pragma once

#include "render/shading_bsdf.h"
#include "scene/types.h"

namespace sol {
namespace optixpt {

using ::sol::BsdfSample;
using ::sol::BsdfEval;
using ::sol::LobeWeights;

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
