// Exit to Diffuse helpers. Slim on purpose: the OptiX ETD pipeline includes
// this with optixTrace and must not pull render/shading.h (MaterialX).
#pragma once

#include "scene/types.h"

namespace sol {

SR_INL SR_HD bool materialWantsExitToDiffuse(const Material& mat) { return mat.exitToDiffuse != 0; }

constexpr int kExitToDiffuseMaxSkips = 16;

SR_INL SR_HD bool exitToDiffuseShouldStart(const Material& mat, int depth) {
    return depth > 0 && materialWantsExitToDiffuse(mat);
}

// Last bounce from a flagged surface — including the camera hit when
// maxDepth == 1. Replaces BSDF with reflect + refract opacity walks; does
// not Lambert this surface.
SR_INL SR_HD bool exitToDiffuseShouldArmBounce(const Material& mat, int depth, int maxDepth) {
    return materialWantsExitToDiffuse(mat) && depth + 1 >= maxDepth;
}

SR_INL SR_HD bool exitToDiffuseWantsRefractWalk(const Material& mat) {
    return mat.transmission > 1e-4f;
}

// Dest NEE is a bounce (card → light). gpuEyeBounceNee(path.depth) is 0 on the
// camera hit (maxDepth == 1), and primary shadows treat glass as opaque — that
// blacks out the walk when the light is on the camera side of the pane.
SR_INL SR_HD int exitToDiffuseEyeBounceNee() { return 1; }

// Same offset as GPU optix_wavefront.cuh offsetRay. Lives here so the shared
// ETD walk can include it without render/integrator.h (MaterialX).
SR_INL SR_HD Vec3 offsetRayOrigin(Vec3 p, Vec3 n, Vec3 dir) {
    const float scale = 1.0f + srMax(fabsf(p.x), srMax(fabsf(p.y), fabsf(p.z)));
    const Vec3 offset = n * (kRayEpsilon * scale);
    return dot(dir, n) > 0.0f ? p + offset : p - offset;
}

SR_INL SR_HD bool exitToDiffuseSkipOpacity(const Material& dest, float u) {
    return dest.opacity <= 1e-6f || (dest.opacity < 0.999f && u > dest.opacity);
}

// Mirror of the incoming ray about Ng facing the ray (no Snell).
SR_INL SR_HD Vec3 exitToDiffuseReflectDirection(Vec3 direction, Vec3 ng) {
    const Vec3 n = faceforward(ng, -direction);
    const Vec3 r = reflect(-direction, n);
    const float len2 = lengthSquared(r);
    if (len2 < 1e-20f) return direction;
    return r * (1.0f / sqrtf(len2));
}

// GPU / one-continuation: pick reflect or through. weightOut is 2 when both
// walks exist so the expectation matches summing both on CPU.
SR_INL SR_HD Vec3 exitToDiffuseSampleEscapeDir(Vec3 incoming, Vec3 ng, const Material& mat, float u,
                                               float& weightOut) {
    const Vec3 refl = exitToDiffuseReflectDirection(incoming, ng);
    if (!exitToDiffuseWantsRefractWalk(mat)) {
        weightOut = 1.0f;
        return refl;
    }
    weightOut = 2.0f;
    return u < 0.5f ? refl : incoming;
}

SR_INL SR_HD bool exitToDiffuseSkipSelf(int escapeMaterialIndex, int hitMaterialIndex, int skips) {
    return escapeMaterialIndex >= 0 && hitMaterialIndex == escapeMaterialIndex &&
           skips < kExitToDiffuseMaxSkips;
}

// Destination Lambert: base_color × base. Glass with base ≈ 0 uses
// transmission_color so a coloured dielectric does not exit as black or chalk.
SR_INL SR_HD Material exitToDiffuseLambert(const Material& mat) {
    Material lambert = defaultMaterial();
    Vec3 albedo = vmax(Vec3(0.0f), mat.baseColor) * srMax(0.0f, mat.baseWeight);
    if (mat.transmission > 1e-4f) {
        const Vec3 tint = vmax(Vec3(0.0f), mat.transmissionColor);
        const float mag = srMax(albedo.x, srMax(albedo.y, albedo.z));
        albedo = mag < 1e-4f ? tint : albedo * tint;
    }
    lambert.baseColor = albedo;
    lambert.baseWeight = 1.0f;
    lambert.specular = 0.0f;
    lambert.metallic = 0.0f;
    lambert.transmission = 0.0f;
    lambert.subsurface = 0.0f;
    lambert.roughness = 1.0f;
    lambert.coat = 0.0f;
    lambert.sheen = 0.0f;
    lambert.diffuseRoughness = 0.0f;
    lambert.doubleSided = mat.doubleSided;
    lambert.shadowOpacity = mat.shadowOpacity;
    lambert.contributeCaustics = 0;
    lambert.exitToDiffuse = 0;
    return lambert;
}

}  // namespace sol
