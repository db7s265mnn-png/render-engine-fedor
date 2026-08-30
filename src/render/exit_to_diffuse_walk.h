// Shared Exit to Diffuse walk (Embree + the dedicated OptiX ETD pipeline).
// Skip-self, opacity, dest maps. Does not flip dest ng/ns (through-glass
// onto a backfacing card matches CPU). Caller runs dest NEE / miss / area
// light and applies clampDirect to the walk sum.
//
// Do not include shading.h / integrator.h / spectral_common.h here — OptiX
// etd.cu traces from this TU; those headers hang cicc / optixModuleCreate.
#pragma once

#include "render/exit_to_diffuse.h"

namespace sol {

enum class EtdWalkKind : int { None = 0, Miss = 1, AreaLight = 2, Dest = 3 };

struct EtdHit {
    Vec3 p{0.0f};
    Vec3 ng{0.0f, 0.0f, 1.0f};
    Vec3 ns{0.0f, 0.0f, 1.0f};
    Vec2 uv{0.0f, 0.0f};
    int materialIndex = -1;
    int lightIndex = -1;
    int instanceIndex = -1;
};

template <typename SurfLike>
SR_INL SR_HD void etdHitFromSurfLike(EtdHit& h, const SurfLike& si) {
    h.p = si.p;
    h.ng = si.ng;
    h.ns = si.ns;
    h.uv = si.uv;
    h.materialIndex = si.materialIndex;
    h.lightIndex = si.lightIndex;
    h.instanceIndex = si.instanceIndex;
}

SR_INL SR_HD int exitToDiffuseDestMedium(const SceneView& scene, int walkMediumIndex, int instanceIndex) {
    if (walkMediumIndex >= 0) return walkMediumIndex;
    if (instanceIndex >= 0 && instanceIndex < scene.instanceCount && scene.instances)
        return scene.instances[instanceIndex].mediumIndex;
    return -1;
}

// Ctx:
//   bool invalid — set when a hit exists but surf rebuild failed (not a miss)
//   bool intersect(Vec3 o, Vec3 d, float tMax, EtdHit& hit)
//   Material evalDestMaps(EtdHit& hit)  // may write hit.ns
//   bool skipOpacity(const Material& dest)
template <typename Ctx>
SR_INL SR_HD EtdWalkKind exitToDiffuseWalkFind(Ctx& ctx, Vec3 origin, Vec3 direction, int escapeMat,
                                               EtdHit& dest, Material& destMat) {
    destMat = defaultMaterial();
    for (int skip = 0; skip < kExitToDiffuseMaxSkips; ++skip) {
        EtdHit hit;
        if (!ctx.intersect(origin, direction, kFloatMax, hit))
            return ctx.invalid ? EtdWalkKind::None : EtdWalkKind::Miss;
        if (hit.lightIndex >= 0) {
            dest = hit;
            return EtdWalkKind::AreaLight;
        }
        if (exitToDiffuseSkipSelf(escapeMat, hit.materialIndex, skip)) {
            origin = offsetRayOrigin(hit.p, hit.ng, direction);
            continue;
        }
        destMat = ctx.evalDestMaps(hit);
        if (ctx.skipOpacity(destMat)) {
            origin = offsetRayOrigin(hit.p, hit.ng, direction);
            continue;
        }
        dest = hit;
        return EtdWalkKind::Dest;
    }
    return EtdWalkKind::None;
}

}  // namespace sol
