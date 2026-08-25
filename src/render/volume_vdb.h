// CPU OpenVDB sampling helpers for the Embree path tracer.
// Free-flight + residual-ratio Tr live in volume_track.h and are shared with OptiX.
#pragma once

#include "core/rng.h"
#include "scene/types.h"
#include "scene/volume_grid.h"
#include "render/spectrum.h"
#include "render/volume.h"
#include "render/volume_track.h"

namespace sol {

inline const VolumeGrid* fogGridAt(const SceneView& scene, int volumeIndex) {
    if (!scene.volumes || volumeIndex < 0 || volumeIndex >= scene.volumeCount ||
        !scene.volumes[volumeIndex])
        return nullptr;
    if (scene.volumes[volumeIndex]->kind() != VolumeGridKind::Fog) return nullptr;
    return scene.volumes[volumeIndex];
}

// MediumData::volumeIndex should be Scene::volumes. If it was overwritten as a
// volumePaths slot, fall back to the only fog grid in the scene.
inline const VolumeGrid* fogGridForMedium(const SceneView& scene, const MediumData& m) {
    if (const VolumeGrid* g = fogGridAt(scene, m.volumeIndex)) return g;
    const VolumeGrid* only = nullptr;
    int nFog = 0;
    for (int i = 0; i < scene.volumeCount; ++i) {
        if (const VolumeGrid* g = fogGridAt(scene, i)) {
            only = g;
            ++nFog;
        }
    }
    return nFog == 1 ? only : nullptr;
}

inline float clipTMaxToFogAabb(const VolumeGrid& fog, Vec3 origin, Vec3 direction, float tMax) {
    const Bounds3 bb = fog.worldBounds();
    if (!bb.valid()) return tMax;
    float tEnter = 0.0f;
    float tExit = tMax;
    if (rayAabbInterval(origin, direction, bb.lo, bb.hi, tEnter, tExit) && tExit >= 0.0f)
        return srMin(tMax, tExit);
    return tMax;
}

// Raymarch an SDF level set (sphere tracing). Returns true on a zero crossing.
// (SDF is a surface hit — not a PBRT participating medium.)
bool intersectSdfVolume(const VolumeGrid& grid, Vec3 origin, Vec3 direction, float tMin, float tMax,
                        float& tHit, Vec3& normal);

// Heterogeneous free-flight via piecewise-majorant delta tracking (PBRT §11.2.1).
// Supervoxels are world-sized (not 8 voxels): empty bricks skip, constant cells
// are analytical, mixed cells use decomposition tracking + a cached OpenVDB accessor.
MediumSample sampleMediumVdbFog(const VolumeGrid& grid, const MediumData& medium, Vec3 origin,
                                Vec3 direction, float tMax, Rng& rng, Vec3& throughput);

// Spectral delta tracking through VDB fog (hero-λ absorb/scatter/null). CPU only.
MediumSample sampleMediumVdbFogSpectral(const VolumeGrid& grid, const MediumData& medium, Vec3 origin,
                                        Vec3 direction, float tMax, Rng& rng, SampledSpectrum& throughput,
                                        const SampledWavelengths& lambda);

// Shadow-ray transmittance via residual ratio tracking (Novak et al. / PBRT §11.2.2):
// control μ_c = σt(min occupancy) is analytic; leftover uses local Λ. Homogeneous
// filled cells reduce to Beer–Lambert. Caller clips to the fog AABB first.
Vec3 mediumShadowTrVdb(const VolumeGrid& grid, const MediumData& medium, Vec3 origin, Vec3 direction,
                       float dist, Rng& rng);

// Hard occluder test: any SDF volume intersected in (tMin, tMax) blocks the shadow ray.
bool shadowOccludedBySdfVolumes(const SceneView& scene, Vec3 origin, Vec3 direction, float tMax);

// Soft fog cast/self shadow: product of ratio-tracked Tr through every Fog volume on the segment.
// Distant lights: each volume is clipped to its world AABB.
Vec3 shadowTransmittanceFogVolumes(const SceneView& scene, Vec3 origin, Vec3 direction, float tMax,
                                   Rng& rng);

}  // namespace sol
