// CPU-only OpenVDB sampling helpers for the Embree path tracer.
// Volume algorithms follow PBRT 4ed Ch.11 (null scattering) and Ch.14 (VolPath).
#pragma once

#include "core/rng.h"
#include "scene/types.h"
#include "scene/volume_grid.h"
#include "render/volume.h"

namespace sol {

// Raymarch an SDF level set (sphere tracing). Returns true on a zero crossing.
// (SDF is a surface hit — not a PBRT participating medium.)
bool intersectSdfVolume(const VolumeGrid& grid, Vec3 origin, Vec3 direction, float tMin, float tMax,
                        float& tHit, Vec3& normal);

// Heterogeneous free-flight via piecewise-majorant delta tracking (PBRT §11.2.1).
// Supervoxels are world-sized (not 8 voxels): empty bricks skip, constant cells
// are analytical, mixed cells use decomposition tracking + a cached OpenVDB accessor.
MediumSample sampleMediumVdbFog(const VolumeGrid& grid, const MediumData& medium, Vec3 origin,
                                Vec3 direction, float tMax, Rng& rng, Vec3& throughput);

// First t in [tMin, tMax] ∩ AABB where occupancy is dense enough to enter the
// medium. False = the ray only sees vacuum / halo (skip the AABB proxy).
// tExit is the analytical AABB far hit when the ray overlaps the bounds.
bool fogRayFirstDenseT(const VolumeGrid& grid, Vec3 origin, Vec3 direction, float tMin, float tMax,
                       float& tDense, float& tExit);

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
