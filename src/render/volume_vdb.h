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

// Heterogeneous free-flight via delta tracking / null collisions (PBRT §11.2.1, §14.1–14.2).
// Majorant from grid; real scatter multiplies throughput by σs/σt (albedo).
MediumSample sampleMediumVdbFog(const VolumeGrid& grid, const MediumData& medium, Vec3 origin,
                                Vec3 direction, float tMax, Rng& rng, Vec3& throughput);

// Shadow-ray transmittance through one fog VDB via ratio tracking (PBRT §11.2.1 Eq. 11.17;
// VolPath §14.2.2 uses this for light-connection shadow rays — not Riemann exp(−τ̂)).
// Caller should clip [origin, origin+dir*dist] to the fog AABB first.
Vec3 mediumShadowTrVdb(const VolumeGrid& grid, const MediumData& medium, Vec3 origin, Vec3 direction,
                       float dist, Rng& rng);

// Hard occluder test: any SDF volume intersected in (tMin, tMax) blocks the shadow ray.
bool shadowOccludedBySdfVolumes(const SceneView& scene, Vec3 origin, Vec3 direction, float tMax);

// Soft fog cast/self shadow: product of ratio-tracked Tr through every Fog volume on the segment.
// Distant lights: each volume is clipped to its world AABB.
Vec3 shadowTransmittanceFogVolumes(const SceneView& scene, Vec3 origin, Vec3 direction, float tMax,
                                   Rng& rng);

}  // namespace sol
