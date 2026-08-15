// CPU-only OpenVDB sampling helpers for the Embree path tracer.
#pragma once

#include "core/rng.h"
#include "scene/types.h"
#include "scene/volume_grid.h"
#include "render/volume.h"

namespace sol {

// Raymarch an SDF level set (sphere tracing). Returns true on a zero crossing.
bool intersectSdfVolume(const VolumeGrid& grid, Vec3 origin, Vec3 direction, float tMin, float tMax,
                        float& tHit, Vec3& normal);

// Heterogeneous delta tracking through a fog VDB. `densityScale` multiplies grid samples.
// `sigmaA/S` are base coefficients from the volume shader (multiplied by density*scale).
MediumSample sampleMediumVdbFog(const VolumeGrid& grid, const MediumData& medium, Vec3 origin,
                                Vec3 direction, float tMax, Rng& rng, Vec3& throughput);

// Transmittance estimate for a shadow segment through one fog VDB (Riemann / ratio-style).
// Integrates only over [origin, origin+dir*dist] — caller should clip to the fog AABB first.
Vec3 mediumShadowTrVdb(const VolumeGrid& grid, const MediumData& medium, Vec3 origin, Vec3 direction,
                       float dist);

// Hard occluder test: any SDF volume intersected in (tMin, tMax) blocks the shadow ray.
bool shadowOccludedBySdfVolumes(const SceneView& scene, Vec3 origin, Vec3 direction, float tMax);

// Soft fog cast/self shadow: product of Tr through every Fog volume along the segment.
// Distant lights: each volume is clipped to its world AABB (infinite tMax is fine).
Vec3 shadowTransmittanceFogVolumes(const SceneView& scene, Vec3 origin, Vec3 direction, float tMax);

}  // namespace sol
