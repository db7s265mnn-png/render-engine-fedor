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

// Transmittance estimate for shadow rays (ratio tracking, capped iterations).
Vec3 mediumShadowTrVdb(const VolumeGrid& grid, const MediumData& medium, Vec3 origin, Vec3 direction,
                       float dist);

}  // namespace sol
