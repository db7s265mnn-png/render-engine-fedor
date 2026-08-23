// CPU OpenVDB sampling helpers for the Embree path tracer.
// Free-flight + residual-ratio Tr live in volume_track.h and are shared with OptiX.
#pragma once

#include "core/rng.h"
#include "scene/types.h"
#include "scene/volume_grid.h"
#include "render/spectrum.h"
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
