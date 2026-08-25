#include "render/volume_vdb.h"
#include "render/spectral_common.h"
#include "render/volume_track.h"

#include "solstice_config.h"

#include <cmath>

#if SOLSTICE_HAVE_OPENVDB
#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/RayIntersector.h>
#endif

namespace sol {
namespace {

#if SOLSTICE_HAVE_OPENVDB
// Cached OpenVDB accessor for a single ray walk. GridSampler<FloatGrid> is uncached
// (tree root every tap); ConstAccessor keeps leaf nodes hot along the ray.
struct FogTrackSampler {
    const openvdb::FloatGrid* vdb = nullptr;
    openvdb::FloatGrid::ConstAccessor acc;
    bool nearest = false;

    FogTrackSampler(openvdb::FloatGrid& grid, bool nearestFilter)
        : vdb(&grid), acc(grid.getConstAccessor()), nearest(nearestFilter) {}

    float sample(Vec3 p) {
        const openvdb::Vec3d is = vdb->worldToIndex(openvdb::Vec3d(p.x, p.y, p.z));
        if (nearest) return float(openvdb::tools::PointSampler::sample(acc, is));
        return float(openvdb::tools::BoxSampler::sample(acc, is));
    }
};
#endif

struct CpuFogGrid {
    const VolumeGrid* grid = nullptr;
#if SOLSTICE_HAVE_OPENVDB
    FogTrackSampler* track = nullptr;
#endif

    Vec3 bmin() const { return grid->worldBounds().lo; }
    Vec3 bmax() const { return grid->worldBounds().hi; }
    float majorant() const { return grid->majorant(); }
    bool hasMajorantGrid() const { return grid->hasMajorantGrid(); }
    bool hasMajorantBricks() const { return grid->hasMajorantBricks(); }
    bool brickEmpty(Vec3 p) const { return grid->hasMajorantBricks() && grid->majorantBrickEmpty(p); }
    float brickExitT(Vec3 o, Vec3 d, float t, float tMax) const {
        return grid->majorantBrickExitT(o, d, t, tMax);
    }
    void occupancy(Vec3 p, float& minD, float& maxD) const { grid->majorantOccupancy(p, minD, maxD); }
    float cellExitT(Vec3 o, Vec3 d, float t, float tMax) const {
        return grid->majorantCellExitT(o, d, t, tMax);
    }
    float sampleOcc(Vec3 p) const {
#if SOLSTICE_HAVE_OPENVDB
        if (track) return srMax(0.0f, track->sample(p));
#endif
        return srMax(0.0f, grid->sampleWorldTracking(p));
    }
};

}  // namespace

bool intersectSdfVolume(const VolumeGrid& grid, Vec3 origin, Vec3 direction, float tMin, float tMax,
                        float& tHit, Vec3& normal) {
    if (!grid.valid() || grid.kind() != VolumeGridKind::Sdf) return false;

#if SOLSTICE_HAVE_OPENVDB
    auto* vdb = static_cast<openvdb::FloatGrid*>(grid.nativeGrid());
    if (!vdb) return false;

    // Prefer OpenVDB's HDDA level-set intersector (robust for narrow bands).
    try {
        openvdb::tools::LevelSetRayIntersector<openvdb::FloatGrid> intersector(*vdb);
        using RayT = openvdb::math::Ray<double>;
        const double t0 = double(srMax(0.0f, tMin));
        const double t1 = double(srMax(t0 + 1e-6, tMax));
        RayT ray(openvdb::Vec3d(origin.x, origin.y, origin.z),
                 openvdb::Vec3d(direction.x, direction.y, direction.z), t0, t1);
        openvdb::Vec3d worldPos, worldNml;
        double wTime = 0.0;
        if (intersector.intersectsWS(ray, worldPos, worldNml, wTime)) {
            tHit = float(wTime);
            normal = Vec3(float(worldNml.x()), float(worldNml.y()), float(worldNml.z()));
            const float len = length(normal);
            if (len > 1e-12f) normal = normal / len;
            else
                normal = grid.gradientWorld(
                    Vec3(float(worldPos.x()), float(worldPos.y()), float(worldPos.z())));
            if (dot(normal, direction) > 0.0f) normal = normal * -1.0f;
            return true;
        }
        return false;
    } catch (...) {
        // Fall through to sphere tracing.
    }
#endif

    // Fallback sphere tracing (also used when OpenVDB is unavailable).
    float aabbEnter = 0.0f;
    float aabbExit = tMax;
    const Bounds3 bb = grid.worldBounds();
    if (bb.valid() && rayAabbInterval(origin, direction, bb.lo, bb.hi, aabbEnter, aabbExit)) {
        tMin = srMax(tMin, aabbEnter);
        tMax = srMin(tMax, aabbExit);
    }
    if (tMax <= tMin) return false;

    const float eps = srMax(1e-4f, grid.voxelSize() * 0.25f);
    float t = srMax(0.0f, tMin);
    for (int i = 0; i < 512; ++i) {
        if (t > tMax) return false;
        const Vec3 p = origin + direction * t;
        const float d = grid.sampleWorld(p);
        if (fabsf(d) < eps) {
            tHit = t;
            normal = grid.gradientWorld(p);
            if (dot(normal, direction) > 0.0f) normal = normal * -1.0f;
            return true;
        }
        const float step = (d > 0.0f) ? srMax(d * 0.9f, eps) : srMax(eps, -d * 0.5f);
        t += step;
    }
    return false;
}

MediumSample sampleMediumVdbFog(const VolumeGrid& grid, const MediumData& medium, Vec3 origin,
                                Vec3 direction, float tMax, Rng& rng, Vec3& throughput) {
    MediumSample out;
    if (!grid.valid() || tMax <= 0.0f) {
        out.t = tMax;
        return out;
    }

    CpuFogGrid view;
    view.grid = &grid;
#if SOLSTICE_HAVE_OPENVDB
    auto* vdb = static_cast<openvdb::FloatGrid*>(grid.nativeGrid());
    if (!vdb) {
        out.t = tMax;
        return out;
    }
    FogTrackSampler track(*vdb, grid.sampleFilter() == VolumeSampleFilter::Nearest);
    view.track = &track;
#endif
    return sampleHeterogeneousFog(view, medium, origin, direction, tMax, rng, throughput);
}

MediumSample sampleMediumVdbFogSpectral(const VolumeGrid& grid, const MediumData& medium, Vec3 origin,
                                        Vec3 direction, float tMax, Rng& rng, SampledSpectrum& throughput,
                                        const SampledWavelengths& lambda) {
    MediumSample out;
    if (!grid.valid() || tMax <= 0.0f) {
        out.t = tMax;
        return out;
    }
    CpuFogGrid view;
    view.grid = &grid;
#if SOLSTICE_HAVE_OPENVDB
    auto* vdb = static_cast<openvdb::FloatGrid*>(grid.nativeGrid());
    if (!vdb) {
        out.t = tMax;
        return out;
    }
    FogTrackSampler track(*vdb, grid.sampleFilter() == VolumeSampleFilter::Nearest);
    view.track = &track;
#endif
    return sampleHeterogeneousFogSpectral(view, medium, origin, direction, tMax, rng, throughput, lambda);
}

Vec3 mediumShadowTrVdb(const VolumeGrid& grid, const MediumData& medium, Vec3 origin, Vec3 direction,
                       float dist, Rng& rng) {
    if (!grid.valid() || dist <= 0.0f) return Vec3(1.0f);

    CpuFogGrid view;
    view.grid = &grid;
#if SOLSTICE_HAVE_OPENVDB
    auto* vdb = static_cast<openvdb::FloatGrid*>(grid.nativeGrid());
    if (!vdb) return Vec3(1.0f);
    FogTrackSampler track(*vdb, grid.sampleFilter() == VolumeSampleFilter::Nearest);
    view.track = &track;
#endif
    return mediumShadowTrHeterogeneous(view, medium, origin, direction, dist, rng);
}

bool shadowOccludedBySdfVolumes(const SceneView& scene, Vec3 origin, Vec3 direction, float tMax) {
    if (!scene.volumes || scene.volumeCount <= 0 || tMax <= 0.0f) return false;
    for (int vi = 0; vi < scene.volumeCount; ++vi) {
        const VolumeGrid* grid = scene.volumes[vi];
        if (!grid || !grid->valid() || grid->kind() != VolumeGridKind::Sdf) continue;
        float tEnter = 0.0f;
        float tExit = tMax;
        const Bounds3 bb = grid->worldBounds();
        if (!bb.valid() || !rayAabbInterval(origin, direction, bb.lo, bb.hi, tEnter, tExit)) continue;
        const float tMin = srMax(srMax(0.0f, tEnter) + 1e-4f, grid->voxelSize() * 0.15f);
        const float tHi = srMin(tMax, tExit);
        if (tHi <= tMin) continue;
        float tHit = 0.0f;
        Vec3 nSdf;
        if (intersectSdfVolume(*grid, origin, direction, tMin, tHi, tHit, nSdf)) return true;
    }
    return false;
}

Vec3 shadowTransmittanceFogVolumes(const SceneView& scene, Vec3 origin, Vec3 direction, float tMax,
                                   Rng& rng) {
    Vec3 Tr(1.0f);
    if (!scene.volumes || !scene.instances || scene.volumeCount <= 0 || tMax <= 0.0f) return Tr;

    // Pair each fog grid with its MediumData via the volume instance (Tr multiplicative — §11.2).
    for (int ii = 0; ii < scene.instanceCount; ++ii) {
        const InstanceData& inst = scene.instances[ii];
        if (inst.volumeIndex < 0 || inst.volumeIndex >= scene.volumeCount) continue;
        const VolumeGrid* grid = scene.volumes[inst.volumeIndex];
        if (!grid || !grid->valid() || grid->kind() != VolumeGridKind::Fog) continue;
        const MediumData* med = getMedium(scene, inst.mediumIndex);
        if (!med) continue;

        float tEnter = 0.0f;
        float tExit = tMax;
        const Bounds3 bb = grid->worldBounds();
        if (!bb.valid() || !rayAabbInterval(origin, direction, bb.lo, bb.hi, tEnter, tExit)) continue;
        const float a = srMax(0.0f, tEnter);
        const float b = srMin(tMax, tExit);
        if (b <= a + 1e-6f) continue;

        Tr = Tr * mediumShadowTrVdb(*grid, *med, origin + direction * a, direction, b - a, rng);
        if (maxComponent(Tr) < 1e-5f) return Vec3(0.0f);
    }
    return Tr;
}

}  // namespace sol
