#include "render/volume_vdb.h"

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

// Slab test: ray vs AABB. Returns true if the ray overlaps [tEnter, tExit] with tExit >= 0.
bool rayAabbInterval(Vec3 origin, Vec3 direction, const Bounds3& b, float& tEnter, float& tExit) {
    if (!b.valid()) return false;
    float t0 = -1.0e30f;
    float t1 = 1.0e30f;
    const float* o = &origin.x;
    const float* d = &direction.x;
    const float* lo = &b.lo.x;
    const float* hi = &b.hi.x;
    for (int axis = 0; axis < 3; ++axis) {
        const float od = d[axis];
        if (fabsf(od) < 1e-20f) {
            if (o[axis] < lo[axis] || o[axis] > hi[axis]) return false;
            continue;
        }
        float inv = 1.0f / od;
        float ta = (lo[axis] - o[axis]) * inv;
        float tb = (hi[axis] - o[axis]) * inv;
        if (ta > tb) {
            const float tmp = ta;
            ta = tb;
            tb = tmp;
        }
        t0 = srMax(t0, ta);
        t1 = srMin(t1, tb);
        if (t0 > t1) return false;
    }
    tEnter = t0;
    tExit = t1;
    return t1 >= 0.0f;
}

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
    if (rayAabbInterval(origin, direction, grid.worldBounds(), aabbEnter, aabbExit)) {
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

    // Clamp the walk to the fog AABB exit so missing Embree backfaces cannot stretch tMax to 1e6.
    float aabbEnter = 0.0f;
    float aabbExit = tMax;
    if (rayAabbInterval(origin, direction, grid.worldBounds(), aabbEnter, aabbExit)) {
        tMax = srMin(tMax, srMax(0.0f, aabbExit));
    }
    if (tMax <= 0.0f) {
        out.t = 0.0f;
        return out;
    }
    // Skip vacuum before the AABB; starting t=0 outside would treat the whole
    // segment as an empty supervoxel and jump to tMax (Tr=1 / no scatters).
    float t = srMax(0.0f, aabbEnter);
    if (t >= tMax) {
        out.t = tMax;
        return out;
    }

    const float densityScale = srMax(0.0f, medium.density);
    const Vec3 sigmaA0 = medium.sigmaA;
    const Vec3 sigmaS0 = medium.sigmaS;
    const Vec3 sigmaT0 = sigmaA0 + sigmaS0;
    const float baseMaj =
        srMax(sigmaT0.x, srMax(sigmaT0.y, sigmaT0.z));
    if (baseMaj <= 1e-12f || densityScale <= 0.0f) {
        out.t = tMax;
        return out;
    }

#if SOLSTICE_HAVE_OPENVDB
    auto* vdb = static_cast<openvdb::FloatGrid*>(grid.nativeGrid());
    if (!vdb) {
        out.t = tMax;
        return out;
    }
    FogTrackSampler track(*vdb, grid.sampleFilter() == VolumeSampleFilter::Nearest);
    auto sampleOcc = [&](Vec3 p) { return srMax(0.0f, track.sample(p)); };
#else
    auto sampleOcc = [&](Vec3 p) { return srMax(0.0f, grid.sampleWorldTracking(p)); };
#endif

    const float gridMaj = srMax(grid.majorant(), 1e-6f);

    auto interact = [&](float occupancy, float tHit) -> bool {
        // Near-zero occupancy is vacuum: albedo σs/σt does not go to 0 with
        // density, so a "real" event here would NEE the dome at weight ~1.
        if (!volumeOccupancyCanScatter(occupancy, gridMaj)) return false;
        const float dens = srMax(0.0f, occupancy) * densityScale;
        const Vec3 sigmaA = sigmaA0 * dens;
        const Vec3 sigmaS = sigmaS0 * dens;
        const Vec3 sigmaT = sigmaA + sigmaS;
        const float saAvg = (sigmaA.x + sigmaA.y + sigmaA.z) * (1.0f / 3.0f);
        const float ssAvg = (sigmaS.x + sigmaS.y + sigmaS.z) * (1.0f / 3.0f);
        const float stSum = srMax(1e-8f, saAvg + ssAvg);
        if (rng.nextFloat() < saAvg / stSum) {
            throughput = Vec3(0.0f);
            out.t = tHit;
            out.absorbed = true;
            return true;
        }
        const Vec3 albedo(sigmaT.x > 1e-8f ? sigmaS.x / sigmaT.x : 0.0f,
                          sigmaT.y > 1e-8f ? sigmaS.y / sigmaT.y : 0.0f,
                          sigmaT.z > 1e-8f ? sigmaS.z / sigmaT.z : 0.0f);
        throughput = throughput * albedo;
        out.t = tHit;
        out.scattered = true;
        return true;
    };

    auto collide = [&](float occupancy, float majorant, float tHit) -> bool {
        if (!volumeOccupancyCanScatter(occupancy, gridMaj)) return false;
        const float dens = srMax(0.0f, occupancy) * densityScale;
        const Vec3 sigmaT = (sigmaA0 + sigmaS0) * dens;
        // Real-vs-null uses the densest channel (matches the max-channel majorant).
        // Mean RGB under-collides the brightest extinction and spikes the others.
        const float stHero = srMax(sigmaT.x, srMax(sigmaT.y, sigmaT.z));
        if (rng.nextFloat() >= stHero / srMax(majorant, 1e-12f)) return false;
        return interact(occupancy, tHit);
    };

    // Control μc and residual pReal use the same max-channel σt as the majorant.
    const float st0 = baseMaj * densityScale;

    constexpr int kNullCollisionMaxIters = 1 << 20;
    for (int iter = 0; iter < kNullCollisionMaxIters && t < tMax; ++iter) {
        const Vec3 pLook = origin + direction * (t + 1e-5f);
        if (grid.hasMajorantBricks() && grid.majorantBrickEmpty(pLook)) {
            const float tBr = grid.majorantBrickExitT(origin, direction, t, tMax);
            t = (tBr > t) ? tBr : t + 1e-4f;
            continue;
        }
        float minD = 0.0f;
        float maxD = grid.majorant();
        grid.majorantOccupancy(pLook, minD, maxD);
        const float tExit = grid.hasMajorantGrid()
                                ? grid.majorantCellExitT(origin, direction, t, tMax)
                                : tMax;
        if (!(tExit > t)) {
            t += 1e-4f;
            continue;
        }
        if (maxD <= 1e-8f) {
            t = tExit;
            continue;
        }

        const float majorant = srMax(1e-8f, baseMaj * maxD * densityScale);
        const bool homog =
            (maxD - minD) <= (1e-3f * srMax(maxD, 1e-6f) + 1e-4f);

        if (homog) {
            // Constant occupancy: analytical Woodcock (0 voxel samples).
            const float occ = 0.5f * (minD + maxD);
            const float u = srMax(1e-6f, 1.0f - rng.nextFloat());
            const float tHit = t + (-logf(u) / majorant);
            if (tHit >= tExit) {
                t = tExit;
                continue;
            }
            if (collide(occ, majorant, tHit)) return out;
            t = tHit;
            continue;
        }

        // Decomposition tracking (Kutz et al.): control μ_c = σt(min) is a real
        // collision; residual uses local Λ − μ_c. min=0 reduces to Woodcock.
        // Vacuum min occupancy must not emit a control scatter (empty AABB corners).
        const float muC =
            volumeOccupancyCanScatter(minD, gridMaj) ? st0 * srMax(0.0f, minD) : 0.0f;
        const float residualMaj = srMax(0.0f, majorant - muC);
        float tLocal = t;
        const float tCtrl = (muC > 1e-8f)
                                ? t + (-logf(srMax(1e-6f, 1.0f - rng.nextFloat())) / muC)
                                : tExit + 1.0f;
        bool leftCell = false;
        while (iter < kNullCollisionMaxIters) {
            float tRes = tExit + 1.0f;
            if (residualMaj > 1e-8f) {
                tLocal += -logf(srMax(1e-6f, 1.0f - rng.nextFloat())) / residualMaj;
                tRes = tLocal;
            }
            const float tEvent = srMin(tCtrl, tRes);
            if (tEvent >= tExit) {
                t = tExit;
                leftCell = true;
                break;
            }
            if (tCtrl <= tRes) {
                if (interact(minD, tCtrl)) return out;
                t = tCtrl;
                leftCell = true;
                break;
            }
            const float occ = clampf(sampleOcc(origin + direction * tRes), minD, maxD);
            ++iter;
            if (!volumeOccupancyCanScatter(occ, gridMaj)) continue;
            const float pReal = (st0 * occ - muC) / srMax(residualMaj, 1e-12f);
            if (rng.nextFloat() >= clampf(pReal, 0.0f, 1.0f)) continue;
            if (interact(occ, tRes)) return out;
            t = tRes;
            leftCell = true;
            break;
        }
        if (leftCell) continue;
        break;
    }
    out.t = tMax;
    return out;
}

// PBRT 4ed residual ratio tracking (Novak et al.): control μ_c = σt(min occupancy)
// is taken out analytically; residual majorant Λ_r = Λ − μ_c walks only the leftover.
// Homogeneous filled cells (min≈max) → Λ_r≈0 → T = exp(−σt L), no voxel samples.
Vec3 mediumShadowTrVdb(const VolumeGrid& grid, const MediumData& medium, Vec3 origin, Vec3 direction,
                       float dist, Rng& rng) {
    if (!grid.valid() || dist <= 0.0f) return Vec3(1.0f);

    const float densityScale = srMax(0.0f, medium.density);
    const Vec3 sigmaT0 = medium.sigmaA + medium.sigmaS;
    const float baseMaj = srMax(sigmaT0.x, srMax(sigmaT0.y, sigmaT0.z));
    if (baseMaj <= 1e-12f || densityScale <= 0.0f) return Vec3(1.0f);

    float aabbEnter = 0.0f;
    float aabbExit = dist;
    if (!rayAabbInterval(origin, direction, grid.worldBounds(), aabbEnter, aabbExit)) return Vec3(1.0f);
    const float tEnd = srMin(dist, srMax(0.0f, aabbExit));
    float t = srMax(0.0f, aabbEnter);
    if (tEnd <= t) return Vec3(1.0f);

#if SOLSTICE_HAVE_OPENVDB
    auto* vdb = static_cast<openvdb::FloatGrid*>(grid.nativeGrid());
    if (!vdb) return Vec3(1.0f);
    FogTrackSampler track(*vdb, grid.sampleFilter() == VolumeSampleFilter::Nearest);
    auto sampleOcc = [&](Vec3 p) { return srMax(0.0f, track.sample(p)); };
#else
    auto sampleOcc = [&](Vec3 p) { return srMax(0.0f, grid.sampleWorldTracking(p)); };
#endif

    Vec3 Tr(1.0f);
    constexpr int kNullCollisionMaxIters = 1 << 20;
    int iter = 0;
    while (t < tEnd && iter < kNullCollisionMaxIters) {
        const Vec3 pLook = origin + direction * (t + 1e-5f);
        if (grid.hasMajorantBricks() && grid.majorantBrickEmpty(pLook)) {
            const float tBr = grid.majorantBrickExitT(origin, direction, t, tEnd);
            t = (tBr > t) ? tBr : t + 1e-4f;
            continue;
        }
        float minD = 0.0f;
        float maxD = grid.majorant();
        grid.majorantOccupancy(pLook, minD, maxD);
        const float tExit =
            grid.hasMajorantGrid() ? grid.majorantCellExitT(origin, direction, t, tEnd) : tEnd;
        const float L = tExit - t;
        if (!(L > 1e-8f)) {
            t += 1e-4f;
            ++iter;
            continue;
        }
        if (maxD <= 1e-8f) {
            t = tExit;
            continue;
        }

        const Vec3 muC = sigmaT0 * (srMax(0.0f, minD) * densityScale);
        const float majorant = srMax(1e-8f, baseMaj * maxD * densityScale);
        const float muCMin = srMin(muC.x, srMin(muC.y, muC.z));
        const float residualMaj = srMax(0.0f, majorant - muCMin);

        Tr = Vec3(Tr.x * expf(-muC.x * L), Tr.y * expf(-muC.y * L), Tr.z * expf(-muC.z * L));

        auto russianRoulette = [&]() -> bool {
            if (maxComponent(Tr) < 1e-6f) {
                Tr = Vec3(0.0f);
                return true;
            }
            return false;
        };
        if (russianRoulette()) return Tr;

        if (residualMaj > 1e-8f) {
            float tRes = t;
            while (iter < kNullCollisionMaxIters) {
                const float u = srMax(1e-6f, 1.0f - rng.nextFloat());
                tRes += -logf(u) / residualMaj;
                if (tRes >= tExit) break;
                const float occ = clampf(sampleOcc(origin + direction * tRes), minD, maxD);
                const Vec3 sigmaT = sigmaT0 * (occ * densityScale);
                const float nx = residualMaj - (sigmaT.x - muC.x);
                const float ny = residualMaj - (sigmaT.y - muC.y);
                const float nz = residualMaj - (sigmaT.z - muC.z);
                Tr = Vec3(Tr.x * clampf(nx / residualMaj, 0.0f, 1.0f),
                          Tr.y * clampf(ny / residualMaj, 0.0f, 1.0f),
                          Tr.z * clampf(nz / residualMaj, 0.0f, 1.0f));
                ++iter;
                if (russianRoulette()) return Tr;
            }
        }
        t = tExit;
        ++iter;
    }
    return Tr;
}

bool shadowOccludedBySdfVolumes(const SceneView& scene, Vec3 origin, Vec3 direction, float tMax) {
    if (!scene.volumes || scene.volumeCount <= 0 || tMax <= 0.0f) return false;
    for (int vi = 0; vi < scene.volumeCount; ++vi) {
        const VolumeGrid* grid = scene.volumes[vi];
        if (!grid || !grid->valid() || grid->kind() != VolumeGridKind::Sdf) continue;
        float tEnter = 0.0f;
        float tExit = tMax;
        if (!rayAabbInterval(origin, direction, grid->worldBounds(), tEnter, tExit)) continue;
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
        if (!rayAabbInterval(origin, direction, grid->worldBounds(), tEnter, tExit)) continue;
        const float a = srMax(0.0f, tEnter);
        const float b = srMin(tMax, tExit);
        if (b <= a + 1e-6f) continue;

        Tr = Tr * mediumShadowTrVdb(*grid, *med, origin + direction * a, direction, b - a, rng);
        if (maxComponent(Tr) < 1e-5f) return Vec3(0.0f);
    }
    return Tr;
}

}  // namespace sol
