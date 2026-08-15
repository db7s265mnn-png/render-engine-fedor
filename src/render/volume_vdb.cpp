#include "render/volume_vdb.h"

#include <cmath>

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

}  // namespace

bool intersectSdfVolume(const VolumeGrid& grid, Vec3 origin, Vec3 direction, float tMin, float tMax,
                        float& tHit, Vec3& normal) {
    if (!grid.valid() || grid.kind() != VolumeGridKind::Sdf) return false;

    // Restrict sphere tracing to the analytical world AABB (proxy triangles can miss exits).
    float aabbEnter = 0.0f;
    float aabbExit = tMax;
    if (rayAabbInterval(origin, direction, grid.worldBounds(), aabbEnter, aabbExit)) {
        tMin = srMax(tMin, aabbEnter);
        tMax = srMin(tMax, aabbExit);
    }
    if (tMax <= tMin) return false;

    const float eps = srMax(1e-4f, grid.voxelSize() * 0.25f);
    float t = srMax(0.0f, tMin);
    // Sphere tracing / raymarch along the AABB segment.
    for (int i = 0; i < 512; ++i) {
        if (t > tMax) return false;
        const Vec3 p = origin + direction * t;
        const float d = grid.sampleWorld(p);
        if (fabsf(d) < eps) {
            tHit = t;
            normal = grid.gradientWorld(p);
            // Ensure normal faces the ray.
            if (dot(normal, direction) > 0.0f) normal = normal * -1.0f;
            return true;
        }
        // Outside: step by distance; inside: step cautiously toward surface.
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
        // Origin is already inside after proxy entry — still clamp far bound.
        tMax = srMin(tMax, srMax(0.0f, aabbExit));
    }
    if (tMax <= 0.0f) {
        out.t = 0.0f;
        return out;
    }

    const float majGrid = srMax(grid.majorant(), 1e-4f);
    const float densityScale = srMax(0.0f, medium.density);
    // Majorant extinction ≈ max(|σa|+|σs|) * majGrid * densityScale
    const Vec3 sigmaA0 = medium.sigmaA;
    const Vec3 sigmaS0 = medium.sigmaS;
    const float baseMaj = srMax(sigmaA0.x + sigmaS0.x, srMax(sigmaA0.y + sigmaS0.y, sigmaA0.z + sigmaS0.z));
    const float majorant = srMax(1e-6f, baseMaj * majGrid * densityScale);
    float t = 0.0f;
    for (int iter = 0; iter < 256; ++iter) {
        const float u = srMax(1e-6f, 1.0f - rng.nextFloat());
        t += -logf(u) / majorant;
        if (t >= tMax) {
            out.t = tMax;
            return out;
        }
        const Vec3 p = origin + direction * t;
        const float dens = srMax(0.0f, grid.sampleWorld(p)) * densityScale;
        const Vec3 sigmaA = sigmaA0 * dens;
        const Vec3 sigmaS = sigmaS0 * dens;
        const Vec3 sigmaT = sigmaA + sigmaS;
        const float stAvg = (sigmaT.x + sigmaT.y + sigmaT.z) * (1.0f / 3.0f);
        const float xi = rng.nextFloat();
        if (xi >= stAvg / majorant) {
            // Null collision.
            continue;
        }
        // Real collision — optional emission in-scatter at event.
        if (!isBlack(medium.emission)) {
            // Tracked as additive contribution by caller if needed; keep throughput path.
        }
        const float saAvg = (sigmaA.x + sigmaA.y + sigmaA.z) * (1.0f / 3.0f);
        const float ssAvg = (sigmaS.x + sigmaS.y + sigmaS.z) * (1.0f / 3.0f);
        const float stSum = srMax(1e-8f, saAvg + ssAvg);
        if (rng.nextFloat() < saAvg / stSum) {
            throughput = Vec3(0.0f);
            out.t = t;
            out.absorbed = true;
            return out;
        }
        // Ratio tracking weight σs/σt ≈ 1 for RGB average path; keep throughput.
        out.t = t;
        out.scattered = true;
        return out;
    }
    out.t = tMax;
    return out;
}

Vec3 mediumShadowTrVdb(const VolumeGrid& grid, const MediumData& medium, Vec3 origin, Vec3 direction,
                       float dist) {
    if (!grid.valid() || dist <= 0.0f) return Vec3(1.0f);
    // Cheap optical-depth estimate: stratified samples along the segment.
    const int n = 8;
    Vec3 tau(0.0f);
    const float densityScale = srMax(0.0f, medium.density);
    for (int i = 0; i < n; ++i) {
        const float t = (float(i) + 0.5f) / float(n) * dist;
        const float dens = srMax(0.0f, grid.sampleWorld(origin + direction * t)) * densityScale;
        tau = tau + (medium.sigmaA + medium.sigmaS) * dens * (dist / float(n));
    }
    return Vec3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
}

}  // namespace sol
