// GPU dense-volume helpers. No optixTrace — used by shade_volume / shade_shadow / init.
#pragma once

#include "render/optix/optix_wavefront.cuh"
#include "render/volume.h"

namespace sol {

__device__ inline bool gpuRayAabb(Vec3 origin, Vec3 direction, Vec3 bmin, Vec3 bmax, float& tEnter,
                                  float& tExit) {
    float t0 = -1.0e30f;
    float t1 = 1.0e30f;
    const float* o = &origin.x;
    const float* d = &direction.x;
    const float* lo = &bmin.x;
    const float* hi = &bmax.x;
    for (int axis = 0; axis < 3; ++axis) {
        const float od = d[axis];
        if (fabsf(od) < 1e-20f) {
            if (o[axis] < lo[axis] || o[axis] > hi[axis]) return false;
            continue;
        }
        float ta = (lo[axis] - o[axis]) / od;
        float tb = (hi[axis] - o[axis]) / od;
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

__device__ inline bool gpuPointInAabb(Vec3 p, Vec3 bmin, Vec3 bmax) {
    return p.x >= bmin.x && p.x <= bmax.x && p.y >= bmin.y && p.y <= bmax.y && p.z >= bmin.z &&
           p.z <= bmax.z;
}

__device__ inline int gpuMediumIndexForVolume(const SceneView& scene, int volumeIndex) {
    if (!scene.media || volumeIndex < 0) return -1;
    for (int i = 0; i < scene.mediumCount; ++i) {
        if (scene.media[i].volumeIndex == volumeIndex) return i;
    }
    return -1;
}

__device__ inline float gpuVolumeAt(const GpuVolumeGrid& g, int ix, int iy, int iz) {
    ix = ix < 0 ? 0 : (ix >= g.nx ? g.nx - 1 : ix);
    iy = iy < 0 ? 0 : (iy >= g.ny ? g.ny - 1 : iy);
    iz = iz < 0 ? 0 : (iz >= g.nz ? g.nz - 1 : iz);
    return g.density[(size_t(iz) * size_t(g.ny) + size_t(iy)) * size_t(g.nx) + size_t(ix)];
}

__device__ inline float sampleGpuVolume(const GpuVolumeGrid& g, Vec3 p) {
    if (!g.density || g.nx <= 0 || g.ny <= 0 || g.nz <= 0) return 0.0f;
    const Vec3 ext = g.bmax - g.bmin;
    if (ext.x <= 1e-12f || ext.y <= 1e-12f || ext.z <= 1e-12f) return 0.0f;
    float u = (p.x - g.bmin.x) / ext.x;
    float v = (p.y - g.bmin.y) / ext.y;
    float w = (p.z - g.bmin.z) / ext.z;
    if (u < 0.0f || v < 0.0f || w < 0.0f || u > 1.0f || v > 1.0f || w > 1.0f) return 0.0f;
    const float x = u * float(g.nx) - 0.5f;
    const float y = v * float(g.ny) - 0.5f;
    const float z = w * float(g.nz) - 0.5f;
    const int x0 = int(floorf(x));
    const int y0 = int(floorf(y));
    const int z0 = int(floorf(z));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    const float fz = z - float(z0);
    const float c000 = gpuVolumeAt(g, x0, y0, z0);
    const float c100 = gpuVolumeAt(g, x0 + 1, y0, z0);
    const float c010 = gpuVolumeAt(g, x0, y0 + 1, z0);
    const float c110 = gpuVolumeAt(g, x0 + 1, y0 + 1, z0);
    const float c001 = gpuVolumeAt(g, x0, y0, z0 + 1);
    const float c101 = gpuVolumeAt(g, x0 + 1, y0, z0 + 1);
    const float c011 = gpuVolumeAt(g, x0, y0 + 1, z0 + 1);
    const float c111 = gpuVolumeAt(g, x0 + 1, y0 + 1, z0 + 1);
    const float c00 = c000 * (1.0f - fx) + c100 * fx;
    const float c10 = c010 * (1.0f - fx) + c110 * fx;
    const float c01 = c001 * (1.0f - fx) + c101 * fx;
    const float c11 = c011 * (1.0f - fx) + c111 * fx;
    const float c0 = c00 * (1.0f - fy) + c10 * fy;
    const float c1 = c01 * (1.0f - fy) + c11 * fy;
    return c0 * (1.0f - fz) + c1 * fz;
}

__device__ inline bool gpuFogFirstDenseT(const GpuVolumeGrid& g, Vec3 origin, Vec3 direction, float tMin,
                                         float tMax, float& tDense) {
    if (!g.density || g.kind != 1 || g.nx <= 0 || g.ny <= 0 || g.nz <= 0) return false;
    const Vec3 ext = g.bmax - g.bmin;
    const float voxel =
        srMax(1e-4f, srMin(ext.x / float(srMax(1, g.nx)),
                           srMin(ext.y / float(srMax(1, g.ny)), ext.z / float(srMax(1, g.nz)))));
    float t = srMax(0.0f, tMin);
    constexpr int kMax = 512;
    for (int i = 0; i < kMax && t < tMax; ++i) {
        const float occ = srMax(0.0f, sampleGpuVolume(g, origin + direction * t));
        if (volumeOccupancyIsDense(occ, g.majorant)) {
            tDense = t;
            return true;
        }
        t += voxel;
    }
    return false;
}

__device__ inline MediumSample sampleGpuFog(const GpuVolumeGrid& g, const MediumData& medium, Vec3 origin,
                                            Vec3 direction, float tMax, Rng& rng, Vec3& throughput) {
    MediumSample out;
    float tEnter = 0.0f;
    float tExit = tMax;
    if (gpuRayAabb(origin, direction, g.bmin, g.bmax, tEnter, tExit)) {
        tMax = srMin(tMax, srMax(0.0f, tExit));
    }
    float t = srMax(0.0f, tEnter);
    if (t >= tMax) {
        out.t = tMax;
        return out;
    }
    const float densityScale = srMax(0.0f, medium.density);
    const Vec3 sigmaT0 = medium.sigmaA + medium.sigmaS;
    const float baseMaj = srMax(sigmaT0.x, srMax(sigmaT0.y, sigmaT0.z));
    const float majorant = srMax(1e-8f, baseMaj * srMax(g.majorant, 1e-6f) * densityScale);
    if (baseMaj <= 1e-12f || densityScale <= 0.0f) {
        out.t = tMax;
        return out;
    }
    constexpr int kMaxIters = 4096;
    for (int iter = 0; iter < kMaxIters; ++iter) {
        const float u = srMax(1e-6f, 1.0f - rng.nextFloat());
        t += -logf(u) / majorant;
        if (t >= tMax) {
            out.t = tMax;
            return out;
        }
        const float occ = srMax(0.0f, sampleGpuVolume(g, origin + direction * t));
        const float dens = occ * densityScale;
        const Vec3 sigmaT = sigmaT0 * dens;
        const float stHero = srMax(sigmaT.x, srMax(sigmaT.y, sigmaT.z));
        if (rng.nextFloat() >= stHero / majorant) continue;
        const Vec3 sigmaA = medium.sigmaA * dens;
        const Vec3 sigmaS = medium.sigmaS * dens;
        const float saAvg = (sigmaA.x + sigmaA.y + sigmaA.z) * (1.0f / 3.0f);
        const float ssAvg = (sigmaS.x + sigmaS.y + sigmaS.z) * (1.0f / 3.0f);
        const float stSum = srMax(1e-8f, saAvg + ssAvg);
        if (rng.nextFloat() < saAvg / stSum) {
            throughput = Vec3(0.0f);
            out.t = t;
            out.occupancy = occ;
            out.dense = volumeOccupancyIsDense(occ, g.majorant);
            out.absorbed = true;
            return out;
        }
        const Vec3 albedo(sigmaT.x > 1e-8f ? sigmaS.x / sigmaT.x : 0.0f,
                          sigmaT.y > 1e-8f ? sigmaS.y / sigmaT.y : 0.0f,
                          sigmaT.z > 1e-8f ? sigmaS.z / sigmaT.z : 0.0f);
        throughput = throughput * albedo;
        out.t = t;
        out.occupancy = occ;
        out.dense = volumeOccupancyIsDense(occ, g.majorant);
        out.scattered = true;
        return out;
    }
    out.t = tMax;
    return out;
}

__device__ inline Vec3 gpuVolumeShadowTr(const LaunchParams& params, Vec3 origin, Vec3 direction, float dist,
                                         int mediumIndex) {
    if (dist <= 1e-6f) return Vec3(1.0f);
    Vec3 Tr(1.0f);
    const SceneView& scene = params.scene;
    if (const MediumData* med = getMedium(scene, mediumIndex)) {
        if (med->type == 1) Tr = Tr * mediumShadowTr(*med, dist);
    }
    if (!params.volumes || params.volumeCount <= 0) return Tr;
    constexpr int kSteps = 48;
    for (int vi = 0; vi < params.volumeCount; ++vi) {
        const GpuVolumeGrid& g = params.volumes[vi];
        if (!g.density || g.kind != 1) continue;
        const int medIndex = gpuMediumIndexForVolume(scene, vi);
        if (medIndex < 0) continue;
        const MediumData& med = scene.media[medIndex];
        float tEnter = 0.0f;
        float tExit = dist;
        if (!gpuRayAabb(origin, direction, g.bmin, g.bmax, tEnter, tExit)) continue;
        const float a = srMax(0.0f, tEnter);
        const float b = srMin(dist, tExit);
        if (b <= a + 1e-6f) continue;
        const Vec3 sigmaT0 = med.sigmaA + med.sigmaS;
        const float maj = srMax(sigmaT0.x, srMax(sigmaT0.y, sigmaT0.z)) * srMax(0.0f, med.density);
        const float dt = (b - a) / float(kSteps);
        float tau = 0.0f;
        for (int s = 0; s < kSteps; ++s) {
            const float t = a + (float(s) + 0.5f) * dt;
            tau += maj * srMax(0.0f, sampleGpuVolume(g, origin + direction * t)) * dt;
        }
        const float att = expf(-tau);
        Tr = Tr * Vec3(att);
        if (maxComponent(Tr) < 1e-5f) return Vec3(0.0f);
    }
    return Tr;
}

__device__ inline bool sphereTraceGpuSdf(const GpuVolumeGrid& g, Vec3 origin, Vec3 direction, float tMin,
                                         float tMax, float& tHit, Vec3& normal) {
    if (!g.density || g.kind != 0) return false;
    float tEnter = tMin;
    float tExit = tMax;
    if (!gpuRayAabb(origin, direction, g.bmin, g.bmax, tEnter, tExit)) return false;
    tMin = srMax(tMin, tEnter);
    tMax = srMin(tMax, tExit);
    if (tMax <= tMin) return false;
    const Vec3 ext = g.bmax - g.bmin;
    const float voxel = srMax(ext.x / float(srMax(1, g.nx)), 1e-4f);
    const float eps = voxel * 0.25f;
    float t = srMax(0.0f, tMin);
    for (int i = 0; i < 256; ++i) {
        if (t > tMax) return false;
        const Vec3 p = origin + direction * t;
        const float d = sampleGpuVolume(g, p);
        if (fabsf(d) < eps) {
            tHit = t;
            const float e = voxel;
            normal = normalize(Vec3(sampleGpuVolume(g, p + Vec3(e, 0, 0)) - sampleGpuVolume(g, p - Vec3(e, 0, 0)),
                                    sampleGpuVolume(g, p + Vec3(0, e, 0)) - sampleGpuVolume(g, p - Vec3(0, e, 0)),
                                    sampleGpuVolume(g, p + Vec3(0, 0, e)) - sampleGpuVolume(g, p - Vec3(0, 0, e))));
            if (dot(normal, direction) > 0.0f) normal = normal * -1.0f;
            return true;
        }
        t += srMax(eps, fabsf(d));
    }
    return false;
}

}  // namespace sol
