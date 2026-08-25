// GPU volume helpers. No optixTrace — used by shade_volume / shade_shadow / init.
// Free-flight is Woodcock / delta tracking (warp-coherent). Shadow Tr stays
// residual-ratio, matching Embree NEE. CPU PT is unchanged.
#pragma once

#include "render/optix/optix_wavefront.cuh"
#include "render/volume.h"
#include "render/volume_track.h"

namespace sol {

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
    if (g.nearest) {
        const int ix = int(floorf(u * float(g.nx)));
        const int iy = int(floorf(v * float(g.ny)));
        const int iz = int(floorf(w * float(g.nz)));
        return gpuVolumeAt(g, ix, iy, iz);
    }
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

struct GpuFogGrid {
    const GpuVolumeGrid* g = nullptr;

    SR_HD Vec3 bmin() const { return g->bmin; }
    SR_HD Vec3 bmax() const { return g->bmax; }
    SR_HD float majorant() const { return g->majorant; }
    SR_HD bool hasMajorantGrid() const { return g->majMin && g->majMax && g->majNx > 0; }
    SR_HD bool hasMajorantBricks() const { return g->bricks && g->brNx > 0; }

    SR_HD bool brickEmpty(Vec3 p) const {
        if (!hasMajorantBricks() || g->majCell <= 1e-12f) return false;
        const int bs = g->brickSize > 0 ? g->brickSize : 4;
        const float cell = g->majCell * float(bs);
        const int bx = int(floorf((p.x - g->majOrigin.x) / cell));
        const int by = int(floorf((p.y - g->majOrigin.y) / cell));
        const int bz = int(floorf((p.z - g->majOrigin.z) / cell));
        if (bx < 0 || by < 0 || bz < 0 || bx >= g->brNx || by >= g->brNy || bz >= g->brNz) return true;
        return g->bricks[size_t(bx + g->brNx * (by + g->brNy * bz))] == 0;
    }

    SR_HD float brickExitT(Vec3 o, Vec3 d, float t, float tMax) const {
        if (!hasMajorantBricks()) return tMax;
        const int bs = g->brickSize > 0 ? g->brickSize : 4;
        return ddaGridExitT(o, d, t, tMax, g->majOrigin, g->majCell * float(bs), g->brNx, g->brNy,
                            g->brNz);
    }

    SR_HD void occupancy(Vec3 p, float& minD, float& maxD) const {
        minD = 0.0f;
        maxD = 0.0f;
        if (!hasMajorantGrid() || g->majCell <= 1e-12f) {
            maxD = g->majorant;
            return;
        }
        const int ix = int(floorf((p.x - g->majOrigin.x) / g->majCell));
        const int iy = int(floorf((p.y - g->majOrigin.y) / g->majCell));
        const int iz = int(floorf((p.z - g->majOrigin.z) / g->majCell));
        if (ix < 0 || iy < 0 || iz < 0 || ix >= g->majNx || iy >= g->majNy || iz >= g->majNz) return;
        const int i = ix + g->majNx * (iy + g->majNy * iz);
        minD = g->majMin[i];
        maxD = g->majMax[i];
    }

    SR_HD float cellExitT(Vec3 o, Vec3 d, float t, float tMax) const {
        if (!hasMajorantGrid()) return tMax;
        return ddaGridExitT(o, d, t, tMax, g->majOrigin, g->majCell, g->majNx, g->majNy, g->majNz);
    }

    SR_HD float sampleOcc(Vec3 p) const { return srMax(0.0f, sampleGpuVolume(*g, p)); }
};

__device__ inline MediumSample sampleGpuFog(const GpuVolumeGrid& g, const MediumData& medium, Vec3 origin,
                                            Vec3 direction, float tMax, Rng& rng, Vec3& throughput) {
    GpuFogGrid view;
    view.g = &g;
    return sampleHeterogeneousFog(view, medium, origin, direction, tMax, rng, throughput);
}

__device__ inline MediumSample sampleGpuFogWl(const GpuVolumeGrid& g, const MediumData& medium, Vec3 origin,
                                              Vec3 direction, float tMax, Rng& rng, float* throughput,
                                              const float* lambda, int n) {
    GpuFogGrid view;
    view.g = &g;
    return sampleHeterogeneousFogWlWoodcock(view, medium, origin, direction, tMax, rng, throughput, lambda, n);
}

__device__ inline Vec3 gpuVolumeShadowTr(const LaunchParams& params, Vec3 origin, Vec3 direction, float dist,
                                         int mediumIndex, Rng& rng) {
    if (dist <= 1e-6f) return Vec3(1.0f);
    Vec3 Tr(1.0f);
    const SceneView& scene = params.scene;
    if (const MediumData* med = getMedium(scene, mediumIndex)) {
        if (med->type == 1) Tr = Tr * mediumShadowTr(*med, dist);
    }
    if (!params.volumes || params.volumeCount <= 0) return Tr;
    for (int vi = 0; vi < params.volumeCount; ++vi) {
        const GpuVolumeGrid& g = params.volumes[vi];
        if (!g.density || g.kind != 1) continue;
        const int medIndex = gpuMediumIndexForVolume(scene, vi);
        if (medIndex < 0) continue;
        const MediumData& med = scene.media[medIndex];
        GpuFogGrid view;
        view.g = &g;
        Tr = Tr * mediumShadowTrHeterogeneous(view, med, origin, direction, dist, rng);
        if (maxComponent(Tr) < 1e-5f) return Vec3(0.0f);
    }
    return Tr;
}

__device__ inline bool sphereTraceGpuSdf(const GpuVolumeGrid& g, Vec3 origin, Vec3 direction, float tMin,
                                         float tMax, float& tHit, Vec3& normal) {
    if (!g.density || g.kind != 0) return false;
    float tEnter = tMin;
    float tExit = tMax;
    if (!rayAabbInterval(origin, direction, g.bmin, g.bmax, tEnter, tExit)) return false;
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
