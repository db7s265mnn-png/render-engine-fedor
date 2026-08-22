// Heterogeneous fog free-flight + residual-ratio shadow Tr (PBRT 4ed §11.2).
// Shared by Embree (OpenVDB) and OptiX (uploaded occupancy + the same majorant
// bricks). Grid adapters must expose:
//   Vec3 bmin()/bmax(), float majorant()
//   bool hasMajorantGrid(), bool hasMajorantBricks()
//   bool brickEmpty(Vec3), float brickExitT(o,d,t,tMax)
//   void occupancy(Vec3, float& minD, float& maxD)
//   float cellExitT(o,d,t,tMax)
//   float sampleOcc(Vec3)  // occupancy ≥ 0
//
// No lambdas: these helpers are __host__ __device__ and nvcc rejects lambdas there.
#pragma once

#include "core/rng.h"
#include "render/volume.h"

namespace sol {

SR_INL SR_HD bool rayAabbInterval(Vec3 origin, Vec3 direction, Vec3 lo, Vec3 hi, float& tEnter,
                                  float& tExit) {
    float t0 = -1.0e30f;
    float t1 = 1.0e30f;
    const float* o = &origin.x;
    const float* d = &direction.x;
    const float* b0 = &lo.x;
    const float* b1 = &hi.x;
    for (int axis = 0; axis < 3; ++axis) {
        const float od = d[axis];
        if (fabsf(od) < 1e-20f) {
            if (o[axis] < b0[axis] || o[axis] > b1[axis]) return false;
            continue;
        }
        float ta = (b0[axis] - o[axis]) / od;
        float tb = (b1[axis] - o[axis]) / od;
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

SR_INL SR_HD int ddaBinClamp(float w, float o, float cell, int dim) {
    int i = int(floorf((w - o) / cell));
    if (i < 0) i = 0;
    if (i >= dim) i = dim - 1;
    return i;
}

SR_INL SR_HD float ddaAxisExit(float o, float d, float orig, float cell, int i, float tMax) {
    if (fabsf(d) < 1e-20f) return tMax;
    const float next = (d > 0.0f) ? (orig + float(i + 1) * cell) : (orig + float(i) * cell);
    return (next - o) / d;
}

SR_INL SR_HD float ddaGridExitT(Vec3 origin, Vec3 direction, float t, float tMax, Vec3 org, float cell,
                                int nx, int ny, int nz) {
    if (tMax <= t || cell <= 1e-12f || nx <= 0 || ny <= 0 || nz <= 0) return tMax;
    const float lookT = t + 1e-5f;
    const Vec3 p = origin + direction * lookT;
    const Vec3 hi(org.x + float(nx) * cell, org.y + float(ny) * cell, org.z + float(nz) * cell);
    if (p.x < org.x || p.y < org.y || p.z < org.z || p.x >= hi.x || p.y >= hi.y || p.z >= hi.z)
        return tMax;
    const int ix = ddaBinClamp(p.x, org.x, cell, nx);
    const int iy = ddaBinClamp(p.y, org.y, cell, ny);
    const int iz = ddaBinClamp(p.z, org.z, cell, nz);
    float tHit = tMax;
    tHit = srMin(tHit, ddaAxisExit(origin.x, direction.x, org.x, cell, ix, tMax));
    tHit = srMin(tHit, ddaAxisExit(origin.y, direction.y, org.y, cell, iy, tMax));
    tHit = srMin(tHit, ddaAxisExit(origin.z, direction.z, org.z, cell, iz, tMax));
    if (!(tHit > t)) tHit = t + 1e-4f;
    return srMin(tHit, tMax);
}

#if defined(__CUDACC__)
constexpr int kNullCollisionMaxIters = 1 << 16;
#else
constexpr int kNullCollisionMaxIters = 1 << 20;
#endif

SR_INL SR_HD bool fogInteract(const MediumData& medium, float densityScale, float occupancy, float tHit,
                              Rng& rng, Vec3& throughput, MediumSample& out) {
    const float dens = srMax(0.0f, occupancy) * densityScale;
    const Vec3 sigmaA = medium.sigmaA * dens;
    const Vec3 sigmaS = medium.sigmaS * dens;
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
}

SR_INL SR_HD bool fogCollide(const MediumData& medium, float densityScale, float occupancy, float majorant,
                             float tHit, Rng& rng, Vec3& throughput, MediumSample& out) {
    const float dens = srMax(0.0f, occupancy) * densityScale;
    const Vec3 sigmaT = (medium.sigmaA + medium.sigmaS) * dens;
    const float stHero = srMax(sigmaT.x, srMax(sigmaT.y, sigmaT.z));
    if (rng.nextFloat() >= stHero / srMax(majorant, 1e-12f)) return false;
    return fogInteract(medium, densityScale, occupancy, tHit, rng, throughput, out);
}

template <typename Grid>
SR_INL SR_HD MediumSample sampleHeterogeneousFog(const Grid& grid, const MediumData& medium, Vec3 origin,
                                                 Vec3 direction, float tMax, Rng& rng, Vec3& throughput) {
    MediumSample out;
    if (tMax <= 0.0f) {
        out.t = tMax;
        return out;
    }

    float aabbEnter = 0.0f;
    float aabbExit = tMax;
    if (rayAabbInterval(origin, direction, grid.bmin(), grid.bmax(), aabbEnter, aabbExit))
        tMax = srMin(tMax, srMax(0.0f, aabbExit));
    if (tMax <= 0.0f) {
        out.t = 0.0f;
        return out;
    }
    float t = srMax(0.0f, aabbEnter);
    if (t >= tMax) {
        out.t = tMax;
        return out;
    }

    const float densityScale = srMax(0.0f, medium.density);
    const Vec3 sigmaT0 = medium.sigmaA + medium.sigmaS;
    const float baseMaj = srMax(sigmaT0.x, srMax(sigmaT0.y, sigmaT0.z));
    if (baseMaj <= 1e-12f || densityScale <= 0.0f) {
        out.t = tMax;
        return out;
    }

    const float st0 = baseMaj * densityScale;

    for (int iter = 0; iter < kNullCollisionMaxIters && t < tMax; ++iter) {
        const Vec3 pLook = origin + direction * (t + 1e-5f);
        if (grid.hasMajorantBricks() && grid.brickEmpty(pLook)) {
            const float tBr = grid.brickExitT(origin, direction, t, tMax);
            t = (tBr > t) ? tBr : t + 1e-4f;
            continue;
        }
        float minD = 0.0f;
        float maxD = grid.majorant();
        grid.occupancy(pLook, minD, maxD);
        const float tCell = grid.hasMajorantGrid() ? grid.cellExitT(origin, direction, t, tMax) : tMax;
        if (!(tCell > t)) {
            t += 1e-4f;
            continue;
        }
        if (maxD <= 1e-8f) {
            t = tCell;
            continue;
        }

        const float majorant = srMax(1e-8f, baseMaj * maxD * densityScale);
        const bool homog = (maxD - minD) <= (1e-3f * srMax(maxD, 1e-6f) + 1e-4f);

        if (homog) {
            const float occ = 0.5f * (minD + maxD);
            const float u = srMax(1e-6f, 1.0f - rng.nextFloat());
            const float tHit = t + (-logf(u) / majorant);
            if (tHit >= tCell) {
                t = tCell;
                continue;
            }
            if (fogCollide(medium, densityScale, occ, majorant, tHit, rng, throughput, out)) return out;
            t = tHit;
            continue;
        }

        const float muC = st0 * srMax(0.0f, minD);
        const float residualMaj = srMax(0.0f, majorant - muC);
        float tLocal = t;
        const float tCtrl =
            (muC > 1e-8f) ? t + (-logf(srMax(1e-6f, 1.0f - rng.nextFloat())) / muC) : tCell + 1.0f;
        bool leftCell = false;
        while (iter < kNullCollisionMaxIters) {
            float tRes = tCell + 1.0f;
            if (residualMaj > 1e-8f) {
                tLocal += -logf(srMax(1e-6f, 1.0f - rng.nextFloat())) / residualMaj;
                tRes = tLocal;
            }
            const float tEvent = srMin(tCtrl, tRes);
            if (tEvent >= tCell) {
                t = tCell;
                leftCell = true;
                break;
            }
            if (tCtrl <= tRes) {
                fogInteract(medium, densityScale, minD, tCtrl, rng, throughput, out);
                return out;
            }
            const float occ = clampf(grid.sampleOcc(origin + direction * tRes), minD, maxD);
            const float pReal = (st0 * occ - muC) / srMax(residualMaj, 1e-12f);
            ++iter;
            if (rng.nextFloat() >= clampf(pReal, 0.0f, 1.0f)) continue;
            fogInteract(medium, densityScale, occ, tRes, rng, throughput, out);
            return out;
        }
        if (leftCell) continue;
        break;
    }
    out.t = tMax;
    return out;
}

template <typename Grid>
SR_INL SR_HD Vec3 mediumShadowTrHeterogeneous(const Grid& grid, const MediumData& medium, Vec3 origin,
                                             Vec3 direction, float dist, Rng& rng) {
    if (dist <= 0.0f) return Vec3(1.0f);

    const float densityScale = srMax(0.0f, medium.density);
    const Vec3 sigmaT0 = medium.sigmaA + medium.sigmaS;
    const float baseMaj = srMax(sigmaT0.x, srMax(sigmaT0.y, sigmaT0.z));
    if (baseMaj <= 1e-12f || densityScale <= 0.0f) return Vec3(1.0f);

    float aabbEnter = 0.0f;
    float aabbExit = dist;
    if (!rayAabbInterval(origin, direction, grid.bmin(), grid.bmax(), aabbEnter, aabbExit))
        return Vec3(1.0f);
    const float tEnd = srMin(dist, srMax(0.0f, aabbExit));
    float t = srMax(0.0f, aabbEnter);
    if (tEnd <= t) return Vec3(1.0f);

    Vec3 Tr(1.0f);
    int iter = 0;
    while (t < tEnd && iter < kNullCollisionMaxIters) {
        const Vec3 pLook = origin + direction * (t + 1e-5f);
        if (grid.hasMajorantBricks() && grid.brickEmpty(pLook)) {
            const float tBr = grid.brickExitT(origin, direction, t, tEnd);
            t = (tBr > t) ? tBr : t + 1e-4f;
            continue;
        }
        float minD = 0.0f;
        float maxD = grid.majorant();
        grid.occupancy(pLook, minD, maxD);
        const float tCell =
            grid.hasMajorantGrid() ? grid.cellExitT(origin, direction, t, tEnd) : tEnd;
        const float L = tCell - t;
        if (!(L > 1e-8f)) {
            t += 1e-4f;
            ++iter;
            continue;
        }
        if (maxD <= 1e-8f) {
            t = tCell;
            continue;
        }

        const Vec3 muC = sigmaT0 * (srMax(0.0f, minD) * densityScale);
        const float majorant = srMax(1e-8f, baseMaj * maxD * densityScale);
        const float muCMin = srMin(muC.x, srMin(muC.y, muC.z));
        const float residualMaj = srMax(0.0f, majorant - muCMin);

        Tr = Vec3(Tr.x * expf(-muC.x * L), Tr.y * expf(-muC.y * L), Tr.z * expf(-muC.z * L));
        if (maxComponent(Tr) < 1e-6f) return Vec3(0.0f);

        if (residualMaj > 1e-8f) {
            float tRes = t;
            while (iter < kNullCollisionMaxIters) {
                const float u = srMax(1e-6f, 1.0f - rng.nextFloat());
                tRes += -logf(u) / residualMaj;
                if (tRes >= tCell) break;
                const float occ = clampf(grid.sampleOcc(origin + direction * tRes), minD, maxD);
                const Vec3 sigmaT = sigmaT0 * (occ * densityScale);
                const float nx = residualMaj - (sigmaT.x - muC.x);
                const float ny = residualMaj - (sigmaT.y - muC.y);
                const float nz = residualMaj - (sigmaT.z - muC.z);
                Tr = Vec3(Tr.x * clampf(nx / residualMaj, 0.0f, 1.0f),
                          Tr.y * clampf(ny / residualMaj, 0.0f, 1.0f),
                          Tr.z * clampf(nz / residualMaj, 0.0f, 1.0f));
                ++iter;
                if (maxComponent(Tr) < 1e-6f) return Vec3(0.0f);
            }
        }
        t = tCell;
        ++iter;
    }
    return Tr;
}

}  // namespace sol
