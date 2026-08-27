// Iray Photoreal photon aiming (Keller et al. 2017, arXiv 1705.01263).
//
// Light-path SampleLe is a mixture of uniform emission and a discrete
// distribution over caustic-caster bounding spheres, weighted by the solid
// angle of each cluster as seen from the camera (correlates with screen-space
// photon density). The estimator stays unbiased because beta uses the mixture
// pdf, not the pdf of the technique that generated the sample.
//
// Device-safe: no STL in the helpers. Host cluster build is behind !__CUDACC__.
#pragma once

#include "core/math.h"

namespace sol {

constexpr int kMaxGpuPhotonClusters = 64;
// Probability of the aiming technique in the mixture. 0 = uniform SampleLe.
constexpr float kGpuPhotonAimMix = 0.85f;

struct GpuPhotonCluster {
    Vec3 center{0.0f};
    float radius = 0.0f;
    float weight = 0.0f;  // discrete pick probability; sums to 1 over the table
    int pad0 = 0;
};

SR_INL SR_HD Vec3 gpuProjectToPlane(Vec3 p, Vec3 planeC, Vec3 planeN) {
    return p - planeN * dot(p - planeC, planeN);
}

SR_INL SR_HD int gpuPickPhotonCluster(const GpuPhotonCluster* clusters, int n, float u) {
    if (!clusters || n <= 0) return 0;
    float x = saturatef(u);
    for (int i = 0; i < n; ++i) {
        x -= clusters[i].weight;
        if (x <= 0.0f) return i;
    }
    return n - 1;
}

// pbrt BoundSubtended: cos of the cone that covers a sphere from p.
// -1 when p is inside the sphere (full sphere / 4π).
SR_INL SR_HD float gpuSphereCosThetaMax(Vec3 p, Vec3 center, float radius) {
    const float r = srMax(0.0f, radius);
    const float d2 = lengthSquared(center - p);
    const float r2 = r * r;
    if (d2 <= r2) return -1.0f;
    return sqrtf(srMax(0.0f, 1.0f - r2 / d2));
}

SR_INL SR_HD float gpuUniformConePdf(float cosThetaMax) {
    if (cosThetaMax >= 1.0f) return 0.0f;
    if (cosThetaMax <= -1.0f) return kInv4Pi;
    const float omega = kTwoPi * (1.0f - cosThetaMax);
    return omega > 1e-20f ? 1.0f / omega : 0.0f;
}

// Solid angle of a bounding sphere as seen from the camera. Larger when the
// caster is closer / bigger — Iray uses this as a proxy for screen-space density.
SR_INL SR_HD float gpuScreenSolidAngle(Vec3 cam, Vec3 center, float radius) {
    const float r = srMax(0.0f, radius);
    const float d2 = lengthSquared(center - cam);
    const float r2 = r * r;
    if (d2 <= r2) return kTwoPi;
    const float cosMax = sqrtf(srMax(0.0f, 1.0f - r2 / d2));
    return kTwoPi * (1.0f - cosMax);
}

SR_INL SR_HD float gpuAimDiskPdf(Vec3 p, Vec3 planeC, Vec3 planeN, const GpuPhotonCluster* clusters,
                                 int n) {
    if (!clusters || n <= 0) return 0.0f;
    const float n2 = lengthSquared(planeN);
    if (n2 < 1e-20f) return 0.0f;
    const Vec3 nn = planeN * (1.0f / sqrtf(n2));
    float pdf = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float rad = srMax(1e-4f, clusters[i].radius);
        const Vec3 cProj = gpuProjectToPlane(clusters[i].center, planeC, nn);
        const Vec3 d = p - cProj;
        const Vec3 dPlane = d - nn * dot(d, nn);
        if (lengthSquared(dPlane) <= rad * rad + 1e-8f)
            pdf += clusters[i].weight / (kPi * rad * rad);
    }
    return pdf;
}

SR_INL SR_HD float gpuPhotonAimMixtureDiskPdf(Vec3 p, Vec3 planeC, Vec3 planeN, float sceneR, float mix,
                                              const GpuPhotonCluster* clusters, int n) {
    const float R = srMax(1e-4f, sceneR);
    const float n2 = lengthSquared(planeN);
    const Vec3 nn = n2 > 1e-20f ? planeN * (1.0f / sqrtf(n2)) : Vec3(0.0f, 1.0f, 0.0f);
    const Vec3 d = p - planeC;
    const Vec3 dPlane = d - nn * dot(d, nn);
    const float uniPdf = lengthSquared(dPlane) <= R * R + 1e-6f ? 1.0f / (kPi * R * R) : 0.0f;
    if (n <= 0 || !clusters || mix <= 0.0f) return uniPdf;
    return (1.0f - mix) * uniPdf + mix * gpuAimDiskPdf(p, planeC, nn, clusters, n);
}

SR_INL SR_HD Vec3 gpuClusterDiskPoint(const GpuPhotonCluster& c, Vec3 planeC, Vec3 planeN, Vec2 unitDisk) {
    const Frame f(planeN);
    const Vec3 cProj = gpuProjectToPlane(c.center, planeC, planeN);
    const float rad = srMax(1e-4f, c.radius);
    return cProj + f.toWorld(Vec3(unitDisk.x * rad, unitDisk.y * rad, 0.0f));
}

SR_INL SR_HD float gpuAimConePdf(Vec3 origin, Vec3 dir, const GpuPhotonCluster* clusters, int n) {
    if (!clusters || n <= 0) return 0.0f;
    const float dir2 = lengthSquared(dir);
    if (dir2 < 1e-20f) return 0.0f;
    const Vec3 w = dir * (1.0f / sqrtf(dir2));
    float pdf = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float rad = srMax(1e-4f, clusters[i].radius);
        const float cosMax = gpuSphereCosThetaMax(origin, clusters[i].center, rad);
        const float conePdf = gpuUniformConePdf(cosMax);
        if (conePdf <= 0.0f) continue;
        const Vec3 axis = clusters[i].center - origin;
        const float a2 = lengthSquared(axis);
        if (a2 < 1e-20f) {
            pdf += clusters[i].weight * conePdf;
            continue;
        }
        if (dot(w, axis * (1.0f / sqrtf(a2))) >= cosMax - 1e-5f) pdf += clusters[i].weight * conePdf;
    }
    return pdf;
}

SR_INL SR_HD bool gpuSamplePhotonAimDir(Vec3 origin, const GpuPhotonCluster& c, float u1, float u2,
                                        Vec3& dir) {
    const Vec3 delta = c.center - origin;
    const float d2 = lengthSquared(delta);
    const float rad = srMax(1e-4f, c.radius);
    if (d2 <= rad * rad) return false;
    const float cosMax = gpuSphereCosThetaMax(origin, c.center, rad);
    const Vec3 axis = delta * (1.0f / sqrtf(d2));
    if (cosMax >= 0.999999f) {
        dir = axis;
        return true;
    }
    const Frame f(axis);
    dir = normalize(f.toWorld(sampleUniformCone(u1, u2, cosMax)));
    return lengthSquared(dir) > 1e-12f;
}

}  // namespace sol

#if !defined(__CUDACC__)
#include <algorithm>
#include <vector>

#include "render/shading_bsdf.h"
#include "scene/types.h"

namespace sol {

// Host: caustic-caster AABBs as Iray clusters. Skips light proxies, disabled
// casters, and thin scene-sized planes (ground / walls that would steal every
// photon). Weights are screen solid angle from the camera; behind-camera
// clusters get weight 0.
inline int fillPhotonAimClusters(const SceneView& scene, GpuPhotonCluster* out, int maxN) {
    if (!out || maxN <= 0) return 0;
    const int cap = maxN < kMaxGpuPhotonClusters ? maxN : kMaxGpuPhotonClusters;
    if (!scene.instances || !scene.meshes || !scene.materials) return 0;
    if (scene.instanceCount <= 0 || scene.meshCount <= 0 || scene.materialCount <= 0) return 0;

    const Vec3 cam = transformPoint(scene.camera.cameraToWorld, Vec3(0.0f, 0.0f, 0.0f));
    const Vec3 viewDir = normalize(transformVector(scene.camera.cameraToWorld, Vec3(0.0f, 0.0f, -1.0f)));
    const float sceneR = scene.worldBounds.valid() ? scene.worldBounds.radius() : 1.0f;
    const float sceneRSafe = sceneR > 1e-3f ? sceneR : 1.0f;

    struct Cand {
        GpuPhotonCluster c;
        float w = 0.0f;
    };
    std::vector<Cand> cands;
    cands.reserve(size_t(scene.instanceCount));

    for (int i = 0; i < scene.instanceCount; ++i) {
        const InstanceData& inst = scene.instances[i];
        if (inst.lightIndex >= 0) continue;
        if (inst.meshIndex < 0 || inst.meshIndex >= scene.meshCount) continue;
        if (inst.materialIndex < 0 || inst.materialIndex >= scene.materialCount) continue;
        const Material& mat = scene.materials[inst.materialIndex];
        if (mat.contributeCaustics == 0) continue;
        const LobeWeights lw = computeLobes(mat);
        if (!isPhotonCausticCasterLobe(lw)) continue;

        const MeshView& mesh = scene.meshes[inst.meshIndex];
        Bounds3 ob;
        ob.lo = mesh.boundsLo;
        ob.hi = mesh.boundsHi;
        if (!ob.valid()) continue;
        const Bounds3 wb = transformBounds(inst.xform, ob);
        if (!wb.valid()) continue;
        const Vec3 extent = wb.extent();
        const float minE = minComponent(extent);
        const float maxE = maxComponent(extent);
        // Thin scene-filling slab (typical ground / wall). A glass Buddha is
        // roughly isotropic and much smaller than the scene radius.
        if (maxE > 1.2f * sceneRSafe && minE < 0.08f * maxE) continue;

        const Vec3 center = wb.center();
        const float radius = srMax(1e-4f, wb.radius());
        const Vec3 toC = center - cam;
        const float d = length(toC);
        if (d > radius && lengthSquared(viewDir) > 1e-12f && dot(toC, viewDir) < -radius) continue;

        const float w = gpuScreenSolidAngle(cam, center, radius);
        if (!(w > 0.0f) || !srIsFinite(w)) continue;
        Cand cand;
        cand.c.center = center;
        cand.c.radius = radius;
        cand.c.weight = 0.0f;
        cand.w = w;
        cands.push_back(cand);
    }

    if (cands.empty()) return 0;
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.w > b.w; });
    const int n = int(cands.size()) < cap ? int(cands.size()) : cap;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += cands[size_t(i)].w;
    if (!(sum > 0.0f)) {
        const float eq = 1.0f / float(n);
        for (int i = 0; i < n; ++i) {
            out[i] = cands[size_t(i)].c;
            out[i].weight = eq;
        }
        return n;
    }
    for (int i = 0; i < n; ++i) {
        out[i] = cands[size_t(i)].c;
        out[i].weight = cands[size_t(i)].w / sum;
    }
    return n;
}

}  // namespace sol
#endif
