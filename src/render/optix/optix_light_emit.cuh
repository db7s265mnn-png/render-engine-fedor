// pbrt SampleLe for OptiX light-path spawn. No BSDF, no optixTrace.
#pragma once

#include "render/lights.h"
#include "render/optix/optix_geom.cuh"

namespace sol {

struct GpuLightEmit {
    Vec3 origin{0.0f};
    Vec3 dir{0.0f, 0.0f, 1.0f};
    Vec3 n{0.0f, 1.0f, 0.0f};
    Vec3 betaRgb{0.0f};
    int lightIndex = -1;
    int infinite = 0;
    int ok = 0;
};

__device__ inline Material gpuMaterialForCausticTransport(const SceneView& scene, int baseIndex) {
    return gpuMaterialForCausticSlot(scene, baseIndex);
}

__device__ inline float gpuLightTraceSplatClamp(const RenderSettingsData& settings) {
    if (settings.clampIndirect <= 0.0f) return 0.0f;
    const int w = settings.resolutionX > 0 ? settings.resolutionX : 1;
    const int h = settings.resolutionY > 0 ? settings.resolutionY : 1;
    return settings.clampIndirect * float(w) * float(h);
}

__device__ inline void gpuPhotonAimState(const GpuPhotonCluster*& clusters, int& n, float& mix) {
    const LaunchParams& p = launchParams();
    clusters = p.photonClusters;
    n = p.photonClusterCount;
    mix = p.photonAimMix;
    if (!clusters || n <= 0 || mix <= 0.0f) {
        clusters = nullptr;
        n = 0;
        mix = 0.0f;
        return;
    }
    if (n > kMaxGpuPhotonClusters) n = kMaxGpuPhotonClusters;
    mix = saturatef(mix);
}

// Infinite-light disk: mixture of the scene disk and caster projections.
__device__ inline Vec3 gpuSampleLeDisk(const SceneView& scene, Vec3 axis, Rng& rng, float& pdfPos) {
    const float r = sceneRadius(scene);
    const Vec3 center = scene.worldBounds.valid() ? scene.worldBounds.center() : Vec3(0.0f);
    const Frame wFrame(axis);
    const GpuPhotonCluster* clusters = nullptr;
    int n = 0;
    float mix = 0.0f;
    gpuPhotonAimState(clusters, n, mix);
    Vec3 pDisk;
    if (mix > 0.0f && rng.nextFloat() < mix) {
        const int ci = gpuPickPhotonCluster(clusters, n, rng.nextFloat());
        pDisk = gpuClusterDiskPoint(clusters[ci], center, axis, sampleConcentricDisk(rng.nextFloat(), rng.nextFloat()));
    } else {
        const Vec2 cd = sampleConcentricDisk(rng.nextFloat(), rng.nextFloat());
        pDisk = center + wFrame.toWorld(Vec3(cd.x, cd.y, 0.0f)) * r;
    }
    pdfPos = gpuPhotonAimMixtureDiskPdf(pDisk, center, axis, r, mix, clusters, n);
    return pDisk;
}

// Device copy of bdpt::startLightPath (pbrt SampleLe). Returns RGB beta = Le / pdfs.
// Direction / disk pdf is the Iray mixture (uniform + photon aiming), not the
// sampled technique alone.
__device__ inline GpuLightEmit gpuSampleLe(const SceneView& scene, Rng& rng) {
    GpuLightEmit out;
    float selectPdf = 0.0f;
    const int li = sampleLightIndex(scene, rng.nextFloat(), selectPdf);
    if (li < 0 || selectPdf <= 0.0f) return out;
    const LightData& l = scene.lights[li];
    out.lightIndex = li;

    const GpuPhotonCluster* clusters = nullptr;
    int nAim = 0;
    float mix = 0.0f;
    gpuPhotonAimState(clusters, nAim, mix);

    if (l.type == kLightDistant || l.type == kLightDome) {
        out.infinite = 1;
        if (l.type == kLightDistant) {
            const Vec3 axis = normalize(lightAxisZ(l));
            if (lengthSquared(axis) < 1e-12f) return out;
            const float halfAngle = radians(srMax(0.0f, l.angle)) * 0.5f;
            const bool deltaDir = halfAngle < kDistantDeltaHalfRad;
            float pdfDirSa = 0.0f;
            Vec3 emitDir;
            if (deltaDir) {
                emitDir = -axis;
                pdfDirSa = 1.0f;
            } else {
                const float cosThetaMax = cosf(halfAngle);
                const float omega = kTwoPi * (1.0f - cosThetaMax);
                if (omega <= 1e-20f) return out;
                const Frame frame(axis);
                emitDir = -normalize(frame.toWorld(
                    sampleUniformCone(rng.nextFloat(), rng.nextFloat(), cosThetaMax)));
                pdfDirSa = 1.0f / omega;
            }
            float pdfPos = 0.0f;
            const Vec3 pDisk = gpuSampleLeDisk(scene, axis, rng, pdfPos);
            const float r = sceneRadius(scene);
            out.origin = pDisk + axis * r;
            out.dir = emitDir;
            out.n = -emitDir;
            const Vec3 Le = deltaDir ? l.emittedRadiance() : lightRadiance(l);
            const float pdfFwd = selectPdf * pdfPos;
            if (isBlack(Le) || pdfFwd <= 0.0f || pdfDirSa <= 0.0f) return out;
            out.betaRgb = Le / srMax(1e-12f, pdfFwd * pdfDirSa);
            out.ok = 1;
            return out;
        }
        float pdfDir = 0.0f;
        Vec3 wiLocal;
        if (l.envIndex >= 0 && l.envIndex < scene.envMapCount && scene.envMaps[l.envIndex].sampled()) {
            wiLocal = envSample(scene.envMaps[l.envIndex], rng.nextFloat(), rng.nextFloat(), pdfDir);
        } else {
            wiLocal = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
            pdfDir = kInv4Pi;
        }
        if (pdfDir <= 0.0f) return out;
        const Vec3 wi = normalize(transformVector(l.xform, wiLocal));
        const Vec3 emitDir = -wi;
        float pdfPos = 0.0f;
        const Vec3 pDisk = gpuSampleLeDisk(scene, wi, rng, pdfPos);
        const float r = sceneRadius(scene);
        out.origin = pDisk + wi * r;
        out.dir = emitDir;
        out.n = -emitDir;
        const Vec3 Le = domeRadiance(scene, l, wi, /*nearestTexel=*/true);
        const float pdfFwd = selectPdf * pdfPos;
        if (isBlack(Le) || pdfFwd <= 0.0f) return out;
        out.betaRgb = Le / srMax(1e-12f, pdfFwd * pdfDir);
        out.ok = 1;
        return out;
    }

    if (l.type == kLightPoint) {
        out.origin = lightOrigin(l);
        out.n = Vec3(0.0f, 1.0f, 0.0f);
        bool aimed = false;
        if (mix > 0.0f && rng.nextFloat() < mix) {
            const int ci = gpuPickPhotonCluster(clusters, nAim, rng.nextFloat());
            aimed = gpuSamplePhotonAimDir(out.origin, clusters[ci], rng.nextFloat(), rng.nextFloat(),
                                          out.dir);
        }
        if (!aimed) out.dir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
        const float pdfDirSa = (1.0f - mix) * kInv4Pi + mix * gpuAimConePdf(out.origin, out.dir, clusters, nAim);
        const float pdfFwd = selectPdf;
        if (pdfDirSa <= 0.0f) return out;
        out.betaRgb = l.emittedRadiance() / srMax(1e-12f, pdfFwd * pdfDirSa);
        out.ok = !isBlack(out.betaRgb);
        return out;
    }

    float area = 0.0f;
    if (l.type == kLightRect) {
        const Vec3 pLocal((rng.nextFloat() - 0.5f) * l.width, (rng.nextFloat() - 0.5f) * l.height, 0.0f);
        out.origin = transformPoint(l.xform, pLocal);
        out.n = areaLightNormal(l);
        area = rectLightArea(l);
    } else if (l.type == kLightDisk) {
        const Vec2 d = sampleConcentricDisk(rng.nextFloat(), rng.nextFloat());
        out.origin = transformPoint(l.xform, Vec3(d.x * l.radius, d.y * l.radius, 0.0f));
        out.n = areaLightNormal(l);
        area = diskLightArea(l);
    } else {
        const Vec3 center = lightOrigin(l);
        const float radius = srMax(1e-5f, sphereLightRadius(l));
        const Vec3 dir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
        out.origin = center + dir * radius;
        out.n = dir;
        area = 4.0f * kPi * radius * radius;
    }
    if (area <= 1e-12f) return out;
    const float pdfFwd = selectPdf / area;
    Vec3 nEmit = out.n;
    if (l.twoSided && rng.nextFloat() < 0.5f) nEmit = -nEmit;

    bool aimed = false;
    if (mix > 0.0f && rng.nextFloat() < mix) {
        const int ci = gpuPickPhotonCluster(clusters, nAim, rng.nextFloat());
        Vec3 aimDir;
        if (gpuSamplePhotonAimDir(out.origin, clusters[ci], rng.nextFloat(), rng.nextFloat(), aimDir)) {
            const float cosN = dot(out.n, aimDir);
            if (l.twoSided) {
                if (fabsf(cosN) > 1e-4f) {
                    out.dir = aimDir;
                    aimed = true;
                }
            } else if (cosN > 1e-4f) {
                out.dir = aimDir;
                aimed = true;
            }
        }
    }
    if (!aimed) {
        const Frame frame(nEmit);
        const Vec3 local = sampleCosineHemisphere(rng.nextFloat(), rng.nextFloat());
        out.dir = normalize(frame.toWorld(local));
    }
    const float cosinePdf = fabsf(dot(nEmit, out.dir)) * kInvPi * (l.twoSided ? 0.5f : 1.0f);
    const float pdfDirSa = (1.0f - mix) * cosinePdf + mix * gpuAimConePdf(out.origin, out.dir, clusters, nAim);
    if (pdfDirSa <= 0.0f) return out;
    out.betaRgb = lightRadiance(l) * fabsf(dot(out.n, out.dir)) / srMax(1e-12f, pdfFwd * pdfDirSa);
    out.ok = !isBlack(out.betaRgb);
    return out;
}

}  // namespace sol
