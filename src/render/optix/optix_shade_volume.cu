// Cycles analogue: integrator_shade_volume.
// Homogeneous media + VDB fog (same residual-ratio tracker as Embree). No optixTrace.
#include "render/lights.h"
#include "render/optix/optix_volume.cuh"

namespace sol {

extern "C" __global__ void __raygen__shade_volume() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    if (path.queue != kQueueShadeVolume) return;

    const SceneView& scene = params.scene;
    GpuHit& hit = params.hits[pixel];
    GpuShadow& shadow = params.shadows[pixel];
    shadow.queue = kShadowIdle;

    const MediumData* med = getMedium(scene, path.mediumIndex);
    if (!med) {
        path.queue = hit.didHit ? kQueueShadeSurface : kQueueShadeBackground;
        return;
    }

    const float tMax = hit.didHit ? srMax(0.0f, hit.t) : 1.0e6f;
    MediumData walk = *med;
    if (scene.settings.volumeSimilarity != 0) walk = mediumWithVolumeSimilarity(*med, path.volumeScatters);

    MediumSample ms;
    if (walk.type == 2) {
        if (walk.volumeIndex >= 0 && walk.volumeIndex < params.volumeCount && params.volumes &&
            params.volumes[walk.volumeIndex].density && params.volumes[walk.volumeIndex].kind == 1) {
            ms = sampleGpuFog(params.volumes[walk.volumeIndex], walk, path.origin, path.direction, tMax,
                              path.rng, path.throughput);
        } else {
            path.queue = hit.didHit ? kQueueShadeSurface : kQueueShadeBackground;
            return;
        }
    } else {
        ms = sampleMediumHomogeneous(walk, tMax, path.rng, path.throughput);
    }

    if (ms.absorbed || isBlack(path.throughput) || !isFinite(path.throughput)) {
        path.queue = kQueueDead;
        return;
    }

    if (ms.scattered) {
        const Vec3 p = path.origin + path.direction * ms.t;
        const Vec3 wo = -path.direction;
        if (scene.lightCount > 0) {
            float selectPdf = 0.0f;
            const int lightIndex = sampleLightIndex(scene, p, path.rng.nextFloat(), selectPdf);
            LightSample ls;
            if (lightIndex >= 0 && selectPdf > 0.0f &&
                sampleLight(scene, lightIndex, p, path.rng.nextFloat(), path.rng.nextFloat(), ls) &&
                ls.pdf > 0.0f && !isBlack(ls.radiance)) {
                const float phase = henyeyGreenstein(dot(wo, ls.wi), walk.g);
                const float lightPdf = ls.pdf * selectPdf;
                const float mis = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, phase);
                Vec3 contrib =
                    path.throughput * ls.radiance * (phase / srMax(1e-8f, lightPdf)) * mis;
                contrib = clampFirefly(contrib, scene.settings.clampDirect);
                if (scene.lights[lightIndex].shadowEnable) {
                    float tSh = 1.0e8f;
                    if (ls.distance < 1.0e7f) tSh = ls.distance * (1.0f - 1e-3f);
                    enqueueShadow(shadow, p, ls.wi, tSh, contrib, path.mediumIndex);
                } else {
                    addRadiance(pixel, contrib);
                }
            }
        }

        float phasePdf = 0.0f;
        const Vec3 wi = sampleHenyeyGreenstein(wo, walk.g, path.rng.nextFloat(), path.rng.nextFloat(), phasePdf);
        path.origin = p;
        path.direction = wi;
        path.bsdfPdf = phasePdf;
        path.specularBounce = 0;
        ++path.depth;
        ++path.volumeScatters;
        if (path.depth >= srMax(1, scene.settings.maxDepth)) {
            path.queue = kQueueDead;
            return;
        }
        if (path.depth >= srMax(1, scene.settings.rrStartDepth)) {
            const float q = volumeRussianRouletteQ(path.throughput);
            if (path.rng.nextFloat() > q) {
                path.queue = kQueueDead;
                return;
            }
            path.throughput /= q;
        }
        path.queue = kQueueIntersectClosest;
        hit = GpuHit{};
        return;
    }

    path.origin = path.origin + path.direction * ms.t;
    // Exited the fog AABB before the surface — leave the medium (same as Embree).
    if (walk.type == 2 && (!hit.didHit || ms.t + 1e-4f < hit.t)) {
        path.mediumIndex = -1;
        ++path.hops;
        path.queue = kQueueIntersectClosest;
        hit = GpuHit{};
        return;
    }
    path.queue = hit.didHit ? kQueueShadeSurface : kQueueShadeBackground;
}

}  // namespace sol
