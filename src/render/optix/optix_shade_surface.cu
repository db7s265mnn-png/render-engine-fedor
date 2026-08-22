// Cycles analogue: integrator_shade_surface.
// No optixTrace — NEE writes a shadow ray for intersect_shadow.
#include "render/lights.h"
#include "render/optix/optix_bsdf.cuh"
#include "render/optix/optix_geom.cuh"

namespace sol {
namespace {

__device__ void enqueueShadow(GpuShadow& shadow, Vec3 origin, Vec3 dir, float tMax, Vec3 contrib) {
    if (!isFinite(contrib) || isBlack(contrib)) {
        shadow.queue = kShadowIdle;
        return;
    }
    shadow.origin = origin;
    shadow.direction = dir;
    shadow.tMax = tMax;
    shadow.contrib = contrib;
    shadow.occluded = 0;
    shadow.queue = kShadowTrace;
}

}  // namespace

extern "C" __global__ void __raygen__shade_surface() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    if (path.queue != kQueueShadeSurface) return;

    const SceneView& scene = params.scene;
    const GpuHit& hit = params.hits[pixel];
    GpuShadow& shadow = params.shadows[pixel];
    shadow.queue = kShadowIdle;

    Surf si;
    if (!buildSurf(scene, hit, path.origin, path.direction, si)) {
        path.queue = kQueueDead;
        return;
    }

    if (si.lightIndex >= 0 && path.depth == 0) {
        const InstanceData& inst = scene.instances[si.instanceIndex];
        if (!inst.visibleCamera) {
            path.origin = offsetRay(si.p, si.ng, path.direction);
            if (++path.hops > 16) {
                path.queue = kQueueDead;
                return;
            }
            path.queue = kQueueIntersectClosest;
            return;
        }
    }

    if (si.lightIndex >= 0) {
        const LightData& light = scene.lights[si.lightIndex];
        const Vec3 lightN = light.type == kLightSphere ? si.ng : areaLightNormal(light);
        Vec3 emitted = areaLightEmission(scene, light, path.direction, lightN);
        if (!isBlack(emitted)) {
            float weight = 1.0f;
            if (!path.specularBounce) {
                const float lp =
                    lightPdfDirection(scene, si.lightIndex, path.origin, path.direction, si.p, lightN) *
                    lightSelectionPdfIndex(scene, path.origin, si.lightIndex);
                weight = powerHeuristic(1.0f, path.bsdfPdf, 1.0f, lp);
            }
            Vec3 contrib = path.throughput * emitted * weight;
            if (path.depth > 0 && !path.specularBounce)
                contrib = clampFirefly(contrib, scene.settings.clampDirect);
            addRadiance(pixel, contrib);
        }
        path.queue = kQueueDead;
        return;
    }

    if (si.materialIndex < 0 || si.materialIndex >= scene.materialCount || !scene.materials) {
        path.queue = kQueueDead;
        return;
    }

    Material mat = optixpt::evaluateMaps(scene, scene.materials[si.materialIndex], si.uv, si.ns);
    if (mat.transmission <= 0.0f && mat.doubleSided && dot(si.ns, -path.direction) < 0.0f) {
        si.ns = -si.ns;
        si.ng = -si.ng;
    }
    if (mat.emissionStrength > 0.0f && !isBlack(mat.emissionColor)) {
        const bool frontFacing = dot(si.ns, -path.direction) > 0.0f;
        if (frontFacing || mat.doubleSided)
            addRadiance(pixel, path.throughput * mat.emissionColor * mat.emissionStrength);
    }
    if (mat.opacity <= 1e-6f || (mat.opacity < 0.999f && path.rng.nextFloat() > mat.opacity)) {
        path.origin = offsetRay(si.p, si.ng, path.direction);
        if (++path.hops > 32) {
            path.queue = kQueueDead;
            return;
        }
        path.queue = kQueueIntersectClosest;
        return;
    }

    const int maxDepth = srMax(1, scene.settings.maxDepth);
    if (path.depth >= maxDepth) {
        path.queue = kQueueDead;
        return;
    }

    const Vec3 wo = -path.direction;
    const Frame frame(si.ns);

    if (scene.lightCount > 0) {
        float selectPdf = 0.0f;
        const int lightIndex = sampleLightIndex(scene, si.p, path.rng.nextFloat(), selectPdf);
        LightSample ls;
        if (lightIndex >= 0 && selectPdf > 0.0f &&
            sampleLight(scene, lightIndex, si.p, path.rng.nextFloat(), path.rng.nextFloat(), ls) &&
            ls.pdf > 0.0f && !isBlack(ls.radiance) &&
            optixpt::shadingNormalConsistent(si.ng, si.ns, wo, ls.wi)) {
            const Vec3 woLocal = frame.toLocal(wo);
            const Vec3 wiLocal = frame.toLocal(ls.wi);
            const optixpt::BsdfEval be = optixpt::bsdfEvalLocal(mat, woLocal, wiLocal);
            if (be.pdf > 0.0f && !isBlack(be.f)) {
                const float lightPdf = ls.pdf * selectPdf;
                const float mis = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, be.pdf);
                Vec3 contrib = path.throughput * ls.radiance * be.f * (fabsf(wiLocal.z) / lightPdf) * mis;
                if (path.depth > 0 && !path.specularBounce)
                    contrib = clampFirefly(contrib, scene.settings.clampDirect);
                if (scene.lights[lightIndex].shadowEnable) {
                    const Vec3 shadowOrigin = offsetRay(si.p, si.ng, ls.wi);
                    float tMax = 1.0e8f;
                    if (ls.distance < 1.0e7f) tMax = ls.distance * (1.0f - 1e-3f);
                    enqueueShadow(shadow, shadowOrigin, ls.wi, tMax, contrib);
                } else {
                    addRadiance(pixel, contrib);
                }
            }
        }
    }

    const optixpt::BsdfSample bs =
        optixpt::bsdfSampleLocal(mat, frame.toLocal(wo), path.rng.nextFloat(), path.rng.nextFloat(),
                                 path.rng.nextFloat(), path.rng.nextFloat());
    if (bs.pdf <= 0.0f || isBlack(bs.weight)) {
        path.queue = kQueueDead;
        return;
    }
    const Vec3 wiWorld = normalize(frame.toWorld(bs.wi));
    if (!optixpt::shadingNormalConsistent(si.ng, si.ns, wo, wiWorld)) {
        path.queue = kQueueDead;
        return;
    }

    path.throughput *= bs.weight;
    if (!isFinite(path.throughput) || isBlack(path.throughput)) {
        path.queue = kQueueDead;
        return;
    }
    path.origin = offsetRay(si.p, si.ng, wiWorld);
    path.direction = wiWorld;
    path.bsdfPdf = bs.pdf;
    path.specularBounce = bs.specular ? 1 : 0;
    ++path.depth;

    if (path.depth >= srMax(1, scene.settings.rrStartDepth)) {
        const float q = clampf(maxComponent(path.throughput), 0.05f, 1.0f);
        if (path.rng.nextFloat() > q) {
            path.queue = kQueueDead;
            return;
        }
        path.throughput /= q;
    }

    path.queue = kQueueIntersectClosest;
}

}  // namespace sol
