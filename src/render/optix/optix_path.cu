// OptiX path-tracing kernel. Dedicated translation unit — does NOT include
// render/integrator.h.
//
// Why this is a separate file (Cycles / Karma XPU / Iray):
//
// * Cycles GPU is wavefront path tracing: OptiX raygen programs are tiny
//   wrappers around intersect_closest / intersect_shadow. Shading lives in
//   other kernels. Heavy paths (MNEE, OSL ray-trace) are a second .cu that
//   is only compiled and loaded when the scene needs them. Shader graphs
//   are SVM bytecode, not inlined into PTX.
// * Karma XPU splits "render kernels" from "user shaders". Kernels compile
//   once and cache; shader graphs compile on first use, not at engine build.
// * Iray JITs MDL to PTX callables per material (optixDirectCall).
//
// This engine does not ship an SVM / MDL compiler. The matching subset:
// one OptiX module that is only unidirectional path tracing + basic surface
// (maps + Lambert/GGX/glass). Volumes, SSS, procedurals, BDPT, MNEE,
// wireframe, AO, and polynomial-optics cameras stay on Embree.
#include "render/optix/optix_common.cuh"
#include "render/blue_noise.h"
#include "render/lights.h"
#include "render/optix/optix_bsdf.cuh"

namespace sol {
namespace {

__device__ inline float3 toFloat3(Vec3 v) { return make_float3(v.x, v.y, v.z); }

struct Hit {
    float t = kFloatMax;
    int instanceIndex = -1;
    uint32_t primIndex = 0;
    float u = 0.0f;
    float v = 0.0f;
};

struct Surf {
    Vec3 p{0.0f};
    Vec3 ng{0.0f, 0.0f, 1.0f};
    Vec3 ns{0.0f, 0.0f, 1.0f};
    Vec2 uv{0.0f, 0.0f};
    int instanceIndex = -1;
    int materialIndex = -1;
    int lightIndex = -1;
};

__device__ bool intersect(Vec3 origin, Vec3 direction, float tMax, Hit& hit) {
    const LaunchParams& params = launchParams();
    unsigned int didHit = 0;
    unsigned int tBits = 0;
    unsigned int instance = 0;
    unsigned int primitive = 0;
    unsigned int uBits = 0;
    unsigned int vBits = 0;
    optixTrace(params.traversable, toFloat3(origin), toFloat3(direction), 0.0f, tMax, 0.0f,
               OptixVisibilityMask(kVisAll), OPTIX_RAY_FLAG_NONE, kRayTypeRadiance, kRayTypeCount,
               kRayTypeRadiance, didHit, tBits, instance, primitive, uBits, vBits);
    if (!didHit) return false;
    hit.t = __uint_as_float(tBits);
    hit.instanceIndex = int(instance);
    hit.primIndex = primitive;
    hit.u = __uint_as_float(uBits);
    hit.v = __uint_as_float(vBits);
    return true;
}

__device__ bool occluded(Vec3 origin, Vec3 direction, float tMax) {
    const LaunchParams& params = launchParams();
    unsigned int blocked = 1;
    optixTrace(params.traversable, toFloat3(origin), toFloat3(direction), 0.0f, tMax, 0.0f,
               OptixVisibilityMask(kVisShadow),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT |
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               kRayTypeShadow, kRayTypeCount, kRayTypeShadow, blocked);
    return blocked != 0;
}

__device__ Vec3 offsetRay(Vec3 p, Vec3 n, Vec3 dir) {
    const float scale = 1.0f + srMax(fabsf(p.x), srMax(fabsf(p.y), fabsf(p.z)));
    const Vec3 offset = n * (kRayEpsilon * scale);
    return dot(dir, n) > 0.0f ? p + offset : p - offset;
}

// Pinhole only. Thin-lens DoF and polynomial optics stay on Embree.
__device__ void pinholeRay(const SceneView& scene, float pixelX, float pixelY, Vec3& origin,
                           Vec3& direction) {
    const CameraData& cam = scene.camera;
    const float resX = float(srMax(1, scene.settings.resolutionX));
    const float resY = float(srMax(1, scene.settings.resolutionY));
    const float sensorHeight = cam.sensorWidth * (resY / resX);
    const float sx = (pixelX / resX - 0.5f) * cam.sensorWidth;
    const float sy = (0.5f - pixelY / resY) * sensorHeight;
    const Vec3 dirCam = normalize(Vec3(sx, sy, -srMax(1e-3f, cam.focalLength)));
    origin = transformPoint(cam.cameraToWorld, Vec3(0.0f, 0.0f, 0.0f));
    direction = normalize(transformVector(cam.cameraToWorld, dirCam));
}

__device__ bool buildSurf(const SceneView& scene, const Hit& hit, Vec3 origin, Vec3 dir, Surf& si) {
    if (hit.instanceIndex < 0 || hit.instanceIndex >= scene.instanceCount) return false;
    const InstanceData& inst = scene.instances[hit.instanceIndex];
    if (inst.meshIndex < 0 || inst.meshIndex >= scene.meshCount) return false;
    const MeshView& mesh = scene.meshes[inst.meshIndex];
    if (hit.primIndex >= mesh.triangleCount || !mesh.indices || !mesh.positions) return false;

    const uint32_t i0 = mesh.indices[hit.primIndex * 3 + 0];
    const uint32_t i1 = mesh.indices[hit.primIndex * 3 + 1];
    const uint32_t i2 = mesh.indices[hit.primIndex * 3 + 2];
    if (i0 >= mesh.vertexCount || i1 >= mesh.vertexCount || i2 >= mesh.vertexCount) return false;
    const Vec3 p0 = mesh.positions[i0];
    const Vec3 p1 = mesh.positions[i1];
    const Vec3 p2 = mesh.positions[i2];
    const float w = 1.0f - hit.u - hit.v;

    si.p = origin + dir * hit.t;
    Vec3 ngLocal = cross(p1 - p0, p2 - p0);
    si.ng = normalize(transformNormalWithInverse(inst.xformInv, ngLocal));
    if (mesh.normals) {
        const Vec3 nLocal = mesh.normals[i0] * w + mesh.normals[i1] * hit.u + mesh.normals[i2] * hit.v;
        Vec3 ns = transformNormalWithInverse(inst.xformInv, nLocal);
        si.ns = lengthSquared(ns) > 0.0f ? normalize(ns) : si.ng;
    } else {
        si.ns = si.ng;
    }
    if (mesh.uvs) {
        const Vec2 uv0 = mesh.uvs[i0], uv1 = mesh.uvs[i1], uv2 = mesh.uvs[i2];
        si.uv = Vec2(uv0.x * w + uv1.x * hit.u + uv2.x * hit.v, uv0.y * w + uv1.y * hit.u + uv2.y * hit.v);
    }
    si.instanceIndex = hit.instanceIndex;
    si.materialIndex = inst.materialIndex;
    si.lightIndex = inst.lightIndex;
    if (dot(si.ns, si.ng) < 0.0f) si.ng = -si.ng;
    return true;
}

__device__ Vec3 clampFirefly(Vec3 contrib, float clampValue) {
    if (clampValue <= 0.0f || !isFinite(contrib)) return isFinite(contrib) ? contrib : Vec3(0.0f);
    const float m = maxComponent(contrib);
    if (m > clampValue) contrib *= clampValue / m;
    return contrib;
}

__device__ Vec3 nextEvent(const SceneView& scene, const Surf& si, const Material& mat, const Frame& frame,
                          Vec3 wo, Rng& rng) {
    if (scene.lightCount <= 0) return Vec3(0.0f);
    float selectPdf = 0.0f;
    const int lightIndex = sampleLightIndex(scene, si.p, rng.nextFloat(), selectPdf);
    if (lightIndex < 0 || selectPdf <= 0.0f) return Vec3(0.0f);
    LightSample ls;
    if (!sampleLight(scene, lightIndex, si.p, rng.nextFloat(), rng.nextFloat(), ls)) return Vec3(0.0f);
    if (ls.pdf <= 0.0f || isBlack(ls.radiance)) return Vec3(0.0f);
    if (!optixpt::shadingNormalConsistent(si.ng, si.ns, wo, ls.wi)) return Vec3(0.0f);
    const Vec3 woLocal = frame.toLocal(wo);
    const Vec3 wiLocal = frame.toLocal(ls.wi);
    const optixpt::BsdfEval be = optixpt::bsdfEvalLocal(mat, woLocal, wiLocal);
    if (be.pdf <= 0.0f || isBlack(be.f)) return Vec3(0.0f);

    float visibility = 1.0f;
    if (scene.lights[lightIndex].shadowEnable) {
        const Vec3 shadowOrigin = offsetRay(si.p, si.ng, ls.wi);
        float tMax = 1.0e8f;
        if (ls.distance < 1.0e7f) tMax = ls.distance * (1.0f - 1e-3f);
        if (occluded(shadowOrigin, ls.wi, tMax)) return Vec3(0.0f);
    }
    const float lightPdf = ls.pdf * selectPdf;
    const float mis = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, be.pdf);
    return ls.radiance * be.f * (fabsf(wiLocal.z) / lightPdf) * visibility * mis;
}

__device__ Vec3 pathTrace(const SceneView& scene, Vec3 origin, Vec3 direction, Rng& rng) {
    Vec3 radiance(0.0f);
    Vec3 throughput(1.0f);
    float bsdfPdf = 0.0f;
    bool specularBounce = true;
    int depth = 0;
    int hops = 0;
    const int maxDepth = srMax(1, scene.settings.maxDepth);

    while (depth <= maxDepth) {
        Hit hit;
        const bool didHit = intersect(origin, direction, kFloatMax, hit);
        if (!didHit) {
            if (scene.domeLightIndex >= 0) {
                const LightData& dome = scene.lights[scene.domeLightIndex];
                const bool primary = depth == 0;
                if (!(primary && (!scene.settings.envVisibleCamera || !dome.visibleCamera))) {
                    Vec3 envL = domeRadiance(scene, dome, direction, /*nearestTexel=*/depth > 0);
                    if (!isBlack(envL)) {
                        float weight = 1.0f;
                        if (!specularBounce) {
                            const float lp =
                                lightPdfDirection(scene, scene.domeLightIndex, origin, direction, origin,
                                                  direction) *
                                lightSelectionPdfIndex(scene, origin, scene.domeLightIndex);
                            weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                        }
                        Vec3 contrib = throughput * envL * weight;
                        if (depth > 0) contrib = clampFirefly(contrib, scene.settings.clampDirect);
                        radiance += contrib;
                    }
                }
            }
            break;
        }

        Surf si;
        if (!buildSurf(scene, hit, origin, direction, si)) break;

        if (si.lightIndex >= 0 && depth == 0) {
            const InstanceData& inst = scene.instances[si.instanceIndex];
            if (!inst.visibleCamera) {
                origin = offsetRay(si.p, si.ng, direction);
                if (++hops > 16) break;
                continue;
            }
        }

        if (si.lightIndex >= 0) {
            const LightData& light = scene.lights[si.lightIndex];
            const Vec3 lightN = light.type == kLightSphere ? si.ng : areaLightNormal(light);
            Vec3 emitted = areaLightEmission(scene, light, direction, lightN);
            if (!isBlack(emitted)) {
                float weight = 1.0f;
                if (!specularBounce) {
                    const float lp = lightPdfDirection(scene, si.lightIndex, origin, direction, si.p, lightN) *
                                     lightSelectionPdfIndex(scene, origin, si.lightIndex);
                    weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                }
                Vec3 contrib = throughput * emitted * weight;
                if (depth > 0 && !specularBounce)
                    contrib = clampFirefly(contrib, scene.settings.clampDirect);
                radiance += contrib;
            }
            break;
        }

        if (si.materialIndex < 0 || si.materialIndex >= scene.materialCount || !scene.materials) break;
        Material mat = optixpt::evaluateMaps(scene, scene.materials[si.materialIndex], si.uv, si.ns);
        if (mat.transmission <= 0.0f && mat.doubleSided && dot(si.ns, -direction) < 0.0f) {
            si.ns = -si.ns;
            si.ng = -si.ng;
        }
        if (mat.emissionStrength > 0.0f && !isBlack(mat.emissionColor)) {
            const bool frontFacing = dot(si.ns, -direction) > 0.0f;
            if (frontFacing || mat.doubleSided)
                radiance += throughput * mat.emissionColor * mat.emissionStrength;
        }
        if (mat.opacity <= 1e-6f || (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity)) {
            origin = offsetRay(si.p, si.ng, direction);
            if (++hops > 32) break;
            continue;
        }
        if (depth >= maxDepth) break;

        const Vec3 wo = -direction;
        const Frame frame(si.ns);
        const Vec3 nee = nextEvent(scene, si, mat, frame, wo, rng);
        Vec3 contrib = throughput * nee;
        if (depth > 0 && !specularBounce) contrib = clampFirefly(contrib, scene.settings.clampDirect);
        radiance += contrib;

        const optixpt::BsdfSample bs =
            optixpt::bsdfSampleLocal(mat, frame.toLocal(wo), rng.nextFloat(), rng.nextFloat(),
                                     rng.nextFloat(), rng.nextFloat());
        if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;
        const Vec3 wiWorld = normalize(frame.toWorld(bs.wi));
        if (!optixpt::shadingNormalConsistent(si.ng, si.ns, wo, wiWorld)) break;

        throughput *= bs.weight;
        if (!isFinite(throughput) || isBlack(throughput)) break;
        origin = offsetRay(si.p, si.ng, wiWorld);
        direction = wiWorld;
        bsdfPdf = bs.pdf;
        specularBounce = bs.specular;
        ++depth;

        if (depth >= srMax(1, scene.settings.rrStartDepth)) {
            const float q = clampf(maxComponent(throughput), 0.05f, 1.0f);
            if (rng.nextFloat() > q) break;
            throughput /= q;
        }
    }

    if (!isFinite(radiance)) return Vec3(0.0f);
    return radiance;
}

}  // namespace

// extern "C" keeps the OptiX entry name unmangled; the function must live in
// namespace sol so it can see the anonymous-namespace helpers above.
extern "C" __global__ void __raygen__path() {
    const LaunchParams& params = launchParams();
    const uint3 launchIndex = optixGetLaunchIndex();
    const int x = int(launchIndex.x);
    const int y = int(launchIndex.y);
    if (x >= params.width || y >= params.height) return;

    const unsigned int pixelIndex = unsigned(y) * unsigned(params.width) + unsigned(x);
    Rng rng = makePixelRng(x, y, params.sampleIndex, params.frameSeed);

    float jitterX = 0.5f, jitterY = 0.5f;
    if (params.pixelSampler == kPixelSamplerBlueNoise) {
        blueNoisePixelJitter(x, y, params.sampleIndex, jitterX, jitterY);
    } else {
        jitterX = rng.nextFloat();
        jitterY = rng.nextFloat();
    }

    Vec3 origin, direction;
    pinholeRay(params.scene, float(x) + jitterX, float(y) + jitterY, origin, direction);
    const Vec3 radiance = pathTrace(params.scene, origin, direction, rng);

    Vec4& pixel = params.accumBuffer[pixelIndex];
    pixel.x += radiance.x;
    pixel.y += radiance.y;
    pixel.z += radiance.z;
    pixel.w += 1.0f;
}

}  // namespace sol
