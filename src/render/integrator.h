// Backend agnostic path tracing kernel.
//
// The integrator is templated on a Tracer that provides
//     bool intersect(Vec3 origin, Vec3 dir, float tMax, RayHit& hit) const;
//     bool occluded(Vec3 origin, Vec3 dir, float tMax) const;
// so the Embree and OptiX backends run the exact same light transport code.
#pragma once

#include "core/rng.h"
#include "render/lights.h"
#include "render/shading.h"
#include "render/volume.h"
#include "scene/types.h"
#include "solstice_config.h"

#if !defined(__CUDACC__)
#include "render/optix/optix_bsdf.cuh"
#endif

#if !defined(__CUDACC__)
#include "render/volume_vdb.h"
#include "scene/volume_grid.h"
#endif

#if !defined(__CUDACC__) && SOLSTICE_HAVE_OPENPGL
#include "render/cpu/path_guiding.h"
#endif

namespace sol {

struct RayHit {
    float t = kFloatMax;
    int instanceIndex = -1;
    uint32_t primIndex = 0;
    float u = 0.0f;
    float v = 0.0f;
    float time = 0.0f;  // shutter time in [0,1] when motion blur is active
};

struct SurfaceInteraction {
    Vec3 p{0.0f, 0.0f, 0.0f};
    Vec3 pObject{0.0f, 0.0f, 0.0f};
    // Pref / Nref (object space, pre-displace cage). Equal to pObject/nObject when absent.
    Vec3 pRef{0.0f, 0.0f, 0.0f};
    Vec3 nRef{0.0f, 0.0f, 1.0f};
    Vec3 ng{0.0f, 0.0f, 1.0f};  // geometric normal (world)
    Vec3 ns{0.0f, 0.0f, 1.0f};  // shading normal (world)
    Vec3 nObject{0.0f, 0.0f, 1.0f};
    Vec2 uv{0.0f, 0.0f};
    // Approximate UV footprint diameter of one camera pixel at the hit (for mip LOD).
    float uvFilterWidth = 0.0f;
    int hasPref = 0;
    int instanceIndex = -1;
    int materialIndex = -1;
    int lightIndex = -1;
};

SR_INL SR_HD Material defaultMaterial() {
    Material m;
    m.baseColor = Vec3(0.7f, 0.7f, 0.7f);
    m.roughness = 0.5f;
    return m;
}

// Incoming ray classification for MaterialX ray_switch_shader — matches Arnold
// aiRaySwitch: the *incoming* ray type selects which surfaceshader port is
// evaluated for that hit (camera rays → camera port, specular transmission
// rays → specular_transmission, etc.). Unconnected ports fall back to the
// base/camera material (-1 in RaySwitchTable).
enum class RayShadeKind : int {
    Camera = 0,
    Shadow,
    DiffuseReflection,
    SpecularReflection,
    DiffuseTransmission,
    SpecularTransmission,
    Sss,
    // Solstice-only convenience for photon / MNEE / BDPT light-tracing through
    // glass. Never used for camera or other eye-path ray types.
    Caustics
};

SR_INL SR_HD int raySwitchSlot(const RaySwitchTable& t, RayShadeKind kind) {
    switch (kind) {
        case RayShadeKind::Camera: return t.camera;
        case RayShadeKind::Shadow: return t.shadow;
        case RayShadeKind::DiffuseReflection: return t.diffuseReflection;
        case RayShadeKind::SpecularReflection: return t.specularReflection;
        case RayShadeKind::DiffuseTransmission: return t.diffuseTransmission;
        case RayShadeKind::SpecularTransmission: return t.specularTransmission;
        case RayShadeKind::Sss: return t.sss;
        case RayShadeKind::Caustics: return t.caustics;
    }
    return -1;
}

// Resolve the Material POD for a hit given the incoming ray kind. `baseIndex` is
// InstanceData::materialIndex (owns the RaySwitchTable).
SR_INL SR_HD Material materialForRay(const SceneView& scene, int baseIndex, RayShadeKind kind) {
    if (baseIndex < 0 || baseIndex >= scene.materialCount) return defaultMaterial();
    const Material& base = scene.materials[baseIndex];
    const int slot = raySwitchSlot(base.raySwitch, kind);
    if (slot < 0 || slot >= scene.materialCount) return base;
    return scene.materials[slot];
}

SR_INL SR_HD Material materialForRay(const SceneView& scene, const SurfaceInteraction& si,
                                     RayShadeKind kind) {
    return materialForRay(scene, si.materialIndex, kind);
}

// Photon / MNEE / BDPT light-path glass: prefer Solstice `caustics` port, else
// Arnold `specular_transmission`, else the camera/base material.
SR_INL SR_HD Material materialForCausticTransport(const SceneView& scene, int baseIndex) {
    if (baseIndex < 0 || baseIndex >= scene.materialCount) return defaultMaterial();
    const Material& base = scene.materials[baseIndex];
    if (base.raySwitch.caustics >= 0) return materialForRay(scene, baseIndex, RayShadeKind::Caustics);
    if (base.raySwitch.specularTransmission >= 0)
        return materialForRay(scene, baseIndex, RayShadeKind::SpecularTransmission);
    return base;
}

// Tag the *next* ray after a BSDF sample (Arnold ray type for the child ray).
SR_INL SR_HD RayShadeKind nextRayShadeKind(const BsdfSample& bs, const LobeWeights& lw) {
    if (bs.transmitted) {
        if (bs.specular || isNearSpecularLobe(lw)) return RayShadeKind::SpecularTransmission;
        return RayShadeKind::DiffuseTransmission;
    }
    if (bs.specular || isNearSpecularLobe(lw)) return RayShadeKind::SpecularReflection;
    return RayShadeKind::DiffuseReflection;
}

// Chiang et al. 2016: map artist multiple-scattering albedo A → single-scattering α.
SR_INL SR_HD float chiangSingleScatterAlbedo(float A) {
    A = saturatef(A);
    return 1.0f - expf(-5.09406f * A + 2.61188f * A * A - 4.31805f * A * A * A);
}

SR_INL SR_HD Vec3 chiangSingleScatterAlbedo(Vec3 A) {
    return Vec3(chiangSingleScatterAlbedo(A.x), chiangSingleScatterAlbedo(A.y),
                chiangSingleScatterAlbedo(A.z));
}

SR_INL SR_HD float vecChannel(Vec3 v, int ch) {
    return ch == 0 ? v.x : (ch == 1 ? v.y : v.z);
}

SR_INL SR_HD Vec3 channelMask(int ch, float value) {
    return ch == 0 ? Vec3(value, 0.0f, 0.0f) : (ch == 1 ? Vec3(0.0f, value, 0.0f) : Vec3(0.0f, 0.0f, value));
}

SR_INL SR_HD Mat4 lerpMat4(const Mat4& a, const Mat4& b, float t) {
    Mat4 r;
    const float u = 1.0f - t;
    for (int i = 0; i < 16; ++i) r.m[i] = a.m[i] * u + b.m[i] * t;
    return r;
}

SR_INL SR_HD void sampleMotionKeys(int keyCount, float time, int& i0, int& i1, float& frac) {
    if (keyCount <= 1) {
        i0 = i1 = 0;
        frac = 0.0f;
        return;
    }
    const float t = saturatef(time) * float(keyCount - 1);
    i0 = int(t);
    if (i0 >= keyCount - 1) {
        i0 = keyCount - 1;
        i1 = i0;
        frac = 0.0f;
        return;
    }
    i1 = i0 + 1;
    frac = t - float(i0);
}

SR_INL SR_HD void instanceXformAtTime(const SceneView& scene, const InstanceData& inst, float time, Mat4& xform,
                                      Mat4& xformInv) {
    if (inst.motionKeyCount <= 1 || !scene.motionXforms) {
        xform = inst.xform;
        xformInv = inst.xformInv;
        return;
    }
    int i0 = 0, i1 = 0;
    float frac = 0.0f;
    sampleMotionKeys(inst.motionKeyCount, time, i0, i1, frac);
    const Mat4& a = scene.motionXforms[inst.motionKeyOffset + i0];
    const Mat4& b = scene.motionXforms[inst.motionKeyOffset + i1];
    xform = (i0 == i1) ? a : lerpMat4(a, b, frac);
    xformInv = inverse(xform);
}

SR_INL SR_HD Mat4 cameraToWorldAtTime(const SceneView& scene, float time) {
    if (scene.cameraMotionKeyCount <= 1 || !scene.cameraMotionXforms) return scene.camera.cameraToWorld;
    int i0 = 0, i1 = 0;
    float frac = 0.0f;
    sampleMotionKeys(scene.cameraMotionKeyCount, time, i0, i1, frac);
    const Mat4& a = scene.cameraMotionXforms[i0];
    const Mat4& b = scene.cameraMotionXforms[i1];
    return (i0 == i1) ? a : lerpMat4(a, b, frac);
}

SR_INL SR_HD Vec3 meshPositionAtTime(const MeshView& mesh, uint32_t vertexIndex, float time) {
    if (!mesh.positions || vertexIndex >= mesh.vertexCount) return Vec3(0.0f);
    if (mesh.motionKeyCount <= 1 || !mesh.motionPositions) return mesh.positions[vertexIndex];
    int i0 = 0, i1 = 0;
    float frac = 0.0f;
    sampleMotionKeys(mesh.motionKeyCount, time, i0, i1, frac);
    const Vec3 a = mesh.motionPositions[uint32_t(i0) * mesh.vertexCount + vertexIndex];
    if (i0 == i1) return a;
    const Vec3 b = mesh.motionPositions[uint32_t(i1) * mesh.vertexCount + vertexIndex];
    return a * (1.0f - frac) + b * frac;
}

SR_INL SR_HD float distancePointToSegment(Vec3 p, Vec3 a, Vec3 b) {
    const Vec3 ab = b - a;
    const float ab2 = lengthSquared(ab);
    if (ab2 < 1e-20f) return length(p - a);
    const float t = clampf(dot(p - a, ab) / ab2, 0.0f, 1.0f);
    return length(p - (a + ab * t));
}

// Screen-space wireframe. Prefers the authored cage overlay (n-gon boundaries only);
// otherwise uses the hit triangle's edges, honoring triEdgeMask to hide diagonals.
SR_INL SR_HD Vec3 shadeWireframe(const SceneView& scene, const RayHit& hit, const SurfaceInteraction& si,
                                 Vec3 direction) {
    const float thicknessPx = srMax(0.25f, scene.settings.wireframeThickness);
    if (hit.instanceIndex < 0 || hit.instanceIndex >= scene.instanceCount || !scene.instances)
        return Vec3(0.05f);
    const InstanceData& inst = scene.instances[hit.instanceIndex];
    if (inst.meshIndex < 0 || inst.meshIndex >= scene.meshCount || !scene.meshes)
        return Vec3(0.05f);
    const MeshView& mesh = scene.meshes[inst.meshIndex];
    if (!mesh.indices || !mesh.positions || hit.primIndex >= mesh.triangleCount)
        return Vec3(0.05f);

    Mat4 xform, xformInv;
    instanceXformAtTime(scene, inst, hit.time, xform, xformInv);

    const float resX = float(srMax(1, scene.settings.resolutionX));
    const float resY = float(srMax(1, scene.settings.resolutionY));
    const float sensorH = scene.camera.sensorWidth * (resY / resX);
    const float pixelAngle = (sensorH / srMax(1e-3f, scene.camera.focalLength)) / resY;
    const float pixelWorld = srMax(1e-8f, hit.t) * pixelAngle;
    const float halfW = thicknessPx * pixelWorld;
    const float aa = 0.5f * pixelWorld;
    const float searchR = halfW + aa + pixelWorld;

    float dEdge = 1.0e30f;
    bool haveEdge = false;

    // Cage overlay: authored face boundaries only (never triangulation diagonals /
    // micropolygon edges). Cage-sized — safe to scan near the hit.
    if (mesh.wireEdgeCount > 0 && mesh.wireIndices && mesh.wirePositions && mesh.wireVertexCount > 0) {
        for (uint32_t e = 0; e < mesh.wireEdgeCount; ++e) {
            const uint32_t a = mesh.wireIndices[e * 2 + 0];
            const uint32_t b = mesh.wireIndices[e * 2 + 1];
            if (a >= mesh.wireVertexCount || b >= mesh.wireVertexCount) continue;
            const Vec3 wa = transformPoint(xform, mesh.wirePositions[a]);
            const Vec3 wb = transformPoint(xform, mesh.wirePositions[b]);
            // Cheap reject: skip edges whose endpoints are both far from the hit.
            const float da = lengthSquared(si.p - wa);
            const float db = lengthSquared(si.p - wb);
            const float r2 = searchR * searchR;
            if (da > r2 && db > r2) {
                // Still check if the segment passes near the hit (midpoint / projection).
                const float dSeg = distancePointToSegment(si.p, wa, wb);
                if (dSeg > searchR) continue;
                dEdge = srMin(dEdge, dSeg);
                haveEdge = true;
                continue;
            }
            dEdge = srMin(dEdge, distancePointToSegment(si.p, wa, wb));
            haveEdge = true;
        }
    }

    if (!haveEdge) {
        // Fallback: edges of the hit triangle, filtered by authored-boundary mask.
        const uint32_t i0 = mesh.indices[hit.primIndex * 3 + 0];
        const uint32_t i1 = mesh.indices[hit.primIndex * 3 + 1];
        const uint32_t i2 = mesh.indices[hit.primIndex * 3 + 2];
        if (i0 >= mesh.vertexCount || i1 >= mesh.vertexCount || i2 >= mesh.vertexCount)
            return Vec3(0.05f);
        const Vec3 p0 = transformPoint(xform, meshPositionAtTime(mesh, i0, hit.time));
        const Vec3 p1 = transformPoint(xform, meshPositionAtTime(mesh, i1, hit.time));
        const Vec3 p2 = transformPoint(xform, meshPositionAtTime(mesh, i2, hit.time));
        uint8_t mask = 7u;
        if (mesh.triEdgeMask) mask = mesh.triEdgeMask[hit.primIndex];
        if (mask & 1u) {
            dEdge = srMin(dEdge, distancePointToSegment(si.p, p0, p1));
            haveEdge = true;
        }
        if (mask & 2u) {
            dEdge = srMin(dEdge, distancePointToSegment(si.p, p1, p2));
            haveEdge = true;
        }
        if (mask & 4u) {
            dEdge = srMin(dEdge, distancePointToSegment(si.p, p2, p0));
            haveEdge = true;
        }
    }

    // 1 on the edge centerline → 0 outside the stroke (+AA).
    float edge = 0.0f;
    if (haveEdge) {
        const float lo = srMax(0.0f, halfW - aa);
        const float hi = halfW + aa;
        if (dEdge <= lo) {
            edge = 1.0f;
        } else if (dEdge < hi) {
            const float t = (dEdge - lo) / srMax(1e-8f, hi - lo);
            edge = 1.0f - t * t * (3.0f - 2.0f * t);
        }
    }

    const float facing = fabsf(dot(si.ns, -direction));
    const Vec3 face = Vec3(0.06f + 0.10f * facing);
    const Vec3 wire = Vec3(0.92f, 0.94f, 0.96f);
    return face * (1.0f - edge) + wire * edge;
}

// Reconstruct shading attributes from a hit record.
SR_INL SR_HD bool buildSurfaceInteraction(const SceneView& scene, const RayHit& hit, Vec3 origin, Vec3 dir,
                                          SurfaceInteraction& si) {
    if (hit.instanceIndex < 0 || hit.instanceIndex >= scene.instanceCount) return false;
    const InstanceData& inst = scene.instances[hit.instanceIndex];
    if (inst.meshIndex < 0 || inst.meshIndex >= scene.meshCount) return false;
    const MeshView& mesh = scene.meshes[inst.meshIndex];
    if (hit.primIndex >= mesh.triangleCount) return false;

    Mat4 xform, xformInv;
    instanceXformAtTime(scene, inst, hit.time, xform, xformInv);

    const uint32_t i0 = mesh.indices[hit.primIndex * 3 + 0];
    const uint32_t i1 = mesh.indices[hit.primIndex * 3 + 1];
    const uint32_t i2 = mesh.indices[hit.primIndex * 3 + 2];
    const Vec3 p0 = meshPositionAtTime(mesh, i0, hit.time);
    const Vec3 p1 = meshPositionAtTime(mesh, i1, hit.time);
    const Vec3 p2 = meshPositionAtTime(mesh, i2, hit.time);
    const float w = 1.0f - hit.u - hit.v;

    const Vec3 pLocal = p0 * w + p1 * hit.u + p2 * hit.v;
    si.pObject = pLocal;
    si.pRef = pLocal;
    si.hasPref = 0;
    si.p = transformPoint(xform, pLocal);
    // The hit distance is authoritative for ray offsets.
    si.p = origin + dir * hit.t;

    Vec3 ngLocal = cross(p1 - p0, p2 - p0);
    si.nObject = lengthSquared(ngLocal) > 0.0f ? normalize(ngLocal) : Vec3(0.0f, 0.0f, 1.0f);
    si.nRef = si.nObject;
    si.ng = normalize(transformNormalWithInverse(xformInv, ngLocal));

    if (mesh.normals) {
        const Vec3 nLocal = mesh.normals[i0] * w + mesh.normals[i1] * hit.u + mesh.normals[i2] * hit.v;
        si.nObject = lengthSquared(nLocal) > 0.0f ? normalize(nLocal) : si.nObject;
        si.nRef = si.nObject;
        Vec3 ns = transformNormalWithInverse(xformInv, nLocal);
        si.ns = lengthSquared(ns) > 0.0f ? normalize(ns) : si.ng;
    } else {
        si.ns = si.ng;
    }

    // Arnold Pref: lock triplanar / noise / autobump to the pre-displace cage.
    if (mesh.restPositions) {
        const Vec3 r0 = mesh.restPositions[i0];
        const Vec3 r1 = mesh.restPositions[i1];
        const Vec3 r2 = mesh.restPositions[i2];
        si.pRef = r0 * w + r1 * hit.u + r2 * hit.v;
        si.hasPref = 1;
        if (mesh.restNormals) {
            const Vec3 rn = mesh.restNormals[i0] * w + mesh.restNormals[i1] * hit.u +
                            mesh.restNormals[i2] * hit.v;
            if (lengthSquared(rn) > 0.0f) si.nRef = normalize(rn);
        } else {
            const Vec3 rn = cross(r1 - r0, r2 - r0);
            if (lengthSquared(rn) > 0.0f) si.nRef = normalize(rn);
        }
    }
    if (mesh.uvs) {
        const Vec2 uv0 = mesh.uvs[i0], uv1 = mesh.uvs[i1], uv2 = mesh.uvs[i2];
        si.uv = Vec2(uv0.x * w + uv1.x * hit.u + uv2.x * hit.v, uv0.y * w + uv1.y * hit.u + uv2.y * hit.v);

        // Pixel footprint → UV filter width for .tx / mip LOD.
        const Vec3 e1w = transformVector(xform, p1 - p0);
        const Vec3 e2w = transformVector(xform, p2 - p0);
        const Vec2 d1 = uv1 - uv0;
        const Vec2 d2 = uv2 - uv0;
        const float lenE1 = length(e1w);
        const float lenE2 = length(e2w);
        const float lenD1 = sqrtf(d1.x * d1.x + d1.y * d1.y);
        const float lenD2 = sqrtf(d2.x * d2.x + d2.y * d2.y);
        const float uvPerWorld =
            srMax(lenE1 > 1e-8f ? lenD1 / lenE1 : 0.0f, lenE2 > 1e-8f ? lenD2 / lenE2 : 0.0f);
        const float resX = float(srMax(1, scene.settings.resolutionX));
        const float resY = float(srMax(1, scene.settings.resolutionY));
        const float sensorH = scene.camera.sensorWidth * (resY / resX);
        const float pixelAngle = (sensorH / srMax(1e-3f, scene.camera.focalLength)) / resY;
        si.uvFilterWidth = srMax(0.0f, hit.t) * pixelAngle * uvPerWorld;
    }
    si.instanceIndex = hit.instanceIndex;
    si.materialIndex = inst.materialIndex;
    si.lightIndex = inst.lightIndex;

    // Keep the shading normal on the same side as the geometric normal.
    if (dot(si.ns, si.ng) < 0.0f) si.ng = -si.ng;
    return true;
}

SR_INL SR_HD Vec3 offsetRayOrigin(Vec3 p, Vec3 n, Vec3 dir) {
    const float scale = 1.0f + srMax(fabsf(p.x), srMax(fabsf(p.y), fabsf(p.z)));
    const Vec3 offset = n * (kRayEpsilon * scale);
    return dot(dir, n) > 0.0f ? p + offset : p - offset;
}

SR_INL SR_HD bool materialSupportsSss(const Material& mat) {
    return saturatef(mat.subsurface) > 1e-4f && mat.transmission <= 1e-4f && mat.metallic < 0.999f;
}

SR_INL SR_HD Material sssSpecularEntryMaterial(const Material& mat) {
    Material specMat = mat;
    specMat.subsurface = 0.0f;
    specMat.transmission = 0.0f;
    if (specMat.metallic < 0.999f) specMat.baseColor = Vec3(0.0f);
    return specMat;
}

SR_INL SR_HD Material sssExitLambertMaterial() {
    Material lambert = defaultMaterial();
    lambert.baseColor = Vec3(1.0f);
    lambert.specular = 0.0f;
    lambert.metallic = 0.0f;
    lambert.transmission = 0.0f;
    lambert.subsurface = 0.0f;
    lambert.roughness = 1.0f;
    return lambert;
}

// Fresnel RR probability of reflecting at the SSS entry instead of entering the body.
SR_INL SR_HD float sssEntrySpecularProb(const Material& specMat, Vec3 woLocal) {
    const LobeWeights specLw = computeLobes(specMat);
    if (specLw.specular <= 1e-5f || saturatef(specMat.specular) <= 1e-5f) return 0.0f;
    const float cosWo = srMax(0.0f, woLocal.z);
    const float fresnelEst = average(fresnelSchlick(specLw.f0, cosWo));
    return clampf(srMax(fresnelEst, specLw.specular * fresnelEst), 0.0f, 0.98f);
}

struct SssWalkResult {
    bool escaped = false;
    Vec3 exitP{0.0f};
    Vec3 exitN{0.0f, 1.0f, 0.0f};
    Vec3 exitWo{0.0f, 1.0f, 0.0f};
    Vec3 pathWeight{1.0f};
};

// Spectral Chiang random-walk BSSRDF (hero-channel MIS). Shared by PT and BDPT.
template <typename Tracer>
SR_INL SssWalkResult sampleSssRandomWalk(const SceneView& scene, const Tracer& tracer,
                                         const SurfaceInteraction& entrySi, Vec3 wo, const Material& mat,
                                         Rng& rng) {
    SssWalkResult out;
    out.exitP = entrySi.p;
    out.exitN = entrySi.ns;
    out.exitWo = wo;
    out.pathWeight = vmax(Vec3(0.0f), mat.subsurfaceColor);

    const Vec3 mfpRGB = vmax(Vec3(0.0f), mat.subsurfaceRadius) * srMax(0.0f, mat.subsurfaceScale);
    const Vec3 multiAlbedo = vmax(Vec3(0.0f), mat.subsurfaceColor);
    if (maxComponent(mfpRGB) < 1e-8f) {
        out.escaped = true;
        out.pathWeight = multiAlbedo;
        return out;
    }

    const Vec3 singleAlbedo = chiangSingleScatterAlbedo(multiAlbedo);
    float sigma[3] = {1.0f / srMax(1e-5f, mfpRGB.x), 1.0f / srMax(1e-5f, mfpRGB.y),
                      1.0f / srMax(1e-5f, mfpRGB.z)};
    float alpha[3] = {srMax(0.0f, singleAlbedo.x), srMax(0.0f, singleAlbedo.y),
                      srMax(0.0f, singleAlbedo.z)};

    float sel[3];
    for (int c = 0; c < 3; ++c) sel[c] = srMax(1e-3f, alpha[c] / sigma[c]);
    const float selSum = sel[0] + sel[1] + sel[2];
    float uSel = rng.nextFloat() * selSum;
    int hero = 0;
    if (uSel >= sel[0]) {
        hero = 1;
        uSel -= sel[0];
    }
    if (hero == 1 && uSel >= sel[1]) hero = 2;
    const float pSel[3] = {sel[0] / selSum, sel[1] / selSum, sel[2] / selSum};
    const float sigmaH = sigma[hero];

    float pathPdf[3] = {pSel[0], pSel[1], pSel[2]};
    float thr[3] = {1.0f, 1.0f, 1.0f};

    Vec3 pWalk = entrySi.p - entrySi.ns * (kRayEpsilon * (1.0f + length(entrySi.p)));
    bool escaped = false;

    constexpr int kMaxWalkSteps = 24;
    for (int step = 0; step < kMaxWalkSteps; ++step) {
        const float stepLen = -logf(srMax(1e-6f, 1.0f - rng.nextFloat())) / sigmaH;

        Vec3 walkDir;
        if (step == 0) {
            const Frame inFrame(-entrySi.ns);
            walkDir = inFrame.toWorld(sampleCosineHemisphere(rng.nextFloat(), rng.nextFloat()));
        } else {
            walkDir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
        }
        if (lengthSquared(walkDir) < 1e-12f) continue;
        walkDir = normalize(walkDir);

        const float rayEps = kRayEpsilon * (1.0f + length(pWalk));
        const Vec3 walkOrigin = pWalk + walkDir * rayEps;
        RayHit walkHit;
        if (!tracer.intersect(walkOrigin, walkDir, stepLen, walkHit)) {
            pWalk = walkOrigin + walkDir * stepLen;
            for (int c = 0; c < 3; ++c) {
                const float tr = expf(-sigma[c] * stepLen);
                const float dens = sigma[c] * tr;
                thr[c] *= alpha[c] * dens;
                pathPdf[c] *= dens;
            }
            const float wHero = thr[hero] / srMax(1e-20f, pathPdf[hero]);
            if (wHero < 1e-5f) {
                thr[0] = thr[1] = thr[2] = 0.0f;
                break;
            }
            if (step >= 8) {
                const float q = clampf(wHero, 0.25f, 1.0f);
                if (rng.nextFloat() > q) {
                    thr[0] = thr[1] = thr[2] = 0.0f;
                    break;
                }
                thr[0] /= q;
                thr[1] /= q;
                thr[2] /= q;
            }
            continue;
        }

        SurfaceInteraction walkSi;
        if (!buildSurfaceInteraction(scene, walkHit, walkOrigin, walkDir, walkSi)) break;

        const float tHit = srMax(0.0f, walkHit.t);
        for (int c = 0; c < 3; ++c) {
            const float tr = expf(-sigma[c] * tHit);
            thr[c] *= tr;
            pathPdf[c] *= tr;
        }

        escaped = true;
        out.exitP = walkSi.p;
        out.exitN = walkSi.ns;
        if (dot(out.exitN, walkDir) < 0.0f) out.exitN = -out.exitN;
        if (lengthSquared(out.exitN) < 1e-12f) out.exitN = walkDir;
        else out.exitN = normalize(out.exitN);

        const float pdfSum = pathPdf[0] + pathPdf[1] + pathPdf[2];
        if (pdfSum > 1e-20f) {
            out.pathWeight = Vec3(thr[0] / pdfSum, thr[1] / pdfSum, thr[2] / pdfSum);
        } else {
            out.pathWeight = Vec3(0.0f);
        }
        const Vec3 toEntry = entrySi.p - out.exitP;
        const float woLen2 = lengthSquared(toEntry);
        out.exitWo = woLen2 > 1e-12f ? normalize(toEntry) : out.exitN;
        break;
    }

    if (!escaped || isBlack(out.pathWeight) || !isFinite(out.pathWeight)) {
        out.escaped = true;
        out.exitP = entrySi.p;
        out.exitN = entrySi.ns;
        out.exitWo = wo;
        out.pathWeight = multiAlbedo;
        return out;
    }
    if (dot(out.exitN, out.exitWo) < 0.0f) out.exitWo = out.exitN;
    out.escaped = true;
    return out;
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
SR_INL SR_HD void generateCameraRay(const SceneView& scene, float pixelX, float pixelY, float lensU, float lensV,
                                    Vec3& origin, Vec3& direction, float shutterTime = 0.0f) {
    const CameraData& cam = scene.camera;
    const Mat4 cameraToWorld = cameraToWorldAtTime(scene, shutterTime);
    const float resX = float(srMax(1, scene.settings.resolutionX));
    const float resY = float(srMax(1, scene.settings.resolutionY));
    const float sensorHeight = cam.sensorWidth * (resY / resX);
    const float sx = (pixelX / resX - 0.5f) * cam.sensorWidth;
    const float sy = (0.5f - pixelY / resY) * sensorHeight;

    Vec3 dirCam = normalize(Vec3(sx, sy, -srMax(1e-3f, cam.focalLength)));
    Vec3 originCam(0.0f, 0.0f, 0.0f);

    if (cam.fStop > 0.0f) {
        // Scene units are assumed to be metres, focal length is in millimetres.
        const float lensRadius = (cam.focalLength * 0.001f) / (2.0f * cam.fStop);
        const Vec2 lens = sampleConcentricDisk(lensU, lensV) * lensRadius;
        const float ft = srMax(1e-4f, cam.focusDistance) / srMax(1e-6f, -dirCam.z);
        const Vec3 focusPoint = dirCam * ft;
        originCam = Vec3(lens.x, lens.y, 0.0f);
        dirCam = normalize(focusPoint - originCam);
    }

    origin = transformPoint(cameraToWorld, originCam);
    direction = normalize(transformVector(cameraToWorld, dirCam));
}

// ---------------------------------------------------------------------------
// Path tracing
// ---------------------------------------------------------------------------

// Default no-op guiding hooks (OptiX / builds without OpenPGL).
struct NullGuiding {
    SR_INL SR_HD bool active() const { return false; }
    SR_INL SR_HD float guideProbability() const { return 0.0f; }
    SR_INL SR_HD bool prepared() const { return false; }
    SR_INL SR_HD bool preparedVolume() const { return false; }
    SR_INL SR_HD bool prepare(Vec3, Vec3, Rng&) { return false; }
    SR_INL SR_HD float pdf(Vec3) const { return 0.0f; }
    SR_INL SR_HD bool sample(float, float, Vec3&, float&) const { return false; }
    SR_INL SR_HD bool prepareVolume(Vec3, Vec3, float, Rng&) { return false; }
    SR_INL SR_HD float pdfVolume(Vec3) const { return 0.0f; }
    SR_INL SR_HD bool sampleVolume(float, float, Vec3&, float&) const { return false; }
    SR_INL SR_HD void beginSegment(Vec3, Vec3) {}
    SR_INL SR_HD void beginVolumeSegment(Vec3, Vec3) {}
    SR_INL SR_HD void recordEmission(Vec3, float) {}
    SR_INL SR_HD void addScattered(Vec3) {}
    SR_INL SR_HD void recordBounce(Vec3, Vec3, float, Vec3, bool, float, float, float, bool = false) {}
    SR_INL SR_HD void setRussianRoulette(float) {}
    SR_INL SR_HD void recordBackground(Vec3, Vec3, Vec3, float) {}
    SR_INL SR_HD void recordLightHit(Vec3, Vec3, Vec3, float) {}
};

// Clamp a path contribution to fight fireflies (bright rare samples).
SR_INL SR_HD Vec3 clampContribution(Vec3 contrib, float clampValue) {
    if (clampValue <= 0.0f || !isFinite(contrib)) return isFinite(contrib) ? contrib : Vec3(0.0f);
    const float m = maxComponent(contrib);
    if (m > clampValue) contrib *= clampValue / m;
    return contrib;
}

SR_INL SR_HD bool xpuEstimatorMatch(const RenderSettingsData& s) { return renderDeviceIsXpu(s.backend); }

SR_INL SR_HD BsdfEval evalBsdfDevice(const RenderSettingsData& s, const Material& mat, Vec3 wo, Vec3 wi) {
#if !defined(__CUDACC__)
    if (xpuEstimatorMatch(s)) {
        const optixpt::BsdfEval e = optixpt::bsdfEvalLocal(mat, wo, wi);
        BsdfEval o;
        o.f = e.f;
        o.pdf = e.pdf;
        return o;
    }
#else
    (void)s;
#endif
    return bsdfEvalLocal(mat, wo, wi);
}

SR_INL SR_HD BsdfSample sampleBsdfDevice(const RenderSettingsData& s, const Material& mat, Vec3 wo, float uLobe,
                                        float u1, float u2, float u3) {
#if !defined(__CUDACC__)
    if (xpuEstimatorMatch(s)) {
        const optixpt::BsdfSample e = optixpt::bsdfSampleLocal(mat, wo, uLobe, u1, u2, u3);
        BsdfSample o;
        o.weight = e.weight;
        o.wi = e.wi;
        o.pdf = e.pdf;
        o.specular = e.specular;
        o.transmitted = e.transmitted;
        return o;
    }
#else
    (void)s;
#endif
    return bsdfSampleLocal(mat, wo, uLobe, u1, u2, u3);
}

// SDS / near-specular firefly cap. `causticClamp` tightens further; when left at 0
// a safety floor of 10 still applies — otherwise `clampContribution(..., 0)` is a
// no-op and roughness-0 glass / BDPT near-spec NEE keep permanent sparkles.
// Test-only: causticClamp < 0 disables the safety floor (unbiased energy compares).
SR_INL SR_HD float causticFireflyCap(const RenderSettingsData& settings) {
    if (settings.causticClamp < 0.0f) return 0.0f;
    return settings.causticClamp > 0.0f ? settings.causticClamp : 10.0f;
}

// BDPT light-tracing deposits include cameraPdfOmega; resolve divides by W·H paths.
// Map Arnold-style Indirect Clamp (pixel radiance) → raw splat threshold.
SR_INL SR_HD float lightTraceSplatClamp(const RenderSettingsData& settings) {
    if (settings.clampIndirect <= 0.0f) return 0.0f;
    const int w = settings.resolutionX > 0 ? settings.resolutionX : 1;
    const int h = settings.resolutionY > 0 ? settings.resolutionY : 1;
    return settings.clampIndirect * float(w) * float(h);
}

// Multi-hit shadow visibility (Embree filter-function style): opaque surfaces
// block fully; transmissive surfaces attenuate by Material::shadowOpacity when
// refractive caustics are enabled (MaterialX / Arnold fake-caustics control).
// VDB: SDF level sets are hard occluders (tested against the field, not the AABB
// proxy). Fog AABBs are skipped here — soft Tr is applied via ratio tracking
// (shadowTransmittanceFogVolumes, PBRT §11.2.1 / VolPath §14.2.2).
template <typename Tracer>
SR_INL SR_HD float shadowVisibility(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 dir,
                                    float tMax) {
#if !defined(__CUDACC__)
    // Field-based SDF cast / self shadow (works from inside the AABB too).
    if (shadowOccludedBySdfVolumes(scene, origin, dir, tMax)) return 0.0f;
#endif

    float visibility = 1.0f;
    Vec3 o = origin;
    float remaining = tMax;
    for (int hop = 0; hop < 24 && visibility > 1e-4f; ++hop) {
        RayHit hit;
        if (!tracer.intersect(o, dir, remaining, hit)) return visibility;
        if (!(hit.t > 1e-5f) || hit.t >= remaining) return visibility;

        SurfaceInteraction si;
        if (!buildSurfaceInteraction(scene, hit, o, dir, si)) return 0.0f;

        // Reached light proxy geometry along the shadow segment — connection ok.
        if (si.lightIndex >= 0) return visibility;

        // Volume AABB proxies exist only to enter SDF/fog on camera rays. They must
        // not occlude NEE / shadow connections (shadowVisibility uses primary-mask
        // intersect, so Embree visibilityMask alone cannot skip them).
        // SDF occlusion is handled by shadowOccludedBySdfVolumes above; fog Tr is
        // applied separately in NEE via shadowTransmittanceFogVolumes.
        if (si.instanceIndex >= 0 && si.instanceIndex < scene.instanceCount) {
            const InstanceData& hitInst = scene.instances[si.instanceIndex];
            if (hitInst.volumeIndex >= 0) {
                const Vec3 p = o + dir * hit.t;
                o = offsetRayOrigin(p, si.ng, dir);
                remaining -= hit.t;
                if (remaining <= 1e-4f) return visibility;
                continue;
            }
        }

        Material mat = materialForRay(scene, si.materialIndex, RayShadeKind::Shadow);
        // Opaque-glass when caustics estimators own transport (caustics / specular_transmission slot).
        const Material matCau = materialForCausticTransport(scene, si.materialIndex);

        float block = 1.0f;
        if (mat.transmission > 1e-3f) {
            // Caustics ON + material contributes: shadow rays treat glass as opaque —
            // transmitted light is delivered by MNEE / BDPT LT / photon gather.
            // Caustics OFF, or material Contribute to Caustics off: fake with shadow_opacity.
            if (scene.settings.caustics == 0 || !materialContributesCaustics(matCau))
                block = saturatef(mat.shadowOpacity);
            else
                block = 1.0f;
        }
        visibility *= (1.0f - block);
        if (block >= 0.999f || visibility <= 1e-5f) return 0.0f;

        const Vec3 p = o + dir * hit.t;
        o = offsetRayOrigin(p, si.ng, dir);
        remaining -= hit.t;
        if (remaining <= 1e-4f) return visibility;
    }
    return 0.0f;
}

template <typename Tracer, typename Guiding>
SR_INL SR_HD Vec3 nextEventEstimationOnce(const SceneView& scene, const Tracer& tracer,
                                          const SurfaceInteraction& si, const Material& mat,
                                          const Frame& frame, Vec3 wo, Rng& rng, Guiding* guiding,
                                          int mediumIndex = -1) {
    Vec3 result(0.0f);
    if (scene.lightCount <= 0) return result;

#if !defined(__CUDACC__)
    if (xpuEstimatorMatch(scene.settings)) {
        float selectPdf = 0.0f;
        const int lightIndex = sampleLightIndex(scene, si.p, rng.nextFloat(), selectPdf);
        LightSample ls;
        if (lightIndex < 0 || selectPdf <= 0.0f) return result;
        if (!sampleLight(scene, lightIndex, si.p, rng.nextFloat(), rng.nextFloat(), ls)) return result;
        if (ls.pdf <= 0.0f || isBlack(ls.radiance)) return result;
        if (!optixpt::shadingNormalConsistent(si.ng, si.ns, wo, ls.wi)) return result;
        const Vec3 woLocal = frame.toLocal(wo);
        const Vec3 wiLocal = frame.toLocal(ls.wi);
        const optixpt::BsdfEval be = optixpt::bsdfEvalLocal(mat, woLocal, wiLocal);
        if (be.pdf <= 0.0f || isBlack(be.f)) return result;
        const float lightPdf = ls.pdf * selectPdf;
        const float mis = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, be.pdf);
        result = ls.radiance * be.f * (fabsf(wiLocal.z) / lightPdf) * mis;
        float visibility = 1.0f;
        Vec3 shadowOrigin = si.p;
        float tMax = 1.0e8f;
        if (scene.lights[lightIndex].shadowEnable) {
            shadowOrigin = offsetRayOrigin(si.p, si.ng, ls.wi);
            if (ls.distance < 1.0e7f) tMax = ls.distance * (1.0f - 1e-3f);
            visibility = shadowVisibility(scene, tracer, shadowOrigin, ls.wi, tMax);
            if (visibility <= 1e-5f) return Vec3(0.0f);
            result = result * visibility;
            result = result * shadowTransmittanceFogVolumes(scene, shadowOrigin, ls.wi, tMax, rng);
        }
        if (const MediumData* med = getMedium(scene, mediumIndex)) {
            if (med->type != 2 && ls.distance < 1.0e7f) result = result * mediumShadowTr(*med, ls.distance);
        }
        return result;
    }
#endif

    const Vec3 woLocal = frame.toLocal(wo);
    LightSample cands[kRisCandidates];
    int lis[kRisCandidates];
    float selPdfs[kRisCandidates];
    Vec3 rgb[kRisCandidates];
    float scatterPdfs[kRisCandidates];
    float ws[kRisCandidates];
    int nOk = 0;
    for (int i = 0; i < kRisCandidates; ++i) {
        float selectPdf = 0.0f;
        const int lightIndex = sampleLightIndex(scene, si.p, rng.nextFloat(), selectPdf);
        if (lightIndex < 0 || selectPdf <= 0.0f) continue;
        LightSample ls;
        if (!sampleLight(scene, lightIndex, si.p, rng.nextFloat(), rng.nextFloat(), ls)) continue;
        if (ls.pdf <= 0.0f || isBlack(ls.radiance)) continue;
        if (!shadingNormalConsistent(si.ng, si.ns, wo, ls.wi)) continue;
        const Vec3 wiLocal = frame.toLocal(ls.wi);
        const BsdfEval be = bsdfEvalLocal(mat, woLocal, wiLocal);
        if (be.pdf <= 0.0f || isBlack(be.f)) continue;
        float scatterPdf = be.pdf;
#if !defined(__CUDACC__)
        if (guiding && guiding->active() && guiding->prepared()) {
            const float pg = guiding->guideProbability();
            const float gPdf = guiding->pdf(ls.wi);
            scatterPdf = pg * gPdf + (1.0f - pg) * be.pdf;
        }
#else
        (void)guiding;
#endif
        const float lightPdf = ls.pdf * selectPdf;
        const Vec3 unshadowed = ls.radiance * be.f * (fabsf(wiLocal.z) / lightPdf);
        const float w = luminance(vmax(unshadowed, Vec3(0.0f)));
        if (w <= 1e-20f) continue;
        cands[nOk] = ls;
        lis[nOk] = lightIndex;
        selPdfs[nOk] = selectPdf;
        rgb[nOk] = unshadowed;
        scatterPdfs[nOk] = scatterPdf;
        ws[nOk] = w;
        ++nOk;
    }
    float wSum = 0.0f;
    const int pick = risPick(ws, nOk, rng.nextFloat(), wSum);
    if (pick < 0) return result;
    const LightSample& ls = cands[pick];
    const int lightIndex = lis[pick];

    float visibility = 1.0f;
    Vec3 shadowOrigin = si.p;
    float tMax = 1.0e8f;
    if (scene.lights[lightIndex].shadowEnable) {
        shadowOrigin = offsetRayOrigin(si.p, si.ng, ls.wi);
        if (ls.distance < 1.0e7f) tMax = ls.distance * (1.0f - 1e-3f);
        visibility = shadowVisibility(scene, tracer, shadowOrigin, ls.wi, tMax);
        if (visibility <= 1e-5f) return result;
    }

    const float lightPdf = ls.pdf * selPdfs[pick];
    const float misWeight = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, scatterPdfs[pick]);
    // RIS: vis * rgb_pick * (Σw) / (M * w_pick). Failed candidates are zeros in the M slots.
    result = rgb[pick] * (visibility * misWeight * wSum / (float(kRisCandidates) * ws[pick]));

#if !defined(__CUDACC__)
    if (scene.lights[lightIndex].shadowEnable)
        result = result * shadowTransmittanceFogVolumes(scene, shadowOrigin, ls.wi, tMax, rng);
#endif
    if (const MediumData* med = getMedium(scene, mediumIndex)) {
        if (med->type != 2 && ls.distance < 1.0e7f) result = result * mediumShadowTr(*med, ls.distance);
    }
    return result;
}

// Continuation pdf at a volume vertex: HG, mixed with OpenPGL volume×HG when
// Indirect Guides is on and the volume field has trained.
template <typename Guiding>
SR_INL SR_HD float volumeScatterPdf(Vec3 woVol, Vec3 wi, const MediumData& med, Guiding* guiding) {
    const float phasePdf = henyeyGreenstein(clampf(dot(woVol, wi), -1.0f, 1.0f), med.g);
#if !defined(__CUDACC__)
    if (guiding && guiding->active() && guiding->preparedVolume()) {
        const float pg = guiding->guideProbability();
        return pg * guiding->pdfVolume(wi) + (1.0f - pg) * phasePdf;
    }
#else
    (void)guiding;
#endif
    return phasePdf;
}

// Volume NEE: M light candidates scored by phase·Le/pdf (product via RIS), one
// shadow ray. M grows with |g| because the env CDF does not see the HG peak.
// Light *selection* is flux × 4π·HG(wo, sun) so the sun is proposed when it
// sits in the HG lobe. Continuation / walk phase stays vanilla HG.
template <typename Tracer, typename Guiding>
SR_INL SR_HD Vec3 nextEventEstimationVolumeOnce(const SceneView& scene, const Tracer& tracer, Vec3 origin,
                                                Vec3 woVol, const MediumData& med, Rng& rng,
                                                Guiding* guiding) {
    Vec3 result(0.0f);
    if (scene.lightCount <= 0) return result;

    const int nCand = volumeRisCandidateCount(med.g);
    LightSample cands[kVolumeRisMax];
    int lis[kVolumeRisMax];
    float selPdfs[kVolumeRisMax];
    Vec3 rgb[kVolumeRisMax];
    float scatterPdfs[kVolumeRisMax];
    float ws[kVolumeRisMax];
    int nOk = 0;
    for (int i = 0; i < nCand; ++i) {
        float selectPdf = 0.0f;
        const int li = sampleVolumeLightIndex(scene, origin, woVol, med.g, rng.nextFloat(), selectPdf);
        if (li < 0 || selectPdf <= 0.0f) continue;
        LightSample ls;
        if (!sampleLight(scene, li, origin, rng.nextFloat(), rng.nextFloat(), ls) || ls.pdf <= 0.0f ||
            isBlack(ls.radiance))
            continue;
        const float cosTheta = clampf(dot(woVol, ls.wi), -1.0f, 1.0f);
        const float phasePdfL = henyeyGreenstein(cosTheta, med.g);
        if (phasePdfL <= 0.0f) continue;
        const float lightPdf = ls.pdf * selectPdf;
        const Vec3 unshadowed = ls.radiance * (phasePdfL / lightPdf);
        const float w = luminance(vmax(unshadowed, Vec3(0.0f)));
        if (w <= 1e-20f) continue;
        cands[nOk] = ls;
        lis[nOk] = li;
        selPdfs[nOk] = selectPdf;
        rgb[nOk] = unshadowed;
        scatterPdfs[nOk] = volumeScatterPdf(woVol, ls.wi, med, guiding);
        ws[nOk] = w;
        ++nOk;
    }
    float wSum = 0.0f;
    const int pick = risPick(ws, nOk, rng.nextFloat(), wSum);
    if (pick < 0) return result;
    const LightSample& ls = cands[pick];
    const int li = lis[pick];

    float vis = 1.0f;
    float tShadow = 1.0e8f;
    if (scene.lights[li].shadowEnable) {
        if (ls.distance < 1.0e7f) tShadow = ls.distance * (1.0f - 1e-3f);
        vis = shadowVisibility(scene, tracer, origin, ls.wi, tShadow);
        if (vis <= 1e-5f) return result;
    }

    const float lightPdf = ls.pdf * selPdfs[pick];
    const float misW = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, scatterPdfs[pick]);
    result = rgb[pick] * (vis * misW * wSum / (float(nCand) * ws[pick]));

#if !defined(__CUDACC__)
    if (scene.lights[li].shadowEnable)
        result = result * shadowTransmittanceFogVolumes(scene, origin, ls.wi, tShadow, rng);
#endif
    if (med.type != 2 && ls.distance < 1.0e7f) result = result * mediumShadowTr(med, ls.distance);
    return result;
}

template <typename Tracer>
SR_INL SR_HD Vec3 nextEventEstimationVolumeOnce(const SceneView& scene, const Tracer& tracer, Vec3 origin,
                                                Vec3 woVol, const MediumData& med, Rng& rng) {
    return nextEventEstimationVolumeOnce<Tracer, NullGuiding>(scene, tracer, origin, woVol, med, rng,
                                                              nullptr);
}

template <typename Tracer, typename Guiding>
SR_INL SR_HD Vec3 nextEventEstimation(const SceneView& scene, const Tracer& tracer, const SurfaceInteraction& si,
                                      const Material& mat, const Frame& frame, Vec3 wo, Rng& rng,
                                      Guiding* guiding, int mediumIndex = -1) {
    const int n = srMax(1, scene.settings.lightSamples);
    Vec3 sum(0.0f);
    for (int i = 0; i < n; ++i)
        sum += nextEventEstimationOnce(scene, tracer, si, mat, frame, wo, rng, guiding, mediumIndex);
    return sum * (1.0f / float(n));
}

template <typename Tracer>
SR_INL SR_HD Vec3 nextEventEstimation(const SceneView& scene, const Tracer& tracer, const SurfaceInteraction& si,
                                      const Material& mat, const Frame& frame, Vec3 wo, Rng& rng,
                                      int mediumIndex = -1) {
    return nextEventEstimation<Tracer, NullGuiding>(scene, tracer, si, mat, frame, wo, rng, nullptr,
                                                    mediumIndex);
}

template <typename Tracer, typename Guiding>
SR_INL SR_HD Vec3 traceRadiance(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                                Rng& rng, Guiding* guiding, DispersionContext* dispersion = nullptr) {
    Vec3 radiance(0.0f);
    Vec3 throughput(1.0f);
    float bsdfPdf = 0.0f;
    bool specularBounce = true;   // primary rays behave like a specular bounce for MIS
    // Caustics OFF: kill diffuse→specular→light transport (matches the dark shadows
    // produced by opaque shadow rays through glass).
    bool suppressCausticLight = false;
    bool sawNonSpecular = false;
    // Specular suffix after a non-specular bounce — per-light Contribute to Caustics.
    bool causticSuffix = false;
    int depth = 0;
    int passThrough = 0;
    int volumeScatterCount = 0;
    // Last volume scatter (for MIS vs HG×sun volume NEE). Cleared on surface BSDF.
    bool volumePhaseMis = false;
    Vec3 volumeMisWo{0.0f, 1.0f, 0.0f};
    float volumeMisG = 0.0f;
    // Homogeneous medium currently surrounding the ray (-1 = vacuum).
    int currentMedium = -1;
    // Arnold ray_switch: incoming ray type selects the surfaceshader port.
    RayShadeKind rayKind = RayShadeKind::Camera;
    const RenderSettingsData& settings = scene.settings;
    const int maxDepth = settings.integrator == kIntegratorDirectLighting ? 1 : srMax(1, settings.maxDepth);

    while (depth <= maxDepth) {
        RayHit hit;
        const bool didHit = tracer.intersect(origin, direction, kFloatMax, hit);

        // Null-scattering / delta tracking through the active medium (homogeneous or VDB fog).
        if (const MediumData* med = getMedium(scene, currentMedium)) {
            float tMax = didHit ? hit.t : 1.0e6f;
            MediumSample ms;
            MediumData medWalk = *med;
            if (settings.volumeSimilarity != 0)
                medWalk = mediumWithVolumeSimilarity(*med, volumeScatterCount);
#if !defined(__CUDACC__)
            if (medWalk.type == 2 && medWalk.volumeIndex >= 0 && medWalk.volumeIndex < scene.volumeCount &&
                scene.volumes && scene.volumes[medWalk.volumeIndex]) {
                // Prefer analytical AABB exit over Embree proxy (exit face is easy to miss).
                const VolumeGrid& fogVol = *scene.volumes[medWalk.volumeIndex];
                const Bounds3 bb = fogVol.worldBounds();
                if (bb.valid()) {
                    float tEnter = 0.0f;
                    float tExit = tMax;
                    const float* o = &origin.x;
                    const float* d = &direction.x;
                    const float* lo = &bb.lo.x;
                    const float* hi = &bb.hi.x;
                    bool hitAabb = true;
                    for (int axis = 0; axis < 3; ++axis) {
                        const float od = d[axis];
                        if (fabsf(od) < 1e-20f) {
                            if (o[axis] < lo[axis] || o[axis] > hi[axis]) {
                                hitAabb = false;
                                break;
                            }
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
                        tEnter = srMax(tEnter, ta);
                        tExit = srMin(tExit, tb);
                        if (tEnter > tExit) {
                            hitAabb = false;
                            break;
                        }
                    }
                    if (hitAabb && tExit >= 0.0f) tMax = srMin(tMax, tExit);
                }
                ms = sampleMediumVdbFog(fogVol, medWalk, origin, direction, tMax, rng, throughput);
            } else {
                ms = sampleMediumHomogeneous(medWalk, tMax, rng, throughput);
            }
#else
            ms = sampleMediumHomogeneous(medWalk, tMax, rng, throughput);
#endif
            if (ms.absorbed || isBlack(throughput)) break;
            if (ms.scattered) {
                origin = origin + direction * ms.t;
#if !defined(__CUDACC__)
                if (!isBlack(med->emission)) radiance += throughput * med->emission;
#endif
                // Incident direction toward the previous vertex (phase frame).
                const Vec3 woVol = -direction;
                volumePhaseMis = true;
                volumeMisWo = woVol;
                volumeMisG = medWalk.g;

#if !defined(__CUDACC__)
                if (guiding && guiding->active()) {
                    guiding->beginVolumeSegment(origin, woVol);
                    guiding->prepareVolume(origin, woVol, medWalk.g, rng);
                }
#endif

                // Volume next-event estimation with MIS vs phase sampling (PBRT VolPath).
                // Unbiased: light strategy weight = powerHeuristic(pdf_light, pdf_phase);
                // the phase→light strategy is the continuing path (MIS on light hits below).
                // Direct Clamp (0 = off) applies to β · NEE / pNee — the pixel deposit —
                // matching env miss. Clamping raw NEE first left spikes of clamp·β/pNee.
                if (scene.lightCount > 0 && depth < maxDepth) {
                    const float pNee = volumeNeeRouletteP(depth);
                    const bool takeNee = pNee >= 1.0f || rng.nextFloat() < pNee;
                    if (takeNee) {
                        if (xpuEstimatorMatch(settings)) {
                            float selectPdf = 0.0f;
                            const int lightIndex =
                                sampleLightIndex(scene, origin, rng.nextFloat(), selectPdf);
                            LightSample ls;
                            if (lightIndex >= 0 && selectPdf > 0.0f &&
                                sampleLight(scene, lightIndex, origin, rng.nextFloat(), rng.nextFloat(),
                                            ls) &&
                                ls.pdf > 0.0f && !isBlack(ls.radiance)) {
                                const float phase = henyeyGreenstein(dot(woVol, ls.wi), medWalk.g);
                                const float lightPdf = ls.pdf * selectPdf;
                                const float mis =
                                    ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, phase);
                                Vec3 contrib = throughput * ls.radiance *
                                               (phase / srMax(1e-8f, lightPdf)) * mis * (1.0f / pNee);
                                contrib = clampContribution(contrib, settings.clampDirect);
                                if (scene.lights[lightIndex].shadowEnable) {
                                    float tSh = 1.0e8f;
                                    if (ls.distance < 1.0e7f) tSh = ls.distance * (1.0f - 1e-3f);
                                    contrib = contrib * shadowVisibility(scene, tracer, origin, ls.wi, tSh);
#if !defined(__CUDACC__)
                                    contrib = contrib * shadowTransmittanceFogVolumes(scene, origin, ls.wi,
                                                                                      tSh, rng);
#endif
                                }
                                if (medWalk.type != 2 && ls.distance < 1.0e7f)
                                    contrib = contrib * mediumShadowTr(medWalk, ls.distance);
                                radiance += contrib;
                            }
                        } else {
                            int nLight = srMax(1, settings.lightSamples);
                            if (depth >= 4) nLight = 1;
                            Vec3 volDirect(0.0f);
                            for (int lsIdx = 0; lsIdx < nLight; ++lsIdx) {
                                volDirect += nextEventEstimationVolumeOnce(scene, tracer, origin, woVol,
                                                                           medWalk, rng, guiding);
                            }
                            volDirect = volDirect * (1.0f / (float(nLight) * pNee));
                            radiance += clampContribution(throughput * volDirect, settings.clampDirect);
#if !defined(__CUDACC__)
                            if (guiding && guiding->active()) guiding->addScattered(volDirect);
#endif
                        }
                    }
                }

                // Continue: HG, mixed with OpenPGL volume×HG when Indirect Guides is on.
                float phasePdf = 0.0f;
                bool gotGuide = false;
#if !defined(__CUDACC__)
                if (guiding && guiding->active() && guiding->preparedVolume()) {
                    const float pg = guiding->guideProbability();
                    if (rng.nextFloat() < pg) {
                        Vec3 wiWorld;
                        float gPdf = 0.0f;
                        if (guiding->sampleVolume(rng.nextFloat(), rng.nextFloat(), wiWorld, gPdf) &&
                            gPdf > 0.0f) {
                            phasePdf = henyeyGreenstein(clampf(dot(woVol, wiWorld), -1.0f, 1.0f), medWalk.g);
                            const float mixPdf = pg * gPdf + (1.0f - pg) * phasePdf;
                            if (mixPdf > 0.0f && phasePdf > 0.0f) {
                                direction = wiWorld;
                                bsdfPdf = mixPdf;
                                throughput = throughput * (phasePdf / mixPdf);
                                gotGuide = true;
                            }
                        }
                    }
                }
#endif
                if (!gotGuide) {
                    direction = sampleHenyeyGreenstein(woVol, medWalk.g, rng.nextFloat(), rng.nextFloat(),
                                                       phasePdf);
                    bsdfPdf = phasePdf;
#if !defined(__CUDACC__)
                    if (guiding && guiding->active() && guiding->preparedVolume() && phasePdf > 0.0f) {
                        const float pg = guiding->guideProbability();
                        const float gPdf = guiding->pdfVolume(direction);
                        const float mixPdf = pg * gPdf + (1.0f - pg) * phasePdf;
                        if (mixPdf > 0.0f) {
                            throughput = throughput * (phasePdf / mixPdf);
                            bsdfPdf = mixPdf;
                        }
                    }
#endif
                }
#if !defined(__CUDACC__)
                if (guiding && guiding->active()) {
                    const float phaseNow =
                        henyeyGreenstein(clampf(dot(woVol, direction), -1.0f, 1.0f), medWalk.g);
                    const Vec3 weight = bsdfPdf > 0.0f ? Vec3(phaseNow / bsdfPdf) : Vec3(1.0f);
                    guiding->recordBounce(woVol, direction, bsdfPdf, weight, false, 1.0f, 1.0f, 1.0f,
                                          true);
                }
#endif
                specularBounce = false;
                sawNonSpecular = true;
                rayKind = RayShadeKind::DiffuseReflection;
                ++depth;
                ++volumeScatterCount;
                if (depth >= settings.rrStartDepth) {
                    const float q = volumeRussianRouletteQ(throughput);
                    if (rng.nextFloat() > q) break;
#if !defined(__CUDACC__)
                    if (guiding && guiding->active()) guiding->setRussianRoulette(q);
#endif
                    throughput = throughput / q;
                }
                continue;
            }
#if !defined(__CUDACC__)
            // Exited the fog AABB (analytical) before hitting any surface — leave the medium.
            if (medWalk.type == 2 && (!didHit || ms.t + 1e-4f < hit.t)) {
                origin = origin + direction * ms.t;
                currentMedium = -1;
                ++passThrough;
                if (passThrough > 32) break;
                continue;
            }
#endif
            // Reached the surface (or infinity) with Tr already in throughput.
        }

        if (!didHit) {
            // Wireframe diagnostic: empty background (no env).
            if (settings.integrator == kIntegratorWireframe) break;
            if (scene.domeLightIndex >= 0) {
                if (!suppressCausticLight) {
                const LightData& dome = scene.lights[scene.domeLightIndex];
                if (!(causticSuffix && !lightContributesCaustics(dome))) {
                const bool primary = depth == 0 && passThrough == 0;
                if (!(primary && (!settings.envVisibleCamera || !dome.visibleCamera))) {
                    Vec3 envL = domeRadiance(scene, dome, direction, /*nearestTexel=*/depth > 0);
                        if (!isBlack(envL)) {
                        float weight = 1.0f;
                        if (!specularBounce) {
                            const float lp = lightPdfDirection(scene, scene.domeLightIndex, origin, direction,
                                                               origin, direction) *
                                             lightSelectionPdfIndexMaybeVolume(
                                                 scene, origin, scene.domeLightIndex, volumePhaseMis,
                                                 volumeMisWo, volumeMisG);
                            weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                        }
                        Vec3 contrib = throughput * envL * weight;
                        // Primary miss keeps the full sun. After a bounce (including
                        // volume phase→sky) bilinear-vs-PDF mismatch is a firefly —
                        // nearest Le + Direct Clamp, same as surface NEE.
                        if (depth > 0) contrib = clampContribution(contrib, settings.clampDirect);
                        radiance += contrib;
#if !defined(__CUDACC__)
                        if (guiding && guiding->active())
                            guiding->recordBackground(origin, direction, envL, weight);
#else
                        (void)guiding;
#endif
                    }
                }
                }
                }
            }
            if (!suppressCausticLight) {
                const bool primarySun = depth == 0 && passThrough == 0;
                if (!(primarySun && !settings.envVisibleCamera)) {
                    const Vec3 sunL = cameraSunDiscRadiance(scene, origin, direction, bsdfPdf,
                                                            specularBounce, primarySun, causticSuffix,
                                                            volumePhaseMis, volumeMisWo, volumeMisG);
                    if (!isBlack(sunL)) {
                        Vec3 contrib = throughput * sunL;
                        radiance += contrib;
                    }
                }
            }
            break;
        }

        SurfaceInteraction si;
        if (!buildSurfaceInteraction(scene, hit, origin, direction, si)) break;

        const InstanceData& inst = scene.instances[si.instanceIndex];

#if !defined(__CUDACC__)
        // Direct VDB rendering: SDF level-set surface or fog volume entry/exit.
        // Wireframe stays on the triangle proxy (bounds silhouette).
        if (settings.integrator != kIntegratorWireframe && inst.volumeIndex >= 0 &&
            inst.volumeIndex < scene.volumeCount && scene.volumes && scene.volumes[inst.volumeIndex]) {
            const VolumeGrid& vol = *scene.volumes[inst.volumeIndex];
            if (vol.kind() == VolumeGridKind::Fog) {
                if (currentMedium == inst.mediumIndex) {
                    // Second hit on the AABB proxy = leaving the volume.
                    currentMedium = -1;
                    origin = offsetRayOrigin(si.p, si.ng, direction);
                } else {
                    // Enter the medium. Empty AABB corners are OK (dens≈0 → null collisions);
                    // container silhouettes are softened by boundary feather in fromPolygons.
                    currentMedium = inst.mediumIndex;
                    origin = offsetRayOrigin(si.p, -si.ng, direction);
                }
                ++passThrough;
                if (passThrough > 32) break;
                continue;
            }
            if (vol.kind() == VolumeGridKind::Sdf) {
                float tSdf = hit.t;
                Vec3 nSdf;
                // Sphere-trace from near the AABB entry; far bound is analytical AABB exit.
                const float tNear = srMax(0.0f, hit.t - vol.voxelSize());
                if (intersectSdfVolume(vol, origin, direction, tNear, hit.t + 1.0e6f, tSdf, nSdf)) {
                    hit.t = tSdf;
                    si.p = origin + direction * tSdf;
                    si.ng = nSdf;
                    si.ns = nSdf;
                    si.nObject = transformVector(inst.xformInv, nSdf);
                    si.pObject = transformPoint(inst.xformInv, si.p);
                } else {
                    origin = offsetRayOrigin(si.p, si.ng, direction);
                    ++passThrough;
                    if (passThrough > 32) break;
                    continue;
                }
            }
        }
#endif

        // Lights that are hidden from the camera let primary rays pass through.
        if (si.lightIndex >= 0 && depth == 0 && !inst.visibleCamera) {
            origin = offsetRayOrigin(si.p, si.ng, direction);
            ++passThrough;
            if (passThrough > 16) break;
            continue;
        }

        // Wireframe: shade every visible surface hit (including area lights).
        if (settings.integrator == kIntegratorWireframe) {
            radiance += throughput * shadeWireframe(scene, hit, si, direction);
            break;
        }

        // Emission from area light geometry.
        if (si.lightIndex >= 0) {
            // Caustics off: no light transport through specular chains after a
            // diffuse vertex (matches the opaque shadow rays through glass).
            if (suppressCausticLight) break;
            const LightData& hitLight = scene.lights[si.lightIndex];
            if (causticSuffix && !lightContributesCaustics(hitLight)) break;
            const LightData& light = scene.lights[si.lightIndex];
            const Vec3 lightN = light.type == kLightSphere ? si.ng : areaLightNormal(light);
            Vec3 emitted = areaLightEmission(scene, light, direction, lightN);
            if (!isBlack(emitted)) {
                float weight = 1.0f;
                if (!specularBounce) {
                    const float lp = lightPdfDirection(scene, si.lightIndex, origin, direction, si.p, lightN) *
                                     lightSelectionPdfIndexMaybeVolume(scene, origin, si.lightIndex,
                                                                        volumePhaseMis, volumeMisWo,
                                                                        volumeMisG);
                    weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                }
                Vec3 contrib = throughput * emitted * weight;
                // Caustic paths (specular chain) keep more energy — clamp less aggressively.
                if (depth > 0 && !specularBounce)
                    contrib = clampContribution(contrib, settings.clampDirect);
                radiance += contrib;
#if !defined(__CUDACC__)
                if (guiding && guiding->active())
                    guiding->recordLightHit(si.p, -direction, emitted, weight);
#endif
            }
            // Light geometry is opaque and is not shaded further.
            break;
        }

        // Arnold ray_switch: shade with the port matching the *incoming* ray type.
        // Camera rays never use the Solstice caustics port.
        Material mat;
#if !defined(__CUDACC__)
        if (xpuEstimatorMatch(settings) && si.materialIndex >= 0 && si.materialIndex < scene.materialCount &&
            scene.materials) {
            mat = optixpt::evaluateMaps(scene, scene.materials[si.materialIndex], si.uv, si.ns);
        } else
#endif
        {
            Material baseMat = materialForRay(scene, si.materialIndex, rayKind);
            mat = evaluateTexturedMaterial(scene, baseMat, si.uv, si.ns, si.pObject, si.nObject,
                                           si.uvFilterWidth, si.pRef, si.nRef, si.hasPref);
            applyDispersion(mat, dispersion);
        }

        // Two sided shading for opaque surfaces. Winding order varies between
        // DCCs, so back faces are shaded as if their normals pointed at us.
        if (mat.transmission <= 0.0f && mat.doubleSided && dot(si.ns, -direction) < 0.0f) {
            si.ns = -si.ns;
            si.ng = -si.ng;
        }

        // Emissive surfaces (evaluated before opacity cutouts so glowing cutouts work).
        if (mat.emissionStrength > 0.0f && !isBlack(mat.emissionColor)) {
            const bool frontFacing = dot(si.ns, -direction) > 0.0f;
            if (frontFacing || mat.doubleSided)
                radiance += throughput * mat.emissionColor * mat.emissionStrength;
        }

        // Arnold presence / opacity: kills ALL lobes including specular & transmission.
        // opacity = 0 → always pass through; partial → stochastic cutout.
        if (mat.opacity <= 1e-6f || (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity)) {
            origin = offsetRayOrigin(si.p, si.ng, direction);
            ++passThrough;
            if (passThrough > 32) break;
            continue;
        }

        if (settings.integrator == kIntegratorAmbientOcclusion) {
            const Frame frame(dot(si.ns, -direction) < 0.0f ? -si.ns : si.ns);
            const Vec3 wi = frame.toWorld(sampleCosineHemisphere(rng.nextFloat(), rng.nextFloat()));
            const Vec3 aoOrigin = offsetRayOrigin(si.p, si.ng, wi);
            const float dist = settings.aoDistance > 0.0f ? settings.aoDistance : kFloatMax;
            const float visibility = tracer.occluded(aoOrigin, wi, dist) ? 0.0f : 1.0f;
            radiance += throughput * Vec3(visibility);
            break;
        }

        if (depth >= maxDepth) break;

        const Vec3 wo = -direction;
        const Frame frame(si.ns);
#if !defined(__CUDACC__)
        if (guiding && guiding->active()) {
            guiding->beginSegment(si.p, wo);
            if (mat.emissionStrength > 0.0f && !isBlack(mat.emissionColor)) {
                const bool frontFacing = dot(si.ns, wo) > 0.0f;
                if (frontFacing || mat.doubleSided)
                    guiding->recordEmission(mat.emissionColor * mat.emissionStrength, 1.0f);
            }
        }
#endif

        // Arnold / Autodesk Standard Surface base mix (MaterialX):
        //   base_mix = (1 - subsurface) * base * base_color * diffuse
        //            + subsurface * SSS
        // Stochastic selection with probability `subsurface` — WITHOUT 1/p
        // compensation — so expectation matches the mix (energy conserving).
        // Specular sits on top of the SSS body via Fresnel RR at entry.
        //
        // Spectral walk: hero RGB channel, MFP = scale * radius[ch], Chiang α.
        const float sssWeight = saturatef(mat.subsurface);
        if (!xpuEstimatorMatch(settings) && materialSupportsSss(mat) && rng.nextFloat() < sssWeight) {
            // Specular-only material at the ENTRY (dielectric F0 / metal base).
            Material specMat = sssSpecularEntryMaterial(mat);

            const Vec3 woLocalEntry = frame.toLocal(wo);
            const LobeWeights specLw = computeLobes(specMat);
            const float pSpec = sssEntrySpecularProb(specMat, woLocalEntry);

            // Direct specular lighting at entry (works for rough GGX; delta → 0).
            if (pSpec > 0.0f) {
                    const Vec3 nee =
                    nextEventEstimation(scene, tracer, si, specMat, frame, wo, rng, guiding,
                                        currentMedium);
                Vec3 contrib = throughput * nee;
                if (depth > 0) contrib = clampContribution(contrib, settings.clampDirect);
                radiance += contrib;
#if !defined(__CUDACC__)
                if (guiding && guiding->active()) guiding->addScattered(nee);
#endif
            }

            // Fresnel lottery: reflect at entry OR enter the SSS body.
            if (pSpec > 0.0f && rng.nextFloat() < pSpec) {
                throughput /= pSpec;
                // Force the specular lobe (uLobe in specular range).
                const float uSpec = specLw.diffuse + specLw.specular * rng.nextFloat();
                const BsdfSample specBs =
                    bsdfSampleLocal(specMat, woLocalEntry, uSpec, rng.nextFloat(), rng.nextFloat(),
                                    rng.nextFloat());
                if (specBs.pdf > 0.0f && !isBlack(specBs.weight)) {
                    const Vec3 wiWorld = normalize(frame.toWorld(specBs.wi));
#if !defined(__CUDACC__)
                    if (guiding && guiding->active())
                        guiding->recordBounce(si.ns, wiWorld, specBs.pdf, specBs.weight, true,
                                              mat.roughness, computeLobes(specMat).eta, 1.0f);
#endif
                    throughput *= specBs.weight;
                    origin = offsetRayOrigin(si.p, si.ng, wiWorld);
                    direction = wiWorld;
                    bsdfPdf = specBs.pdf;
                    specularBounce = specBs.specular;
                    ++depth;
                    continue;
                }
                break;
            }
            if (pSpec > 0.0f && pSpec < 0.999f) throughput /= (1.0f - pSpec);

            const SssWalkResult walk = sampleSssRandomWalk(scene, tracer, si, wo, mat, rng);
            Material lambert = sssExitLambertMaterial();
            SurfaceInteraction ssSi = si;
            ssSi.p = walk.exitP;
            ssSi.ns = walk.exitN;
            ssSi.ng = walk.exitN;
            const Frame ssFrame(walk.exitN);
            // NEE at SSS exit (lightSamples is handled inside nextEventEstimation).
            const Vec3 nee =
                nextEventEstimation(scene, tracer, ssSi, lambert, ssFrame, walk.exitWo, rng, guiding,
                                    currentMedium);
            Vec3 contrib = throughput * walk.pathWeight * nee;
            if (depth > 0) contrib = clampContribution(contrib, settings.clampDirect);
            radiance += contrib;
#if !defined(__CUDACC__)
            if (guiding && guiding->active()) guiding->addScattered(walk.pathWeight * nee);
#endif
            const BsdfSample ssBs =
                bsdfSampleLocal(lambert, ssFrame.toLocal(walk.exitWo), rng.nextFloat(), rng.nextFloat(),
                                rng.nextFloat(), rng.nextFloat());
            if (ssBs.pdf > 0.0f && !isBlack(ssBs.weight)) {
                const Vec3 wiWorld = normalize(ssFrame.toWorld(ssBs.wi));
#if !defined(__CUDACC__)
                if (guiding && guiding->active())
                    guiding->recordBounce(walk.exitN, wiWorld, ssBs.pdf, walk.pathWeight * ssBs.weight,
                                          false, 1.0f, 1.0f, 1.0f);
#endif
                throughput *= walk.pathWeight * ssBs.weight;
                origin = offsetRayOrigin(walk.exitP, walk.exitN, wiWorld);
                direction = wiWorld;
                bsdfPdf = ssBs.pdf;
                specularBounce = false;
                ++depth;
                continue;
            }
            break;
        }

        // Complementary BRDF path: selected with probability (1 - subsurface).
        // No 1/(1-w) boost — that previously made diffuse+SSS additive.

        const LobeWeights lw = computeLobes(mat);
#if !defined(__CUDACC__)
        const bool guideReady =
            guiding && guiding->active() && !lw.delta && !isNearSpecularLobe(lw) &&
            lw.diffuse > 1e-4f && guiding->prepare(si.p, si.ns, rng);
#else
        const bool guideReady = false;
#endif

        // NEE on diffuse after a caustic-disabled specular/transmission bounce is suppressed.
        if (!(suppressCausticLight && !specularBounce)) {
            const Vec3 nee = nextEventEstimation(scene, tracer, si, mat, frame, wo, rng, guiding,
                                                 currentMedium);
            Vec3 contrib = throughput * nee;
            if (depth > 0 && !specularBounce)
                contrib = clampContribution(contrib, settings.clampDirect);
            radiance += contrib;
#if !defined(__CUDACC__)
            if (guiding && guiding->active()) guiding->addScattered(nee);
#endif
        }
        const Vec3 woLocal = frame.toLocal(wo);
        BsdfSample bs;
        bool gotSample = false;
#if !defined(__CUDACC__)
        if (guideReady) {
            const float pg = guiding->guideProbability();
            if (rng.nextFloat() < pg) {
                Vec3 wiWorld;
                float gPdf = 0.0f;
                if (guiding->sample(rng.nextFloat(), rng.nextFloat(), wiWorld, gPdf) && gPdf > 0.0f) {
                    const Vec3 wiLocal = frame.toLocal(wiWorld);
                    const BsdfEval be = bsdfEvalLocal(mat, woLocal, wiLocal);
                    if (be.pdf > 0.0f && !isBlack(be.f)) {
                        const float mixPdf = pg * gPdf + (1.0f - pg) * be.pdf;
                        if (mixPdf > 0.0f) {
                            bs.wi = wiLocal;
                            bs.pdf = mixPdf;
                            bs.weight = be.f * (fabsf(wiLocal.z) / mixPdf);
                            bs.specular = false;
                            bs.transmitted = wiLocal.z < 0.0f;
                            gotSample = true;
                        }
                    }
                }
            }
        }
#endif
        if (!gotSample) {
            bs = sampleBsdfDevice(settings, mat, woLocal, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                 rng.nextFloat());
#if !defined(__CUDACC__)
            if (bs.pdf > 0.0f && guideReady && !bs.specular) {
                const float pg = guiding->guideProbability();
                const float gPdf = guiding->pdf(normalize(frame.toWorld(bs.wi)));
                const float mixPdf = pg * gPdf + (1.0f - pg) * bs.pdf;
                if (mixPdf > 0.0f) {
                    bs.weight *= bs.pdf / mixPdf;
                    bs.pdf = mixPdf;
                }
            }
#endif
        }
        if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;

        // Indirect Clamp applies only to path *contributions* (pixel radiance), like
        // Arnold/Karma — never to BSDF bounce weights (those are not in pixel units).
        const Vec3 weight = bs.weight;

        const Vec3 wiWorld = normalize(frame.toWorld(bs.wi));
        if (!shadingNormalConsistent(si.ng, si.ns, wo, wiWorld)) break;
#if !defined(__CUDACC__)
        // Always record the bounce so radiance can propagate through delta /
        // near-spec glass (OpenPGL needs isDelta segments). Sampling the guide
        // still stays diffuse-only (guideReady above).
        if (guiding && guiding->active()) {
            const bool deltaSeg = bs.specular || isNearSpecularLobe(lw);
            guiding->recordBounce(si.ns, wiWorld, bs.pdf, weight, deltaSeg, mat.roughness, lw.eta,
                                  1.0f);
        }
#endif

        throughput *= weight;
        if (bs.transmitted && !xpuEstimatorMatch(settings))
            throughput = applyFakeDispersionThroughput(throughput, mat, dispersion);
        if (!isFinite(throughput) || isBlack(throughput)) break;

        origin = offsetRayOrigin(si.p, si.ng, wiWorld);
        direction = wiWorld;
        bsdfPdf = bs.pdf;
        specularBounce = bs.specular;
        volumePhaseMis = false;
        // Homogeneous volume interior: transmission across a geo-authored medium
        // boundary enters/exits the medium (PBRT MediumInterface on the prim).
        if (bs.transmitted && inst.mediumIndex >= 0 && mediumIsActive(scene, inst.mediumIndex)) {
            const bool entering = dot(si.ng, wiWorld) < 0.0f;
            currentMedium = entering ? inst.mediumIndex : -1;
        }
        // Child ray type for the next hit (Arnold ray_switch on that surface).
        rayKind = nextRayShadeKind(bs, lw);
        // Caustic bookkeeping follows the near-specular classification (same as BDPT)
        // so low-roughness glass counts as a caustic chain, not as diffuse transport.
        const bool causticBounce = bs.specular || isNearSpecularLobe(lw);
        if (settings.caustics == 0 && causticBounce && sawNonSpecular) suppressCausticLight = true;
        if (causticBounce && sawNonSpecular) causticSuffix = true;
        if (!causticBounce) {
            sawNonSpecular = true;
            causticSuffix = false;
        }
        ++depth;

        // Russian roulette.
        if (depth >= srMax(1, settings.rrStartDepth)) {
            const float q = clampf(maxComponent(throughput), 0.05f, 1.0f);
            if (rng.nextFloat() > q) break;
#if !defined(__CUDACC__)
            if (guiding && guiding->active()) guiding->setRussianRoulette(q);
#endif
            throughput /= q;
        }
    }

    if (!isFinite(radiance)) return Vec3(0.0f);
    return radiance;
}

template <typename Tracer>
SR_INL SR_HD Vec3 traceRadiance(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                                Rng& rng, DispersionContext* dispersion = nullptr) {
    return traceRadiance<Tracer, NullGuiding>(scene, tracer, origin, direction, rng, nullptr, dispersion);
}

}  // namespace sol
