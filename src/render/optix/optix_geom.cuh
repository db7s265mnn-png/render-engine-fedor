// Surface reconstruction + thin-lens camera for shade / init kernels.
#pragma once

#include "render/optix/optix_wavefront.cuh"

namespace sol {

struct Surf {
    Vec3 p{0.0f};
    Vec3 ng{0.0f, 0.0f, 1.0f};
    Vec3 ns{0.0f, 0.0f, 1.0f};
    Vec2 uv{0.0f, 0.0f};
    int instanceIndex = -1;
    int materialIndex = -1;
    int lightIndex = -1;
};

__device__ inline void cameraRay(const SceneView& scene, float pixelX, float pixelY, float lensU,
                                 float lensV, Vec3& origin, Vec3& direction) {
    const CameraData& cam = scene.camera;
    const float resX = float(srMax(1, scene.settings.resolutionX));
    const float resY = float(srMax(1, scene.settings.resolutionY));
    const float sensorHeight = cam.sensorWidth * (resY / resX);
    const float sx = (pixelX / resX - 0.5f) * cam.sensorWidth;
    const float sy = (0.5f - pixelY / resY) * sensorHeight;
    Vec3 dirCam = normalize(Vec3(sx, sy, -srMax(1e-3f, cam.focalLength)));
    Vec3 originCam(0.0f, 0.0f, 0.0f);
    if (cam.fStop > 0.0f) {
        const float lensRadius = (cam.focalLength * 0.001f) / (2.0f * cam.fStop);
        const Vec2 lens = sampleConcentricDisk(lensU, lensV) * lensRadius;
        const float ft = srMax(1e-4f, cam.focusDistance) / srMax(1e-6f, -dirCam.z);
        const Vec3 focusPoint = dirCam * ft;
        originCam = Vec3(lens.x, lens.y, 0.0f);
        dirCam = normalize(focusPoint - originCam);
    }
    origin = transformPoint(cam.cameraToWorld, originCam);
    direction = normalize(transformVector(cam.cameraToWorld, dirCam));
}

__device__ inline void pinholeRay(const SceneView& scene, float pixelX, float pixelY, Vec3& origin,
                                  Vec3& direction) {
    cameraRay(scene, pixelX, pixelY, 0.5f, 0.5f, origin, direction);
}

__device__ inline bool buildSurf(const SceneView& scene, const GpuHit& hit, Vec3 origin, Vec3 dir,
                                 Surf& si) {
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

__device__ inline Material gpuMaterialAt(const SceneView& scene, int index) {
    Material fallback;
    fallback.baseColor = Vec3(0.7f, 0.7f, 0.7f);
    fallback.roughness = 0.5f;
    if (index < 0 || index >= scene.materialCount || !scene.materials) return fallback;
    return scene.materials[index];
}

// Shadow-port material (Arnold ray_switch). Does not include shading.h.
__device__ inline Material gpuMaterialForShadow(const SceneView& scene, int baseIndex) {
    const Material base = gpuMaterialAt(scene, baseIndex);
    const int slot = base.raySwitch.shadow;
    if (slot < 0 || slot >= scene.materialCount || !scene.materials) return base;
    return scene.materials[slot];
}

__device__ inline Material gpuMaterialForCausticSlot(const SceneView& scene, int baseIndex) {
    const Material base = gpuMaterialAt(scene, baseIndex);
    int slot = base.raySwitch.caustics;
    if (slot < 0) slot = base.raySwitch.specularTransmission;
    if (slot < 0 || slot >= scene.materialCount || !scene.materials) return base;
    return scene.materials[slot];
}

}  // namespace sol
