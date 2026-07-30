#include "render/motion_blur.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "core/log.h"
#include "nodes/stage.h"

namespace sol {
namespace {

}  // namespace

void attachMotionBlurKeys(NodeGraph& graph, const CookContext& centerContext, Scene& scene) {
    if (!scene.settings.motionBlur) return;
    if (scene.settings.backend == kBackendGpuOptix) {
        logWarning("Motion blur is supported on CPU / Embree only; ignoring for OptiX");
        return;
    }

    int keys = std::clamp(scene.settings.motionKeys, 2, 8);
    const float shutterLen = std::max(0.0f, scene.settings.shutterLength);
    if (shutterLen <= 1e-8f) {
        scene.settings.motionBlur = 0;
        return;
    }

    double openTime = 0.0, closeTime = 0.0;
    shutterIntervalSeconds(centerContext.time, centerContext.fps, shutterLen, openTime, closeTime);

    scene.camera.shutterOpen = 0.0f;
    scene.camera.shutterClose = 1.0f;

    // Map prim path -> instance / mesh for the center scene.
    std::unordered_map<std::string, int> pathToInstance;
    std::unordered_map<std::string, int> pathToMesh;
    for (const PrimRecord& prim : scene.prims) {
        if (prim.instanceIndex >= 0) pathToInstance[prim.path] = prim.instanceIndex;
    }
    // Prefer matching by instance's mesh via prim records that carry mesh geometry.
    for (const PrimRecord& prim : scene.prims) {
        if (prim.instanceIndex < 0) continue;
        const int instIdx = prim.instanceIndex;
        if (instIdx < 0 || instIdx >= int(scene.instances.size())) continue;
        const int meshIdx = scene.instances[size_t(instIdx)].meshIndex;
        if (meshIdx >= 0) pathToMesh[prim.path] = meshIdx;
    }

    const int instanceCount = int(scene.instances.size());
    scene.motionXforms.assign(size_t(instanceCount) * size_t(keys), Mat4::identity());
    scene.cameraMotionXforms.assign(size_t(keys), scene.camera.cameraToWorld);

    // Seed key slots with the center-frame transforms (stable if a key cook fails).
    for (int i = 0; i < instanceCount; ++i) {
        for (int k = 0; k < keys; ++k)
            scene.motionXforms[size_t(i) * size_t(keys) + size_t(k)] = scene.instances[size_t(i)].xform;
        scene.instances[size_t(i)].motionKeyOffset = i * keys;
        scene.instances[size_t(i)].motionKeyCount = keys;
    }

    // Deformation slots (lazy — only allocated when topology matches).
    std::vector<std::vector<std::vector<Vec3>>> meshKeyPositions(scene.meshes.size());

    for (int k = 0; k < keys; ++k) {
        const double t = (keys == 1) ? centerContext.time
                                     : openTime + (closeTime - openTime) * (double(k) / double(keys - 1));
        CookContext ctx = centerContext;
        ctx.time = t;
        ctx.frame = 1 + int(std::lround(t * centerContext.fps));
        ctx.hasSuggestedRange = false;
        ctx.warnings.clear();
        ctx.errors.clear();

        graph.markTimeDependentDirty();
        StagePtr stage = graph.cookDisplay(ctx);
        if (!stage) continue;
        ScenePtr keyScene = stage->toScene();
        if (!keyScene) continue;

        // Camera at this shutter sample.
        scene.cameraMotionXforms[size_t(k)] = keyScene->camera.cameraToWorld;

        // Match instances / meshes by prim path.
        for (const PrimRecord& prim : keyScene->prims) {
            if (prim.instanceIndex < 0) continue;
            auto instIt = pathToInstance.find(prim.path);
            if (instIt == pathToInstance.end()) continue;
            const int dstInst = instIt->second;
            const int srcInst = prim.instanceIndex;
            if (srcInst < 0 || srcInst >= int(keyScene->instances.size())) continue;
            if (dstInst < 0 || dstInst >= instanceCount) continue;

            const Mat4 xform = keyScene->instances[size_t(srcInst)].xform;
            scene.motionXforms[size_t(dstInst) * size_t(keys) + size_t(k)] = xform;

            // Deformation: copy positions when topology matches.
            auto meshIt = pathToMesh.find(prim.path);
            if (meshIt == pathToMesh.end()) continue;
            const int dstMesh = meshIt->second;
            const int srcMesh = keyScene->instances[size_t(srcInst)].meshIndex;
            if (dstMesh < 0 || dstMesh >= int(scene.meshes.size())) continue;
            if (srcMesh < 0 || srcMesh >= int(keyScene->meshes.size())) continue;
            const MeshPtr& src = keyScene->meshes[size_t(srcMesh)];
            const MeshPtr& dst = scene.meshes[size_t(dstMesh)];
            if (!src || !dst) continue;
            if (src->positions.size() != dst->positions.size() || src->indices.size() != dst->indices.size())
                continue;

            if (meshKeyPositions[size_t(dstMesh)].empty())
                meshKeyPositions[size_t(dstMesh)].assign(size_t(keys), std::vector<Vec3>());
            meshKeyPositions[size_t(dstMesh)][size_t(k)] = src->positions;
        }
    }

    // Attach deformation keys onto center meshes (key 0 = current positions).
    for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
        auto& keyPos = meshKeyPositions[mi];
        if (keyPos.empty()) continue;
        MeshPtr& mesh = scene.meshes[mi];
        if (!mesh) continue;
        bool complete = true;
        for (int k = 0; k < keys; ++k) {
            if (keyPos[size_t(k)].size() != mesh->positions.size()) {
                complete = false;
                break;
            }
        }
        if (!complete) continue;
        // Replace center positions with key 0 sample; store remaining keys as motion.
        mesh->positions = std::move(keyPos[0]);
        mesh->motionPositions.clear();
        mesh->motionPositions.reserve(size_t(keys - 1));
        for (int k = 1; k < keys; ++k) mesh->motionPositions.push_back(std::move(keyPos[size_t(k)]));
        mesh->computeBounds();
    }

    // Keep InstanceData.xform as shutter-center (middle key) for non-MB code paths.
    if (keys >= 2) {
        const int mid = keys / 2;
        for (int i = 0; i < instanceCount; ++i) {
            scene.instances[size_t(i)].xform = scene.motionXforms[size_t(i) * size_t(keys) + size_t(mid)];
            scene.instances[size_t(i)].xformInv = inverse(scene.instances[size_t(i)].xform);
        }
        scene.camera.cameraToWorld = scene.cameraMotionXforms[size_t(mid)];
    }

    logInfo("Motion blur: " + std::to_string(keys) + " keys, shutter length " +
            std::to_string(shutterLen) + " frames (centered)");
    scene.refreshMeshViews();
}

}  // namespace sol
