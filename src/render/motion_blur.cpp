#include "render/motion_blur.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "core/log.h"
#include "nodes/stage.h"

namespace sol {
namespace {

Mat4 cameraWorldFromPrim(const StagePrim& prim) {
    // Stage::toScene assigns camera.cameraToWorld = prim.xform.
    return prim.xform;
}

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

    // Never mutate Mesh objects still owned by the node-graph cache.
    for (MeshPtr& mesh : scene.meshes) {
        if (mesh && mesh.use_count() > 1) mesh = std::make_shared<Mesh>(*mesh);
        if (mesh) mesh->motionPositions.clear();
    }

    std::unordered_map<std::string, int> pathToInstance;
    std::unordered_map<std::string, int> pathToMesh;
    for (const PrimRecord& prim : scene.prims) {
        if (prim.instanceIndex < 0) continue;
        const int instIdx = prim.instanceIndex;
        if (instIdx < 0 || instIdx >= int(scene.instances.size())) continue;
        pathToInstance[prim.path] = instIdx;
        const int meshIdx = scene.instances[size_t(instIdx)].meshIndex;
        if (meshIdx >= 0 && meshIdx < int(scene.meshes.size())) pathToMesh[prim.path] = meshIdx;
    }

    const int instanceCount = int(scene.instances.size());
    try {
        scene.motionXforms.assign(size_t(instanceCount) * size_t(keys), Mat4::identity());
        scene.cameraMotionXforms.assign(size_t(keys), scene.camera.cameraToWorld);
        // Snapshot shutter-center xforms before key cooks mutate geometry.
        scene.pickXforms.resize(size_t(instanceCount));
        for (int i = 0; i < instanceCount; ++i) scene.pickXforms[size_t(i)] = scene.instances[size_t(i)].xform;
    } catch (const std::bad_alloc&) {
        logError("Motion blur: out of memory allocating transform keys; disabling");
        scene.settings.motionBlur = 0;
        scene.motionXforms.clear();
        scene.cameraMotionXforms.clear();
        scene.pickXforms.clear();
        return;
    }

    for (int i = 0; i < instanceCount; ++i) {
        for (int k = 0; k < keys; ++k)
            scene.motionXforms[size_t(i) * size_t(keys) + size_t(k)] = scene.instances[size_t(i)].xform;
        scene.instances[size_t(i)].motionKeyOffset = i * keys;
        scene.instances[size_t(i)].motionKeyCount = keys;
    }

    bool needsResample = false;
    for (const NodePtr& node : graph.nodes()) {
        if (node && node->dependsOnTime()) {
            needsResample = true;
            break;
        }
    }
    // Also resample Alembic/USD on the first MB enable while cache is still unknown.
    if (!needsResample) {
        for (const NodePtr& node : graph.nodes()) {
            if (!node) continue;
            const QString type = node->typeName();
            if (type == QLatin1String("alembic") || type == QLatin1String("usd")) {
                needsResample = true;
                break;
            }
        }
    }

    if (!needsResample) {
        // Nothing moves with time — keep duplicated center keys and return.
        logInfo("Motion blur: no time-dependent geometry; using static shutter keys");
        scene.refreshMeshViews();
        return;
    }

    std::vector<std::vector<std::vector<Vec3>>> meshKeyPositions(scene.meshes.size());
    const QString renderCameraPath;  // empty → first authored camera in each key stage

    for (int k = 0; k < keys; ++k) {
        const double t =
            openTime + (closeTime - openTime) * (double(k) / double(std::max(1, keys - 1)));
        CookContext ctx = centerContext;
        ctx.time = t;
        ctx.frame = 1 + int(std::lround(t * std::max(1e-6, centerContext.fps)));
        ctx.hasSuggestedRange = false;
        ctx.warnings.clear();
        ctx.errors.clear();

        for (const NodePtr& node : graph.nodes()) {
            if (!node) continue;
            const QString type = node->typeName();
            if (type == QLatin1String("alembic") || type == QLatin1String("usd") || node->dependsOnTime())
                graph.markDirty(node.get());
        }

        StagePtr stage;
        try {
            stage = graph.cookDisplay(ctx);
        } catch (const std::bad_alloc&) {
            logError("Motion blur: out of memory while cooking shutter key " + std::to_string(k));
            scene.settings.motionBlur = 0;
            scene.motionXforms.clear();
            scene.cameraMotionXforms.clear();
            scene.pickXforms.clear();
            for (MeshPtr& mesh : scene.meshes) {
                if (mesh) mesh->motionPositions.clear();
            }
            for (InstanceData& inst : scene.instances) {
                inst.motionKeyCount = 1;
                inst.motionKeyOffset = 0;
            }
            return;
        }
        if (!stage) continue;

        // Camera key from stage (same rule as Stage::toScene).
        bool gotCamera = false;
        for (const StagePrim& prim : stage->prims) {
            if (prim.type != PrimType::Camera) continue;
            const bool selected = renderCameraPath.isEmpty() ? !gotCamera : (prim.path == renderCameraPath);
            if (!selected) continue;
            scene.cameraMotionXforms[size_t(k)] = cameraWorldFromPrim(prim);
            gotCamera = true;
        }

        for (const StagePrim& prim : stage->prims) {
            if (prim.type != PrimType::Mesh || !prim.mesh) continue;
            const std::string path = prim.path.toStdString();
            auto instIt = pathToInstance.find(path);
            if (instIt == pathToInstance.end()) continue;
            const int dstInst = instIt->second;
            if (dstInst < 0 || dstInst >= instanceCount) continue;

            scene.motionXforms[size_t(dstInst) * size_t(keys) + size_t(k)] = prim.xform;

            auto meshIt = pathToMesh.find(path);
            if (meshIt == pathToMesh.end()) continue;
            const int dstMesh = meshIt->second;
            if (dstMesh < 0 || dstMesh >= int(scene.meshes.size())) continue;
            const MeshPtr& dst = scene.meshes[size_t(dstMesh)];
            if (!dst) continue;
            if (prim.mesh->positions.size() != dst->positions.size() ||
                prim.mesh->indices.size() != dst->indices.size())
                continue;

            try {
                if (meshKeyPositions[size_t(dstMesh)].empty())
                    meshKeyPositions[size_t(dstMesh)].assign(size_t(keys), std::vector<Vec3>());
                meshKeyPositions[size_t(dstMesh)][size_t(k)] = prim.mesh->positions;
            } catch (const std::bad_alloc&) {
                logError("Motion blur: out of memory copying deformation key; "
                         "falling back to transform blur only");
                meshKeyPositions[size_t(dstMesh)].clear();
            }
        }
    }

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
        // Keep shutter-center bounds for framing / UI even after key0 becomes open.
        const Bounds3 centerBounds = mesh->bounds;
        try {
            mesh->positions = std::move(keyPos[0]);
            mesh->motionPositions.clear();
            mesh->motionPositions.reserve(size_t(keys - 1));
            for (int k = 1; k < keys; ++k) mesh->motionPositions.push_back(std::move(keyPos[size_t(k)]));
            if (centerBounds.valid())
                mesh->bounds = centerBounds;
            else
                mesh->computeBounds();
        } catch (const std::bad_alloc&) {
            logError("Motion blur: out of memory installing deformation keys");
            mesh->motionPositions.clear();
        }
    }

    // Keep InstanceData::xform at shutter center for picks / gizmos / framing.
    // Embree beauty still uses motionXforms when motionKeyCount > 1.
    for (int i = 0; i < instanceCount; ++i) {
        if (size_t(i) < scene.pickXforms.size())
            scene.instances[size_t(i)].xform = scene.pickXforms[size_t(i)];
        scene.instances[size_t(i)].xformInv = inverse(scene.instances[size_t(i)].xform);
    }

    logInfo("Motion blur: " + std::to_string(keys) + " keys, shutter length " +
            std::to_string(shutterLen) + " frames (centered)");
    scene.refreshMeshViews();
}

}  // namespace sol
