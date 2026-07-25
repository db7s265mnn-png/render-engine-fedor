#include "nodes/stage.h"

#include <QStringList>
#include <map>

namespace sol {

QString StagePrim::typeName() const {
    switch (type) {
        case PrimType::Mesh: return "Mesh";
        case PrimType::Camera: return "Camera";
        case PrimType::Scope: return "Scope";
        case PrimType::Light:
            switch (light.type) {
                case kLightDistant: return "DistantLight";
                case kLightRect: return "RectLight";
                case kLightDisk: return "DiskLight";
                case kLightSphere: return "SphereLight";
                case kLightDome: return "DomeLight";
                case kLightPoint: return "PointLight";
                default: return "Light";
            }
    }
    return "Prim";
}

StagePrim* Stage::find(const QString& path) {
    for (StagePrim& prim : prims) {
        if (prim.path == path) return &prim;
    }
    return nullptr;
}

const StagePrim* Stage::find(const QString& path) const {
    for (const StagePrim& prim : prims) {
        if (prim.path == path) return &prim;
    }
    return nullptr;
}

QString Stage::uniquePath(const QString& desiredPath) const {
    if (!find(desiredPath)) return desiredPath;
    for (int i = 1; i < 100000; ++i) {
        const QString candidate = desiredPath + QString::number(i);
        if (!find(candidate)) return candidate;
    }
    return desiredPath;
}

void Stage::addPrim(StagePrim prim) {
    prim.path = uniquePath(prim.path);
    prims.push_back(std::move(prim));
}

void Stage::appendFrom(const Stage& other) {
    for (const StagePrim& prim : other.prims) addPrim(prim);
    if (other.settingsAuthored && !settingsAuthored) {
        settings = other.settings;
        settingsAuthored = true;
    }
    if (renderCameraPath.isEmpty()) renderCameraPath = other.renderCameraPath;
}

int Stage::countOfType(PrimType type) const {
    int count = 0;
    for (const StagePrim& prim : prims) {
        if (prim.type == type && prim.active) ++count;
    }
    return count;
}

QStringList Stage::paths() const {
    QStringList result;
    result.reserve(int(prims.size()));
    for (const StagePrim& prim : prims) result << prim.path;
    return result;
}

ScenePtr Stage::toScene() const {
    auto scene = std::make_shared<Scene>();
    scene->settings = settings;

    std::map<const Mesh*, int> meshIndexCache;
    std::map<const EnvironmentMap*, int> envIndexCache;

    for (const StagePrim& prim : prims) {
        if (!prim.active) continue;
        switch (prim.type) {
            case PrimType::Mesh: {
                if (!prim.mesh || prim.mesh->indices.empty()) break;
                int meshIndex = -1;
                auto it = meshIndexCache.find(prim.mesh.get());
                if (it != meshIndexCache.end()) {
                    meshIndex = it->second;
                } else {
                    meshIndex = scene->addMesh(prim.mesh);
                    meshIndexCache[prim.mesh.get()] = meshIndex;
                }
                const int materialIndex = scene->addMaterial(prim.material);
                InstanceData inst;
                inst.xform = prim.xform;
                inst.xformInv = inverse(prim.xform);
                inst.meshIndex = meshIndex;
                inst.materialIndex = materialIndex;
                inst.lightIndex = -1;
                inst.visibleCamera = 1;
                scene->instances.push_back(inst);

                PrimRecord record;
                record.path = prim.path.toStdString();
                record.type = prim.typeName().toStdString();
                record.sourceNode = prim.sourceNode.toStdString();
                record.instanceIndex = int(scene->instances.size()) - 1;
                record.pointCount = static_cast<long long>(prim.mesh->positions.size());
                record.triangleCount = static_cast<long long>(prim.mesh->triangleCount());
                scene->prims.push_back(record);
                break;
            }
            case PrimType::Light: {
                LightData light = prim.light;
                light.xform = prim.xform;
                light.xformInv = inverse(prim.xform);
                if (prim.environment) {
                    auto it = envIndexCache.find(prim.environment.get());
                    if (it != envIndexCache.end()) {
                        light.envIndex = it->second;
                    } else {
                        light.envIndex = scene->addEnvMap(prim.environment);
                        envIndexCache[prim.environment.get()] = light.envIndex;
                    }
                } else {
                    light.envIndex = -1;
                }
                scene->lights.push_back(light);

                PrimRecord record;
                record.path = prim.path.toStdString();
                record.type = prim.typeName().toStdString();
                record.sourceNode = prim.sourceNode.toStdString();
                record.lightIndex = int(scene->lights.size()) - 1;
                scene->prims.push_back(record);
                break;
            }
            case PrimType::Camera: {
                const bool selected = renderCameraPath.isEmpty() ? !scene->cameraAuthored
                                                                 : prim.path == renderCameraPath;
                if (selected) {
                    scene->camera = prim.camera;
                    scene->camera.cameraToWorld = prim.xform;
                    scene->cameraAuthored = true;
                }
                PrimRecord record;
                record.path = prim.path.toStdString();
                record.type = "Camera";
                record.sourceNode = prim.sourceNode.toStdString();
                scene->prims.push_back(record);
                break;
            }
            case PrimType::Scope: {
                PrimRecord record;
                record.path = prim.path.toStdString();
                record.type = "Scope";
                record.sourceNode = prim.sourceNode.toStdString();
                scene->prims.push_back(record);
                break;
            }
        }
    }

    scene->finalize();
    if (!scene->cameraAuthored) scene->frameCameraOnContents();
    return scene;
}

}  // namespace sol
