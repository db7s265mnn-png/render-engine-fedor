#include "nodes/stage.h"

#include <QStringList>
#include <map>
#include <vector>

#include "scene/displace.h"

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

QString Stage::addPrim(StagePrim prim) {
    prim.path = uniquePath(prim.path);
    const QString path = prim.path;
    prims.push_back(std::move(prim));
    return path;
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
                Material material = prim.material;
                material.baseColorTex = scene->addTexture(prim.baseColorTexture);
                material.roughnessTex = scene->addTexture(prim.roughnessTexture);
                material.metallicTex = scene->addTexture(prim.metallicTexture);
                material.opacityTex = scene->addTexture(prim.opacityTexture);
                material.emissionTex = scene->addTexture(prim.emissionTexture);
                material.normalTex = scene->addTexture(prim.normalTexture);
                material.bumpTex = scene->addTexture(prim.bumpTexture);
                material.displacementTex = scene->addTexture(prim.displacementTexture);
                material.subsurfaceTex = scene->addTexture(prim.subsurfaceTexture);
                material.specularColorTex = scene->addTexture(prim.specularColorTexture);
                material.transmissionColorTex = scene->addTexture(prim.transmissionColorTexture);

                // Append shade-time procedurals; remap local child / texture indices.
                if (!prim.procedurals.empty()) {
                    // Map each proceduralImages[i] → absolute scene texture index
                    // (addTexture may skip empty images — never assume dense texBase+i).
                    std::vector<int> localTexToScene(prim.proceduralImages.size(), -1);
                    for (size_t i = 0; i < prim.proceduralImages.size(); ++i)
                        localTexToScene[i] = scene->addTexture(prim.proceduralImages[i]);
                    const int procBase = int(scene->procedurals.size());
                    auto remapProc = [&](int& idx) {
                        if (idx >= 0) idx += procBase;
                    };
                    auto remapTex = [&](int& idx) {
                        if (idx < 0) return;
                        if (idx >= int(localTexToScene.size())) {
                            idx = -1;
                            return;
                        }
                        idx = localTexToScene[size_t(idx)];
                    };
                    auto remapRoot = [&](int& idx) {
                        if (idx >= 0) idx += procBase;
                    };
                    for (ProceduralNode node : prim.procedurals) {
                        switch (node.op) {
                            case kProcImage:
                                remapTex(node.in0);
                                remapProc(node.in1);  // texcoord / place2d graph
                                break;
                            case kProcPlace2d:
                                remapProc(node.in0);  // texcoord child
                                break;
                            case kProcTriplanar:
                                remapTex(node.in0);
                                remapTex(node.in1);
                                remapTex(node.in2);
                                // Optional future: position/normal children on in3 — keep remap for safety.
                                remapProc(node.in3);
                                break;
                            default:
                                remapProc(node.in0);
                                remapProc(node.in1);
                                remapProc(node.in2);
                                remapProc(node.in3);
                                break;
                        }
                        scene->procedurals.push_back(node);
                    }
                    remapRoot(material.baseColorProc);
                    remapRoot(material.roughnessProc);
                    remapRoot(material.metallicProc);
                    remapRoot(material.opacityProc);
                    remapRoot(material.emissionProc);
                    remapRoot(material.normalProc);
                    remapRoot(material.subsurfaceProc);
                    remapRoot(material.bumpProc);
                    remapRoot(material.displacementProc);
                    remapRoot(material.specularColorProc);
                    remapRoot(material.transmissionColorProc);
                    auto clampRoot = [&](int& idx) {
                        if (idx >= int(scene->procedurals.size())) idx = -1;
                    };
                    clampRoot(material.baseColorProc);
                    clampRoot(material.roughnessProc);
                    clampRoot(material.metallicProc);
                    clampRoot(material.opacityProc);
                    clampRoot(material.emissionProc);
                    clampRoot(material.normalProc);
                    clampRoot(material.subsurfaceProc);
                    clampRoot(material.bumpProc);
                    clampRoot(material.displacementProc);
                    clampRoot(material.specularColorProc);
                    clampRoot(material.transmissionColorProc);
                }

                // Cages only at cook — tessellation + displace run at Render start.
                int meshIndex = -1;
                MeshPtr renderMesh = prim.mesh;
                if (renderMesh) {
                    // Stamp authored tessellation onto the mesh (may share cage pointers;
                    // params are overwritten each cook from the owning prim).
                    renderMesh->subdivType = prim.subdivType;
                    renderMesh->subdivIterations = prim.subdivIterations;
                    renderMesh->dicingQuality = prim.dicingQuality;
                    renderMesh->boundsPadding = prim.boundsPadding;
                    renderMesh->timeDependent = prim.timeDependent;
                }
                if (materialHasGeometricDisplacement(material)) {
                    // Unique mesh entry so Render can tessellate without sharing cages.
                    auto cageCopy = std::make_shared<Mesh>(*renderMesh);
                    cageCopy->subdivType = prim.subdivType;
                    cageCopy->subdivIterations = prim.subdivIterations;
                    cageCopy->dicingQuality = prim.dicingQuality;
                    cageCopy->boundsPadding = prim.boundsPadding;
                    cageCopy->timeDependent = prim.timeDependent;
                    meshIndex = scene->addMesh(cageCopy);
                    renderMesh = cageCopy;
                } else {
                    auto it = meshIndexCache.find(prim.mesh.get());
                    if (it != meshIndexCache.end()) {
                        meshIndex = it->second;
                    } else {
                        meshIndex = scene->addMesh(prim.mesh);
                        meshIndexCache[prim.mesh.get()] = meshIndex;
                    }
                }

                // Ray-switch branches: add as separate Materials, rewrite local → scene indices.
                auto remapSwitchSlot = [&](int& slot) {
                    if (slot < 0) return;
                    if (slot >= int(prim.raySwitchBranches.size())) {
                        slot = -1;
                        return;
                    }
                    Material branch = prim.raySwitchBranches[size_t(slot)];
                    branch.raySwitch = RaySwitchTable{};
                    // Branches currently carry scalar/params only (no private textures).
                    slot = scene->addMaterial(branch);
                };
                remapSwitchSlot(material.raySwitch.camera);
                remapSwitchSlot(material.raySwitch.shadow);
                remapSwitchSlot(material.raySwitch.diffuseReflection);
                remapSwitchSlot(material.raySwitch.specularReflection);
                remapSwitchSlot(material.raySwitch.diffuseTransmission);
                remapSwitchSlot(material.raySwitch.specularTransmission);
                remapSwitchSlot(material.raySwitch.sss);
                remapSwitchSlot(material.raySwitch.caustics);

                const int materialIndex = scene->addMaterial(material);
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
                record.pointCount = static_cast<long long>(renderMesh->positions.size());
                record.triangleCount = static_cast<long long>(renderMesh->triangleCount());
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
