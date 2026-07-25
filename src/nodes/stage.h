// The Stage is what flows along the node network, in the spirit of a USD layer
// stack in Houdini Solaris: an ordered list of prims addressed by path that
// every node can add to, edit or prune.
#pragma once

#include <QString>
#include <memory>
#include <vector>

#include "scene/scene.h"

namespace sol {

enum class PrimType { Mesh, Light, Camera, Scope };

struct StagePrim {
    QString path;
    PrimType type = PrimType::Mesh;
    QString sourceNode;
    bool active = true;

    Mat4 xform;  // world space transform

    // Geometry
    MeshPtr mesh;
    Material material;
    bool materialAssigned = false;
    QString materialName;

    // Light
    LightData light;
    std::shared_ptr<EnvironmentMap> environment;

    // Camera
    CameraData camera;

    QString typeName() const;
};

class Stage {
public:
    std::vector<StagePrim> prims;
    RenderSettingsData settings;
    bool settingsAuthored = false;
    QString renderCameraPath;

    StagePrim* find(const QString& path);
    const StagePrim* find(const QString& path) const;
    QString uniquePath(const QString& desiredPath) const;
    void addPrim(StagePrim prim);
    void appendFrom(const Stage& other);

    int countOfType(PrimType type) const;
    QStringList paths() const;

    // Flattens the stage into a renderable scene.
    ScenePtr toScene() const;
};

using StagePtr = std::shared_ptr<Stage>;

}  // namespace sol
