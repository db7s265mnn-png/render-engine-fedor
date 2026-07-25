// USD scene import (USDA text and USDC binary via a lightweight reader).
#pragma once

#include <string>
#include <vector>

#include "scene/scene.h"

namespace sol {

struct UsdLoadOptions {
    double time = 0.0;
    bool importNormals = true;
    bool importUvs = true;
    float scale = 1.0f;
    std::string pathFilter;
};

struct UsdPrim {
    enum class Type { Mesh, Camera, Light };

    std::string path;
    Type type = Type::Mesh;
    MeshPtr mesh;
    Mat4 transform = Mat4::identity();
    CameraData camera;
    LightData light;
    bool hasCamera = false;
    bool hasLight = false;
};

struct UsdContents {
    std::vector<UsdPrim> prims;
    std::string archiveInfo;
};

bool loadUsd(const std::string& filePath, const UsdLoadOptions& options, UsdContents& out, std::string& error);
bool usdSupportAvailable();

}  // namespace sol
