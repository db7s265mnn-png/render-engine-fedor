// MaterialX document helpers: create a Solaris-like default graph, serialize
// user nodes, and evaluate standard_surface (+ image maps) into Solstice Material.
#pragma once

#include <QString>
#include <memory>
#include <string>
#include <unordered_map>

#include "scene/types.h"

namespace sol {

class Image;

struct MaterialXEvalResult {
    Material material;
    std::shared_ptr<Image> baseColorTexture;
    std::shared_ptr<Image> roughnessTexture;
    std::shared_ptr<Image> metallicTexture;
    std::shared_ptr<Image> opacityTexture;
    std::shared_ptr<Image> emissionTexture;
    std::shared_ptr<Image> normalTexture;
    std::shared_ptr<Image> subsurfaceTexture;
    QString error;
    bool ok = false;
};

bool materialXAvailable();

// Absolute path to the MaterialX data libraries (stdlib/pbrlib/bxdf).
QString materialXLibraryRoot();

// Default Houdini-like MaterialX graph: image* → standard_surface → surfacematerial("surface").
QString createDefaultMaterialXDocument();

// Validate / normalize stored XML (returns empty on hard failure).
QString normalizeMaterialXDocument(const QString& xml);

// Evaluate a MaterialX user graph into our path-tracer Material + textures.
MaterialXEvalResult evaluateMaterialXDocument(const QString& xml, const QString& searchDirectory);

}  // namespace sol
