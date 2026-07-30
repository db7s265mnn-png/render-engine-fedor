// MaterialX document helpers: create a Solaris-like default graph, serialize
// user nodes, and evaluate standard_surface (+ image maps) into Solstice Material.
#pragma once

#include <QHash>
#include <QString>
#include <QStringList>
#include <QVector>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "scene/types.h"

namespace sol {

class Image;

struct MaterialXEvalResult {
    Material material;
    // Extra materials from ray_switch_shader branches (not including `material`).
    // `material.raySwitch.*` holds local indices into this vector (-1 = use material).
    std::vector<Material> raySwitchBranches;
    std::shared_ptr<Image> baseColorTexture;
    std::shared_ptr<Image> roughnessTexture;
    std::shared_ptr<Image> metallicTexture;
    std::shared_ptr<Image> opacityTexture;
    std::shared_ptr<Image> emissionTexture;
    std::shared_ptr<Image> normalTexture;
    std::shared_ptr<Image> bumpTexture;
    std::shared_ptr<Image> displacementTexture;
    std::shared_ptr<Image> subsurfaceTexture;
    std::shared_ptr<Image> specularColorTexture;
    std::shared_ptr<Image> transmissionColorTexture;
    // Shade-time procedural graphs (noise / math / triplanar). Indices on `material.*Proc`
    // are local to `procedurals`. Image leaves store local indices into `proceduralImages`.
    std::vector<ProceduralNode> procedurals;
    std::vector<std::shared_ptr<Image>> proceduralImages;
    QString error;
    bool ok = false;
};

struct MaterialXNodeInputDef {
    QString name;
    QString type;
    QString value;
};

struct MaterialXNodeCatalogEntry {
    QString category;
    QString type;              // preferred default signature
    QStringList typeVariants;  // all available output types for this category
    QString group;
    QString label;
    // Inputs keyed by output type (signature). Prefer inputsFor(type).
    QHash<QString, QVector<MaterialXNodeInputDef>> inputsByType;

    QVector<MaterialXNodeInputDef> inputsFor(const QString& signature) const {
        const QString key = signature.isEmpty() ? type : signature;
        auto it = inputsByType.constFind(key);
        if (it != inputsByType.constEnd()) return it.value();
        if (!type.isEmpty()) {
            it = inputsByType.constFind(type);
            if (it != inputsByType.constEnd()) return it.value();
        }
        if (!inputsByType.isEmpty()) return inputsByType.constBegin().value();
        return {};
    }
};

bool materialXAvailable();

// Absolute path to the MaterialX data libraries (stdlib/pbrlib/bxdf).
QString materialXLibraryRoot();

// All instantiable MaterialX nodedefs from the loaded libraries (stdlib/pbrlib/bxdf).
QVector<MaterialXNodeCatalogEntry> listMaterialXNodeCatalog();

// Default Houdini-like MaterialX graph: image* → standard_surface → surfacematerial("surface").
QString createDefaultMaterialXDocument();

// Validate / normalize stored XML (returns empty on hard failure).
QString normalizeMaterialXDocument(const QString& xml);

// Evaluate a MaterialX user graph into our path-tracer Material + textures.
MaterialXEvalResult evaluateMaterialXDocument(const QString& xml, const QString& searchDirectory);

// UI graph model helpers — always go through MaterialX I/O so filename tokens
// like <UDIM> survive (MaterialX intentionally keeps angle brackets unescaped).
struct MaterialXGraphInput {
    QString name;
    QString type;
    QString value;
    QString nodename;
};

struct MaterialXGraphNode {
    QString name;
    QString category;
    QString type;
    double xpos = std::numeric_limits<double>::quiet_NaN();
    double ypos = std::numeric_limits<double>::quiet_NaN();
    QVector<MaterialXGraphInput> inputs;
};

bool parseMaterialXGraph(const QString& xml, QVector<MaterialXGraphNode>& outNodes, QString* error = nullptr,
                         QVector<int>* outUdimSet = nullptr);
// Serialize user nodes (+ optional MaterialX geominfo udimset stringarray).
QString serializeMaterialXGraph(const QVector<MaterialXGraphNode>& nodes, const QVector<int>& udimSet = {});

}  // namespace sol
