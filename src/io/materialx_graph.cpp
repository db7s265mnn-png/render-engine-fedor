#include "io/materialx_graph.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QHash>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <unordered_map>

#include "core/log.h"
#include "io/image_io.h"
#include "io/materialx_compile.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_MATERIALX
#  include <MaterialXCore/Document.h>
#  include <MaterialXCore/Geom.h>
#  include <MaterialXCore/Interface.h>
#  include <MaterialXCore/Node.h>
#  include <MaterialXCore/Value.h>
#  include <MaterialXFormat/Util.h>
#  include <MaterialXFormat/XmlIo.h>
#endif

namespace sol {
namespace {

#if SOLSTICE_HAVE_MATERIALX
namespace mx = MaterialX;

QString findLibraryRoot() {
    QStringList candidates;
#ifdef SOLSTICE_MATERIALX_LIBRARIES
    candidates << QString::fromUtf8(SOLSTICE_MATERIALX_LIBRARIES);
#endif
    candidates << QDir::currentPath() + "/materialx/libraries"
               << QDir::currentPath() + "/../materialx/libraries"
               << "/tmp/mtlx-src/libraries"
               << "/tmp/mtlx-install/libraries";
    if (QCoreApplication::instance()) {
        const QString appDir = QCoreApplication::applicationDirPath();
        candidates.prepend(appDir + "/../materialx/libraries");
        candidates.prepend(appDir + "/materialx/libraries");
    }
    for (const QString& path : candidates) {
        if (path.isEmpty()) continue;
        if (QDir(path + "/stdlib").exists()) return QFileInfo(path).absoluteFilePath();
    }
    return {};
}

mx::DocumentPtr libraryDocument(std::string& error) {
    static std::mutex mutex;
    static mx::DocumentPtr cached;
    std::lock_guard<std::mutex> lock(mutex);
    if (cached) return cached;

    const QString root = findLibraryRoot();
    if (root.isEmpty()) {
        error = "MaterialX libraries not found (stdlib/pbrlib/bxdf)";
        return nullptr;
    }
    mx::FileSearchPath searchPath;
    searchPath.append(mx::FilePath(root.toStdString()));
    auto doc = mx::createDocument();
    try {
        mx::loadLibraries({"targets", "stdlib", "pbrlib", "bxdf"}, searchPath, doc);
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
    cached = doc;
    return cached;
}

mx::DocumentPtr makeLibraryDocument(std::string& error) {
    mx::DocumentPtr libs = libraryDocument(error);
    if (!libs) return nullptr;
    // Fresh document per call so concurrent cooks / UI catalog never share mutable state.
    auto doc = mx::createDocument();
    try {
        doc->importLibrary(libs);
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
    return doc;
}

mx::DocumentPtr loadUserDocument(const QString& xml, std::string& error) {
    auto doc = makeLibraryDocument(error);
    if (!doc) return nullptr;
    try {
        mx::readFromXmlString(doc, xml.toStdString());
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
    return doc;
}

QString writeUserNodesOnly(const mx::DocumentPtr& doc) {
    auto out = mx::createDocument();
    out->setVersionString(doc->getVersionString());
    for (const mx::NodePtr& node : doc->getNodes()) {
        // Library nodes carry a source URI; skip them.
        if (!node->getSourceUri().empty()) continue;
        mx::NodePtr copy = out->addNode(node->getCategory(), node->getName(), node->getType());
        copy->copyContentFrom(node);
    }
    for (const mx::NodeGraphPtr& graph : doc->getNodeGraphs()) {
        if (!graph->getSourceUri().empty()) continue;
        mx::NodeGraphPtr copy = out->addNodeGraph(graph->getName());
        copy->copyContentFrom(graph);
    }
    // Preserve geominfo (e.g. udimset) — MaterialX UDIM assets declare tiles there.
    for (const mx::GeomInfoPtr& geomInfo : doc->getGeomInfos()) {
        if (!geomInfo || !geomInfo->getSourceUri().empty()) continue;
        mx::GeomInfoPtr copy = out->addGeomInfo(geomInfo->getName());
        copy->copyContentFrom(geomInfo);
    }
    return QString::fromStdString(mx::writeToXmlString(out));
}

std::vector<int> readUdimSet(const mx::DocumentPtr& doc) {
    std::vector<int> udims;
    if (!doc) return udims;
    mx::ValuePtr value = doc->getGeomPropValue(mx::UDIM_SET_PROPERTY);
    if (!value) return udims;
    if (value->isA<mx::StringVec>()) {
        for (const std::string& id : value->asA<mx::StringVec>()) {
            try {
                const int udim = std::stoi(id);
                if (udim >= 1001 && udim < 2000) udims.push_back(udim);
            } catch (...) {
            }
        }
    } else if (value->isA<std::string>()) {
        // Some documents store a comma-separated string.
        QStringList parts = QString::fromStdString(value->asA<std::string>()).split(',', Qt::SkipEmptyParts);
        for (QString part : parts) {
            bool ok = false;
            const int udim = part.trimmed().toInt(&ok);
            if (ok && udim >= 1001 && udim < 2000) udims.push_back(udim);
        }
    }
    std::sort(udims.begin(), udims.end());
    udims.erase(std::unique(udims.begin(), udims.end()), udims.end());
    return udims;
}

bool parseColor3(const std::string& value, Vec3& out) {
    float x = 0, y = 0, z = 0;
    if (std::sscanf(value.c_str(), "%f,%f,%f", &x, &y, &z) == 3 ||
        std::sscanf(value.c_str(), "%f %f %f", &x, &y, &z) == 3) {
        out = Vec3(x, y, z);
        return true;
    }
    if (std::sscanf(value.c_str(), "%f", &x) == 1) {
        out = Vec3(x);
        return true;
    }
    return false;
}

bool parseFloat(const std::string& value, float& out) {
    try {
        out = std::stof(value);
        return true;
    } catch (...) {
        return false;
    }
}

mx::NodePtr resolveConnectedNode(const mx::NodePtr& node, const std::string& inputName) {
    if (!node) return nullptr;
    mx::InputPtr input = node->getInput(inputName);
    if (!input) return nullptr;
    // Ignore defaultgeomprop-only ports (Pobject / UV0 / …) — they are not graph wires.
    if (!input->hasNodeName() || input->getNodeName().empty()) return nullptr;
    return input->getConnectedNode();
}

std::string inputValueString(const mx::NodePtr& node, const std::string& inputName) {
    if (!node) return {};
    mx::InputPtr input = node->getInput(inputName);
    if (!input) return {};
    if (input->hasValueString()) return input->getValueString();
    mx::ValuePtr value = input->getValue();
    return value ? value->getValueString() : std::string();
}

std::shared_ptr<Image> loadTextureFromImageNode(const mx::NodePtr& imageNode, const QString& searchDirectory,
                                                 const std::vector<int>& udimSet, std::string& error,
                                                 bool srgbColor = true) {
    if (!imageNode) return nullptr;
    const std::string category = imageNode->getCategory();
    if (category != "image" && category != "tiledimage") return nullptr;
    // MaterialX keeps <UDIM> unresolved on the input; resolution happens at bind/load time
    // (View: setUdimString per tile; here: discover tiles + atlas bake).
    std::string file = inputValueString(imageNode, "file");
    if (file.empty()) return nullptr;
    QString pattern;
    std::vector<int> discovered;
    const QString fileQ = QString::fromStdString(file);
    if (resolveUdimPattern(fileQ, searchDirectory, pattern, discovered)) {
        std::vector<int> tiles = udimSet;
        if (tiles.empty()) tiles = discovered;
        std::string ids;
        for (size_t i = 0; i < tiles.size(); ++i) {
            if (i) ids += ",";
            ids += std::to_string(tiles[i]);
        }
        logInfo("MaterialX image file='" + file + "' → pattern='" + pattern.toStdString() + "' tiles=[" + ids +
                "]");
        return loadImageOrUdim(pattern, searchDirectory, error, tiles, srgbColor);
    }
    logInfo("MaterialX image file='" + file + "'");
    return loadImageOrUdim(fileQ, searchDirectory, error, udimSet, srgbColor);
}

// Walk through multiply/mix/normalmap/bump wrappers to find an image node.
mx::NodePtr findImageNode(mx::NodePtr node) {
    for (int depth = 0; node && depth < 8; ++depth) {
        const std::string cat = node->getCategory();
        if (cat == "image" || cat == "tiledimage") return node;
        if (cat == "normalmap") {
            node = resolveConnectedNode(node, "in");
            continue;
        }
        if (cat == "bump") {
            node = resolveConnectedNode(node, "height");
            if (!node) node = resolveConnectedNode(node, "in");
            continue;
        }
        if (cat == "multiply" || cat == "mix") {
            mx::NodePtr a = resolveConnectedNode(node, "in1");
            if (!a) a = resolveConnectedNode(node, "fg");
            if (!a) a = resolveConnectedNode(node, "bg");
            node = a;
            continue;
        }
        break;
    }
    return nullptr;
}

void applyStandardSurface(const mx::NodePtr& ss, Material& material) {
    auto setColor = [&](const char* name, Vec3& dst) {
        if (resolveConnectedNode(ss, name)) return;  // textured — keep default tint
        Vec3 v;
        if (parseColor3(inputValueString(ss, name), v)) dst = v;
    };
    auto setFloat = [&](const char* name, float& dst) {
        if (resolveConnectedNode(ss, name)) return;
        float v = 0;
        if (parseFloat(inputValueString(ss, name), v)) dst = v;
    };

    setColor("base_color", material.baseColor);
    // Autodesk Standard Surface: diffuse weight is `base` * `base_color`.
    float baseWeight = 0.8f;
    setFloat("base", baseWeight);
    material.baseWeight = srMax(0.0f, baseWeight);
    setFloat("specular_roughness", material.roughness);
    setFloat("metalness", material.metallic);
    setFloat("specular", material.specular);
    setFloat("specular_IOR", material.ior);
    setFloat("transmission", material.transmission);
    material.roughness = saturatef(material.roughness);
    material.metallic = saturatef(material.metallic);
    material.specular = saturatef(material.specular);
    material.ior = clampf(material.ior, 0.0f, 5.0f);
    material.transmission = saturatef(material.transmission);
    {
        // MaterialX opacity is color3; use average as scalar cutout weight.
        if (!resolveConnectedNode(ss, "opacity")) {
            Vec3 opacityColor(1.0f);
            const std::string raw = inputValueString(ss, "opacity");
            if (raw.find(',') != std::string::npos || raw.find(' ') != std::string::npos) {
                if (parseColor3(raw, opacityColor)) material.opacity = average(opacityColor);
            } else {
                float opacityF = 1.0f;
                if (parseFloat(raw, opacityF)) material.opacity = opacityF;
            }
        }
    }
    // Fake-caustics style: how opaque a cast shadow is (1=black shadow, 0=none).
    float shadowOpacity = 1.0f;
    setFloat("shadow_opacity", shadowOpacity);
    material.shadowOpacity = saturatef(shadowOpacity);

    setColor("emission_color", material.emissionColor);
    setFloat("emission", material.emissionStrength);
    setFloat("subsurface", material.subsurface);
    setColor("subsurface_color", material.subsurfaceColor);
    setColor("subsurface_radius", material.subsurfaceRadius);
    // Arnold: MFP (metres) = subsurface_scale * subsurface_radius (random-walk SSS).
    float subsurfaceScale = 1.0f;
    setFloat("subsurface_scale", subsurfaceScale);
    material.subsurface = saturatef(material.subsurface);
    // Scale is metric (scene units); allow large values for cm-authored Radius maps.
    material.subsurfaceScale = srMax(0.0f, subsurfaceScale);
}

#endif  // SOLSTICE_HAVE_MATERIALX

}  // namespace

bool materialXAvailable() {
#if SOLSTICE_HAVE_MATERIALX
    return !findLibraryRoot().isEmpty();
#else
    return false;
#endif
}

QString materialXLibraryRoot() {
#if SOLSTICE_HAVE_MATERIALX
    return findLibraryRoot();
#else
    return {};
#endif
}

namespace {

QString catalogGroupFor(const QString& category, const QString& type, const QString& nodeGroup) {
    const QString group = nodeGroup.toLower();
    const QString cat = category.toLower();
    if (!group.isEmpty()) {
        if (group.contains("texture") || group.contains("image")) return "Texture";
        if (group.contains("pbr") || group.contains("shader") || group.contains("bxdf") ||
            group.contains("material"))
            return "PBR / Shading";
        if (group.contains("geometric") || group.contains("geometry")) return "Geometric";
        if (group.contains("procedural") || group.contains("noise")) return "Procedural";
        if (group.contains("color") || group.contains("adjustment")) return "Color";
        if (group.contains("math") || group.contains("commutative") || group.contains("conditional"))
            return "Math";
        if (group.contains("compositing") || group.contains("channel")) return "Compositing";
        if (group.contains("light")) return "Lights";
        return nodeGroup;
    }
    if (type == "surfaceshader" || type == "material" || type == "displacementshader" ||
        type == "volumeshader" || type == "lightshader" || cat.contains("surface") ||
        cat.contains("material") || cat.contains("bsdf") || cat.contains("edf") || cat.contains("vdf"))
        return "PBR / Shading";
    if (cat.contains("image") || cat.contains("texture") || cat.contains("triplanar") ||
        cat == "normalmap" || cat == "bump" || cat == "tangent")
        return "Texture";
    if (cat.contains("noise") || cat.contains("fractal") || cat.contains("cell") || cat.contains("ramp") ||
        cat.contains("checker") || cat.contains("worley"))
        return "Procedural";
    if (cat.contains("position") || cat.contains("normal") || cat.contains("tangent") ||
        cat.contains("texcoord") || cat.contains("geom") || cat == "viewdirection" || cat == "time")
        return "Geometric";
    if (type.startsWith("color") || cat.contains("hsv") || cat.contains("luminance") ||
        cat.contains("saturate") || cat.contains("contrast"))
        return "Color";
    if (type.startsWith("float") || type.startsWith("vector") || type.startsWith("matrix") ||
        cat.contains("add") || cat.contains("multiply") || cat.contains("mix") || cat.contains("clamp") ||
        cat.contains("dot") || cat.contains("cross") || cat.contains("normalize"))
        return "Math";
    return "Utility";
}

int materialXTypePreferenceRank(const QString& type) {
    if (type == "color3" || type == "surfaceshader" || type == "material") return 0;
    if (type == "float") return 1;
    if (type == "vector3") return 2;
    if (type == "color4") return 3;
    if (type == "vector2") return 4;
    if (type == "vector4") return 5;
    if (type == "integer" || type == "boolean") return 6;
    return 10;
}

#if SOLSTICE_HAVE_MATERIALX
QVector<MaterialXNodeInputDef> inputsFromNodeDef(const mx::NodeDefPtr& def) {
    QVector<MaterialXNodeInputDef> inputs;
    if (!def) return inputs;
    for (const mx::InputPtr& input : def->getActiveInputs()) {
        if (!input) continue;
        MaterialXNodeInputDef in;
        in.name = QString::fromStdString(input->getName());
        in.type = QString::fromStdString(input->getType());
        if (input->hasValueString()) in.value = QString::fromStdString(input->getValueString());
        else if (input->getValue()) in.value = QString::fromStdString(input->getValue()->getValueString());
        if (in.type == "filename") in.value.clear();
        inputs.push_back(in);
    }
    return inputs;
}
#endif

void finalizeCatalogEntry(MaterialXNodeCatalogEntry& entry, const QString& nodeGroup = QString()) {
    std::sort(entry.typeVariants.begin(), entry.typeVariants.end(),
              [](const QString& a, const QString& b) {
                  const int ra = materialXTypePreferenceRank(a);
                  const int rb = materialXTypePreferenceRank(b);
                  if (ra != rb) return ra < rb;
                  return a < b;
              });
    if (entry.type.isEmpty() && !entry.typeVariants.isEmpty()) entry.type = entry.typeVariants.front();
    else if (!entry.typeVariants.isEmpty()) {
        QString best = entry.typeVariants.front();
        for (const QString& variant : entry.typeVariants) {
            if (materialXTypePreferenceRank(variant) < materialXTypePreferenceRank(best)) best = variant;
        }
        entry.type = best;
    }
    entry.group = catalogGroupFor(entry.category, entry.type, nodeGroup);
    entry.label = entry.category;
}

QVector<MaterialXNodeCatalogEntry> fallbackMaterialXCatalog() {
    QHash<QString, MaterialXNodeCatalogEntry> byCategory;
    auto add = [&](const char* category, const char* type, const char* group,
                   std::initializer_list<MaterialXNodeInputDef> inputs) {
        const QString cat = QString::fromUtf8(category);
        const QString typ = QString::fromUtf8(type);
        MaterialXNodeCatalogEntry& e = byCategory[cat];
        e.category = cat;
        if (!e.typeVariants.contains(typ)) e.typeVariants.push_back(typ);
        QVector<MaterialXNodeInputDef> defs;
        for (const MaterialXNodeInputDef& input : inputs) defs.push_back(input);
        e.inputsByType.insert(typ, defs);
        if (e.group.isEmpty()) e.group = QString::fromUtf8(group);
    };

    add("image", "color3", "Texture",
        {{"file", "filename", {}},
         {"texcoord", "vector2", {}},
         {"uvtiling", "vector2", "1, 1"},
         {"uvoffset", "vector2", "0, 0"},
         {"default", "color3", "0, 0, 0"}});
    add("image", "color4", "Texture",
        {{"file", "filename", {}},
         {"texcoord", "vector2", {}},
         {"uvtiling", "vector2", "1, 1"},
         {"uvoffset", "vector2", "0, 0"},
         {"default", "color4", "0, 0, 0, 1"}});
    add("image", "float", "Texture",
        {{"file", "filename", {}},
         {"texcoord", "vector2", {}},
         {"uvtiling", "vector2", "1, 1"},
         {"uvoffset", "vector2", "0, 0"},
         {"default", "float", "0"}});
    add("image", "vector2", "Texture",
        {{"file", "filename", {}},
         {"texcoord", "vector2", {}},
         {"uvtiling", "vector2", "1, 1"},
         {"uvoffset", "vector2", "0, 0"},
         {"default", "vector2", "0, 0"}});
    add("image", "vector3", "Texture",
        {{"file", "filename", {}},
         {"texcoord", "vector2", {}},
         {"uvtiling", "vector2", "1, 1"},
         {"uvoffset", "vector2", "0, 0"},
         {"default", "vector3", "0, 0, 0"}});
    add("image", "vector4", "Texture",
        {{"file", "filename", {}},
         {"texcoord", "vector2", {}},
         {"uvtiling", "vector2", "1, 1"},
         {"uvoffset", "vector2", "0, 0"},
         {"default", "vector4", "0, 0, 0, 1"}});
    add("tiledimage", "color3", "Texture",
        {{"file", "filename", {}},
         {"texcoord", "vector2", {}},
         {"uvtiling", "vector2", "1, 1"},
         {"uvoffset", "vector2", "0, 0"},
         {"default", "color3", "0, 0, 0"}});
    add("tiledimage", "float", "Texture",
        {{"file", "filename", {}},
         {"texcoord", "vector2", {}},
         {"uvtiling", "vector2", "1, 1"},
         {"uvoffset", "vector2", "0, 0"},
         {"default", "float", "0"}});
    add("tiledimage", "vector3", "Texture",
        {{"file", "filename", {}},
         {"texcoord", "vector2", {}},
         {"uvtiling", "vector2", "1, 1"},
         {"uvoffset", "vector2", "0, 0"},
         {"default", "vector3", "0, 0, 0"}});
    add("constant", "color3", "Math", {{"value", "color3", "1, 1, 1"}});
    add("constant", "color4", "Math", {{"value", "color4", "1, 1, 1, 1"}});
    add("constant", "float", "Math", {{"value", "float", "1"}});
    add("constant", "vector2", "Math", {{"value", "vector2", "1, 1"}});
    add("constant", "vector3", "Math", {{"value", "vector3", "1, 1, 1"}});
    add("constant", "vector4", "Math", {{"value", "vector4", "1, 1, 1, 1"}});
    add("multiply", "color3", "Math", {{"in1", "color3", "1, 1, 1"}, {"in2", "color3", "1, 1, 1"}});
    add("multiply", "float", "Math", {{"in1", "float", "1"}, {"in2", "float", "1"}});
    add("multiply", "vector2", "Math", {{"in1", "vector2", "1, 1"}, {"in2", "vector2", "1, 1"}});
    add("multiply", "vector3", "Math", {{"in1", "vector3", "1, 1, 1"}, {"in2", "vector3", "1, 1, 1"}});
    add("add", "color3", "Math", {{"in1", "color3", "0, 0, 0"}, {"in2", "color3", "0, 0, 0"}});
    add("add", "float", "Math", {{"in1", "float", "0"}, {"in2", "float", "0"}});
    add("add", "vector2", "Math", {{"in1", "vector2", "0, 0"}, {"in2", "vector2", "0, 0"}});
    add("add", "vector3", "Math", {{"in1", "vector3", "0, 0, 0"}, {"in2", "vector3", "0, 0, 0"}});
    add("mix", "color3", "Math",
        {{"bg", "color3", "0, 0, 0"}, {"fg", "color3", "1, 1, 1"}, {"mix", "float", "0.5"}});
    add("mix", "float", "Math", {{"bg", "float", "0"}, {"fg", "float", "1"}, {"mix", "float", "0.5"}});
    add("mix", "vector3", "Math",
        {{"bg", "vector3", "0, 0, 0"}, {"fg", "vector3", "1, 1, 1"}, {"mix", "float", "0.5"}});
    add("range", "float", "Math",
        {{"in", "float", "0"}, {"inlow", "float", "0"}, {"inhigh", "float", "1"}, {"outlow", "float", "0"},
         {"outhigh", "float", "1"}, {"gamma", "float", "1"}, {"doclamp", "boolean", "false"}});
    add("range", "color3", "Math",
        {{"in", "color3", "0, 0, 0"}, {"inlow", "color3", "0, 0, 0"}, {"inhigh", "color3", "1, 1, 1"},
         {"outlow", "color3", "0, 0, 0"}, {"outhigh", "color3", "1, 1, 1"}, {"gamma", "color3", "1, 1, 1"},
         {"doclamp", "boolean", "false"}});
    add("range", "vector3", "Math",
        {{"in", "vector3", "0, 0, 0"}, {"inlow", "vector3", "0, 0, 0"}, {"inhigh", "vector3", "1, 1, 1"},
         {"outlow", "vector3", "0, 0, 0"}, {"outhigh", "vector3", "1, 1, 1"}, {"gamma", "vector3", "1, 1, 1"},
         {"doclamp", "boolean", "false"}});
    add("normalmap", "vector3", "Texture", {{"in", "vector3", {}}, {"scale", "float", "1"}});
    add("bump", "vector3", "Geometric",
        {{"height", "float", "0"}, {"scale", "float", "1"}, {"normal", "vector3", {}}, {"tangent", "vector3", {}},
         {"bitangent", "vector3", {}}});
    add("texcoord", "vector2", "Geometric", {{"index", "integer", "0"}});
    add("place2d", "vector2", "Geometric",
        {{"texcoord", "vector2", {}},
         {"pivot", "vector2", "0.5, 0.5"},
         {"scale", "vector2", "1, 1"},
         {"rotate", "float", "0"},
         {"offset", "vector2", "0, 0"}});
    add("rotate2d", "vector2", "Geometric",
        {{"in", "vector2", {}}, {"texcoord", "vector2", {}}, {"amount", "float", "0"}, {"pivot", "vector2", "0.5, 0.5"}});
    add("position", "vector3", "Geometric", {{"space", "string", "object"}});
    add("normal", "vector3", "Geometric", {{"space", "string", "object"}});
    add("standard_surface", "surfaceshader", "PBR / Shading",
        {{"base_color", "color3", "0.8, 0.8, 0.8"},
         {"base", "float", "0.8"},
         {"specular_roughness", "float", "0.35"},
         {"metalness", "float", "0"},
         {"specular", "float", "0.5"},
         {"specular_IOR", "float", "1.5"},
         {"transmission", "float", "0"},
         {"opacity", "color3", "1, 1, 1"},
         {"shadow_opacity", "float", "1"},
         {"emission", "float", "0"},
         {"emission_color", "color3", "1, 1, 1"},
         {"normal", "vector3", {}},
         {"subsurface", "float", "0"},
         {"subsurface_color", "color3", "1, 0.75, 0.55"},
         {"subsurface_radius", "color3", "1, 0.35, 0.2"},
         {"subsurface_scale", "float", "1"}});
    add("triplanarprojection", "color3", "Texture",
        {{"file", "filename", {}},
         {"input_per_axis", "boolean", "false"},
         {"filex", "filename", {}},
         {"filey", "filename", {}},
         {"filez", "filename", {}},
         {"scale", "vector3", "1, 1, 1"},
         {"rotate", "float", "0"},
         {"offset", "vector3", "0, 0, 0"},
         {"blend", "float", "0.1"},
         {"default", "color3", "0.2, 0.5, 0.8"}});
    add("surfacematerial", "material", "PBR / Shading", {{"surfaceshader", "surfaceshader", {}}});

    QVector<MaterialXNodeCatalogEntry> out;
    out.reserve(byCategory.size());
    for (auto it = byCategory.begin(); it != byCategory.end(); ++it) {
        finalizeCatalogEntry(it.value());
        out.push_back(it.value());
    }
    std::sort(out.begin(), out.end(), [](const MaterialXNodeCatalogEntry& a, const MaterialXNodeCatalogEntry& b) {
        if (a.group != b.group) return a.group < b.group;
        return a.category < b.category;
    });
    return out;
}

}  // namespace

QVector<MaterialXNodeCatalogEntry> listMaterialXNodeCatalog() {
#if SOLSTICE_HAVE_MATERIALX
    static std::mutex catalogMutex;
    static QVector<MaterialXNodeCatalogEntry> cached;
    static bool ready = false;
    {
        std::lock_guard<std::mutex> lock(catalogMutex);
        if (ready) return cached;
    }

    std::string error;
    auto doc = makeLibraryDocument(error);
    if (!doc) {
        logWarning("MaterialX catalog: " + error);
        return fallbackMaterialXCatalog();
    }

    // One menu entry per category; all typed nodedef signatures live as typeVariants.
    struct Bucket {
        QString nodeGroup;
        QHash<QString, QVector<MaterialXNodeInputDef>> inputsByType;
        QStringList typeVariants;
    };
    QHash<QString, Bucket> byCategory;

    for (const mx::NodeDefPtr& def : doc->getNodeDefs()) {
        if (!def) continue;
        const std::string categoryStd = def->getNodeString().empty() ? def->getName() : def->getNodeString();
        if (categoryStd.empty()) continue;
        if (categoryStd == "backdrop" || categoryStd == "tokengraph" || categoryStd == "nodedef") continue;
        const std::string typeStd = def->getType();
        if (typeStd.empty() || typeStd == "none" || typeStd == "multioutput") continue;

        const QString category = QString::fromStdString(categoryStd);
        const QString type = QString::fromStdString(typeStd);
        Bucket& bucket = byCategory[category];
        if (bucket.nodeGroup.isEmpty()) bucket.nodeGroup = QString::fromStdString(def->getNodeGroup());
        if (!bucket.typeVariants.contains(type)) bucket.typeVariants.push_back(type);
        if (!bucket.inputsByType.contains(type)) bucket.inputsByType.insert(type, inputsFromNodeDef(def));
    }

    QVector<MaterialXNodeCatalogEntry> entries;
    entries.reserve(byCategory.size());
    for (auto it = byCategory.begin(); it != byCategory.end(); ++it) {
        MaterialXNodeCatalogEntry entry;
        entry.category = it.key();
        entry.typeVariants = it.value().typeVariants;
        entry.inputsByType = it.value().inputsByType;
        finalizeCatalogEntry(entry, it.value().nodeGroup);
        entries.push_back(entry);
    }

    std::sort(entries.begin(), entries.end(), [](const MaterialXNodeCatalogEntry& a,
                                                 const MaterialXNodeCatalogEntry& b) {
        if (a.group != b.group) return a.group < b.group;
        return a.category < b.category;
    });

    std::lock_guard<std::mutex> lock(catalogMutex);
    cached = entries;
    ready = true;
    return cached;
#else
    return fallbackMaterialXCatalog();
#endif
}

QString createDefaultMaterialXDocument() {
#if SOLSTICE_HAVE_MATERIALX
    std::string error;
    auto doc = makeLibraryDocument(error);
    if (!doc) {
        logWarning("MaterialX: " + error);
        return {};
    }
    mx::NodePtr ss = doc->addNode("standard_surface", "standard_surface1", "surfaceshader");
    // Author Diffuse Color before weight so XML/UI port order stays stable.
    ss->setInputValue("base_color", mx::Color3(0.8f, 0.8f, 0.8f));
    ss->setInputValue("base", 0.8f);
    ss->setInputValue("specular_roughness", 0.35f);
    ss->setInputValue("metalness", 0.0f);
    ss->setInputValue("specular", 0.5f);
    ss->setInputValue("specular_IOR", 1.5f);
    ss->setInputValue("transmission", 0.0f);
    ss->setInputValue("opacity", mx::Color3(1.0f, 1.0f, 1.0f));
    ss->setInputValue("emission", 0.0f);
    ss->setInputValue("emission_color", mx::Color3(1.0f, 1.0f, 1.0f));
    ss->setInputValue("subsurface", 0.0f);
    ss->setInputValue("subsurface_color", mx::Color3(1.0f, 0.75f, 0.55f));
    ss->setInputValue("subsurface_radius", mx::Color3(1.0f, 0.35f, 0.2f));
    ss->setInputValue("subsurface_scale", 1.0f);

    // Place nodes left-to-right like Houdini Solaris MaterialX.
    ss->setAttribute("xpos", "0.0");
    ss->setAttribute("ypos", "0.0");

    mx::NodePtr surface = doc->addNode("surfacematerial", "surface", "material");
    surface->setConnectedNode("surfaceshader", ss);
    surface->setAttribute("xpos", "4.0");
    surface->setAttribute("ypos", "0.0");

    return writeUserNodesOnly(doc);
#else
    return {};
#endif
}

QString normalizeMaterialXDocument(const QString& xml) {
#if SOLSTICE_HAVE_MATERIALX
    if (xml.trimmed().isEmpty()) return createDefaultMaterialXDocument();
    std::string error;
    auto doc = loadUserDocument(xml, error);
    if (!doc) {
        logWarning("MaterialX normalize failed: " + error);
        return createDefaultMaterialXDocument();
    }
    // Ensure a surfacematerial named "surface" exists.
    mx::NodePtr surface = doc->getNode("surface");
    if (!surface) {
        for (const mx::NodePtr& node : doc->getNodes()) {
            if (node->getCategory() == "surfacematerial") {
                surface = node;
                break;
            }
        }
    }
    if (!surface) {
        mx::NodePtr ss = doc->getNode("standard_surface1");
        if (!ss) {
            for (const mx::NodePtr& node : doc->getNodes()) {
                if (node->getCategory() == "standard_surface") {
                    ss = node;
                    break;
                }
            }
        }
        if (!ss) ss = doc->addNode("standard_surface", "standard_surface1", "surfaceshader");
        surface = doc->addNode("surfacematerial", "surface", "material");
        surface->setConnectedNode("surfaceshader", ss);
    }
    return writeUserNodesOnly(doc);
#else
    Q_UNUSED(xml);
    return {};
#endif
}

MaterialXEvalResult evaluateMaterialXDocument(const QString& xml, const QString& searchDirectory) {
    MaterialXEvalResult result;
#if SOLSTICE_HAVE_MATERIALX
    try {
        std::string error;
        auto doc = loadUserDocument(xml.isEmpty() ? createDefaultMaterialXDocument() : xml, error);
        if (!doc) {
            result.error = QString::fromStdString(error);
            return result;
        }

        mx::NodePtr surface;
        for (const mx::NodePtr& node : doc->getNodes()) {
            if (node->getName() == "surface" && node->getCategory() == "surfacematerial") {
                surface = node;
                break;
            }
        }
        if (!surface) {
            for (const mx::NodePtr& node : doc->getNodes()) {
                if (node->getCategory() == "surfacematerial") {
                    surface = node;
                    break;
                }
            }
        }

        mx::NodePtr ss;
        if (surface) ss = resolveConnectedNode(surface, "surfaceshader");
        if (!ss) {
            for (const mx::NodePtr& node : doc->getNodes()) {
                if (node->getCategory() == "standard_surface") {
                    ss = node;
                    break;
                }
            }
        }
        if (!ss) {
            result.error = "MaterialX graph has no standard_surface";
            return result;
        }

        applyStandardSurface(ss, result.material);

        const std::vector<int> udimSet = readUdimSet(doc);
        if (!udimSet.empty()) {
            std::string ids;
            for (size_t i = 0; i < udimSet.size(); ++i) {
                if (i) ids += ",";
                ids += std::to_string(udimSet[i]);
            }
            logInfo("MaterialX udimset=[" + ids + "]");
        }

        auto readNodeFloat = [](const mx::NodePtr& node, const char* name, float fallback) -> float {
            if (!node) return fallback;
            float v = fallback;
            if (parseFloat(inputValueString(node, name), v)) return v;
            return fallback;
        };

        // Reuse one compile when the same upstream node feeds multiple ports.
        std::unordered_map<const mx::Node*, int> compiledRoots;
        auto compileProc = [&](mx::NodePtr connected, int& procIndex, const char* inputName) {
            if (!connected) return;
            auto it = compiledRoots.find(connected.get());
            if (it != compiledRoots.end()) {
                procIndex = it->second;
                return;
            }
            std::string compileError;
            const int localRoot = compileMaterialXNode(connected, searchDirectory, udimSet, result.procedurals,
                                                       result.proceduralImages, compileError);
            if (localRoot >= 0) {
                procIndex = localRoot;
                compiledRoots[connected.get()] = localRoot;
                logInfo(std::string("MaterialX: compiled ") + connected->getCategory() + " → " + inputName +
                        " (shade-time procedural, " + std::to_string(result.procedurals.size()) + " ops)");
            } else if (!compileError.empty()) {
                logWarning("MaterialX procedural (" + std::string(inputName) + "): " + compileError);
            }
        };

        auto bindSlot = [&](const char* inputName, std::shared_ptr<Image>& slot, int& procIndex,
                            bool dataMap = false) {
            mx::NodePtr connected = resolveConnectedNode(ss, inputName);
            if (!connected) return;
            const std::string cat = connected->getCategory();
            const bool srgbColor = !dataMap;

            // Pure image maps keep the texture path (mips / UDIM) unless a UV graph
            // (texcoord/place2d/math) or uvtiling/uvoffset requires shade-time sampling.
            if (cat == "image" || cat == "tiledimage") {
                if (materialXImageNeedsProceduralBind(connected)) {
                    compileProc(connected, procIndex, inputName);
                } else {
                    std::string texError;
                    slot = loadTextureFromImageNode(connected, searchDirectory, udimSet, texError, srgbColor);
                    if (!slot && !texError.empty()) logWarning("MaterialX: " + texError);
                }
                return;
            }
            if (cat == "normalmap") {
                result.material.normalScale = readNodeFloat(connected, "scale", 1.0f);
                mx::NodePtr inNode = resolveConnectedNode(connected, "in");
                if (inNode && (inNode->getCategory() == "image" || inNode->getCategory() == "tiledimage") &&
                    materialXImageNeedsProceduralBind(inNode)) {
                    compileProc(inNode, procIndex, inputName);
                    return;
                }
                if (mx::NodePtr image = findImageNode(connected)) {
                    // Prefer procedural when any upstream UV graph exists under the map.
                    if (materialXNodeIsProcedural(connected) &&
                        (materialXImageNeedsProceduralBind(image) || resolveConnectedNode(image, "texcoord"))) {
                        compileProc(inNode ? inNode : connected, procIndex, inputName);
                        return;
                    }
                    std::string texError;
                    slot = loadTextureFromImageNode(image, searchDirectory, udimSet, texError, false);
                    if (!slot && !texError.empty()) logWarning("MaterialX: " + texError);
                    return;
                }
                // Procedural/vector upstream of normalmap → compile as RGB tangent map.
                if (materialXNodeIsProcedural(connected)) {
                    compileProc(inNode ? inNode : connected, procIndex, inputName);
                }
                return;
            }
            if (cat == "bump") {
                result.material.normalScale = readNodeFloat(connected, "scale", 1.0f);
                mx::NodePtr height = resolveConnectedNode(connected, "height");
                if (!height) height = resolveConnectedNode(connected, "in");
                if (height && (height->getCategory() == "image" || height->getCategory() == "tiledimage")) {
                    if (materialXImageNeedsProceduralBind(height)) {
                        compileProc(height, result.material.bumpProc, "bump.height");
                    } else {
                        std::string texError;
                        result.bumpTexture =
                            loadTextureFromImageNode(height, searchDirectory, udimSet, texError, false);
                        if (!result.bumpTexture && !texError.empty()) logWarning("MaterialX: " + texError);
                    }
                    return;
                }
                if (height && materialXNodeIsProcedural(height)) {
                    compileProc(height, result.material.bumpProc, "bump.height");
                    return;
                }
                if (materialXNodeIsProcedural(connected)) {
                    compileProc(connected, result.material.bumpProc, "bump");
                }
                return;
            }

            // Noise / math / triplanar / image blends → shade-time procedural (not UV bake).
            if (materialXNodeIsProcedural(connected)) {
                compileProc(connected, procIndex, inputName);
                return;
            }

            logWarning(std::string("MaterialX: unsupported upstream node '") + connected->getCategory() +
                       "' on " + inputName + " (connect image / noise / triplanar / math / bump / normalmap)");
        };

        bindSlot("base_color", result.baseColorTexture, result.material.baseColorProc);
        bindSlot("specular_roughness", result.roughnessTexture, result.material.roughnessProc);
        bindSlot("metalness", result.metallicTexture, result.material.metallicProc);
        bindSlot("opacity", result.opacityTexture, result.material.opacityProc);
        bindSlot("emission_color", result.emissionTexture, result.material.emissionProc);
        bindSlot("normal", result.normalTexture, result.material.normalProc, true);
        bindSlot("subsurface_color", result.subsurfaceTexture, result.material.subsurfaceProc);

        // Drop dangling roots if a compile failed mid-way.
        auto sanitize = [&](int& idx) {
            if (idx >= 0 && idx >= int(result.procedurals.size())) idx = -1;
        };
        sanitize(result.material.baseColorProc);
        sanitize(result.material.roughnessProc);
        sanitize(result.material.metallicProc);
        sanitize(result.material.opacityProc);
        sanitize(result.material.emissionProc);
        sanitize(result.material.normalProc);
        sanitize(result.material.subsurfaceProc);
        sanitize(result.material.bumpProc);

        result.ok = true;
        return result;
    } catch (const std::exception& e) {
        result.error = QString("MaterialX evaluate failed: %1").arg(e.what());
        result.ok = false;
        return result;
    } catch (...) {
        result.error = "MaterialX evaluate failed: unknown error";
        result.ok = false;
        return result;
    }
#else
    Q_UNUSED(xml);
    Q_UNUSED(searchDirectory);
    result.error = "this build has no MaterialX support";
    return result;
#endif
}

bool parseMaterialXGraph(const QString& xml, QVector<MaterialXGraphNode>& outNodes, QString* error,
                         QVector<int>* outUdimSet) {
    outNodes.clear();
    if (outUdimSet) outUdimSet->clear();
#if SOLSTICE_HAVE_MATERIALX
    try {
        std::string err;
        // MaterialX's pugixml fork accepts raw <UDIM> and escaped &lt;UDIM&gt;.
        auto doc = loadUserDocument(xml, err);
        if (!doc) {
            if (error) *error = QString::fromStdString(err);
            return false;
        }
        if (outUdimSet) {
            for (int udim : readUdimSet(doc)) outUdimSet->push_back(udim);
        }
        for (const mx::NodePtr& node : doc->getNodes()) {
            if (!node || !node->getSourceUri().empty()) continue;
            MaterialXGraphNode graphNode;
            graphNode.name = QString::fromStdString(node->getName());
            graphNode.category = QString::fromStdString(node->getCategory());
            graphNode.type = QString::fromStdString(node->getType());
            if (node->hasAttribute("xpos"))
                graphNode.xpos = QString::fromStdString(node->getAttribute("xpos")).toDouble();
            else
                graphNode.xpos = std::numeric_limits<double>::quiet_NaN();
            if (node->hasAttribute("ypos"))
                graphNode.ypos = QString::fromStdString(node->getAttribute("ypos")).toDouble();
            else
                graphNode.ypos = std::numeric_limits<double>::quiet_NaN();
            for (const mx::InputPtr& input : node->getInputs()) {
                if (!input) continue;
                MaterialXGraphInput graphInput;
                graphInput.name = QString::fromStdString(input->getName());
                graphInput.type = QString::fromStdString(input->getType());
                graphInput.nodename = QString::fromStdString(input->getNodeName());
                if (input->hasValueString())
                    graphInput.value = QString::fromStdString(input->getValueString());
                else if (input->getValue())
                    graphInput.value = QString::fromStdString(input->getValue()->getValueString());
                graphNode.inputs.push_back(graphInput);
            }
            outNodes.push_back(graphNode);
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = QString::fromStdString(e.what());
        return false;
    } catch (...) {
        if (error) *error = "unknown MaterialX parse failure";
        return false;
    }
#else
    if (error) *error = "MaterialX unavailable";
    Q_UNUSED(xml);
    return false;
#endif
}

QString serializeMaterialXGraph(const QVector<MaterialXGraphNode>& nodes, const QVector<int>& udimSet) {
#if SOLSTICE_HAVE_MATERIALX
    try {
        auto out = mx::createDocument();
        out->setVersionString("1.38");
        for (const MaterialXGraphNode& graphNode : nodes) {
            if (graphNode.category.isEmpty() || graphNode.name.isEmpty()) continue;
            mx::NodePtr node = out->addNode(graphNode.category.toStdString(), graphNode.name.toStdString(),
                                            graphNode.type.toStdString());
            if (!std::isnan(graphNode.xpos)) node->setAttribute("xpos", std::to_string(graphNode.xpos));
            if (!std::isnan(graphNode.ypos)) node->setAttribute("ypos", std::to_string(graphNode.ypos));
            for (const MaterialXGraphInput& graphInput : graphNode.inputs) {
                if (graphInput.name.isEmpty()) continue;
                // Skip empty unconnected ports — MaterialX nodedefs supply defaults
                // (including defaultgeomprop like Pobject). Authoring empty typed
                // inputs has crashed some MaterialX builds on cook/validate.
                if (graphInput.nodename.isEmpty() && graphInput.value.isEmpty()) continue;
                mx::InputPtr input = node->addInput(graphInput.name.toStdString(), graphInput.type.toStdString());
                if (!graphInput.nodename.isEmpty()) {
                    input->setNodeName(graphInput.nodename.toStdString());
                } else if (!graphInput.value.isEmpty()) {
                    // Keep MaterialX filename tokens such as <UDIM> unresolved.
                    input->setValueString(graphInput.value.toStdString());
                }
            }
        }
        // MaterialX UDIM assets declare the tile list on geominfo/udimset (see TestSuite udim.mtlx).
        if (!udimSet.isEmpty()) {
            mx::GeomInfoPtr geomInfo = out->addGeomInfo("udim_geom");
            geomInfo->setGeom("/");
            mx::StringVec ids;
            ids.reserve(size_t(udimSet.size()));
            for (int udim : udimSet) ids.push_back(std::to_string(udim));
            geomInfo->setGeomPropValue(mx::UDIM_SET_PROPERTY, ids);
        }
        return QString::fromStdString(mx::writeToXmlString(out));
    } catch (const std::exception& e) {
        logWarning(std::string("MaterialX serialize failed: ") + e.what());
        return {};
    } catch (...) {
        logWarning("MaterialX serialize failed: unknown error");
        return {};
    }
#else
    Q_UNUSED(nodes);
    Q_UNUSED(udimSet);
    return {};
#endif
}

}  // namespace sol
