#include "io/materialx_graph.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include "core/log.h"
#include "io/image_io.h"
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

mx::DocumentPtr makeLibraryDocument(std::string& error) {
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
                                                 const std::vector<int>& udimSet, std::string& error) {
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
        return loadImageOrUdim(pattern, searchDirectory, error, tiles);
    }
    logInfo("MaterialX image file='" + file + "'");
    return loadImageOrUdim(fileQ, searchDirectory, error, udimSet);
}

// Walk through multiply/mix/normalmap wrappers to find an image node.
mx::NodePtr findImageNode(mx::NodePtr node) {
    for (int depth = 0; node && depth < 8; ++depth) {
        const std::string cat = node->getCategory();
        if (cat == "image" || cat == "tiledimage") return node;
        if (cat == "normalmap") {
            node = resolveConnectedNode(node, "in");
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
    setColor("emission_color", material.emissionColor);
    setFloat("emission", material.emissionStrength);
    setFloat("subsurface", material.subsurface);
    setColor("subsurface_color", material.subsurfaceColor);
    setColor("subsurface_radius", material.subsurfaceRadius);
    // Arnold: world-space MFP = subsurface_scale * subsurface_radius (random-walk SSS).
    float subsurfaceScale = 1.0f;
    setFloat("subsurface_scale", subsurfaceScale);
    material.subsurface = saturatef(material.subsurface);
    material.subsurfaceScale = clampf(subsurfaceScale, 0.0f, 100.0f);
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
        cat == "normalmap" || cat == "tangent")
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

QVector<MaterialXNodeCatalogEntry> fallbackMaterialXCatalog() {
    const auto entry = [](const char* category, const char* type, const char* group,
                          std::initializer_list<MaterialXNodeInputDef> inputs) {
        MaterialXNodeCatalogEntry e;
        e.category = QString::fromUtf8(category);
        e.type = QString::fromUtf8(type);
        e.group = QString::fromUtf8(group);
        e.label = e.category + " (" + e.type + ")";
        for (const MaterialXNodeInputDef& input : inputs) e.inputs.push_back(input);
        return e;
    };
    return {
        entry("image", "color3", "Texture", {{"file", "filename", {}}, {"default", "color3", "0, 0, 0"}}),
        entry("tiledimage", "color3", "Texture",
              {{"file", "filename", {}}, {"uvtiling", "vector2", "1, 1"}, {"default", "color3", "0, 0, 0"}}),
        entry("constant", "color3", "Math", {{"value", "color3", "1, 1, 1"}}),
        entry("multiply", "color3", "Math",
              {{"in1", "color3", "1, 1, 1"}, {"in2", "color3", "1, 1, 1"}}),
        entry("add", "color3", "Math", {{"in1", "color3", "0, 0, 0"}, {"in2", "color3", "0, 0, 0"}}),
        entry("mix", "color3", "Math",
              {{"bg", "color3", "0, 0, 0"}, {"fg", "color3", "1, 1, 1"}, {"mix", "float", "0.5"}}),
        entry("normalmap", "vector3", "Texture", {{"in", "vector3", {}}, {"scale", "float", "1"}}),
        entry("texcoord", "vector2", "Geometric", {{"index", "integer", "0"}}),
        entry("standard_surface", "surfaceshader", "PBR / Shading",
              {{"base_color", "color3", "0.8, 0.8, 0.8"},
               {"specular_roughness", "float", "0.35"},
               {"metalness", "float", "0"},
               {"specular", "float", "0.5"},
               {"specular_IOR", "float", "1.5"},
               {"transmission", "float", "0"},
               {"opacity", "color3", "1, 1, 1"},
               {"emission", "float", "0"},
               {"emission_color", "color3", "1, 1, 1"},
               {"subsurface", "float", "0"},
               {"subsurface_color", "color3", "1, 0.75, 0.55"},
               {"subsurface_radius", "color3", "1, 0.35, 0.2"},
               {"subsurface_scale", "float", "1"},
               {"normal", "vector3", {}}}),
        entry("surfacematerial", "material", "PBR / Shading", {{"surfaceshader", "surfaceshader", {}}}),
    };
}

}  // namespace

QVector<MaterialXNodeCatalogEntry> listMaterialXNodeCatalog() {
#if SOLSTICE_HAVE_MATERIALX
    std::string error;
    auto doc = makeLibraryDocument(error);
    if (!doc) {
        logWarning("MaterialX catalog: " + error);
        return fallbackMaterialXCatalog();
    }

    // Prefer useful typed variants: for a category keep one nodedef per output type,
    // preferring color3 / float / vector3 / surfaceshader / material when available.
    std::map<std::pair<std::string, std::string>, mx::NodeDefPtr> chosen;
    for (const mx::NodeDefPtr& def : doc->getNodeDefs()) {
        if (!def) continue;
        const std::string category = def->getNodeString().empty() ? def->getName() : def->getNodeString();
        if (category.empty()) continue;
        // Skip interface / token helpers that are not graph nodes.
        if (category == "backdrop" || category == "tokengraph" || category == "nodedef") continue;
        const std::string type = def->getType();
        if (type.empty() || type == "none" || type == "multioutput") continue;
        const auto key = std::make_pair(category, type);
        if (!chosen.count(key)) chosen.emplace(key, def);
    }

    QVector<MaterialXNodeCatalogEntry> entries;
    entries.reserve(int(chosen.size()));
    for (const auto& [key, def] : chosen) {
        MaterialXNodeCatalogEntry entry;
        entry.category = QString::fromStdString(key.first);
        entry.type = QString::fromStdString(key.second);
        entry.group = catalogGroupFor(entry.category, entry.type, QString::fromStdString(def->getNodeGroup()));
        entry.label = entry.category + " (" + entry.type + ")";
        for (const mx::InputPtr& input : def->getActiveInputs()) {
            if (!input) continue;
            MaterialXNodeInputDef in;
            in.name = QString::fromStdString(input->getName());
            in.type = QString::fromStdString(input->getType());
            if (input->hasValueString()) in.value = QString::fromStdString(input->getValueString());
            else if (input->getValue()) in.value = QString::fromStdString(input->getValue()->getValueString());
            // Filename defaults are usually empty placeholders — keep empty for dialogs.
            if (in.type == "filename") in.value.clear();
            entry.inputs.push_back(in);
        }
        entries.push_back(entry);
    }

    std::sort(entries.begin(), entries.end(), [](const MaterialXNodeCatalogEntry& a,
                                                 const MaterialXNodeCatalogEntry& b) {
        if (a.group != b.group) return a.group < b.group;
        if (a.category != b.category) return a.category < b.category;
        return a.type < b.type;
    });
    return entries;
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
    ss->setInputValue("base_color", mx::Color3(0.8f, 0.8f, 0.8f));
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

    auto bindTex = [&](const char* inputName, std::shared_ptr<Image>& slot) {
        mx::NodePtr connected = resolveConnectedNode(ss, inputName);
        mx::NodePtr image = findImageNode(connected);
        if (!image) return;
        std::string texError;
        slot = loadTextureFromImageNode(image, searchDirectory, udimSet, texError);
        if (!slot && !texError.empty()) logWarning("MaterialX: " + texError);
    };

    bindTex("base_color", result.baseColorTexture);
    bindTex("specular_roughness", result.roughnessTexture);
    bindTex("metalness", result.metallicTexture);
    bindTex("opacity", result.opacityTexture);
    bindTex("emission_color", result.emissionTexture);
    bindTex("normal", result.normalTexture);
    bindTex("subsurface_color", result.subsurfaceTexture);

    result.ok = true;
    return result;
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
#else
    if (error) *error = "MaterialX unavailable";
    Q_UNUSED(xml);
    return false;
#endif
}

QString serializeMaterialXGraph(const QVector<MaterialXGraphNode>& nodes, const QVector<int>& udimSet) {
#if SOLSTICE_HAVE_MATERIALX
    auto out = mx::createDocument();
    out->setVersionString("1.38");
    for (const MaterialXGraphNode& graphNode : nodes) {
        if (graphNode.category.isEmpty() || graphNode.name.isEmpty()) continue;
        mx::NodePtr node =
            out->addNode(graphNode.category.toStdString(), graphNode.name.toStdString(), graphNode.type.toStdString());
        if (!std::isnan(graphNode.xpos)) node->setAttribute("xpos", std::to_string(graphNode.xpos));
        if (!std::isnan(graphNode.ypos)) node->setAttribute("ypos", std::to_string(graphNode.ypos));
        for (const MaterialXGraphInput& graphInput : graphNode.inputs) {
            if (graphInput.name.isEmpty()) continue;
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
#else
    Q_UNUSED(nodes);
    Q_UNUSED(udimSet);
    return {};
#endif
}

}  // namespace sol
