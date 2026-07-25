#include "io/materialx_graph.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>

#include "core/log.h"
#include "io/image_io.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_MATERIALX
#  include <MaterialXCore/Document.h>
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
    return QString::fromStdString(mx::writeToXmlString(out));
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
                                                 std::string& error) {
    if (!imageNode) return nullptr;
    const std::string category = imageNode->getCategory();
    if (category != "image" && category != "tiledimage") return nullptr;
    std::string file = inputValueString(imageNode, "file");
    if (file.empty()) return nullptr;
    QString path = QString::fromStdString(file);
    QFileInfo info(path);
    if (!info.isAbsolute() && !searchDirectory.isEmpty())
        path = QDir(searchDirectory).absoluteFilePath(path);
    if (!QFileInfo::exists(path)) {
        error = "texture not found: " + path.toStdString();
        return nullptr;
    }
    auto image = std::make_shared<Image>();
    std::string loadError;
    if (!loadImage(path.toStdString(), *image, loadError)) {
        error = loadError;
        return nullptr;
    }
    return image;
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
    material.subsurfaceScale = 0.05f * srMax(1e-4f, maxComponent(material.subsurfaceRadius));
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

    auto bindTex = [&](const char* inputName, std::shared_ptr<Image>& slot) {
        mx::NodePtr connected = resolveConnectedNode(ss, inputName);
        mx::NodePtr image = findImageNode(connected);
        if (!image) return;
        std::string texError;
        slot = loadTextureFromImageNode(image, searchDirectory, texError);
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

}  // namespace sol
