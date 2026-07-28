// Built in node types. The set mirrors the LOP network of Houdini Solaris:
// geometry sources, transforms, material assignment, lights, camera and render
// settings, all of which edit the stage flowing through the network.
#include <QDir>
#include <QFileInfo>
#include <QString>
#include <algorithm>
#include <map>

#include "core/log.h"
#include "core/units.h"
#include "io/alembic_loader.h"
#include "io/image_io.h"
#include "io/materialx_graph.h"
#include "io/usd_loader.h"
#include "nodes/node_registry.h"

namespace sol {
namespace {

QString resolvePath(const CookContext& context, const QString& path) {
    if (path.isEmpty()) return path;
    QFileInfo info(path);
    if (info.isAbsolute() || context.sceneDirectory.isEmpty()) return path;
    return QDir(context.sceneDirectory).absoluteFilePath(path);
}

QString primPathFor(const Node& node, const QString& group, const QString& explicitName) {
    QString leaf = explicitName.trimmed();
    if (leaf.isEmpty()) leaf = node.name();
    leaf.replace('/', '_');
    return "/" + group + "/" + leaf;
}

bool matchesPattern(const QString& pattern, const QString& path) {
    if (pattern.trimmed().isEmpty() || pattern.trimmed() == "*") return true;
    const QStringList tokens = pattern.split(' ', Qt::SkipEmptyParts);
    for (const QString& token : tokens) {
        if (globMatch(token.toStdString(), path.toStdString())) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Geometry sources
// ---------------------------------------------------------------------------

class AlembicNode : public Node {
public:
    explicit AlembicNode(const QString& name) : Node("alembic", name) {
        setInputLabels({"Input"});
        addParameter(Parameter::makeFile("file", "Alembic File", "", "Alembic (*.abc)")
                         .withTooltip("Path to an .abc archive; polygon meshes and subdivision cages are read"));
        addParameter(Parameter::makeString("primpath", "Prim Path", "/geo")
                         .withTooltip("Scene graph location the imported prims are placed under"));
        addParameter(Parameter::makeString("pathfilter", "Path Filter", "")
                         .withTooltip("Only import Alembic objects whose path matches this glob"));
        addParameter(Parameter::makeFloat("time", "Time", 0.0, 0.0, 100.0, false)
                         .withTooltip("Sample time in seconds; the nearest stored sample is used"));
        addParameter(Parameter::makeFloat("importscale", "Import Scale", 1.0, 0.001, 100.0, false)
                         .withTooltip(units::importScaleTooltip()));
        addParameter(Parameter::makeBool("importnormals", "Import Normals", true));
        addParameter(Parameter::makeBool("importuvs", "Import UVs", true));
        addTransformParameters(*this);
    }

    void cook(CookContext& context, const std::vector<StagePtr>&, Stage& stage) override {
        const QString file = resolvePath(context, stringValue("file"));
        if (file.isEmpty()) {
            context.reportWarning(this, "no Alembic file set");
            return;
        }
        if (!alembicSupportAvailable()) {
            context.reportError(this, "this build has no Alembic support");
            return;
        }
        if (!QFileInfo::exists(file)) {
            context.reportError(this, "file not found: " + file);
            return;
        }

        AlembicLoadOptions options;
        options.time = floatValue("time");
        options.scale = float(floatValue("importscale", 1.0));
        options.importNormals = boolValue("importnormals", true);
        options.importUvs = boolValue("importuvs", true);
        options.pathFilter = stringValue("pathfilter").toStdString();

        const QString cacheKey = file + "|" + QString::number(options.time) + "|" +
                                 QString::number(double(options.scale)) + "|" +
                                 QString::fromStdString(options.pathFilter) + "|" +
                                 (options.importNormals ? "n" : "-") + (options.importUvs ? "u" : "-");
        if (cacheKey != cacheKey_ || cache_.prims.empty()) {
            AlembicContents contents;
            std::string error;
            if (!loadAlembic(file.toStdString(), options, contents, error)) {
                context.reportError(this, QString::fromStdString(error));
                return;
            }
            cache_ = std::move(contents);
            cacheKey_ = cacheKey;
        }

        const Mat4 nodeTransform = transformFromParameters(*this);
        const QString root = stringValue("primpath", "/geo");
        for (const AlembicPrim& prim : cache_.prims) {
            StagePrim out;
            out.type = PrimType::Mesh;
            out.mesh = prim.mesh;
            out.xform = nodeTransform * prim.transform;
            out.sourceNode = name();
            QString leaf = QString::fromStdString(prim.path);
            if (leaf.startsWith('/')) leaf.remove(0, 1);
            leaf.replace('/', '_');
            out.path = (root.endsWith('/') ? root : root + "/") + leaf;
            out.material = Material();
            stage.addPrim(std::move(out));
        }
    }

private:
    QString cacheKey_;
    AlembicContents cache_;
};

class UsdNode : public Node {
public:
    explicit UsdNode(const QString& name) : Node("usd", name) {
        setInputLabels({"Input"});
        addParameter(Parameter::makeFile("file", "USD File", "", "USD (*.usd *.usda *.usdc)")
                         .withTooltip("Path to a USDA scene; meshes, cameras and lights are imported"));
        addParameter(Parameter::makeString("primpath", "Prim Path", "/geo")
                         .withTooltip("Scene graph location the imported prims are placed under"));
        addParameter(Parameter::makeString("pathfilter", "Path Filter", "")
                         .withTooltip("Only import prims whose path matches this glob"));
        addParameter(Parameter::makeFloat("importscale", "Import Scale", 1.0, 0.001, 100.0, false)
                         .withTooltip(units::importScaleTooltip()));
        addParameter(Parameter::makeBool("importnormals", "Import Normals", true));
        addParameter(Parameter::makeBool("importuvs", "Import UVs", true));
        addTransformParameters(*this);
    }

    void cook(CookContext& context, const std::vector<StagePtr>&, Stage& stage) override {
        const QString file = resolvePath(context, stringValue("file"));
        if (file.isEmpty()) {
            context.reportWarning(this, "no USD file set");
            return;
        }
        if (!usdSupportAvailable()) {
            context.reportError(this, "this build has no USD support");
            return;
        }
        if (!QFileInfo::exists(file)) {
            context.reportError(this, "file not found: " + file);
            return;
        }

        UsdLoadOptions options;
        options.scale = float(floatValue("importscale", 1.0));
        options.importNormals = boolValue("importnormals", true);
        options.importUvs = boolValue("importuvs", true);
        options.pathFilter = stringValue("pathfilter").toStdString();

        const QString cacheKey = file + "|" + QString::number(double(options.scale)) + "|" +
                                 QString::fromStdString(options.pathFilter);
        if (cacheKey != cacheKey_ || cache_.prims.empty()) {
            UsdContents contents;
            std::string error;
            if (!loadUsd(file.toStdString(), options, contents, error)) {
                context.reportError(this, QString::fromStdString(error));
                return;
            }
            cache_ = std::move(contents);
            cacheKey_ = cacheKey;
        }

        const Mat4 nodeTransform = transformFromParameters(*this);
        const QString meshRoot = stringValue("primpath", "/geo");
        for (const UsdPrim& prim : cache_.prims) {
            QString leaf = QString::fromStdString(prim.path);
            if (leaf.startsWith('/')) leaf.remove(0, 1);
            leaf.replace('/', '_');

            if (prim.type == UsdPrim::Type::Mesh && prim.mesh) {
                StagePrim out;
                out.type = PrimType::Mesh;
                out.mesh = prim.mesh;
                out.xform = nodeTransform * prim.transform;
                out.sourceNode = name();
                out.path = (meshRoot.endsWith('/') ? meshRoot : meshRoot + "/") + leaf;
                out.material = Material();
                stage.addPrim(std::move(out));
            } else if (prim.type == UsdPrim::Type::Camera && prim.hasCamera) {
                StagePrim out;
                out.type = PrimType::Camera;
                out.camera = prim.camera;
                out.camera.cameraToWorld = nodeTransform * prim.transform;
                out.sourceNode = name();
                out.path = "/cameras/" + leaf;
                stage.addPrim(std::move(out));
            } else if (prim.type == UsdPrim::Type::Light && prim.hasLight) {
                StagePrim out;
                out.type = PrimType::Light;
                out.light = prim.light;
                out.xform = nodeTransform * prim.transform;
                out.sourceNode = name();
                out.path = "/lights/" + leaf;
                stage.addPrim(std::move(out));
            }
        }
    }

private:
    QString cacheKey_;
    UsdContents cache_;
};

class PrimitiveNode : public Node {
public:
    enum class Shape { Sphere, Grid, Box, Tube };

    PrimitiveNode(const QString& typeName, const QString& name, Shape shape)
        : Node(typeName, name), shape_(shape) {
        setInputLabels({"Input"});
        addParameter(Parameter::makeString("primname", "Prim Name", name));
        switch (shape_) {
            case Shape::Sphere:
                addParameter(Parameter::makeFloat("radius", "Radius", 1.0, 0.001, 100.0, false));
                addParameter(Parameter::makeInt("segments", "Segments", 64, 4, 512));
                break;
            case Shape::Grid:
                addParameter(Parameter::makeFloat("sizex", "Size X", 10.0, 0.001, 1000.0, false));
                addParameter(Parameter::makeFloat("sizez", "Size Z", 10.0, 0.001, 1000.0, false));
                addParameter(Parameter::makeInt("divisions", "Divisions", 1, 1, 512));
                break;
            case Shape::Box:
                addParameter(Parameter::makeVec3("size", "Size", Vec3(1.0f, 1.0f, 1.0f)));
                break;
            case Shape::Tube:
                addParameter(Parameter::makeFloat("radius", "Radius", 0.5, 0.001, 100.0, false));
                addParameter(Parameter::makeFloat("height", "Height", 2.0, 0.001, 100.0, false));
                addParameter(Parameter::makeInt("segments", "Segments", 48, 3, 512));
                break;
        }
        addTransformParameters(*this);
    }

    void cook(CookContext&, const std::vector<StagePtr>&, Stage& stage) override {
        MeshPtr mesh;
        switch (shape_) {
            case Shape::Sphere:
                mesh = makeSphereMesh(float(floatValue("radius", 1.0)), intValue("segments", 64),
                                      std::max(3, intValue("segments", 64) / 2));
                break;
            case Shape::Grid:
                mesh = makeGridMesh(float(floatValue("sizex", 10.0)), float(floatValue("sizez", 10.0)),
                                    intValue("divisions", 1), intValue("divisions", 1));
                break;
            case Shape::Box: mesh = makeBoxMesh(vec3Value("size", Vec3(1.0f))); break;
            case Shape::Tube:
                mesh = makeTubeMesh(float(floatValue("radius", 0.5)), float(floatValue("height", 2.0)),
                                    intValue("segments", 48));
                break;
        }
        if (!mesh) return;
        StagePrim prim;
        prim.type = PrimType::Mesh;
        prim.mesh = std::move(mesh);
        prim.xform = transformFromParameters(*this);
        prim.sourceNode = name();
        prim.path = primPathFor(*this, "geo", stringValue("primname"));
        stage.addPrim(std::move(prim));
    }

private:
    Shape shape_;
};

// ---------------------------------------------------------------------------
// Stage editing
// ---------------------------------------------------------------------------

class TransformNode : public Node {
public:
    explicit TransformNode(const QString& name) : Node("transform", name) {
        addParameter(Parameter::makeString("pattern", "Prim Pattern", "*")
                         .withTooltip("Space separated globs; matching prims are transformed"));
        addTransformParameters(*this);
    }

    void cook(CookContext&, const std::vector<StagePtr>&, Stage& stage) override {
        const Mat4 transform = transformFromParameters(*this);
        const QString pattern = stringValue("pattern", "*");
        for (StagePrim& prim : stage.prims) {
            if (!matchesPattern(pattern, prim.path)) continue;
            prim.xform = transform * prim.xform;
        }
    }
};

class MergeNode : public Node {
public:
    explicit MergeNode(const QString& name) : Node("merge", name) {
        setInputLabels({"Input 1", "Input 2", "Input 3", "Input 4"});
    }

    bool copiesFirstInput() const override { return false; }

    void cook(CookContext&, const std::vector<StagePtr>& inputs, Stage& stage) override {
        for (const StagePtr& input : inputs) {
            if (input) stage.appendFrom(*input);
        }
    }
};

class SwitchNode : public Node {
public:
    explicit SwitchNode(const QString& name) : Node("switch", name) {
        setInputLabels({"Input 1", "Input 2", "Input 3", "Input 4"});
        addParameter(Parameter::makeInt("index", "Input Index", 0, 0, 3));
    }

    bool copiesFirstInput() const override { return false; }

    void cook(CookContext&, const std::vector<StagePtr>& inputs, Stage& stage) override {
        const int index = std::clamp(intValue("index", 0), 0, int(inputs.size()) - 1);
        if (index >= 0 && index < int(inputs.size()) && inputs[size_t(index)]) stage = *inputs[size_t(index)];
    }
};

class PruneNode : public Node {
public:
    explicit PruneNode(const QString& name) : Node("prune", name) {
        addParameter(Parameter::makeString("pattern", "Prim Pattern", ""));
        addParameter(Parameter::makeBool("invert", "Invert Selection", false));
    }

    void cook(CookContext&, const std::vector<StagePtr>&, Stage& stage) override {
        const QString pattern = stringValue("pattern");
        if (pattern.trimmed().isEmpty()) return;
        const bool invert = boolValue("invert", false);
        for (StagePrim& prim : stage.prims) {
            const bool matched = matchesPattern(pattern, prim.path);
            if (matched != invert) prim.active = false;
        }
    }
};

class NullNode : public Node {
public:
    explicit NullNode(const QString& name) : Node("null", name) {}
    void cook(CookContext&, const std::vector<StagePtr>&, Stage&) override {}
};

class MaterialNode : public Node {
public:
    explicit MaterialNode(const QString& name) : Node("material", name) {
        addParameter(Parameter::makeString("pattern", "Assign To", "*")
                         .withTooltip("Prim path or glob (e.g. /geo/sphere1). Copy from Scene Graph with "
                                      "Ctrl+C and paste here with Ctrl+V, or drag the prim name into this field"));
        addParameter(Parameter::makeLabel("mtlx_hint",
                                          "Edit shading in the Material Network tab "
                                          "(MaterialX, left → right like Houdini Solaris)")
                         .withGroup("MaterialX"));
        // Hidden from the parameter panel; edited by MaterialNetworkView.
        addParameter(Parameter::makeString("mtlx", "MaterialX Document", ""));
    }

    void cook(CookContext& context, const std::vector<StagePtr>&, Stage& stage) override {
        QString xml = stringValue("mtlx");
        if (xml.trimmed().isEmpty()) {
            xml = createDefaultMaterialXDocument();
            if (!xml.isEmpty()) setParameterValue("mtlx", xml);
        }

        MaterialXEvalResult evaluated = evaluateMaterialXDocument(xml, context.sceneDirectory);
        if (!evaluated.ok) {
            if (!evaluated.error.isEmpty()) context.reportError(this, evaluated.error);
            // Fallback constant material so the scene still renders.
            evaluated.material = Material{};
            evaluated.material.baseColor = Vec3(0.8f);
            evaluated.material.roughness = 0.35f;
        }

        const QString pattern = stringValue("pattern", "*");
        for (StagePrim& prim : stage.prims) {
            if (prim.type != PrimType::Mesh) continue;
            if (!matchesPattern(pattern, prim.path)) continue;
            prim.material = evaluated.material;
            prim.materialAssigned = true;
            prim.materialName = name();
            prim.baseColorTexture = evaluated.baseColorTexture;
            prim.roughnessTexture = evaluated.roughnessTexture;
            prim.metallicTexture = evaluated.metallicTexture;
            prim.opacityTexture = evaluated.opacityTexture;
            prim.emissionTexture = evaluated.emissionTexture;
            prim.normalTexture = evaluated.normalTexture;
            prim.subsurfaceTexture = evaluated.subsurfaceTexture;
            prim.procedurals = evaluated.procedurals;
            prim.proceduralImages = evaluated.proceduralImages;
        }
    }
};

// ---------------------------------------------------------------------------
// Lights
// ---------------------------------------------------------------------------

class LightNode : public Node {
public:
    LightNode(const QString& typeName, const QString& name, LightType type) : Node(typeName, name), type_(type) {
        addParameter(Parameter::makeString("primname", "Prim Name", name));
        addParameter(Parameter::makeBool("enabled", "Enabled", true)
                         .withGroup("Light")
                         .withTooltip("Uncheck to turn this light off without deleting the node"));
        addParameter(Parameter::makeColor("color", "Color", Vec3(1.0f, 1.0f, 1.0f)).withGroup("Light"));
        addParameter(Parameter::makeFloat("intensity", "Intensity", defaultIntensity(), 0.0, 100.0, false)
                         .withGroup("Light"));
        addParameter(Parameter::makeFloat("exposure", "Exposure", 0.0, -10.0, 10.0).withGroup("Light"));
        addParameter(Parameter::makeBool("shadows", "Cast Shadows", true)
                         .withGroup("Light")
                         .withTooltip("When off, this light ignores occluders (no shadows). "
                                      "For HDRI/dome lights, off removes hard environment shadows"));

        switch (type_) {
            case kLightRect:
                addParameter(Parameter::makeFloat("width", "Width", 2.0, 0.001, 100.0, false).withGroup("Shape"));
                addParameter(Parameter::makeFloat("height", "Height", 2.0, 0.001, 100.0, false).withGroup("Shape"));
                addParameter(Parameter::makeBool("normalize", "Normalize", true).withGroup("Shape"));
                addParameter(Parameter::makeBool("twosided", "Two Sided", false).withGroup("Shape"));
                addParameter(Parameter::makeBool("visiblecamera", "Visible To Camera", true).withGroup("Shape"));
                addParameter(Parameter::makeBool("selfshadows", "Self Shadows", false)
                                 .withGroup("Shape")
                                 .withTooltip("When off, this light's own geometry does not cast shadows"));
                break;
            case kLightDisk:
            case kLightSphere:
                addParameter(Parameter::makeFloat("radius", "Radius", 0.5, 0.001, 100.0, false).withGroup("Shape"));
                addParameter(Parameter::makeBool("normalize", "Normalize", true).withGroup("Shape"));
                addParameter(Parameter::makeBool("twosided", "Two Sided", false).withGroup("Shape"));
                addParameter(Parameter::makeBool("visiblecamera", "Visible To Camera", true).withGroup("Shape"));
                addParameter(Parameter::makeBool("selfshadows", "Self Shadows", false)
                                 .withGroup("Shape")
                                 .withTooltip("When off, this light's own geometry does not cast shadows"));
                break;
            case kLightDistant:
                addParameter(Parameter::makeFloat("angle", "Angular Diameter", 0.53, 0.0, 20.0).withGroup("Shape"));
                // Normalising by the solid angle makes "intensity" behave as
                // irradiance, which is what the sun disc size should not change.
                addParameter(Parameter::makeBool("normalize", "Normalize", true).withGroup("Shape"));
                break;
            case kLightDome:
                addParameter(Parameter::makeFile("texture", "HDRI Texture", "",
                                                 "Environment maps (*.hdr *.exr *.png *.jpg *.jpeg)")
                                 .withGroup("Environment"));
                addParameter(Parameter::makeBool("visiblecamera", "Visible To Camera", true)
                                 .withGroup("Environment")
                                 .withTooltip("Show the HDRI as the background"));
                break;
            default: break;
        }

        if (type_ == kLightDistant) {
            addTransformParameters(*this, Vec3(0.0f), Vec3(-45.0f, -35.0f, 0.0f), Vec3(1.0f));
        } else if (type_ == kLightRect) {
            addTransformParameters(*this, Vec3(0.0f, 4.0f, 0.0f), Vec3(90.0f, 0.0f, 0.0f), Vec3(1.0f));
        } else if (type_ == kLightDome) {
            addTransformParameters(*this);
        } else {
            addTransformParameters(*this, Vec3(0.0f, 3.0f, 0.0f), Vec3(0.0f), Vec3(1.0f));
        }
    }

    void cook(CookContext& context, const std::vector<StagePtr>&, Stage& stage) override {
        if (!boolValue("enabled", true)) return;

        StagePrim prim;
        prim.type = PrimType::Light;
        prim.sourceNode = name();
        prim.path = primPathFor(*this, "lights", stringValue("primname"));
        prim.xform = transformFromParameters(*this);

        LightData light;
        light.type = type_;
        light.color = vec3Value("color", Vec3(1.0f));
        light.intensity = float(floatValue("intensity", defaultIntensity()));
        light.exposure = float(floatValue("exposure", 0.0));
        light.shadowEnable = boolValue("shadows", true) ? 1 : 0;
        light.selfShadowEnable = boolValue("selfshadows", false) ? 1 : 0;
        light.normalize = boolValue("normalize", type_ != kLightDistant) ? 1 : 0;
        light.twoSided = boolValue("twosided", false) ? 1 : 0;
        light.visibleCamera = boolValue("visiblecamera", true) ? 1 : 0;
        light.width = float(floatValue("width", 1.0));
        light.height = float(floatValue("height", 1.0));
        light.radius = float(floatValue("radius", 0.5));
        light.angle = float(floatValue("angle", 0.53));

        if (type_ == kLightDome) {
            const QString texture = resolvePath(context, stringValue("texture"));
            if (!texture.isEmpty()) {
                if (texture != envPath_ || !environment_) {
                    auto env = std::make_shared<EnvironmentMap>();
                    std::string error;
                    if (!loadImage(texture.toStdString(), env->image, error)) {
                        context.reportError(this, QString::fromStdString(error));
                    } else {
                        env->path = texture.toStdString();
                        env->buildSamplingTables();
                        environment_ = std::move(env);
                        envPath_ = texture;
                        logInfo("Dome light loaded " + texture.toStdString() + " (" +
                                std::to_string(environment_->image.width()) + "x" +
                                std::to_string(environment_->image.height()) + ")");
                    }
                }
                prim.environment = environment_;
            } else {
                environment_.reset();
                envPath_.clear();
            }
        }

        prim.light = light;
        stage.addPrim(std::move(prim));
    }

private:
    double defaultIntensity() const {
        switch (type_) {
            case kLightDistant: return 3.0;
            // A white dome without a texture acts as a soft ambient fill, so it
            // is dialled back to keep the default lighting readable.
            case kLightDome: return 0.35;
            case kLightRect: return 40.0;
            case kLightDisk: return 30.0;
            case kLightSphere: return 40.0;
            default: return 10.0;
        }
    }

    LightType type_;
    QString envPath_;
    std::shared_ptr<EnvironmentMap> environment_;
};

// ---------------------------------------------------------------------------
// Camera and render settings
// ---------------------------------------------------------------------------

class CameraNode : public Node {
public:
    explicit CameraNode(const QString& name) : Node("camera", name) {
        addParameter(Parameter::makeString("primname", "Prim Name", name));
        addParameter(Parameter::makeFloat("focal", "Focal Length (mm)", 50.0, 8.0, 300.0)
                         .withGroup("Lens")
                         .withTooltip("Focal length in millimetres (Houdini camera convention). "
                                      "Use the preset menu for common lens lengths"));
        addParameter(Parameter::makeFloat("aperture", "Sensor Width (mm)", 36.0, 4.0, 100.0)
                         .withGroup("Lens")
                         .withTooltip("Horizontal aperture in millimetres"));
        addParameter(Parameter::makeFloat("fstop", "F-Stop", 0.0, 0.0, 64.0)
                         .withGroup("Lens")
                         .withTooltip("0 disables depth of field. Use the preset menu for common stops"));
        addParameter(Parameter::makeFloat("focusdistance", "Focus Distance", 5.0, 0.01, 1000.0, false)
                         .withGroup("Lens")
                         .withTooltip(units::focusDistanceTooltip()));
        addParameter(Parameter::makeBool("uselookat", "Use Look At", true).withGroup("Placement"));
        addParameter(Parameter::makeVec3("eye", "Eye", Vec3(6.0f, 4.0f, 9.0f))
                         .withGroup("Placement")
                         .withTooltip(units::lengthTooltip()));
        addParameter(Parameter::makeVec3("target", "Look At", Vec3(0.0f, 1.0f, 0.0f))
                         .withGroup("Placement")
                         .withTooltip(units::lengthTooltip()));
        addParameter(Parameter::makeVec3("up", "Up", Vec3(0.0f, 1.0f, 0.0f)).withGroup("Placement"));
        addTransformParameters(*this, Vec3(0.0f, 2.0f, 8.0f));
    }

    void cook(CookContext&, const std::vector<StagePtr>&, Stage& stage) override {
        StagePrim prim;
        prim.type = PrimType::Camera;
        prim.sourceNode = name();
        prim.path = primPathFor(*this, "cameras", stringValue("primname"));

        CameraData camera;
        camera.focalLength = float(floatValue("focal", 50.0));
        camera.sensorWidth = float(floatValue("aperture", 36.0));
        camera.fStop = float(floatValue("fstop", 0.0));
        camera.focusDistance = float(floatValue("focusdistance", 5.0));

        if (boolValue("uselookat", true)) {
            const Vec3 eye = vec3Value("eye", Vec3(6.0f, 4.0f, 9.0f));
            const Vec3 target = vec3Value("target", Vec3(0.0f, 1.0f, 0.0f));
            const Vec3 up = vec3Value("up", Vec3(0.0f, 1.0f, 0.0f));
            prim.xform = lookAtMatrix(eye, target, up);
            if (floatValue("focusdistance", 5.0) <= 0.0) camera.focusDistance = length(target - eye);
        } else {
            prim.xform = transformFromParameters(*this);
        }
        camera.cameraToWorld = prim.xform;
        prim.camera = camera;
        // addPrim may rename on collision — bind render camera to the final path.
        stage.renderCameraPath = stage.addPrim(std::move(prim));
    }
};

class RenderSettingsNode : public Node {
public:
    explicit RenderSettingsNode(const QString& name) : Node("rendersettings", name) {
        addParameter(Parameter::makeInt("resx", "Resolution X", 960, 16, 8192, false).withGroup("Image"));
        addParameter(Parameter::makeInt("resy", "Resolution Y", 540, 16, 8192, false).withGroup("Image"));
        addParameter(Parameter::makeInt("samples", "Samples Per Pixel", 128, 1, 100000, false).withGroup("Image"));
        addParameter(Parameter::makeMenu("backend", "Render Backend", {"CPU (Embree)", "GPU (OptiX)"}, 0)
                         .withGroup("Engine"));
        addParameter(Parameter::makeMenu("integrator", "Integrator",
                                         {"Path Tracer", "Direct Lighting", "Ambient Occlusion"}, 0)
                         .withGroup("Engine"));
        addParameter(Parameter::makeInt("maxdepth", "Max Ray Depth", 8, 1, 64).withGroup("Engine"));
        addParameter(Parameter::makeInt("rrdepth", "Russian Roulette Depth", 3, 1, 64).withGroup("Engine"));
        addParameter(Parameter::makeInt("lightsamples", "Light Samples", 2, 1, 16)
                         .withGroup("Engine")
                         .withTooltip("Next-event estimation samples per bounce (MIS with BSDF). "
                                      "Higher = less light/reflection noise, slower."));
        addParameter(Parameter::makeFloat("clamp", "Indirect Clamp", 10.0, 0.0, 1000.0, false)
                         .withGroup("Engine")
                         .withTooltip("Caps bright indirect path contributions (fireflies). "
                                      "0 disables. Lower = cleaner, slightly darker caustics."));
        addParameter(Parameter::makeInt("seed", "Seed", 0, 0, 100000, false).withGroup("Engine"));
        addParameter(Parameter::makeInt("threads", "CPU Threads", 0, 0, 256, false)
                         .withGroup("Engine")
                         .withTooltip("0 uses every available core"));
        addParameter(Parameter::makeInt("tilesize", "Tile Size", 32, 8, 256).withGroup("Engine"));
        addParameter(Parameter::makeFloat("aodistance", "AO Distance", 1.0, 0.01, 100.0, false).withGroup("Engine"));
        addParameter(Parameter::makeBool("pathguiding", "Path Guiding (OpenPGL)", false)
                         .withGroup("Engine")
                         .withTooltip("Learn incident radiance while rendering and guide BSDF "
                                      "samples (CPU / Embree only). Off by default."));
        addParameter(Parameter::makeMenu("tonemap", "Tone Map", {"None", "Reinhard", "ACES"}, 2).withGroup("Film"));
        addParameter(Parameter::makeFloat("exposure", "Exposure", 0.0, -8.0, 8.0).withGroup("Film"));
        addParameter(Parameter::makeFloat("gamma", "Gamma", 2.2, 1.0, 4.0).withGroup("Film"));
        addParameter(Parameter::makeBool("envvisible", "Environment Visible To Camera", true).withGroup("Film"));
    }

    void cook(CookContext&, const std::vector<StagePtr>&, Stage& stage) override {
        RenderSettingsData settings;
        settings.resolutionX = intValue("resx", 960);
        settings.resolutionY = intValue("resy", 540);
        settings.samplesPerPixel = intValue("samples", 128);
        settings.backend = intValue("backend", 0) == 1 ? kBackendGpuOptix : kBackendCpuEmbree;
        settings.integrator = intValue("integrator", 0);
        settings.maxDepth = intValue("maxdepth", 8);
        settings.rrStartDepth = intValue("rrdepth", 3);
        settings.lightSamples = std::max(1, intValue("lightsamples", 2));
        settings.clampIndirect = float(floatValue("clamp", 10.0));
        settings.seed = intValue("seed", 0);
        settings.threads = intValue("threads", 0);
        settings.tileSize = intValue("tilesize", 32);
        settings.aoDistance = float(floatValue("aodistance", 1.0));
        settings.pathGuiding = boolValue("pathguiding", false) ? 1 : 0;
        settings.toneMapper = intValue("tonemap", 2);
        settings.exposure = float(floatValue("exposure", 0.0));
        settings.gamma = float(floatValue("gamma", 2.2));
        settings.envVisibleCamera = boolValue("envvisible", true) ? 1 : 0;
        stage.settings = settings;
        stage.settingsAuthored = true;
    }
};

template <typename NodeT, typename... Args>
NodeTypeInfo makeType(const QString& typeName, const QString& label, const QString& category,
                      const QString& description, const QString& color, Args... args) {
    NodeTypeInfo info;
    info.typeName = typeName;
    info.label = label;
    info.category = category;
    info.description = description;
    info.colorHex = color;
    info.factory = [args...](const QString& name) -> NodePtr { return std::make_unique<NodeT>(name, args...); };
    return info;
}

}  // namespace

void registerBuiltinNodes() {
    static bool registered = false;
    if (registered) return;
    registered = true;

    NodeRegistry& registry = NodeRegistry::instance();

    registry.registerType(makeType<AlembicNode>("alembic", "Alembic Import", "Geometry",
                                                "Loads polygon meshes from an .abc archive", "#a8aaae"));
    registry.registerType(makeType<UsdNode>("usd", "USD Import", "Geometry",
                                            "Loads meshes, cameras and lights from a USDA scene", "#a8aaae"));

    {
        NodeTypeInfo info;
        info.typeName = "sphere";
        info.label = "Sphere";
        info.category = "Geometry";
        info.description = "Polygonal sphere primitive";
        info.colorHex = "#a8aaae";
        info.factory = [](const QString& name) -> NodePtr {
            return std::make_unique<PrimitiveNode>("sphere", name, PrimitiveNode::Shape::Sphere);
        };
        registry.registerType(info);

        info.typeName = "grid";
        info.label = "Grid";
        info.description = "Ground plane primitive";
        info.factory = [](const QString& name) -> NodePtr {
            return std::make_unique<PrimitiveNode>("grid", name, PrimitiveNode::Shape::Grid);
        };
        registry.registerType(info);

        info.typeName = "box";
        info.label = "Box";
        info.description = "Box primitive";
        info.factory = [](const QString& name) -> NodePtr {
            return std::make_unique<PrimitiveNode>("box", name, PrimitiveNode::Shape::Box);
        };
        registry.registerType(info);

        info.typeName = "tube";
        info.label = "Tube";
        info.description = "Open cylinder primitive";
        info.factory = [](const QString& name) -> NodePtr {
            return std::make_unique<PrimitiveNode>("tube", name, PrimitiveNode::Shape::Tube);
        };
        registry.registerType(info);
    }

    registry.registerType(makeType<TransformNode>("transform", "Transform", "Utility",
                                                  "Transforms prims matching a pattern", "#6a5f8a"));
    registry.registerType(
        makeType<MergeNode>("merge", "Merge", "Utility", "Combines up to four stages", "#6a5f8a"));
    registry.registerType(
        makeType<SwitchNode>("switch", "Switch", "Utility", "Passes through one selected input", "#6a5f8a"));
    registry.registerType(
        makeType<PruneNode>("prune", "Prune", "Utility", "Deactivates prims matching a pattern", "#6a5f8a"));
    registry.registerType(makeType<NullNode>("null", "Null", "Utility", "Pass through / bookmark", "#565b63"));
    registry.registerType(makeType<MaterialNode>("material", "Material", "Material",
                                                 "Principled surface assigned by prim pattern", "#8a6a3f"));

    {
        NodeTypeInfo info;
        info.category = "Lighting";
        info.colorHex = "#d67a2a";

        info.typeName = "domelight";
        info.label = "Dome Light";
        info.description = "HDRI environment light";
        info.factory = [](const QString& name) -> NodePtr {
            return std::make_unique<LightNode>("domelight", name, kLightDome);
        };
        registry.registerType(info);

        info.typeName = "distantlight";
        info.label = "Distant Light";
        info.description = "Sun style directional light";
        info.factory = [](const QString& name) -> NodePtr {
            return std::make_unique<LightNode>("distantlight", name, kLightDistant);
        };
        registry.registerType(info);

        info.typeName = "rectlight";
        info.label = "Rect Light";
        info.description = "Rectangular area light";
        info.factory = [](const QString& name) -> NodePtr {
            return std::make_unique<LightNode>("rectlight", name, kLightRect);
        };
        registry.registerType(info);

        info.typeName = "disklight";
        info.label = "Disk Light";
        info.description = "Disk area light";
        info.factory = [](const QString& name) -> NodePtr {
            return std::make_unique<LightNode>("disklight", name, kLightDisk);
        };
        registry.registerType(info);

        info.typeName = "spherelight";
        info.label = "Sphere Light";
        info.description = "Spherical area light";
        info.factory = [](const QString& name) -> NodePtr {
            return std::make_unique<LightNode>("spherelight", name, kLightSphere);
        };
        registry.registerType(info);
    }

    registry.registerType(
        makeType<CameraNode>("camera", "Camera", "Camera", "Render camera with lens controls", "#3a76b2"));
    registry.registerType(makeType<RenderSettingsNode>("rendersettings", "Render Settings", "Render",
                                                       "Resolution, sampling and backend selection", "#8a4550"));
}

}  // namespace sol
