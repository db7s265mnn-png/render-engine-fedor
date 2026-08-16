// Built in node types. The set mirrors the LOP network of Houdini Solaris:
// geometry sources, transforms, material assignment, lights, camera and render
// settings, all of which edit the stage flowing through the network.
#include <QDir>
#include <QFileInfo>
#include <QString>
#include <QRegularExpression>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

#include "core/expr_eval.h"
#include "core/log.h"
#include "core/units.h"
#include "io/alembic_loader.h"
#include "io/image_io.h"
#include "io/materialx_graph.h"
#include "render/metal_spectra.h"
#include "io/usd_loader.h"
#include "nodes/node_registry.h"
#include "render/cpu/polynomial_optics.h"
#include "render/physical_sky.h"
#include "render/render_device.h"

namespace sol {

void registerVdbNodes(NodeRegistry& registry);

namespace {

QString resolvePath(const CookContext& context, const QString& path) {
    if (path.isEmpty()) return path;
    QString expanded = path.contains(QStringLiteral("$F"))
                           ? resolveFramePathExisting(path, context.frame)
                           : expandStringExpression(path, context.frame);
    QFileInfo info(expanded);
    if (info.isAbsolute() || context.sceneDirectory.isEmpty()) return expanded;
    return QDir(context.sceneDirectory).absoluteFilePath(expanded);
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
        addParameter(Parameter::makeFloat("time", "Time Offset", 0.0, -10000.0, 10000.0, false)
                         .withTooltip("Seconds added to the global timeline time when sampling "
                                      "the Alembic archive"));
        addParameter(Parameter::makeFloat("importscale", "Import Scale", 1.0, 0.001, 100.0, false)
                         .withTooltip(units::importScaleTooltip()));
        addParameter(Parameter::makeBool("importnormals", "Import Normals", true));
        addParameter(Parameter::makeBool("importuvs", "Import UVs", true));
        addTessellationParameters(*this);
        addMediumParameters(*this);
        addTransformParameters(*this);
    }

    bool dependsOnTime() const override {
        // Unknown until the first successful cook; afterwards only animated archives.
        if (cache_.prims.empty()) return true;
        return cache_.animated;
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
        options.time = context.time + floatValue("time");
        options.scale = float(floatValue("importscale", 1.0));
        options.importNormals = boolValue("importnormals", true);
        options.importUvs = boolValue("importuvs", true);
        options.pathFilter = stringValue("pathfilter").toStdString();

        // Options that force a reload even for static archives.
        const QString optionsKey = file + "|" + QString::number(double(options.scale)) + "|" +
                                   QString::fromStdString(options.pathFilter) + "|" +
                                   (options.importNormals ? "n" : "-") + (options.importUvs ? "u" : "-");
        const QString timedKey = optionsKey + "|" + QString::number(options.time, 'g', 12);

        const bool canReuseStatic =
            optionsKey == optionsKey_ && !cache_.animated && !cache_.prims.empty();
        if (!canReuseStatic && (timedKey != cacheKey_ || cache_.prims.empty())) {
            AlembicContents contents;
            std::string error;
            if (!loadAlembic(file.toStdString(), options, contents, error)) {
                context.reportError(this, QString::fromStdString(error));
                return;
            }
            cache_ = std::move(contents);
            cacheKey_ = timedKey;
            optionsKey_ = optionsKey;
        }
        if (cache_.animated) context.suggestPlaybackRange(cache_.startTime, cache_.endTime);

        const Mat4 nodeTransform = transformFromParameters(*this);
        const QString root = stringValue("primpath", "/geo");
        for (const AlembicPrim& prim : cache_.prims) {
            StagePrim out;
            out.type = PrimType::Mesh;
            out.mesh = prim.mesh;
            out.xform = nodeTransform * prim.transform;
            out.timeDependent = prim.timeDependent;
            out.sourceNode = name();
            QString leaf = QString::fromStdString(prim.path);
            if (leaf.startsWith('/')) leaf.remove(0, 1);
            leaf.replace('/', '_');
            out.path = (root.endsWith('/') ? root : root + "/") + leaf;
            out.material = Material();
            applyTessellationParameters(*this, out);
            applyMediumParameters(*this, out);
            stage.addPrim(std::move(out));
        }
    }

private:
    QString cacheKey_;
    QString optionsKey_;
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
        addParameter(Parameter::makeFloat("time", "Time Offset", 0.0, -10000.0, 10000.0, false)
                         .withTooltip("Seconds added to the global timeline time when sampling "
                                      "animated USD (timeSamples)"));
        addParameter(Parameter::makeFloat("importscale", "Import Scale", 1.0, 0.001, 100.0, false)
                         .withTooltip(units::importScaleTooltip()));
        addParameter(Parameter::makeBool("importnormals", "Import Normals", true));
        addParameter(Parameter::makeBool("importuvs", "Import UVs", true));
        addTessellationParameters(*this);
        addMediumParameters(*this);
        addTransformParameters(*this);
    }

    bool dependsOnTime() const override {
        // After we observe that a time change did not alter geometry, stop dirtying.
        if (cache_.prims.empty()) return true;
        return !staticReuseOk_;
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
        options.time = context.time + floatValue("time");
        options.scale = float(floatValue("importscale", 1.0));
        options.importNormals = boolValue("importnormals", true);
        options.importUvs = boolValue("importuvs", true);
        options.pathFilter = stringValue("pathfilter").toStdString();

        const QString optionsKey = file + "|" + QString::number(double(options.scale)) + "|" +
                                   QString::fromStdString(options.pathFilter) + "|" +
                                   (options.importNormals ? "n" : "-") + (options.importUvs ? "u" : "-");
        const QString timedKey = optionsKey + "|" + QString::number(options.time, 'g', 12);

        if (optionsKey == optionsKey_ && staticReuseOk_ && !cache_.prims.empty()) {
            // Static USD: ignore timeline time for reloads.
        } else if (timedKey != cacheKey_ || cache_.prims.empty()) {
            UsdContents contents;
            std::string error;
            if (!loadUsd(file.toStdString(), options, contents, error)) {
                context.reportError(this, QString::fromStdString(error));
                return;
            }
            if (optionsKey == optionsKey_ && !cache_.prims.empty() &&
                contents.prims.size() == cache_.prims.size()) {
                // Same options + same prim count at a new time → treat as static.
                staticReuseOk_ = true;
            } else {
                cache_ = std::move(contents);
                staticReuseOk_ = false;
            }
            cacheKey_ = timedKey;
            optionsKey_ = optionsKey;
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
                // USD loader has weak timeSamples support; treat as animated until
                // proven static (staticReuseOk_).
                out.timeDependent = !staticReuseOk_;
                out.sourceNode = name();
                out.path = (meshRoot.endsWith('/') ? meshRoot : meshRoot + "/") + leaf;
                out.material = Material();
                applyTessellationParameters(*this, out);
                applyMediumParameters(*this, out);
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
    QString optionsKey_;
    bool staticReuseOk_ = false;
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
                addParameter(Parameter::makeInt("segments", "Segments", 96, 4, 512));
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
        addTessellationParameters(*this);
        addMediumParameters(*this);
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
        applyTessellationParameters(*this, prim);
        applyMediumParameters(*this, prim);
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
        // Legacy tag (pre conductor_eta/k). Hidden; migrated into MaterialX on cook.
        addParameter(Parameter::makeMenu("spectralmetalpreset", "Spectral Metal Preset",
                                         {"None", "Gold (Au)", "Silver (Ag)", "Copper (Cu)", "Aluminium (Al)"},
                                         0)
                         .withVisibleWhen("false"));
    }

    void cook(CookContext& context, const std::vector<StagePtr>&, Stage& stage) override {
        try {
            QString xml = stringValue("mtlx");
            if (xml.trimmed().isEmpty()) {
                xml = createDefaultMaterialXDocument();
                if (!xml.isEmpty()) setParameterValue("mtlx", xml);
            }

            // Density / filter tweaks on an upstream VDB dirty this node even when
            // the MaterialX document is unchanged. Reuse the last eval so we do not
            // re-parse XML (or reload textures) on every slider tick.
            MaterialXEvalResult evaluated;
            const bool cacheHit = evalCacheValid_ && cachedXml_ == xml &&
                                  cachedSearchDir_ == context.sceneDirectory;
            if (cacheHit) {
                evaluated = cachedEval_;
            } else {
                evaluated = evaluateMaterialXDocument(xml, context.sceneDirectory);
                if (!evaluated.ok) {
                    if (!evaluated.error.isEmpty()) context.reportError(this, evaluated.error);
                    // Fallback constant material so the scene still renders.
                    evaluated.material = Material{};
                    evaluated.material.baseColor = Vec3(0.8f);
                    evaluated.material.roughness = 0.35f;
                    evaluated.procedurals.clear();
                    evaluated.proceduralImages.clear();
                }
            }
            // Legacy spectralMetalPreset → conductor_eta / conductor_k in MaterialX.
            const int legacyPreset = intValue("spectralmetalpreset", 0);
            if (legacyPreset > 0) {
                const char* name = "Au";
                if (legacyPreset == 2) name = "Ag";
                else if (legacyPreset == 3) name = "Cu";
                else if (legacyPreset == 4) name = "Al";
                Vec3 etaRgb, kRgb;
                metalNkRgbPreset(name, etaRgb, kRgb);
                evaluated.material.conductorEta = etaRgb;
                evaluated.material.conductorK = kRgb;
                auto fmt3 = [](Vec3 v) {
                    return QString("%1, %2, %3")
                        .arg(v.x, 0, 'g', 6)
                        .arg(v.y, 0, 'g', 6)
                        .arg(v.z, 0, 'g', 6);
                };
                auto upsertColor = [](QString doc, const QString& input, const QString& value) {
                    const QRegularExpression re(
                        QStringLiteral("<input\\s+name=\"%1\"\\s+type=\"color3\"\\s+value=\"[^\"]*\"/>")
                            .arg(input));
                    const QString tag =
                        QStringLiteral("<input name=\"%1\" type=\"color3\" value=\"%2\"/>").arg(input, value);
                    if (re.match(doc).hasMatch())
                        return doc.replace(re, tag);
                    const QString needle = QStringLiteral("</standard_surface>");
                    int at = doc.indexOf(needle);
                    if (at >= 0)
                        doc.insert(at, QStringLiteral("    %1\n").arg(tag));
                    return doc;
                };
                xml = upsertColor(xml, QStringLiteral("conductor_eta"), fmt3(etaRgb));
                xml = upsertColor(xml, QStringLiteral("conductor_k"), fmt3(kRgb));
                setParameterValue("mtlx", xml, false);
                setParameterValue("spectralmetalpreset", 0, false);
            }

            // Drop dangling procedural roots (corrupt XML / failed partial compiles).
            auto sanitizeProc = [&](int& idx) {
                if (idx < 0) return;
                if (idx >= int(evaluated.procedurals.size())) idx = -1;
            };
            sanitizeProc(evaluated.material.baseColorProc);
            sanitizeProc(evaluated.material.roughnessProc);
            sanitizeProc(evaluated.material.metallicProc);
            sanitizeProc(evaluated.material.opacityProc);
            sanitizeProc(evaluated.material.emissionProc);
            sanitizeProc(evaluated.material.normalProc);
            sanitizeProc(evaluated.material.subsurfaceProc);
            sanitizeProc(evaluated.material.bumpProc);
            sanitizeProc(evaluated.material.displacementProc);
            sanitizeProc(evaluated.material.specularColorProc);
            sanitizeProc(evaluated.material.transmissionColorProc);

            const QString pattern = stringValue("pattern", "*");
            for (StagePrim& prim : stage.prims) {
                if (prim.type != PrimType::Mesh && prim.type != PrimType::Volume) continue;
                if (!matchesPattern(pattern, prim.path)) continue;
                prim.material = evaluated.material;
                prim.raySwitchBranches = evaluated.raySwitchBranches;
                prim.materialAssigned = true;
                prim.materialName = name();
                prim.baseColorTexture = evaluated.baseColorTexture;
                prim.roughnessTexture = evaluated.roughnessTexture;
                prim.metallicTexture = evaluated.metallicTexture;
                prim.opacityTexture = evaluated.opacityTexture;
                prim.emissionTexture = evaluated.emissionTexture;
                prim.normalTexture = evaluated.normalTexture;
                prim.bumpTexture = evaluated.bumpTexture;
                prim.displacementTexture = evaluated.displacementTexture;
                prim.subsurfaceTexture = evaluated.subsurfaceTexture;
                prim.specularColorTexture = evaluated.specularColorTexture;
                prim.transmissionColorTexture = evaluated.transmissionColorTexture;
                prim.procedurals = evaluated.procedurals;
                prim.proceduralImages = evaluated.proceduralImages;
            }

            cachedXml_ = xml;
            cachedSearchDir_ = context.sceneDirectory;
            cachedEval_ = std::move(evaluated);
            evalCacheValid_ = true;
        } catch (const std::exception& e) {
            evalCacheValid_ = false;
            context.reportError(this, QString("MaterialX cook failed: %1").arg(e.what()));
        } catch (...) {
            evalCacheValid_ = false;
            context.reportError(this, "MaterialX cook failed: unknown error");
        }
    }

private:
    QString cachedXml_;
    QString cachedSearchDir_;
    MaterialXEvalResult cachedEval_;
    bool evalCacheValid_ = false;
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
        addParameter(Parameter::makeFloat("colortemperature", "Color Temperature (K)", 0.0, 0.0, 20000.0)
                         .withGroup("Light")
                         .withTooltip("Blackbody CCT in Kelvin for spectral integrators.\n"
                                      "0 = off (RGB Color only). Typical: 2700 warm, 6500 daylight.\n"
                                      "RGB Path Tracer ignores this (uses Color)."));
        addParameter(Parameter::makeFloat("intensity", "Intensity", defaultIntensity(), 0.0, 100.0, false)
                         .withGroup("Light"));
        addParameter(Parameter::makeFloat("exposure", "Exposure", 0.0, -10.0, 10.0).withGroup("Light"));
        addParameter(Parameter::makeBool("shadows", "Cast Shadows", true)
                         .withGroup("Light")
                         .withTooltip("When off, this light ignores occluders (no shadows). "
                                      "For HDRI/dome lights, off removes hard environment shadows"));
        addParameter(Parameter::makeBool("caustics", "Contribute to Caustics", true)
                         .withGroup("Light")
                         .withTooltip("When off, this light still illuminates surfaces directly but "
                                      "does not cast caustics through glass (MNEE / BDPT / photon map / "
                                      "specular→light paths). Works for area, sun, and dome lights."));

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
                addParameter(Parameter::makeString("colorspace", "Color Space", "auto")
                                 .withGroup("Environment")
                                 .withTooltip("Arnold-style input colour space. Cook converts to ACEScg.\n"
                                              "auto: HDR/EXR → Utility - Linear - sRGB, 8-bit → sRGB Texture.\n"
                                              "ACES - ACEScg / Utility - Raw: no convert.\n"
                                              "Utility - Linear - sRGB: Rec.709 primaries → ACEScg (typical HDRI)."));
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
        light.colorTemperatureK = float(floatValue("colortemperature", 0.0));
        light.shadowEnable = boolValue("shadows", true) ? 1 : 0;
        light.selfShadowEnable = boolValue("selfshadows", false) ? 1 : 0;
        light.contributeCaustics = boolValue("caustics", true) ? 1 : 0;
        light.normalize = boolValue("normalize", type_ != kLightDistant) ? 1 : 0;
        light.twoSided = boolValue("twosided", false) ? 1 : 0;
        light.visibleCamera = boolValue("visiblecamera", true) ? 1 : 0;
        light.width = float(floatValue("width", 1.0));
        light.height = float(floatValue("height", 1.0));
        light.radius = float(floatValue("radius", 0.5));
        light.angle = float(floatValue("angle", 0.53));

        if (type_ == kLightDome) {
            const QString texture = resolvePath(context, stringValue("texture"));
            const QString cs = stringValue("colorspace", "auto");
            if (!texture.isEmpty()) {
                if (texture != envPath_ || cs != envCs_ || !environment_) {
                    auto env = std::make_shared<EnvironmentMap>();
                    std::string error;
                    const std::string texPath = texture.toStdString();
                    if (!loadImage(texPath, env->image, error, /*srgbColor=*/true, cs.toStdString())) {
                        context.reportError(this, QString::fromStdString(error));
                    } else {
                        env->path = texture.toStdString();
                        env->buildSamplingTables();
                        environment_ = std::move(env);
                        envPath_ = texture;
                        envCs_ = cs;
                        logInfo("Dome light loaded " + texture.toStdString() + " colorspace='" +
                                cs.toStdString() + "' (" +
                                std::to_string(environment_->image.width()) + "x" +
                                std::to_string(environment_->image.height()) + ")");
                    }
                }
                prim.environment = environment_;
            } else {
                environment_.reset();
                envPath_.clear();
                envCs_.clear();
            }
        }

        prim.light = light;
        stage.addPrim(std::move(prim));
    }

private:
    double defaultIntensity() const {
        switch (type_) {
            case kLightDistant: return 3.0;
            // Textured HDRI needs intensity 1 so the sun in the map is not dimmed.
            // An untextured white dome is also 1 (user can lower it).
            case kLightDome: return 1.0;
            case kLightRect: return 40.0;
            case kLightDisk: return 30.0;
            case kLightSphere: return 40.0;
            default: return 10.0;
        }
    }

    LightType type_;
    QString envPath_;
    QString envCs_;
    std::shared_ptr<EnvironmentMap> environment_;
};

// Hosek–Wilkie Physical Sky: RGB-data sky + solar disc on the dome (one env light).
class PhysicalSkyLightNode : public Node {
public:
    explicit PhysicalSkyLightNode(const QString& name) : Node("physicalskylight", name) {
        addParameter(Parameter::makeString("primname", "Prim Name", name));
        addParameter(Parameter::makeBool("enabled", "Enabled", true)
                         .withGroup("Light")
                         .withTooltip("Uncheck to turn this light off without deleting the node"));
        addParameter(Parameter::makeFloat("intensity", "Intensity", 1.0, 0.0, 100.0, false)
                         .withGroup("Light")
                         .withTooltip("Overall scale for both the sky dome and the sun.\n"
                                      "Sky Intensity and Sun Intensity are extra multipliers"));
        addParameter(Parameter::makeFloat("exposure", "Exposure", 0.0, -10.0, 10.0)
                         .withGroup("Light")
                         .withTooltip("2^exposure multiplier on sky and sun"));
        addParameter(Parameter::makeBool("shadows", "Cast Shadows", true).withGroup("Light"));
        addParameter(Parameter::makeBool("caustics", "Contribute to Caustics", true)
                         .withGroup("Light")
                         .withTooltip("When off, sky and sun still light surfaces directly but "
                                      "do not cast caustics through glass"));
        addParameter(Parameter::makeBool("visiblecamera", "Visible To Camera", true)
                         .withGroup("Light")
                         .withTooltip("Karma Render Light Geometry: show the sky and sun disc "
                                      "in the camera"));

        addParameter(Parameter::makeFloat("turbidity", "Turbidity", 3.0, 1.0, 10.0)
                         .withGroup("Sky")
                         .withTooltip("Hosek–Wilkie aerosol content (Karma Physical Sky).\n"
                                      "2 = arctic, 3 = clear temperate, 6 = warm/moist, 10 = hazy"));
        addParameter(Parameter::makeColor("groundalbedo", "Ground Albedo", Vec3(0.1f, 0.1f, 0.1f))
                         .withGroup("Sky")
                         .withTooltip("Ground reflectivity that affects sky colour"));
        addParameter(Parameter::makeBool("computegroundcolor", "Compute Ground Color", true)
                         .withGroup("Sky")
                         .withTooltip("On: ground from albedo × sky. Off: use Ground Color"));
        addParameter(Parameter::makeColor("groundcolor", "Ground Color", Vec3(0.1f, 0.1f, 0.1f))
                         .withGroup("Sky")
                         .withVisibleWhen("computegroundcolor==0")
                         .withTooltip("Visual ground colour when Compute Ground Color is off"));
        addParameter(Parameter::makeFloat("horizonblur", "Horizon Blur Falloff", 5.0, 0.0, 30.0)
                         .withGroup("Sky")
                         .withTooltip("Degrees below the horizon over which sky blends into ground"));
        addParameter(Parameter::makeColor("skytint", "Sky Tint", Vec3(1.0f, 1.0f, 1.0f))
                         .withGroup("Sky")
                         .withTooltip("Karma Sky Color — tints the dome (not the sun disc)"));
        addParameter(Parameter::makeFloat("elevation", "Solar Altitude", 45.0, 0.0, 90.0)
                         .withGroup("Sky")
                         .withTooltip("Vertical angle of the sun from the horizon (90 = zenith)"));
        addParameter(Parameter::makeFloat("azimuth", "Solar Azimuth", 90.0, 0.0, 360.0)
                         .withGroup("Sky")
                         .withTooltip("Horizontal angle along the horizon. 0 = −Z, 90 = +X"));
        addParameter(Parameter::makeBool("enablesky", "Enable Sky Light", true)
                         .withGroup("Sky")
                         .withTooltip("Creates the sky dome. Off keeps the sun only"));
        addParameter(Parameter::makeFloat("skyintensity", "Sky Intensity", 1.0, 0.0, 100.0, false)
                         .withGroup("Sky")
                         .withVisibleWhen("enablesky")
                         .withTooltip("Extra multiplier on the sky dome only"));

        addParameter(Parameter::makeBool("enablesun", "Enable Sun Light", true)
                         .withGroup("Sun")
                         .withTooltip("Solar disc on the sky (Hosek RGB data + 2013 solar radiance).\n"
                                      "Off keeps the sky colour from the sun position but no disc"));
        addParameter(Parameter::makeFloat("sunintensity", "Sun Intensity", 1.0, 0.0, 100.0, false)
                         .withGroup("Sun")
                         .withVisibleWhen("enablesun")
                         .withTooltip("Extra multiplier on the sun only"));
        addParameter(Parameter::makeColor("suntint", "Sun Color", Vec3(1.0f, 1.0f, 1.0f))
                         .withGroup("Sun")
                         .withVisibleWhen("enablesun"));
        addParameter(Parameter::makeFloat("sunsize", "Angular Size", 0.53, 0.01, 10.0)
                         .withGroup("Sun")
                         .withVisibleWhen("enablesun")
                         .withTooltip("Visual size of the sun disc on the sky, in degrees.\n"
                                      "Scene brightness stays the same (irradiance is normalized)"));

        addTransformParameters(*this);
    }

    void cook(CookContext& context, const std::vector<StagePtr>&, Stage& stage) override {
        if (!boolValue("enabled", true)) return;

        PhysicalSkyParams params;
        params.turbidity = float(floatValue("turbidity", 3.0));
        params.groundAlbedo = vec3Value("groundalbedo", Vec3(0.1f));
        params.skyTint = vmax(Vec3(0.0f), vec3Value("skytint", Vec3(1.0f)));
        params.elevationDeg = float(floatValue("elevation", 45.0));
        params.azimuthDeg = float(floatValue("azimuth", 90.0));
        params.intensity = float(floatValue("intensity", 1.0));
        params.skyIntensity = float(floatValue("skyintensity", 1.0));
        params.exposure = float(floatValue("exposure", 0.0));
        params.enableSky = boolValue("enablesky", true);
        params.enableSun = boolValue("enablesun", true);
        params.sunIntensity = float(floatValue("sunintensity", 1.0));
        params.sunTint = vmax(Vec3(0.0f), vec3Value("suntint", Vec3(1.0f)));
        params.sunSizeDeg = float(floatValue("sunsize", 0.53));
        params.computeGroundColor = boolValue("computegroundcolor", true);
        params.groundColor = vec3Value("groundcolor", Vec3(0.1f));
        params.horizonBlurDeg = float(floatValue("horizonblur", 5.0));

        const Mat4 nodeXform = transformFromParameters(*this);
        const int shadows = boolValue("shadows", true) ? 1 : 0;
        const int caustics = boolValue("caustics", true) ? 1 : 0;
        const int visible = boolValue("visiblecamera", true) ? 1 : 0;

        const bool needEnv = params.enableSky || params.enableSun;
        if (needEnv && !skyCacheValid(params)) {
            auto env = std::make_shared<EnvironmentMap>();
            bakePhysicalSkyEnv(env->image, params, 2048, 1024);
            env->path = "physical_sky";
            env->buildSamplingTables();
            environment_ = std::move(env);
            cacheTurbidity_ = params.turbidity;
            cacheElevation_ = params.elevationDeg;
            cacheAzimuth_ = params.azimuthDeg;
            cacheAlbedo_ = params.groundAlbedo;
            cacheGroundColor_ = params.groundColor;
            cacheSkyTint_ = params.skyTint;
            cacheSunTint_ = params.sunTint;
            cacheComputeGround_ = params.computeGroundColor;
            cacheHorizonBlur_ = params.horizonBlurDeg;
            cacheSkyIntensity_ = params.skyIntensity;
            cacheSunIntensity_ = params.sunIntensity;
            cacheSunSize_ = params.sunSizeDeg;
            cacheEnableSky_ = params.enableSky;
            cacheEnableSun_ = params.enableSun;
            logInfo("Physical Sky (Hosek–Wilkie) baked " + std::to_string(environment_->image.width()) + "x" +
                    std::to_string(environment_->image.height()) + " (turbidity " +
                    std::to_string(params.turbidity) + ")");
        }
        if (needEnv && !environment_) {
            context.reportError(this, "Physical Sky bake failed");
            return;
        }

        if (needEnv) {
            StagePrim domePrim;
            domePrim.type = PrimType::Light;
            domePrim.sourceNode = name();
            domePrim.path = primPathFor(*this, "lights", stringValue("primname"));
            domePrim.xform = nodeXform;
            domePrim.environment = environment_;

            LightData dome;
            dome.type = kLightDome;
            dome.color = Vec3(1.0f);
            dome.intensity = params.intensity;
            dome.exposure = params.exposure;
            dome.shadowEnable = shadows;
            dome.contributeCaustics = caustics;
            dome.visibleCamera = visible;
            domePrim.light = dome;
            stage.addPrim(std::move(domePrim));
        }
    }

private:
    bool skyCacheValid(const PhysicalSkyParams& p) const {
        if (!environment_ || environment_->image.empty()) return false;
        return std::fabs(cacheTurbidity_ - p.turbidity) < 1e-4f &&
               std::fabs(cacheElevation_ - p.elevationDeg) < 1e-4f &&
               std::fabs(cacheAzimuth_ - p.azimuthDeg) < 1e-4f &&
               lengthSquared(cacheAlbedo_ - p.groundAlbedo) < 1e-8f &&
               lengthSquared(cacheGroundColor_ - p.groundColor) < 1e-8f &&
               lengthSquared(cacheSkyTint_ - p.skyTint) < 1e-8f &&
               lengthSquared(cacheSunTint_ - p.sunTint) < 1e-8f &&
               cacheComputeGround_ == p.computeGroundColor &&
               std::fabs(cacheHorizonBlur_ - p.horizonBlurDeg) < 1e-4f &&
               std::fabs(cacheSkyIntensity_ - p.skyIntensity) < 1e-4f &&
               std::fabs(cacheSunIntensity_ - p.sunIntensity) < 1e-4f &&
               std::fabs(cacheSunSize_ - p.sunSizeDeg) < 1e-4f &&
               cacheEnableSky_ == p.enableSky && cacheEnableSun_ == p.enableSun;
    }

    std::shared_ptr<EnvironmentMap> environment_;
    float cacheTurbidity_ = -1.0f;
    float cacheElevation_ = -1.0f;
    float cacheAzimuth_ = -1.0f;
    Vec3 cacheAlbedo_{};
    Vec3 cacheGroundColor_{};
    Vec3 cacheSkyTint_{};
    Vec3 cacheSunTint_{};
    bool cacheComputeGround_ = true;
    float cacheHorizonBlur_ = -1.0f;
    float cacheSkyIntensity_ = -1.0f;
    float cacheSunIntensity_ = -1.0f;
    float cacheSunSize_ = -1.0f;
    bool cacheEnableSky_ = true;
    bool cacheEnableSun_ = true;
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
                         .withTooltip("Aperture. Lower = stronger bokeh. 0 = wide open. "
                                      "Polynomial Optics cannot open wider than the real lens "
                                      "(e.g. f/1 on an f/1.1 optic = wide open)."));
        addParameter(Parameter::makeFloat("focusdistance", "Focus Distance", 5.0, 0.01, 1000.0, false)
                         .withGroup("Lens")
                         .withTooltip(units::focusDistanceTooltip()));
        addParameter(Parameter::makeMenu("opticalmodel", "Camera Model",
                                         QStringList{"Thin Lens", "Polynomial Optics (Embree)"}, 0)
                         .withGroup("Optics")
                         .withTooltip("Polynomial Optics uses Lentil-style fitted real lenses "
                                      "(Embree only; OptiX falls back to Thin Lens)"));
        {
            QStringList lenses;
            for (const std::string& name : polynomialOpticsLensNames()) lenses << QString::fromStdString(name);
            addParameter(Parameter::makeMenu("lensmodel", "Lens", lenses, 19)
                             .withGroup("Optics")
                             .withTooltip("Real lens prescription (polynomial optics)"));
        }
        addParameter(Parameter::makeFloat("wavelength", "Wavelength (nm)", 550.0, 380.0, 780.0)
                         .withGroup("Optics")
                         .withTooltip("Wavelength for polynomial evaluation when Chromatic Aberration is off"));
        addParameter(Parameter::makeBool("chromatic", "Chromatic Aberration", false)
                         .withGroup("Optics")
                         .withTooltip("RGB chromatic aberration via per-sample R/G/B wavelengths "
                                      "(Polynomial Optics + Embree only)"));
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
        camera.opticalModel = intValue("opticalmodel", 0);
        camera.lensModel = intValue("lensmodel", 19);
        camera.opticalWavelengthNm = float(floatValue("wavelength", 550.0));
        camera.chromaticAberration = boolValue("chromatic", false) ? 1 : 0;

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
        // Group order follows first appearance: Image → Sampling → Engine → …

        // --- Image --------------------------------------------------------------------
        addParameter(Parameter::makeInt("resx", "Resolution X", 960, 16, 8192, false).withGroup("Image"));
        addParameter(Parameter::makeInt("resy", "Resolution Y", 540, 16, 8192, false).withGroup("Image"));
        addParameter(Parameter::makeFile("outputpath", "Output Path", "render.exr",
                                         "OpenEXR (*.exr);;All Files (*)")
                         .withGroup("Image")
                         .withFileSaveMode(true)
                         .withTooltip("Still-frame output path. Relative paths resolve next to the "
                                      "scene file (or the current working directory if unsaved).\n"
                                      "Default: render.exr. Existing files are overwritten."));
        addParameter(Parameter::makeButton("render", "Render")
                         .withGroup("Image")
                         .withTooltip("One full pass from scratch with the current settings. "
                                      "When all spp finish, writes a display-view EXR to Output Path."));
        addParameter(Parameter::makeMenu("bitdepth", "Bit Depth", {"8", "16", "32"}, 1)
                         .withGroup("Image")
                         .withTooltip("Bit depth for viewport resolve and EXR/PNG save.\n"
                                      "Accumulation stays float. Default 16 (half)."));
        addParameter(Parameter::makeBool("enabletxcache", "Convert Textures to TX", true)
                         .withGroup("Image")
                         .withTooltip("On Start / Render, convert source textures to .tx mipmaps "
                                      "via maketx (→ ACEScg when input colour space needs it)."));
        addParameter(Parameter::makeFile("txcachedir", "TX Cache Directory", "tx_cache", QString())
                         .withGroup("Image")
                         .withDirectoryMode(true)
                         .withTooltip("Directory for converted .tx files (default tx_cache). "
                                      "Names match the source basename; collisions use _copy_N."));

        // --- Sampling -----------------------------------------------------------------
        addParameter(Parameter::makeInt("samples", "Samples Per Pixel", 128, 1, 100000, false)
                         .withGroup("Sampling"));
        addParameter(Parameter::makeMenu("pixelsampler", "Pixel Sampler",
                                         {"Sobol (Owen)", "Blue Noise", "Xorshift", "GenPnt2D",
                                          "Manual-Test"},
                                         0)
                         .withGroup("Sampling")
                         .withTooltip(
                             "Camera AA / DoF / shutter primary samples.\n"
                             "Path bounces always use Owen-scrambled Sobol (PBRT4).\n"
                             "Sobol: stratified per pixel — recommended.\n"
                             "Blue Noise: CP dither from a 64×64 mask with per-tile phase.\n"
                             "Xorshift: Marsaglia xorshift32 white jitter.\n"
                             "GenPnt2D: plastic-number R2 (Roberts), n = sampleIndex + "
                             "per-pixel CP phase; shutter uses golden 1D.\n"
                             "Manual-Test: sample at pixel center, then add U(-1,1)×Mult "
                             "per axis (clamped to the pixel). For grid diagnostics.\n"
                             "Active sampler is shown in the viewport spp overlay."));
        addParameter(Parameter::makeFloat("manualtestmult", "Manual-Test Mult", 0.0, 0.0, 2.0, false)
                         .withGroup("Sampling")
                         .withVisibleWhen("pixelsampler==4")
                         .withTooltip("Manual-Test only. Pixel jitter = 0.5 + U(-1,1)×Mult on X and Y "
                                      "(same Mult), then clamped to [0,1).\n"
                                      "0 = exact pixel center. Raise to scatter samples inside "
                                      "(and, if Mult>0.5, would leave the pixel — but we clamp)."));
        // Hidden: marks menus that include Manual-Test (index 4). Older files used 4/5 for R2 modes.
        addParameter(Parameter::makeBool("_pixel_sampler_manual_v1", "", true));
        // Hidden: old menu was Legacy / FilmTile / Progressive (0/1/2).
        addParameter(Parameter::makeBool("_sampling_type_v2", "", true));
        addParameter(Parameter::makeMenu("samplingengine", "Sampling Type",
                                         {"Buckets", "Progressive"}, 0)
                         .withGroup("Sampling")
                         .withTooltip("How the frame is scheduled.\n"
                                      "Buckets: PBRT FilmTile — local bucket accum, then merge; "
                                      "strong (x,y,spp) seed.\n"
                                      "Progressive: no buckets — parallel scanlines, whole frame "
                                      "densifies evenly."));
        addParameter(Parameter::makeInt("tilesize", "Bucket Size (px)", 32, 0, 256, false)
                         .withGroup("Sampling")
                         .withVisibleWhen("samplingengine==0")
                         .withTooltip("Bucket size in pixels for Sampling Type = Buckets.\n"
                                      "0 = Auto (~8× threads tiles, side 8/16/32/64).\n"
                                      "Ignored by Progressive."));
        addParameter(Parameter::makeInt("seed", "Seed", 0, 0, 100000, false).withGroup("Sampling"));
        addParameter(Parameter::makeInt("lightsamples", "Light Samples", 2, 1, 16)
                         .withGroup("Sampling")
                         .withTooltip("Next-event estimation samples per bounce (MIS with BSDF). "
                                      "Higher = less light/reflection noise, slower."));
        addParameter(Parameter::makeFloat("clampdirect", "Direct Clamp", 10.0, 0.0, 1000000.0, false)
                         .withGroup("Sampling")
                         .withTooltip("Caps eye-path sample contributions in linear pixel radiance "
                                      "(Arnold Direct Clamp). Applies to PT/BDPT eye paths, NEE "
                                      "(including the first volume scatter on the lit side of fog), "
                                      "MNEE, and photon gather.\n"
                                      "Default 10; ~100 is a soft look. 0 disables (unbiased)."));
        addParameter(Parameter::makeFloat("clamp", "Indirect Clamp", 10.0, 0.0, 1000000.0, false)
                         .withGroup("Sampling")
                         .withTooltip("Caps BDPT light-tracing splat contributions in linear pixel "
                                      "radiance (Arnold Indirect Clamp). Raw LT deposits carry "
                                      "camera PDF — they are scaled to radiance before clamping.\n"
                                      "Affects BDPT / BDPT Spectral caustics from LT. 0 disables."));
        addParameter(Parameter::makeMenu("pixelfilter", "Pixel Filter",
                                         {"Box", "Triangle", "Gaussian", "Mitchell"}, 0)
                         .withGroup("Sampling")
                         .withTooltip("Film reconstruction filter — how each continuous sample is "
                                      "weighted into neighbouring pixels (PBRT / Arnold).\n"
                                      "Box (default): hard 1×1 pixels — sharp but makes 1spp noise "
                                      "look blocky when zoomed.\n"
                                      "Triangle / Gaussian / Mitchell: softer AA; softens the "
                                      "visible pixel grid at low spp.\n"
                                      "Does not change the Pixel Sampler (Sobol / GenPnt2D / …)."));
        addParameter(Parameter::makeFloat("filterradius", "Filter Radius", 0.5, 0.0, 8.0, false)
                         .withGroup("Sampling")
                         .withTooltip("Filter support in pixels. Changing Pixel Filter fills the "
                                      "PBRT-v4 recommended radius (Box 0.5, Triangle 2, "
                                      "Gaussian 1.5, Mitchell 2). You can still override.\n"
                                      "Larger = softer / more blur."));

        // --- Engine -------------------------------------------------------------------
        // OptiX is optional at compile time — label the menu so artists know when
        // this binary has no GPU backend (Windows CI historically shipped Embree-only).
        const QStringList backends =
            optixBackendCompiledIn()
                ? QStringList{"CPU (Embree)", "GPU (OptiX)"}
                : QStringList{"CPU (Embree)", "GPU (OptiX) — not in this build"};
        addParameter(Parameter::makeMenu("backend", "Render Backend", backends, 0)
                         .withGroup("Engine")
                         .withTooltip(optixBackendCompiledIn()
                                          ? QStringLiteral("CPU uses Embree. GPU requires an NVIDIA GPU "
                                                           "and a build with OptiX enabled.")
                                          : QStringLiteral("This executable was built without OptiX/CUDA. "
                                                          "GPU (OptiX) will fall back to Embree.")));
        addParameter(Parameter::makeMenu("integrator", "Integrator",
                                         {"Path Tracer", "BDPT (Bidirectional)", "Direct Lighting",
                                          "Ambient Occlusion", "PT Spectral", "BDPT Spectral",
                                          "Wireframe"},
                                         0)
                         .withGroup("Engine")
                         .withTooltip("Path Tracer: unidirectional (+ MNEE or Photon caustics).\n"
                                      "BDPT: bidirectional + light-tracing / Photon caustics "
                                      "(CPU only — OptiX falls back to Path Tracer).\n"
                                      "PT Spectral: hero-wavelength path tracer (CPU / Embree).\n"
                                      "BDPT Spectral: bidirectional + spectral transport "
                                      "(LT / MNEE / Photon + Indirect Guides; CPU / Embree).\n"
                                      "Wireframe: triangle edges with screen-space thickness "
                                      "(see Wireframe Thickness).\n"
                                      "Pick the caustics estimator under Caustics Engine.\n"
                                      "The log reports which caustics mode is active."));
        // Hidden migration marker: legacy menu was PT / DL / AO / BDPT.
        // New nodes default to v2; legacy files without this key are remapped on load.
        addParameter(Parameter::makeBool("_integrator_menu_v2", "", true));
        addParameter(Parameter::makeInt("spectralsamples", "Spectral Samples", 4, 2, 16)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator==4||integrator==5")
                         .withTooltip("Spectral integrators: number of hero wavelengths per path "
                                      "(2–16, default 4). Higher = cleaner colour, slower."));
        addParameter(Parameter::makeInt("spectralbins", "Spectral Bins", 16, 8, 32)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator==4||integrator==5")
                         .withTooltip("Spectral: fixed wavelength bins for multilayer spectral "
                                      "EXR / false-color (8–32)."));
        addParameter(Parameter::makeBool("spectralexr", "Write Spectral EXR Layers", false)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator==4||integrator==5")
                         .withTooltip("When saving EXR with a spectral integrator, also write "
                                      "fixed spectral bin layers (S0..Sn)."));
        addParameter(Parameter::makeMenu("spectralcolorspace", "Spectral Color Space",
                                         {"sRGB Linear", "ACEScg", "Rec.2020", "Display P3"}, 0)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator==4||integrator==5")
                         .withTooltip("Color space for spectral → beauty RGB (default sRGB Linear).\n"
                                      "Uses tabulated CIE XYZ + RGBColorSpace matrices."));
        addParameter(Parameter::makeMenu("spectralwavesamp", "Wavelength Sampling",
                                         {"Visible (importance)", "Uniform"}, 0)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator==4||integrator==5")
                         .withTooltip("Visible: pbrt-style PDF peaked near 538 nm (less colour noise).\n"
                                      "Uniform: stratified across 360–830 nm."));
        addParameter(Parameter::makeInt("maxdepth", "Max Ray Depth", 8, 1, 4096)
                         .withGroup("Engine")
                         .withTooltip("Max path bounces after the camera (surfaces + volume scatters).\n"
                                      "Dense fog / clouds with deep multiple scattering need 1000+.\n"
                                      "Also raise Russian Roulette Depth, or RR will kill deep paths early."));
        addParameter(Parameter::makeInt("rrdepth", "Russian Roulette Depth", 3, 1, 4096)
                         .withGroup("Engine")
                         .withTooltip("Start Russian roulette after this depth.\n"
                                      "For deep volume multiple scattering, set near Max Ray Depth."));
        addParameter(Parameter::makeInt("threads", "CPU Threads", 0, 0, 256, false)
                         .withGroup("Engine")
                         .withTooltip("0 uses every available core"));
        addParameter(Parameter::makeFloat("aodistance", "AO Distance", 1.0, 0.01, 100.0, false)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator==3")
                         .withTooltip("Ambient Occlusion: max occlusion ray length in scene units."));
        addParameter(Parameter::makeFloat("wireframethickness", "Wireframe Thickness", 1.0, 0.25, 8.0,
                                          false)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator==6")
                         .withTooltip("Wireframe: edge half-width in screen pixels (anti-aliased). "
                                      "Higher = thicker mesh lines."));
        addParameter(Parameter::makeBool("caustics", "Caustics", true)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator==0||integrator==1||integrator==5")
                         .withTooltip("Enable caustic light transport (light focused through glass "
                                      "and off mirrors).\n"
                                      "Engine picks the estimator (MNEE / MNEE+Photon / Photon).\n"
                                      "Per-light and per-material Contribute to Caustics can disable "
                                      "individual sources or casters.\n"
                                      "Off: glass casts dark shadows (soften with shadow_opacity)."));
        addParameter(Parameter::makeMenu("causticsengine", "Caustics Engine",
                                         {"MNEE (manifolds)", "MNEE+Photon", "Photon / VCM"}, 1)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator==0||integrator==1||integrator==5")
                         .withTooltip("MNEE: manifold next-event — best for near-delta glass + "
                                      "area lights.\n"
                                      "MNEE+Photon: picks one estimator for the scene — delta-only "
                                      "glass → MNEE; if any rough refractive caster exists → "
                                      "Photon / VCM. When Photon is active, MNEE / LT / eye-path "
                                      "BSDF caustics are turned off (no stacking).\n"
                                      "Photon / VCM: caustic-only photon map — rough glass and "
                                      "black bases through refraction."));
        // Hidden migration: legacy menu was Automatic / MNEE / Photon (0/1/2).
        addParameter(Parameter::makeBool("_caustics_engine_menu_v2", "", true));
        addParameter(Parameter::makeInt("photoncount", "Photon Count", 100000, 1000, 5000000, false)
                         .withGroup("Engine")
                         .withVisibleWhen("caustics==1&&causticsengine==1||caustics==1&&causticsengine==2")
                         .withTooltip("Photons emitted per progressive pass when Caustics Engine "
                                      "is Photon / VCM or MNEE+Photon (Auto→Photon)."));
        addParameter(Parameter::makeFloat("photonradius", "Photon Radius", 0.08, 0.001, 10.0, false)
                         .withGroup("Engine")
                         .withVisibleWhen("caustics==1&&causticsengine==1||caustics==1&&causticsengine==2")
                         .withTooltip("Initial gather radius (scene units) for the caustic photon "
                                      "map. Shrinks as samples accumulate."));
        addParameter(Parameter::makeFloat("causticclamp", "Caustic Firefly Clamp", 0.0, 0.0, 1000.0, false)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator==0||integrator==1||integrator==5")
                         .withTooltip("Extra cap on paths that look through glass/mirrors at a light "
                                      "(the sparkle inside refractive objects).\n"
                                      "Even at 0, a safety cap of 10 is applied to those paths — they "
                                      "never converge with more samples when the light is small.\n"
                                      "Raise to tighten further; the caustic on the floor is not capped."));
        addParameter(Parameter::makeMenu(
                         "dispersionmode", "Dispersion Mode",
                         {"Hero (default)", "Optimized", "Spectral RGB ×3", "Fake tint"}, 0)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator!=4&&integrator!=5")
                         .withTooltip(
                             "How chromatic dispersion (dispersion_abbe) is sampled.\n"
                             "Hero: one random RGB channel per sample; masks the whole path "
                             "(noisy; legacy).\n"
                             "Optimized: stratified channel + hero mask only if the path hit "
                             "dispersing glass + IOR change limited to first N interfaces. "
                             "Fixes ray_switch: camera-only Abbe no longer tints shadows.\n"
                             "Spectral RGB ×3: trace R+G+B and average (~3× slower, clean).\n"
                             "Fake tint: no ray bending — chromatic transmission tint only."));
        addParameter(Parameter::makeInt("dispersionmaxiface", "Dispersion Max Interfaces", 2, 1, 16)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator!=4&&integrator!=5&&dispersionmode==1")
                         .withTooltip("Optimized mode only: how many dispersing glass interfaces "
                                      "may change IOR (enter+exit of one pane ≈ 2)."));
        addParameter(Parameter::makeBool("pathguiding", "Indirect Guides (OpenPGL)", false)
                         .withGroup("Engine")
                         .withVisibleWhen("integrator==0||integrator==1||integrator==5")
                         .withTooltip("Learn incident radiance while rendering and guide eye-path "
                                      "BSDF samples (CPU / Embree). Works with Path Tracer, BDPT, "
                                      "and BDPT Spectral.\n"
                                      "Diffuse receivers only — glass/mirrors stay BSDF-sampled; "
                                      "caustic radiance (MNEE / photons / paths through glass) trains "
                                      "the field at the floor so guides learn bright caustic regions.\n"
                                      "Kicks in after the first training passes."));
        addParameter(Parameter::makeBool("motionblur", "Enable Motion Blur", false)
                         .withGroup("Motion Blur")
                         .withTooltip("Camera and geometry motion blur across the shutter "
                                      "(CPU / Embree). Uses the timeline frame as shutter center."));
        addParameter(Parameter::makeInt("motionkeys", "Motion Keys", 2, 2, 8)
                         .withGroup("Motion Blur")
                         .withTooltip("Number of transform / deformation samples across the "
                                      "shutter. 2 = open+close; higher = smoother blur."));
        addParameter(Parameter::makeFloat("shutterlength", "Shutter Length", 0.5, 0.0, 2.0, false)
                         .withGroup("Motion Blur")
                         .withTooltip("Shutter open duration in frames (Arnold-style, centered on "
                                      "the current frame). Default 0.5 ≈ 180° shutter / ~1/48 s at "
                                      "24 fps (close to 1/50). Open=-Length/2, Close=+Length/2."));
        addParameter(Parameter::makeBool("frustumcull", "Frustum Cull", true)
                         .withGroup("Subdivision")
                         .withTooltip("Meshes outside the dicing-camera frustum (plus padding) "
                                      "skip subdivision and only displace the cage. "
                                      "Close-ups on large faces still count as inside "
                                      "(screen-covering triangles / camera rays)."));
        addParameter(Parameter::makeFloat("frustumpadding", "Frustum Padding (%)", 10.0, 0.0, 100.0, false)
                         .withGroup("Subdivision")
                         .withTooltip("Screen-space margin as a percent of resolution width/height."));
        addParameter(Parameter::makeBool("screenadaptive", "Screen Adaptive", false)
                         .withGroup("Subdivision")
                         .withTooltip(
                             "Karma / Mantra / RenderMan-style raster dicing: split until projected "
                             "edge length ≈ 1/DicingQuality pixels (Quality 1 ≈ 1 micropolygon per "
                             "pixel). Only camera-visible faces (plus Frustum Padding). "
                             "Per-mesh Subdiv Iterations are ignored — density comes from Quality. "
                             "Re-tessellates on Start; animated Alembic/USD also re-dice on frame "
                             "change while Start is armed."));
        addParameter(Parameter::makeBool("enabledisplacement", "Enable Displacement", true)
                         .withGroup("Subdivision")
                         .withTooltip(
                             "Master switch for geometric displacement and densify/dicing. "
                             "Off: render authored cages with no subdiv and no displace "
                             "(fast Play / lookdev). On: normal tessellation + displace."));
        addParameter(Parameter::makeInt("dicingpolylimitm", "Dicing Poly Limit (M)", 10, 1, 200)
                         .withGroup("Subdivision")
                         .withTooltip("Safety fuse: stop densify / Screen Adaptive once the mesh "
                                      "reaches this many million triangles (default 10)."));
        addParameter(Parameter::makeMenu("dicingcamera", "Dicing Camera",
                                         {"Render Camera", "Custom"}, 0)
                         .withGroup("Subdivision")
                         .withTooltip("Camera used for frustum cull and screen-space dicing. "
                                      "Custom locks density to another camera prim."));
        addParameter(Parameter::makeString("dicingcamerapath", "Dicing Camera Path", "")
                         .withGroup("Subdivision")
                         .withTooltip("Stage prim path of the custom dicing camera "
                                      "(e.g. /cameras/dice). Empty falls back to the render camera."));
        addParameter(Parameter::makeMenu("workingspace", "Working Space",
                                         {"sRGB Linear", "ACEScg"}, 1)
                         .withGroup("Film")
                         .withTooltip("Render working colour space.\n"
                                      "ACEScg (default): Arnold-style ACES workflow "
                                      "(textures convert to ACEScg).\n"
                                      "sRGB Linear: legacy linear RGB working space.\n"
                                      "Viewport View (next to World) controls monitor display."));
        // Marks scenes authored after ACEScg became the default working space.
        addParameter(Parameter::makeBool("_working_space_aces_v1", "", true));
        addParameter(Parameter::makeBool("ociousenv", "Use OCIO from Environment", true)
                         .withGroup("Film")
                         .withTooltip("On: read config from the OCIO environment variable "
                                      "(instead of the path below).\n"
                                      "Off: use Film → OCIO Config path.\n"
                                      "LUTs stay next to your config — not shipped with Grendizer_Render."));
        addParameter(Parameter::makeFile("ocioconfig", "OCIO Config", "",
                                         "OCIO Config (*.ocio);;All Files (*)")
                         .withGroup("Film")
                         .withVisibleWhen("ociousenv==0")
                         .withTooltip("Path to config.ocio when Use OCIO from Environment is off.\n"
                                      "Example: C:/OCIO/aces_1.0.3/config.ocio"));
        addParameter(Parameter::makeBool("envvisible", "Environment Visible To Camera", true).withGroup("Film"));

        addParameter(Parameter::makeMenu("samplingdebug", "Sampling Debug",
                                         {"Off", "Pixel Jitter XY", "Path RNG u0", "Bucket ID",
                                          "Pixel Hash"},
                                         0)
                         .withGroup("Diagnostic")
                         .withTooltip("Replace beauty with a sampling/seed visualisation "
                                      "(no light transport).\n"
                                      "Pixel Jitter XY: R=jx G=jy from Pixel Sampler — Blue Noise "
                                      "shows a 64px period; Sobol/Xorshift/GenPnt2D should not.\n"
                                      "Path RNG u0: first Owen-Sobol / PCG float — look for "
                                      "faint seams from correlated seeds.\n"
                                      "Bucket ID: color by Bucket Size tiles (threading only).\n"
                                      "Pixel Hash: RGB from the per-pixel seed hash.\n"
                                      "Tip: set View to Raw for a clearer diagnostic."));
        addParameter(Parameter::makeBool("filmfalsecolor", "Spectral False Color", false)
                         .withGroup("Diagnostic")
                         .withVisibleWhen("integrator==4||integrator==5")
                         .withTooltip("Spectral integrators: visualise one spectral bin as "
                                      "false-color instead of beauty RGB (debug)."));
        addParameter(Parameter::makeInt("filmfalsecolorbin", "False Color Bin", 0, 0, 31)
                         .withGroup("Diagnostic")
                         .withVisibleWhen("integrator==4&&filmfalsecolor==1||integrator==5&&filmfalsecolor==1")
                         .withTooltip("Which spectral bin to show when Spectral False Color is on."));
    }

    void cook(CookContext&, const std::vector<StagePtr>&, Stage& stage) override {
        RenderSettingsData settings;
        settings.resolutionX = intValue("resx", 960);
        settings.resolutionY = intValue("resy", 540);
        settings.samplesPerPixel = intValue("samples", 128);
        settings.backend = intValue("backend", 0) == 1 ? kBackendGpuOptix : kBackendCpuEmbree;
        settings.integrator = std::clamp(intValue("integrator", 0), 0, 6);
        settings.maxDepth = std::clamp(intValue("maxdepth", 8), 1, 4096);
        settings.rrStartDepth = std::clamp(intValue("rrdepth", 3), 1, 4096);
        settings.lightSamples = std::max(1, intValue("lightsamples", 2));
        settings.clampDirect = float(floatValue("clampdirect", 10.0));
        settings.clampIndirect = float(floatValue("clamp", 10.0));
        settings.seed = intValue("seed", 0);
        settings.threads = intValue("threads", 0);
        settings.tileSize = std::clamp(intValue("tilesize", 32), 0, 256);
        // Pixel Sampler. Before Manual-Test, indices 4/5 were R2 salt/linear → GenPnt2D.
        {
            int ps = intValue("pixelsampler", 0);
            if (!boolValue("_pixel_sampler_manual_v1", false) && ps >= 4) ps = 3;
            settings.pixelSampler = std::clamp(ps, 0, 4);
        }
        settings.manualTestMult = float(floatValue("manualtestmult", 0.0));
        // Sampling Type v2: Buckets=0 / Progressive=1.
        // Old menu was Legacy=0 / FilmTile=1 / Progressive=2.
        if (!boolValue("_sampling_type_v2", false)) {
            const int old = intValue("samplingengine", 1);
            settings.samplingEngine =
                (old == 2) ? kSamplingEngineProgressive : kSamplingEngineBuckets;
        } else {
            settings.samplingEngine = std::clamp(intValue("samplingengine", 0), 0, 1);
        }
        settings.aoDistance = float(floatValue("aodistance", 1.0));
        settings.wireframeThickness =
            std::clamp(float(floatValue("wireframethickness", 1.0)), 0.25f, 8.0f);
        settings.pathGuiding = boolValue("pathguiding", false) ? 1 : 0;
        settings.caustics = boolValue("caustics", true) ? 1 : 0;
        settings.causticsEngine = std::clamp(intValue("causticsengine", 1), 0, 2);
        settings.causticClamp = float(floatValue("causticclamp", 0.0));
        settings.dispersionMode = intValue("dispersionmode", 0);
        settings.dispersionMaxInterfaces = std::max(1, intValue("dispersionmaxiface", 2));
        settings.photonCount = std::max(1000, intValue("photoncount", 100000));
        settings.photonRadius = float(floatValue("photonradius", 0.08));
        settings.motionBlur = boolValue("motionblur", false) ? 1 : 0;
        settings.motionKeys = std::clamp(intValue("motionkeys", 2), 2, 8);
        settings.shutterLength = float(floatValue("shutterlength", 0.5));
        settings.frustumCull = boolValue("frustumcull", true) ? 1 : 0;
        settings.frustumPadding = float(floatValue("frustumpadding", 10.0));
        settings.screenAdaptive = boolValue("screenadaptive", false) ? 1 : 0;
        settings.enableDisplacement = boolValue("enabledisplacement", true) ? 1 : 0;
        settings.dicingPolyLimitM = std::clamp(intValue("dicingpolylimitm", 10), 1, 200);
        settings.dicingCameraMode =
            intValue("dicingcamera", 0) == 1 ? kDicingCameraCustom : kDicingCameraRender;
        settings.spectralSamples = std::clamp(intValue("spectralsamples", 4), 2, 16);
        settings.spectralBins = std::clamp(intValue("spectralbins", 16), 8, 32);
        settings.spectralExr = boolValue("spectralexr", false) ? 1 : 0;
        settings.spectralColorSpace = std::clamp(intValue("spectralcolorspace", 0), 0, 3);
        settings.spectralWavelengthSampling = std::clamp(intValue("spectralwavesamp", 0), 0, 1);
        settings.workingSpace = std::clamp(intValue("workingspace", 1), 0, 1);
        {
            // Menu indices 0/1/2 → bit depths 8/16/32.
            const int bitIdx = std::clamp(intValue("bitdepth", 1), 0, 2);
            settings.bitDepth = bitIdx == 0 ? 8 : (bitIdx == 2 ? 32 : 16);
        }
        settings.pixelFilter = std::clamp(intValue("pixelfilter", 0), 0, 3);
        settings.filterRadius = float(floatValue("filterradius", 0.5));
        settings.envVisibleCamera = boolValue("envvisible", true) ? 1 : 0;
        settings.filmFalseColor = boolValue("filmfalsecolor", false) ? 1 : 0;
        settings.filmFalseColorBin = std::clamp(intValue("filmfalsecolorbin", 0), 0, 31);
        settings.samplingDebug = std::clamp(intValue("samplingdebug", 0), 0, 4);
        settings.enableTxCache = boolValue("enabletxcache", true) ? 1 : 0;
        settings.ocioUseEnv = boolValue("ociousenv", true) ? 1 : 0;
        {
            const std::string dir = stringValue("txcachedir", "tx_cache").toStdString();
            const size_t maxLen = sizeof(settings.txCacheDir) - 1;
            std::strncpy(settings.txCacheDir, dir.c_str(), maxLen);
            settings.txCacheDir[maxLen] = '\0';
        }
        {
            const std::string cfg = stringValue("ocioconfig", "").toStdString();
            const size_t maxLen = sizeof(settings.ocioConfigPath) - 1;
            std::strncpy(settings.ocioConfigPath, cfg.c_str(), maxLen);
            settings.ocioConfigPath[maxLen] = '\0';
        }
        stage.settings = settings;
        stage.settingsAuthored = true;
        stage.dicingCameraPath = stringValue("dicingcamerapath");
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

        info.typeName = "physicalskylight";
        info.label = "Physical Sky";
        info.description = "Hosek–Wilkie sky with solar disc on the dome";
        info.factory = [](const QString& name) -> NodePtr {
            return std::make_unique<PhysicalSkyLightNode>(name);
        };
        registry.registerType(info);
    }

    registry.registerType(
        makeType<CameraNode>("camera", "Camera", "Camera", "Render camera with lens controls", "#3a76b2"));
    registry.registerType(makeType<RenderSettingsNode>("rendersettings", "Render Settings", "Render",
                                                       "Resolution, sampling and backend selection", "#8a4550"));

    registerVdbNodes(registry);
}

}  // namespace sol
