// Houdini-like VDB SOP nodes: VDB from Polygons, SDF to Polygons (VDB / DCSDD),
// and VDB File.
#include "nodes/node_registry.h"

#include <QDir>
#include <QFileInfo>

#include "core/log.h"
#include "io/alembic_loader.h"
#include "nodes/node.h"
#include "scene/dcsdd_contouring.h"
#include "scene/volume_grid.h"

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
// VDB from Polygons  (Houdini: VDB from Polygons SOP)
// ---------------------------------------------------------------------------
class VdbFromPolygonsNode : public Node {
public:
    explicit VdbFromPolygonsNode(const QString& name) : Node("vdbfrompolygons", name) {
        setInputLabels({"Input"});
        addParameter(Parameter::makeString("pattern", "Prim Pattern", "*")
                         .withTooltip("Mesh prims to convert (glob). Output is VDB only."));
        addParameter(Parameter::makeMenu("mode", "Mode", {"SDF", "Fog Volume"}, 0)
                         .withTooltip("SDF = level set; Fog Volume = density fill inside closed mesh"));
        addParameter(Parameter::makeFloat("voxelsize", "Voxel Size", 0.05, 0.0001, 10.0, false)
                         .withTooltip("World-space voxel size"));
        addParameter(Parameter::makeFloat("exteriorband", "Exterior Band", 3.0, 1.0, 64.0, false)
                         .withTooltip("Narrow-band width outside the surface (voxels)"));
        addParameter(Parameter::makeFloat("interiorband", "Interior Band", 3.0, 1.0, 64.0, false)
                         .withTooltip("Narrow-band width inside the surface (voxels)"));
        addParameter(Parameter::makeFloat("filldensity", "Fill Density", 1.0, 0.0, 100.0, false)
                         .withTooltip("Fog Volume: density value inside the mesh"));
        addParameter(Parameter::makeString("primname", "Prim Name", "vdb")
                         .withTooltip("Leaf name for the output VDB prim"));
    }

    bool copiesFirstInput() const override { return false; }

    void cook(CookContext& context, const std::vector<StagePtr>& inputs, Stage& stage) override {
        if (inputs.empty() || !inputs[0]) {
            context.reportWarning(this, "no input stage");
            return;
        }
        const Stage& in = *inputs[0];
        const QString pattern = stringValue("pattern");
        VolumeFromPolygonsSettings settings;
        settings.kind = intValue("mode") == 0 ? VolumeGridKind::Sdf : VolumeGridKind::Fog;
        settings.voxelSize = float(floatValue("voxelsize"));
        settings.exteriorBand = float(floatValue("exteriorband"));
        settings.interiorBand = float(floatValue("interiorband"));
        settings.fillDensity = float(floatValue("filldensity"));

        int created = 0;
        for (const StagePrim& prim : in.prims) {
            if (!prim.active || prim.type != PrimType::Mesh || !prim.mesh) continue;
            if (!matchesPattern(pattern, prim.path)) continue;
            std::string err;
            VolumeGridPtr grid = VolumeGrid::fromPolygons(*prim.mesh, prim.xform, settings, &err);
            if (!grid || !grid->valid()) {
                context.reportWarning(this, QString("failed to convert %1: %2")
                                                .arg(prim.path)
                                                .arg(QString::fromStdString(err)));
                continue;
            }
            StagePrim out;
            out.type = PrimType::Volume;
            out.sourceNode = this->name();
            out.volume = grid;
            out.xform = Mat4::identity();  // grid already in world space
            out.material = prim.material;
            out.materialAssigned = prim.materialAssigned;
            out.path = primPathFor(*this, "volume", stringValue("primname"));
            if (created > 0) out.path = out.path + QString::number(created);
            stage.addPrim(std::move(out));
            ++created;
        }
        if (created == 0) context.reportWarning(this, "no mesh prims matched pattern");
    }
};

// ---------------------------------------------------------------------------
// VDB File
// ---------------------------------------------------------------------------
class VdbFileNode : public Node {
public:
    explicit VdbFileNode(const QString& name) : Node("vdbfile", name) {
        setInputLabels({"Input"});
        addParameter(Parameter::makeFile("file", "VDB File", "", "OpenVDB (*.vdb)")
                         .withTooltip("Load a .vdb from disk into a Volume prim"));
        addParameter(Parameter::makeString("primname", "Prim Name", "vdb"));
        addTransformParameters(*this);
    }

    bool copiesFirstInput() const override { return true; }

    void cook(CookContext& context, const std::vector<StagePtr>&, Stage& stage) override {
        const QString file = resolvePath(context, stringValue("file"));
        if (file.isEmpty()) {
            context.reportWarning(this, "no VDB file set");
            return;
        }
        if (!QFileInfo::exists(file)) {
            context.reportError(this, "file not found: " + file);
            return;
        }
        std::string err;
        VolumeGridPtr grid = VolumeGrid::loadVdb(file.toStdString(), &err);
        if (!grid) {
            context.reportError(this, QString::fromStdString(err.empty() ? "failed to load VDB" : err));
            return;
        }
        StagePrim prim;
        prim.type = PrimType::Volume;
        prim.sourceNode = this->name();
        prim.volume = grid;
        prim.path = primPathFor(*this, "volume", stringValue("primname"));
        prim.xform = transformFromParameters(*this);
        stage.addPrim(std::move(prim));
    }
};

// ---------------------------------------------------------------------------
// SDF to Polygons — OpenVDB mesher
// ---------------------------------------------------------------------------
class SdfToPolygonsVdbNode : public Node {
public:
    explicit SdfToPolygonsVdbNode(const QString& name) : Node("sdftopolygons_vdb", name) {
        setInputLabels({"Input"});
        addParameter(Parameter::makeString("pattern", "Prim Pattern", "*"));
        addParameter(Parameter::makeFloat("isovalue", "Isovalue", 0.0, -10.0, 10.0, false));
        addParameter(Parameter::makeFloat("adaptivity", "Adaptivity", 0.0, 0.0, 1.0, false)
                         .withTooltip("OpenVDB volumeToMesh adaptivity (0 = densest)"));
        addParameter(Parameter::makeString("primname", "Prim Name", "mesh"));
    }

    bool copiesFirstInput() const override { return false; }

    void cook(CookContext& context, const std::vector<StagePtr>& inputs, Stage& stage) override {
        if (inputs.empty() || !inputs[0]) return;
        const float iso = float(floatValue("isovalue"));
        const float adapt = float(floatValue("adaptivity"));
        int created = 0;
        for (const StagePrim& prim : inputs[0]->prims) {
            if (!prim.active || prim.type != PrimType::Volume || !prim.volume) continue;
            if (!matchesPattern(stringValue("pattern"), prim.path)) continue;
            MeshPtr mesh = prim.volume->toPolygonsOpenVDB(iso, adapt);
            if (!mesh || mesh->indices.empty()) {
                context.reportWarning(this, "OpenVDB mesher produced empty mesh for " + prim.path);
                continue;
            }
            StagePrim out;
            out.type = PrimType::Mesh;
            out.sourceNode = this->name();
            out.mesh = mesh;
            out.xform = prim.xform;
            out.material = prim.material;
            out.materialAssigned = prim.materialAssigned;
            out.path = primPathFor(*this, "geo", stringValue("primname"));
            if (created > 0) out.path = out.path + QString::number(created);
            stage.addPrim(std::move(out));
            ++created;
        }
        if (created == 0) context.reportWarning(this, "no volume prims matched");
    }
};

// ---------------------------------------------------------------------------
// SDF to Polygons — Dual Contouring of Signed Distance Data
// ---------------------------------------------------------------------------
class SdfToPolygonsDcsddNode : public Node {
public:
    explicit SdfToPolygonsDcsddNode(const QString& name) : Node("sdftopolygons_dcsdd", name) {
        setInputLabels({"Input"});
        addParameter(Parameter::makeString("pattern", "Prim Pattern", "*"));
        addParameter(Parameter::makeFloat("voxelsize", "Sample Voxel Size", 0.0, 0.0, 10.0, false)
                         .withTooltip("0 = use the VDB voxel size; otherwise resample the SDF"));
        addParameter(Parameter::makeInt("outeriters", "Outer Iterations", 40, 1, 200));
        addParameter(Parameter::makeInt("inneriters", "Inner Iterations", 20, 1, 200));
        addParameter(Parameter::makeString("primname", "Prim Name", "mesh"));
    }

    bool copiesFirstInput() const override { return false; }

    void cook(CookContext& context, const std::vector<StagePtr>& inputs, Stage& stage) override {
        if (inputs.empty() || !inputs[0]) return;
        DcsddOptions opt;
        opt.outerIters = intValue("outeriters");
        opt.innerIters = intValue("inneriters");
        const float sampleVs = float(floatValue("voxelsize"));
        int created = 0;
        for (const StagePrim& prim : inputs[0]->prims) {
            if (!prim.active || prim.type != PrimType::Volume || !prim.volume) continue;
            if (!matchesPattern(stringValue("pattern"), prim.path)) continue;
            std::string err;
            MeshPtr mesh = dcsddContourVolume(*prim.volume, sampleVs, opt, &err);
            if (!mesh) {
                context.reportWarning(this, QString("DCSDD failed for %1: %2")
                                                .arg(prim.path)
                                                .arg(QString::fromStdString(err)));
                continue;
            }
            StagePrim out;
            out.type = PrimType::Mesh;
            out.sourceNode = this->name();
            out.mesh = mesh;
            out.xform = prim.xform;
            out.material = prim.material;
            out.materialAssigned = prim.materialAssigned;
            out.path = primPathFor(*this, "geo", stringValue("primname"));
            if (created > 0) out.path = out.path + QString::number(created);
            stage.addPrim(std::move(out));
            ++created;
        }
        if (created == 0) context.reportWarning(this, "no volume prims matched");
    }
};

}  // namespace

void registerVdbNodes(NodeRegistry& registry) {
    auto make = [&registry](auto factory, const char* type, const char* label, const char* desc) {
        NodeTypeInfo info;
        info.typeName = type;
        info.label = label;
        info.category = "Volume";
        info.description = desc;
        info.colorHex = "#2f6f6a";
        info.factory = factory;
        registry.registerType(info);
    };
    make([](const QString& n) -> NodePtr { return std::make_unique<VdbFromPolygonsNode>(n); },
         "vdbfrompolygons", "VDB from Polygons",
         "Convert polygon meshes to OpenVDB SDF or Fog Volume (output VDB only)");
    make([](const QString& n) -> NodePtr { return std::make_unique<VdbFileNode>(n); }, "vdbfile", "VDB File",
         "Load an OpenVDB (.vdb) file from disk");
    make([](const QString& n) -> NodePtr { return std::make_unique<SdfToPolygonsVdbNode>(n); },
         "sdftopolygons_vdb", "SDF to Polygons (VDB)",
         "Extract a polygon mesh from an SDF VDB via OpenVDB volumeToMesh");
    make([](const QString& n) -> NodePtr { return std::make_unique<SdfToPolygonsDcsddNode>(n); },
         "sdftopolygons_dcsdd", "SDF to Polygons (DCSDD)",
         "Dual Contouring of Signed Distance Data (Carrera et al. 2026) — sharp features");
}

}  // namespace sol
