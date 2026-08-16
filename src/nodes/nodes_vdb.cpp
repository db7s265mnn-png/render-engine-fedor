// Houdini-like VDB SOP nodes: VDB from Polygons, SDF to Polygons (VDB / DCSDD),
// and VDB File.
#include "nodes/node_registry.h"

#include <QDir>
#include <QFileInfo>
#include <QHash>

#include <cmath>

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
        addParameter(Parameter::makeBool("autovoxelsize", "Auto Voxel Size", true)
                         .withTooltip("Derive voxel size from mesh bounds (~1/128 of diagonal). "
                                      "Turn off to use Voxel Size below."));
        addParameter(Parameter::makeFloat("voxelsize", "Voxel Size", 0.05, 0.0001, 10.0, false)
                         .withTooltip("World-space voxel size (used when Auto Voxel Size is off)"));
        addParameter(Parameter::makeFloat("exteriorband", "Exterior Band", 3.0, 1.0, 64.0, false)
                         .withTooltip("Narrow-band width outside the surface (voxels)"));
        addParameter(Parameter::makeFloat("interiorband", "Interior Band", 3.0, 1.0, 64.0, false)
                         .withTooltip("Narrow-band width inside the surface (voxels)"));
        addParameter(Parameter::makeFloat("filldensity", "Density", 1.0, 0.0, 100.0, false)
                         .withTooltip("Runtime density multiplier for Fog (MediumData::density).\n"
                                      "Does NOT rebuild the VDB — only scales sampling / shadows.\n"
                                      "Multiplies MaterialX standard_volume.density when assigned."));
        addParameter(Parameter::makeMenu("filter", "Sample Filter",
                                         {"Nearest", "Linear", "Quadratic"}, 1)
                         .withTooltip("Voxel reconstruction filter when sampling the VDB.\n"
                                      "Nearest = 1 tap (blocky); Linear = 8-tap trilinear;\n"
                                      "Quadratic = 27-tap triquadratic (smoothest, ~3–4× slower per\n"
                                      "sample — costly under fog delta tracking / SDF gradients).\n"
                                      "Changing filter does not rebuild the grid."));
        addParameter(Parameter::makeString("primname", "Prim Name", "vdb")
                         .withTooltip("Leaf name for the output VDB prim"));
    }

    bool copiesFirstInput() const override { return false; }

    void cook(CookContext& context, const std::vector<StagePtr>& inputs, Stage& stage) override {
        if (!VolumeGrid::openVdbAvailable()) {
            context.reportError(this,
                                "OpenVDB is not linked in this build — VDB from Polygons cannot create "
                                "volumes. Use a Windows build that ships openvdb.dll (CI after OpenVDB "
                                "enablement), or a Linux build with libopenvdb.");
            return;
        }
        if (inputs.empty() || !inputs[0]) {
            context.reportWarning(this, "no input stage");
            return;
        }
        const Stage& in = *inputs[0];
        const QString pattern = stringValue("pattern");
        const bool autoVoxel = boolValue("autovoxelsize", true);
        const VolumeGridKind kind = intValue("mode") == 0 ? VolumeGridKind::Sdf : VolumeGridKind::Fog;
        const float authoredVoxel = float(floatValue("voxelsize"));
        const float exteriorBand = float(floatValue("exteriorband"));
        const float interiorBand = float(floatValue("interiorband"));
        const float densityScale = float(floatValue("filldensity", 1.0));
        const int filterIdx = intValue("filter", 1);
        const VolumeSampleFilter filter = filterIdx <= 0   ? VolumeSampleFilter::Nearest
                                          : filterIdx >= 2 ? VolumeSampleFilter::Quadratic
                                                           : VolumeSampleFilter::Linear;

        // Drop cache entries whose source mesh is no longer in the input.
        {
            std::vector<QString> keep;
            for (const StagePrim& prim : in.prims) {
                if (prim.active && prim.type == PrimType::Mesh && prim.mesh &&
                    matchesPattern(pattern, prim.path))
                    keep.push_back(prim.path);
            }
            for (auto it = gridCache_.begin(); it != gridCache_.end();) {
                bool found = false;
                for (const QString& p : keep) {
                    if (p == it.key()) {
                        found = true;
                        break;
                    }
                }
                if (!found) it = gridCache_.erase(it);
                else ++it;
            }
        }

        int created = 0;
        for (const StagePrim& prim : in.prims) {
            if (!prim.active || prim.type != PrimType::Mesh || !prim.mesh) continue;
            if (!matchesPattern(pattern, prim.path)) continue;

            float voxelSize = authoredVoxel;
            if (autoVoxel) {
                Bounds3 b;
                for (const Vec3& p : prim.mesh->positions) b.extend(transformPoint(prim.xform, p));
                const float diag = length(b.hi - b.lo);
                voxelSize = srMax(1e-4f, diag / 128.0f);
            }

            // Rebuild key excludes Density + Sample Filter (runtime / cheap).
            const GridCacheKey key{kind, voxelSize, exteriorBand, interiorBand,
                                   uintptr_t(prim.mesh.get()), prim.mesh->positions.size(),
                                   prim.mesh->indices.size(), prim.mesh->faceVertexIndices.size()};

            VolumeGridPtr grid;
            auto cacheIt = gridCache_.find(prim.path);
            if (cacheIt != gridCache_.end() && cacheIt.value().key == key && cacheIt.value().grid &&
                cacheIt.value().grid->valid()) {
                grid = cacheIt.value().grid;
            } else {
                VolumeFromPolygonsSettings local;
                local.kind = kind;
                local.voxelSize = voxelSize;
                local.exteriorBand = exteriorBand;
                local.interiorBand = interiorBand;
                local.fillDensity = 1.0f;  // occupancy only; density is a medium multiplier
                local.filter = filter;
                std::string err;
                grid = VolumeGrid::fromPolygons(*prim.mesh, prim.xform, local, &err);
                if (!grid || !grid->valid()) {
                    context.reportError(this, QString("failed to convert %1: %2")
                                                  .arg(prim.path)
                                                  .arg(QString::fromStdString(
                                                      err.empty() ? "unknown error" : err)));
                    continue;
                }
                CachedGrid entry;
                entry.key = key;
                entry.grid = grid;
                gridCache_.insert(prim.path, entry);
            }
            grid->setSampleFilter(filter);

            StagePrim out;
            out.type = PrimType::Volume;
            out.sourceNode = this->name();
            out.volume = grid;
            out.xform = Mat4::identity();  // grid already in world space
            out.material = prim.material;
            out.materialAssigned = prim.materialAssigned;
            // Runtime density scale — toScene multiplies into MediumData::density.
            out.mediumAssigned = true;
            out.medium.type = (kind == VolumeGridKind::Fog) ? 2 : 3;
            out.medium.density = densityScale;
            out.path = primPathFor(*this, "volume", stringValue("primname"));
            if (created > 0) out.path = out.path + QString::number(created);
            stage.addPrim(std::move(out));
            ++created;
        }
        if (created == 0) context.reportWarning(this, "no mesh prims matched pattern");
    }

private:
    struct GridCacheKey {
        VolumeGridKind kind = VolumeGridKind::Sdf;
        float voxelSize = 0.0f;
        float exteriorBand = 0.0f;
        float interiorBand = 0.0f;
        uintptr_t meshPtr = 0;
        size_t positionCount = 0;
        size_t indexCount = 0;
        size_t faceIndexCount = 0;

        bool operator==(const GridCacheKey& o) const {
            return kind == o.kind && meshPtr == o.meshPtr && positionCount == o.positionCount &&
                   indexCount == o.indexCount && faceIndexCount == o.faceIndexCount &&
                   std::fabs(voxelSize - o.voxelSize) < 1e-8f &&
                   std::fabs(exteriorBand - o.exteriorBand) < 1e-5f &&
                   std::fabs(interiorBand - o.interiorBand) < 1e-5f;
        }
    };
    struct CachedGrid {
        GridCacheKey key;
        VolumeGridPtr grid;
    };
    QHash<QString, CachedGrid> gridCache_;
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
        addParameter(Parameter::makeMenu("filter", "Sample Filter",
                                         {"Nearest", "Linear", "Quadratic"}, 1)
                         .withTooltip("Voxel reconstruction filter when sampling the VDB.\n"
                                      "Nearest = blocky; Linear = trilinear; Quadratic = smoothest."));
        addParameter(Parameter::makeString("primname", "Prim Name", "vdb"));
        addTransformParameters(*this);
    }

    bool copiesFirstInput() const override { return true; }

    void cook(CookContext& context, const std::vector<StagePtr>&, Stage& stage) override {
        if (!VolumeGrid::openVdbAvailable()) {
            context.reportError(this, "OpenVDB is not linked in this build — cannot load .vdb files");
            return;
        }
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
        const int filterIdx = intValue("filter", 1);
        grid->setSampleFilter(filterIdx <= 0   ? VolumeSampleFilter::Nearest
                              : filterIdx >= 2 ? VolumeSampleFilter::Quadratic
                                               : VolumeSampleFilter::Linear);
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
