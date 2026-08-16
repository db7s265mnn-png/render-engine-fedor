#include "scene/volume_grid.h"

#include "core/log.h"
#include "scene/triangulate.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPENVDB
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/io/File.h>
#endif

#include <cmath>
#include <limits>

namespace sol {

#if SOLSTICE_HAVE_OPENVDB

namespace {

void ensureOpenVdb() {
    static bool once = false;
    if (!once) {
        openvdb::initialize();
        once = true;
    }
}

openvdb::math::Transform::Ptr makeXform(float voxelSize) {
    return openvdb::math::Transform::createLinearTransform(double(srMax(1e-6f, voxelSize)));
}

}  // namespace

struct VolumeGrid::Impl {
    openvdb::FloatGrid::Ptr grid;
};

bool VolumeGrid::openVdbAvailable() { return true; }

VolumeGrid::VolumeGrid() : impl_(std::make_unique<Impl>()) {}
VolumeGrid::~VolumeGrid() = default;
VolumeGrid::VolumeGrid(VolumeGrid&&) noexcept = default;
VolumeGrid& VolumeGrid::operator=(VolumeGrid&&) noexcept = default;

bool VolumeGrid::valid() const { return impl_ && impl_->grid && !impl_->grid->empty(); }

void* VolumeGrid::nativeGrid() const { return impl_ ? static_cast<void*>(impl_->grid.get()) : nullptr; }

float VolumeGrid::sampleWorld(const Vec3& p) const {
    if (!valid()) return (kind_ == VolumeGridKind::Sdf) ? 1e6f : 0.0f;
    const openvdb::Vec3d wp(p.x, p.y, p.z);
    switch (sampleFilter_) {
        case VolumeSampleFilter::Nearest: {
            openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::PointSampler> sampler(
                *impl_->grid);
            return float(sampler.wsSample(wp));
        }
        case VolumeSampleFilter::Quadratic: {
            openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::QuadraticSampler> sampler(
                *impl_->grid);
            return float(sampler.wsSample(wp));
        }
        case VolumeSampleFilter::Linear:
        default: {
            openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(
                *impl_->grid);
            return float(sampler.wsSample(wp));
        }
    }
}

Vec3 VolumeGrid::gradientWorld(const Vec3& p) const {
    const float eps = srMax(1e-4f, voxelSize_ * 0.5f);
    const float dx = sampleWorld(p + Vec3(eps, 0, 0)) - sampleWorld(p - Vec3(eps, 0, 0));
    const float dy = sampleWorld(p + Vec3(0, eps, 0)) - sampleWorld(p - Vec3(0, eps, 0));
    const float dz = sampleWorld(p + Vec3(0, 0, eps)) - sampleWorld(p - Vec3(0, 0, eps));
    const Vec3 g(dx, dy, dz);
    const float len = length(g);
    return len > 1e-12f ? g / len : Vec3(0, 1, 0);
}

bool VolumeGrid::saveVdb(const std::string& path) const {
    if (!valid()) return false;
    ensureOpenVdb();
    try {
        openvdb::io::File file(path);
        openvdb::GridPtrVec grids;
        grids.push_back(impl_->grid);
        file.write(grids);
        file.close();
        return true;
    } catch (const std::exception& e) {
        logWarning(std::string("OpenVDB save failed: ") + e.what());
        return false;
    }
}

std::shared_ptr<VolumeGrid> VolumeGrid::loadVdb(const std::string& path, std::string* error) {
    ensureOpenVdb();
    try {
        openvdb::io::File file(path);
        file.open();
        openvdb::GridBase::Ptr base = file.readGrid(file.beginName().gridName());
        file.close();
        openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
        if (!grid) {
            if (error) *error = "VDB does not contain a FloatGrid";
            return nullptr;
        }
        auto out = std::make_shared<VolumeGrid>();
        out->impl_->grid = grid;
        const std::string className = grid->getGridClass() == openvdb::GRID_LEVEL_SET ? "levelset" : "fog";
        out->kind_ = (grid->getGridClass() == openvdb::GRID_LEVEL_SET) ? VolumeGridKind::Sdf : VolumeGridKind::Fog;
        out->name_ = grid->getName().empty() ? path : grid->getName();
        out->voxelSize_ = float(grid->voxelSize()[0]);
        const openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
        const openvdb::Vec3d w0 = grid->indexToWorld(bbox.min().asVec3d());
        const openvdb::Vec3d w1 = grid->indexToWorld(bbox.max().asVec3d() + openvdb::Vec3d(1));
        out->bounds_.lo = Vec3(float(std::min(w0.x(), w1.x())), float(std::min(w0.y(), w1.y())),
                               float(std::min(w0.z(), w1.z())));
        out->bounds_.hi = Vec3(float(std::max(w0.x(), w1.x())), float(std::max(w0.y(), w1.y())),
                               float(std::max(w0.z(), w1.z())));
        float maj = 0.0f;
        for (auto it = grid->cbeginValueOn(); it; ++it) maj = srMax(maj, fabsf(float(*it)));
        out->majorant_ = srMax(maj, 1e-4f);
        (void)className;
        return out;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return nullptr;
    }
}

std::shared_ptr<VolumeGrid> VolumeGrid::fromPolygons(const Mesh& mesh, const Mat4& xform,
                                                     const VolumeFromPolygonsSettings& settings,
                                                     std::string* error) {
    ensureOpenVdb();
    Mesh local = mesh;
    local.ensureRenderTriangles();
    if (local.indices.size() < 3 || local.positions.empty()) {
        if (error) *error = "Mesh is empty";
        return nullptr;
    }

    // meshToLevelSet expects world-space points.
    std::vector<openvdb::Vec3s> points(local.positions.size());
    for (size_t i = 0; i < local.positions.size(); ++i) {
        const Vec3 p = transformPoint(xform, local.positions[i]);
        points[i] = openvdb::Vec3s(p.x, p.y, p.z);
    }
    std::vector<openvdb::Vec3I> tris;
    tris.reserve(local.indices.size() / 3);
    for (size_t t = 0; t + 2 < local.indices.size(); t += 3) {
        tris.emplace_back(local.indices[t], local.indices[t + 1], local.indices[t + 2]);
    }

    openvdb::math::Transform::Ptr xformVdb = makeXform(settings.voxelSize);
    const float halfWidth =
        srMax(1.0f, srMax(settings.exteriorBand, settings.interiorBand));
    openvdb::FloatGrid::Ptr grid;
    try {
        openvdb::FloatGrid::Ptr sdf =
            openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(*xformVdb, points, tris, halfWidth);
        if (settings.kind == VolumeGridKind::Sdf) {
            grid = sdf;
            grid->setGridClass(openvdb::GRID_LEVEL_SET);
            grid->setName("sdf");
        } else {
            // Fog = filled density, not a hollow narrow-band shell.
            // meshToLevelSet signed-flood-fills inactive interior (negative);
            // sdfToFogVolume: interior → dens 1, interior band ramps 0→1, exterior 0.
            grid = sdf;
            const float cutoffWorld =
                settings.voxelSize * srMax(1.0f, settings.interiorBand);
            openvdb::tools::sdfToFogVolume(*grid, cutoffWorld);
            grid->setGridClass(openvdb::GRID_FOG_VOLUME);
            grid->setName("density");
            // Mild AABB-edge fade only (2 voxels) — softens hard container cuts
            // without carving planar holes into a filled interior.
            const openvdb::CoordBBox densBox = grid->evalActiveVoxelBoundingBox();
            if (!densBox.empty()) {
                constexpr int kFeatherVox = 2;
                for (auto it = grid->beginValueOn(); it; ++it) {
                    const openvdb::Coord c = it.getCoord();
                    const int dx = std::min(c.x() - densBox.min().x(), densBox.max().x() - c.x());
                    const int dy = std::min(c.y() - densBox.min().y(), densBox.max().y() - c.y());
                    const int dz = std::min(c.z() - densBox.min().z(), densBox.max().z() - c.z());
                    const int distEdge = std::min(dx, std::min(dy, dz));
                    if (distEdge >= kFeatherVox) continue;
                    const float u = clampf(float(distEdge) / float(kFeatherVox), 0.0f, 1.0f);
                    const float fade = u * u * (3.0f - 2.0f * u);
                    const float v = float(*it) * fade;
                    if (v > 1e-8f) it.setValue(v);
                    else it.setValueOff();
                }
            }
            grid->pruneGrid();
        }
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return nullptr;
    }

    auto out = std::make_shared<VolumeGrid>();
    out->impl_->grid = grid;
    out->kind_ = settings.kind;
    out->sampleFilter_ = settings.filter;
    out->voxelSize_ = settings.voxelSize;
    out->name_ = (settings.kind == VolumeGridKind::Sdf) ? "sdf" : "density";
    const openvdb::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
    if (!bbox.empty()) {
        const openvdb::Vec3d w0 = grid->indexToWorld(bbox.min().asVec3d());
        const openvdb::Vec3d w1 = grid->indexToWorld(bbox.max().asVec3d() + openvdb::Vec3d(1));
        out->bounds_.lo = Vec3(float(std::min(w0.x(), w1.x())), float(std::min(w0.y(), w1.y())),
                               float(std::min(w0.z(), w1.z())));
        out->bounds_.hi = Vec3(float(std::max(w0.x(), w1.x())), float(std::max(w0.y(), w1.y())),
                               float(std::max(w0.z(), w1.z())));
    }
    float maj = 0.0f;
    for (auto it = grid->cbeginValueOn(); it; ++it) maj = srMax(maj, fabsf(float(*it)));
    // Occupancy is normalized to ~1; runtime density multiplies in the integrator.
    out->majorant_ = srMax(maj, 1.0f);
    return out;
}

MeshPtr VolumeGrid::toPolygonsOpenVDB(float isovalue, float adaptivity) const {
    if (!valid()) return nullptr;
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> triangles;
    std::vector<openvdb::Vec4I> quads;
    openvdb::tools::volumeToMesh(*impl_->grid, points, triangles, quads, double(isovalue), double(adaptivity));
    // Convert quads → two triangles.
    for (const openvdb::Vec4I& q : quads) {
        triangles.emplace_back(q[0], q[1], q[2]);
        triangles.emplace_back(q[0], q[2], q[3]);
    }
    if (points.empty() || triangles.empty()) return nullptr;
    auto mesh = std::make_shared<Mesh>();
    mesh->positions.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i)
        mesh->positions[i] = Vec3(points[i].x(), points[i].y(), points[i].z());
    mesh->faceVertexCounts.reserve(triangles.size());
    mesh->faceVertexIndices.reserve(triangles.size() * 3);
    mesh->indices.reserve(triangles.size() * 3);
    for (const openvdb::Vec3I& t : triangles) {
        mesh->faceVertexCounts.push_back(3);
        mesh->faceVertexIndices.push_back(t[0]);
        mesh->faceVertexIndices.push_back(t[1]);
        mesh->faceVertexIndices.push_back(t[2]);
        mesh->indices.push_back(t[0]);
        mesh->indices.push_back(t[1]);
        mesh->indices.push_back(t[2]);
    }
    mesh->validate();
    return mesh;
}

#else  // !SOLSTICE_HAVE_OPENVDB

struct VolumeGrid::Impl {};
VolumeGrid::VolumeGrid() : impl_(std::make_unique<Impl>()) {}
VolumeGrid::~VolumeGrid() = default;
VolumeGrid::VolumeGrid(VolumeGrid&&) noexcept = default;
VolumeGrid& VolumeGrid::operator=(VolumeGrid&&) noexcept = default;
bool VolumeGrid::openVdbAvailable() { return false; }
bool VolumeGrid::valid() const { return false; }
void* VolumeGrid::nativeGrid() const { return nullptr; }
float VolumeGrid::sampleWorld(const Vec3&) const { return kind_ == VolumeGridKind::Sdf ? 1e6f : 0.0f; }
Vec3 VolumeGrid::gradientWorld(const Vec3&) const { return Vec3(0, 1, 0); }
bool VolumeGrid::saveVdb(const std::string&) const { return false; }
std::shared_ptr<VolumeGrid> VolumeGrid::loadVdb(const std::string&, std::string* error) {
    if (error) *error = "OpenVDB support not built into this binary";
    return nullptr;
}
std::shared_ptr<VolumeGrid> VolumeGrid::fromPolygons(const Mesh&, const Mat4&, const VolumeFromPolygonsSettings&,
                                                     std::string* error) {
    if (error) *error = "OpenVDB support not built into this binary";
    return nullptr;
}
MeshPtr VolumeGrid::toPolygonsOpenVDB(float, float) const { return nullptr; }

#endif

}  // namespace sol
