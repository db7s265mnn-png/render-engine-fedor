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

#include <algorithm>
#include <cmath>
#include <vector>

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

float ddaGridExitT(Vec3 origin, Vec3 direction, float t, float tMax, Vec3 org, float cell, int nx,
                   int ny, int nz) {
    if (tMax <= t || cell <= 1e-12f || nx <= 0 || ny <= 0 || nz <= 0) return tMax;
    const float lookT = t + 1e-5f;
    const Vec3 p = origin + direction * lookT;
    const Vec3 hi(org.x + float(nx) * cell, org.y + float(ny) * cell, org.z + float(nz) * cell);
    if (p.x < org.x || p.y < org.y || p.z < org.z || p.x >= hi.x || p.y >= hi.y || p.z >= hi.z)
        return tMax;
    auto binClamp = [&](float w, float o, int dim) -> int {
        int i = int(std::floor(double((w - o) / cell)));
        if (i < 0) i = 0;
        if (i >= dim) i = dim - 1;
        return i;
    };
    const int ix = binClamp(p.x, org.x, nx);
    const int iy = binClamp(p.y, org.y, ny);
    const int iz = binClamp(p.z, org.z, nz);
    auto axisExit = [&](float o, float d, float orig, int i) {
        if (fabsf(d) < 1e-20f) return tMax;
        const float next = (d > 0.0f) ? (orig + float(i + 1) * cell) : (orig + float(i) * cell);
        return (next - o) / d;
    };
    float tExit = tMax;
    tExit = srMin(tExit, axisExit(origin.x, direction.x, org.x, ix));
    tExit = srMin(tExit, axisExit(origin.y, direction.y, org.y, iy));
    tExit = srMin(tExit, axisExit(origin.z, direction.z, org.z, iz));
    if (!(tExit > t)) tExit = t + 1e-4f;
    return srMin(tExit, tMax);
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

float VolumeGrid::sampleWorldTracking(const Vec3& p) const {
    if (!valid()) return 0.0f;
    if (sampleFilter_ == VolumeSampleFilter::Nearest) return sampleWorld(p);
    // Linear (8-tap) even when the authored filter is Quadratic — deep MS walks
    // would otherwise pay 27 taps per null collision.
    const openvdb::Vec3d wp(p.x, p.y, p.z);
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*impl_->grid);
    return float(sampler.wsSample(wp));
}

void VolumeGrid::rebuildMajorantGrid() {
    majNx_ = majNy_ = majNz_ = 0;
    majCell_ = 0.0f;
    majMin_.clear();
    majMax_.clear();
    brNx_ = brNy_ = brNz_ = 0;
    brOcc_.clear();
    if (kind_ != VolumeGridKind::Fog || !valid() || !bounds_.valid()) return;

    constexpr int kHaloVox = 2;
    const Vec3 ext = bounds_.hi - bounds_.lo;
    const float longAxis = srMax(ext.x, srMax(ext.y, ext.z));
    const int voxelsLong = srMax(1, int(std::lround(double(longAxis / srMax(voxelSize_, 1e-8f)))));
    int sv = 8;
    if (voxelsLong < 64) sv = 4;
    if (voxelsLong < 24) sv = 2;
    // OpenVDB leaves are 8³ — keep majorants that tight (PBRT: smaller Λ is faster).
    majCell_ = srMax(voxelSize_ * float(sv), srMax(voxelSize_, 1e-6f));
    majOrigin_ = bounds_.lo;
    majNx_ = srMax(1, int(std::ceil(double(ext.x / majCell_))));
    majNy_ = srMax(1, int(std::ceil(double(ext.y / majCell_))));
    majNz_ = srMax(1, int(std::ceil(double(ext.z / majCell_))));
    const int n = majNx_ * majNy_ * majNz_;
    majMin_.assign(size_t(n), 1.0e30f);
    majMax_.assign(size_t(n), 0.0f);
    std::vector<int> counts(size_t(n), 0);

    auto idx = [&](int x, int y, int z) -> int { return x + majNx_ * (y + majNy_ * z); };
    auto inRange = [&](int x, int y, int z) {
        return x >= 0 && y >= 0 && z >= 0 && x < majNx_ && y < majNy_ && z < majNz_;
    };
    auto bin = [&](float w, float o) -> int {
        return int(std::floor(double((w - o) / majCell_)));
    };

    const openvdb::FloatGrid& grid = *impl_->grid;
    const float haloMax = voxelSize_ * float(kHaloVox);
    const float haloMin = voxelSize_;  // Linear 8-tap support; 2-vox would poison interior μc
    for (auto it = grid.cbeginValueOn(); it; ++it) {
        const float v = fabsf(float(*it));
        const openvdb::Coord c = it.getCoord();
        const openvdb::Vec3d w = grid.indexToWorld(c.asVec3d() + openvdb::Vec3d(0.5));
        const int ix = bin(float(w.x()), majOrigin_.x);
        const int iy = bin(float(w.y()), majOrigin_.y);
        const int iz = bin(float(w.z()), majOrigin_.z);
        if (inRange(ix, iy, iz)) {
            const int i = idx(ix, iy, iz);
            majMin_[size_t(i)] = std::min(majMin_[size_t(i)], v);
            majMax_[size_t(i)] = std::max(majMax_[size_t(i)], v);
            counts[size_t(i)] += 1;
        }
        auto stampHalo = [&](float halo, bool bumpMin, bool bumpMax) {
            const float xLo = float(w.x()) - halo;
            const float xHi = float(w.x()) + halo;
            const float yLo = float(w.y()) - halo;
            const float yHi = float(w.y()) + halo;
            const float zLo = float(w.z()) - halo;
            const float zHi = float(w.z()) + halo;
            const int hx0 = bin(xLo, majOrigin_.x);
            const int hx1 = bin(xHi, majOrigin_.x);
            const int hy0 = bin(yLo, majOrigin_.y);
            const int hy1 = bin(yHi, majOrigin_.y);
            const int hz0 = bin(zLo, majOrigin_.z);
            const int hz1 = bin(zHi, majOrigin_.z);
            for (int z = hz0; z <= hz1; ++z) {
                for (int y = hy0; y <= hy1; ++y) {
                    for (int x = hx0; x <= hx1; ++x) {
                        if (!inRange(x, y, z)) continue;
                        const int i = idx(x, y, z);
                        if (bumpMax) majMax_[size_t(i)] = std::max(majMax_[size_t(i)], v);
                        if (bumpMin) majMin_[size_t(i)] = std::min(majMin_[size_t(i)], v);
                    }
                }
            }
        };
        // Halo: trilinear can see this voxel from neighbouring cells.
        stampHalo(haloMax, false, true);
        stampHalo(haloMin, true, false);
    }

    const int voxelsPerCell =
        srMax(1, int(std::lround(double(majCell_ / srMax(voxelSize_, 1e-8f)))));
    const int expected = voxelsPerCell * voxelsPerCell * voxelsPerCell;
    const int filledThresh = srMax(1, int(float(expected) * 0.85f));
    float globalMaj = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (counts[size_t(i)] <= 0 || majMax_[size_t(i)] <= 0.0f) {
            majMin_[size_t(i)] = 0.0f;
            majMax_[size_t(i)] = 0.0f;
            continue;
        }
        if (counts[size_t(i)] < filledThresh) majMin_[size_t(i)] = 0.0f;
        else majMin_[size_t(i)] = srMax(0.0f, majMin_[size_t(i)]);
        // Mixed cells: slight max inflation for reconstruction overshoot.
        if (majMax_[size_t(i)] > majMin_[size_t(i)] + 1e-4f)
            majMax_[size_t(i)] *= 1.15f;
        globalMaj = srMax(globalMaj, majMax_[size_t(i)]);
    }
    // Tracking uses Linear (8-tap). A sample in a filled cell next to empty
    // interpolates toward 0 and can go below voxel-min — residual ratio then
    // yields Tr>1. Any cell touching empty space cannot use a positive control μc.
    if (sampleFilter_ != VolumeSampleFilter::Nearest) {
        auto cellEmpty = [&](int x, int y, int z) -> bool {
            if (!inRange(x, y, z)) return true;
            return majMax_[size_t(idx(x, y, z))] <= 1e-8f;
        };
        for (int z = 0; z < majNz_; ++z) {
            for (int y = 0; y < majNy_; ++y) {
                for (int x = 0; x < majNx_; ++x) {
                    if (cellEmpty(x - 1, y, z) || cellEmpty(x + 1, y, z) || cellEmpty(x, y - 1, z) ||
                        cellEmpty(x, y + 1, z) || cellEmpty(x, y, z - 1) || cellEmpty(x, y, z + 1))
                        majMin_[size_t(idx(x, y, z))] = 0.0f;
                }
            }
        }
    }
    if (globalMaj > 0.0f) majorant_ = srMax(majorant_, globalMaj);

    brNx_ = (majNx_ + kMajBrick - 1) / kMajBrick;
    brNy_ = (majNy_ + kMajBrick - 1) / kMajBrick;
    brNz_ = (majNz_ + kMajBrick - 1) / kMajBrick;
    brOcc_.assign(size_t(brNx_ * brNy_ * brNz_), 0);
    for (int z = 0; z < majNz_; ++z) {
        for (int y = 0; y < majNy_; ++y) {
            for (int x = 0; x < majNx_; ++x) {
                const int i = idx(x, y, z);
                if (majMax_[size_t(i)] <= 1e-8f) continue;
                const int bx = x / kMajBrick;
                const int by = y / kMajBrick;
                const int bz = z / kMajBrick;
                brOcc_[size_t(bx + brNx_ * (by + brNy_ * bz))] = 1;
            }
        }
    }
}

void VolumeGrid::majorantOccupancy(const Vec3& p, float& minD, float& maxD) const {
    minD = 0.0f;
    maxD = 0.0f;
    if (!hasMajorantGrid()) {
        maxD = majorant_;
        return;
    }
    auto bin = [&](float w, float o) -> int {
        return int(std::floor(double((w - o) / majCell_)));
    };
    const int ix = bin(p.x, majOrigin_.x);
    const int iy = bin(p.y, majOrigin_.y);
    const int iz = bin(p.z, majOrigin_.z);
    if (ix < 0 || iy < 0 || iz < 0 || ix >= majNx_ || iy >= majNy_ || iz >= majNz_) return;
    const int i = ix + majNx_ * (iy + majNy_ * iz);
    minD = majMin_[size_t(i)];
    maxD = majMax_[size_t(i)];
}

float VolumeGrid::majorantCellExitT(Vec3 origin, Vec3 direction, float t, float tMax) const {
    if (!hasMajorantGrid()) return tMax;
    return ddaGridExitT(origin, direction, t, tMax, majOrigin_, majCell_, majNx_, majNy_, majNz_);
}

bool VolumeGrid::majorantBrickEmpty(const Vec3& p) const {
    if (brOcc_.empty() || majCell_ <= 1e-12f) return false;
    const float cell = majCell_ * float(kMajBrick);
    auto bin = [&](float w, float o, int dim) -> int {
        const int i = int(std::floor(double((w - o) / cell)));
        if (i < 0 || i >= dim) return -1;
        return i;
    };
    const int bx = bin(p.x, majOrigin_.x, brNx_);
    const int by = bin(p.y, majOrigin_.y, brNy_);
    const int bz = bin(p.z, majOrigin_.z, brNz_);
    if (bx < 0 || by < 0 || bz < 0) return true;
    return brOcc_[size_t(bx + brNx_ * (by + brNy_ * bz))] == 0;
}

float VolumeGrid::majorantBrickExitT(Vec3 origin, Vec3 direction, float t, float tMax) const {
    if (brOcc_.empty()) return tMax;
    return ddaGridExitT(origin, direction, t, tMax, majOrigin_, majCell_ * float(kMajBrick), brNx_,
                        brNy_, brNz_);
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
        if (out->kind_ == VolumeGridKind::Fog) out->rebuildMajorantGrid();
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
    if (out->kind_ == VolumeGridKind::Fog) out->rebuildMajorantGrid();
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
float VolumeGrid::sampleWorldTracking(const Vec3&) const { return 0.0f; }
Vec3 VolumeGrid::gradientWorld(const Vec3&) const { return Vec3(0, 1, 0); }
void VolumeGrid::rebuildMajorantGrid() {
    majNx_ = majNy_ = majNz_ = 0;
    majMin_.clear();
    majMax_.clear();
    brNx_ = brNy_ = brNz_ = 0;
    brOcc_.clear();
}
void VolumeGrid::majorantOccupancy(const Vec3&, float& minD, float& maxD) const {
    minD = 0.0f;
    maxD = majorant_;
}
float VolumeGrid::majorantCellExitT(Vec3, Vec3, float, float tMax) const { return tMax; }
bool VolumeGrid::majorantBrickEmpty(const Vec3&) const { return false; }
float VolumeGrid::majorantBrickExitT(Vec3, Vec3, float, float tMax) const { return tMax; }
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
