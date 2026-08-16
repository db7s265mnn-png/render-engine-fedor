// Host-side OpenVDB grid wrapper (SDF level set or fog/density volume).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/math.h"
#include "scene/scene.h"

namespace sol {

enum class VolumeGridKind : int {
    Sdf = 0,   // OpenVDB level set (signed distance)
    Fog = 1,   // Fog / density volume
};

// OpenVDB GridSampler filter — controls fog/SDF sample smoothness.
enum class VolumeSampleFilter : int {
    Nearest = 0,    // PointSampler — blocky voxels
    Linear = 1,     // BoxSampler — trilinear (default)
    Quadratic = 2,  // QuadraticSampler — smoother
};

struct VolumeFromPolygonsSettings {
    VolumeGridKind kind = VolumeGridKind::Sdf;
    float voxelSize = 0.05f;
    float exteriorBand = 3.0f;  // voxels (Houdini-like)
    float interiorBand = 3.0f;  // voxels
    float fillDensity = 1.0f;   // unused for voxel bake — runtime MediumData::density scale
    VolumeSampleFilter filter = VolumeSampleFilter::Linear;
};

class VolumeGrid {
public:
    VolumeGrid();
    ~VolumeGrid();
    VolumeGrid(VolumeGrid&&) noexcept;
    VolumeGrid& operator=(VolumeGrid&&) noexcept;
    VolumeGrid(const VolumeGrid&) = delete;
    VolumeGrid& operator=(const VolumeGrid&) = delete;

    VolumeGridKind kind() const { return kind_; }
    VolumeSampleFilter sampleFilter() const { return sampleFilter_; }
    void setSampleFilter(VolumeSampleFilter f) { sampleFilter_ = f; }
    const std::string& name() const { return name_; }
    void setName(std::string n) { name_ = std::move(n); }
    Bounds3 worldBounds() const { return bounds_; }
    float voxelSize() const { return voxelSize_; }
    bool valid() const;

    // True when this binary was linked against OpenVDB (false → fromPolygons/loadVdb are stubs).
    static bool openVdbAvailable();

    // Sample scalar field in world space (SDF distance or fog density).
    float sampleWorld(const Vec3& p) const;
    // Density sample for delta / residual tracking. Quadratic is 27-tap and
    // dominates deep multiple scattering; tracking always uses Linear (8-tap).
    // SDF gradients and the authored Sample Filter still use sampleWorld().
    float sampleWorldTracking(const Vec3& p) const;
    // Finite-difference gradient (world space). Useful for SDF normals.
    Vec3 gradientWorld(const Vec3& p) const;
    // Max |density| / majorant estimate for delta tracking (fog).
    float majorant() const { return majorant_; }

    // Piecewise min/max occupancy (supervoxels). Empty / constant cells skip
    // voxel sampling during Woodcock walks. Fog only. Cell size is world-space
    // (floored so a dense VDB does not get 8-voxel micro-cells).
    bool hasMajorantGrid() const { return majNx_ > 0 && int(majMax_.size()) == majNx_ * majNy_ * majNz_; }
    float majorantCellSize() const { return majCell_; }
    int majorantDimX() const { return majNx_; }
    int majorantDimY() const { return majNy_; }
    int majorantDimZ() const { return majNz_; }
    void majorantOccupancy(const Vec3& p, float& minD, float& maxD) const;
    // Ray parameter of the current coarse-cell far face, clamped to tMax.
    float majorantCellExitT(Vec3 origin, Vec3 direction, float t, float tMax) const;
    // 4³ bricks of supervoxels: skip empty AABB regions without visiting every cell.
    bool hasMajorantBricks() const { return !brOcc_.empty(); }
    bool majorantBrickEmpty(const Vec3& p) const;
    float majorantBrickExitT(Vec3 origin, Vec3 direction, float t, float tMax) const;

    // Serialize helpers
    bool saveVdb(const std::string& path) const;
    static std::shared_ptr<VolumeGrid> loadVdb(const std::string& path, std::string* error = nullptr);

    static std::shared_ptr<VolumeGrid> fromPolygons(const Mesh& mesh, const Mat4& xform,
                                                    const VolumeFromPolygonsSettings& settings,
                                                    std::string* error = nullptr);

    // Convert SDF level set → triangle mesh (OpenVDB volumeToMesh).
    MeshPtr toPolygonsOpenVDB(float isovalue = 0.0f, float adaptivity = 0.0f) const;

    // Opaque access for advanced callers (may be null when OpenVDB is disabled).
    void* nativeGrid() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    VolumeGridKind kind_ = VolumeGridKind::Sdf;
    VolumeSampleFilter sampleFilter_ = VolumeSampleFilter::Linear;
    std::string name_;
    Bounds3 bounds_;
    float voxelSize_ = 0.05f;
    float majorant_ = 1.0f;

    void rebuildMajorantGrid();
    int majNx_ = 0;
    int majNy_ = 0;
    int majNz_ = 0;
    float majCell_ = 0.0f;
    Vec3 majOrigin_{};
    std::vector<float> majMin_;
    std::vector<float> majMax_;
    static constexpr int kMajBrick = 4;
    int brNx_ = 0;
    int brNy_ = 0;
    int brNz_ = 0;
    std::vector<uint8_t> brOcc_;
};

using VolumeGridPtr = std::shared_ptr<VolumeGrid>;

}  // namespace sol
