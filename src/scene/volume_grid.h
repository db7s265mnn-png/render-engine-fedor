// Host-side OpenVDB grid wrapper (SDF level set or fog/density volume).
#pragma once

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

struct VolumeFromPolygonsSettings {
    VolumeGridKind kind = VolumeGridKind::Sdf;
    float voxelSize = 0.05f;
    float exteriorBand = 3.0f;  // voxels (Houdini-like)
    float interiorBand = 3.0f;  // voxels
    float fillDensity = 1.0f;   // fog fill value inside closed mesh
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
    const std::string& name() const { return name_; }
    void setName(std::string n) { name_ = std::move(n); }
    Bounds3 worldBounds() const { return bounds_; }
    float voxelSize() const { return voxelSize_; }
    bool valid() const;

    // Sample scalar field in world space (SDF distance or fog density).
    float sampleWorld(const Vec3& p) const;
    // Finite-difference gradient (world space). Useful for SDF normals.
    Vec3 gradientWorld(const Vec3& p) const;
    // Max |density| / majorant estimate for delta tracking (fog).
    float majorant() const { return majorant_; }

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
    std::string name_;
    Bounds3 bounds_;
    float voxelSize_ = 0.05f;
    float majorant_ = 1.0f;
};

using VolumeGridPtr = std::shared_ptr<VolumeGrid>;

}  // namespace sol
