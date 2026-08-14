// Dual Contouring of Signed Distance Data (Carrera et al., SIGGRAPH 2026) —
// CPU port of the paper algorithm over a dense regular SDF grid (no Python /
// libigl / Polyscope). Samples may come from an OpenVDB level set.
#pragma once

#include "scene/scene.h"

namespace sol {

struct DcsddOptions {
    int outerIters = 40;
    int innerIters = 20;
    float mu = 0.1f;
    float dcWeight = 0.02f;
    float hermitePosWeight = 0.2f;
    float hermiteNormalWeight = 0.2f;
};

// `sdf` is row-major with size resX*resY*resZ; sample (i,j,k) at
// origin + Vec3(i,j,k)*voxelSize. Negative = interior (OpenVDB convention).
MeshPtr dcsddContourSdfGrid(const std::vector<float>& sdf, int resX, int resY, int resZ, Vec3 origin,
                            float voxelSize, const DcsddOptions& options = {});

// Sample an SDF VolumeGrid onto a dense grid covering its active bounds, then contour.
MeshPtr dcsddContourVolume(const VolumeGrid& volume, float sampleVoxelSize = 0.0f,
                           const DcsddOptions& options = {}, std::string* error = nullptr);

}  // namespace sol
