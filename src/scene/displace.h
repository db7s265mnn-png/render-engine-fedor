// Arnold-style geometric displacement: uniform triangle subdivision, then
// vertex offset from a height/vector map, optional bounds padding.
#pragma once

#include <cmath>

#include "scene/scene.h"
#include "scene/types.h"

namespace sol {

inline bool materialHasGeometricDisplacement(const Material& m) {
    if (m.displacementProc >= 0 || m.displacementTex >= 0) return true;
    return std::fabs(m.displacementHeight * m.displacementScale) > 1.0e-8f;
}

// Copy `src`, optionally subdivide, displace vertices, pad bounds.
// Samples `mat.displacementTex` / `mat.displacementProc` via scene textures/procs
// (indices must already be absolute in `scene`).
MeshPtr applyArnoldDisplacement(const Mesh& src, const Material& mat, const Scene& scene);

}  // namespace sol
