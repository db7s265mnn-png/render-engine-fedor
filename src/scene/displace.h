// Geometric displacement: vertex offset from height/vector maps, Pref storage,
// bounds padding. Tessellation (subdiv / dicing) lives in tessellate.cpp.
#pragma once

#include <cmath>

#include "scene/scene.h"
#include "scene/types.h"

namespace sol {

inline bool materialHasGeometricDisplacement(const Material& m) {
    if (m.displacementProc >= 0 || m.displacementTex >= 0) return true;
    // Rare: constant height authored in MaterialX without a map.
    return std::fabs(m.displacementHeight * m.displacementScale) > 1.0e-8f;
}

// Displace vertices, store Pref/Nref, pad bounds. No subdivision.
// Prefer the by-value overload when the caller can move a refined mesh in
// (avoids an extra full copy before Render).
MeshPtr displaceMeshOnly(const Mesh& src, const Material& mat, const Scene& scene);
MeshPtr displaceMeshOnly(Mesh&& src, const Material& mat, const Scene& scene);

}  // namespace sol
