// Concave-safe polygon triangulation (Mapbox earcut) for n-gon cages.
#pragma once

#include <cstdint>
#include <vector>

#include "core/math.h"

namespace sol {

// Triangulate a single planar (or nearly planar) n-gon in 3D.
// `faceVerts` are indices into `positions`. Output is a flat triangle index list
// (3 indices per triangle) referring to the same vertex buffer.
// Returns false if the face cannot be triangulated (degenerate / <3 verts).
bool triangulatePolygon(const std::vector<Vec3>& positions, const uint32_t* faceVerts, size_t faceVertCount,
                        std::vector<uint32_t>& outTriangles);

// densify: for each face in faceVertexCounts / faceVertexIndices, append triangles
// into `outIndices`. Clears outIndices first.
// If `outEdgeMask` is non-null, it is cleared and filled with one uint8 per triangle:
// bits 0/1/2 mark edges (i0,i1)/(i1,i2)/(i2,i0) that lie on the authored face boundary
// (so wireframe can hide triangulation diagonals).
bool triangulateMeshFaces(const std::vector<Vec3>& positions, const std::vector<uint32_t>& faceVertexCounts,
                          const std::vector<uint32_t>& faceVertexIndices, std::vector<uint32_t>& outIndices,
                          std::vector<uint8_t>* outEdgeMask = nullptr);

}  // namespace sol
