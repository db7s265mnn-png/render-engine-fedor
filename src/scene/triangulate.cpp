#include "scene/triangulate.h"

#include <cmath>
#include <array>

#include "earcut.hpp"

namespace sol {
namespace {

Vec3 faceNormal(const std::vector<Vec3>& positions, const uint32_t* faceVerts, size_t n) {
    // Newell's method — robust for non-convex planar polygons.
    Vec3 nrm(0.0f);
    for (size_t i = 0; i < n; ++i) {
        const Vec3& cur = positions[faceVerts[i]];
        const Vec3& nxt = positions[faceVerts[(i + 1) % n]];
        nrm.x += (cur.y - nxt.y) * (cur.z + nxt.z);
        nrm.y += (cur.z - nxt.z) * (cur.x + nxt.x);
        nrm.z += (cur.x - nxt.x) * (cur.y + nxt.y);
    }
    const float len = length(nrm);
    return len > 1e-20f ? nrm / len : Vec3(0.0f, 1.0f, 0.0f);
}

void buildTangentFrame(Vec3 n, Vec3& t, Vec3& b) {
    const Vec3 a = (fabsf(n.x) > 0.9f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
    t = normalize(cross(n, a));
    b = cross(n, t);
}

}  // namespace

bool triangulatePolygon(const std::vector<Vec3>& positions, const uint32_t* faceVerts, size_t faceVertCount,
                        std::vector<uint32_t>& outTriangles) {
    if (!faceVerts || faceVertCount < 3) return false;
    for (size_t i = 0; i < faceVertCount; ++i) {
        if (faceVerts[i] >= positions.size()) return false;
    }

    if (faceVertCount == 3) {
        outTriangles.push_back(faceVerts[0]);
        outTriangles.push_back(faceVerts[1]);
        outTriangles.push_back(faceVerts[2]);
        return true;
    }

    // Fast path: convex-looking quads stay as two tris when diagonal is valid.
    if (faceVertCount == 4) {
        const uint32_t i0 = faceVerts[0], i1 = faceVerts[1], i2 = faceVerts[2], i3 = faceVerts[3];
        const Vec3 n0 = cross(positions[i1] - positions[i0], positions[i2] - positions[i0]);
        const Vec3 n1 = cross(positions[i2] - positions[i0], positions[i3] - positions[i0]);
        if (dot(n0, n1) >= 0.0f && lengthSquared(n0) > 0.0f && lengthSquared(n1) > 0.0f) {
            outTriangles.push_back(i0);
            outTriangles.push_back(i1);
            outTriangles.push_back(i2);
            outTriangles.push_back(i0);
            outTriangles.push_back(i2);
            outTriangles.push_back(i3);
            return true;
        }
    }

    const Vec3 n = faceNormal(positions, faceVerts, faceVertCount);
    Vec3 t, b;
    buildTangentFrame(n, t, b);

    using Point = std::array<double, 2>;
    std::vector<std::vector<Point>> polygon(1);
    polygon[0].reserve(faceVertCount);
    const Vec3 origin = positions[faceVerts[0]];
    for (size_t i = 0; i < faceVertCount; ++i) {
        const Vec3 d = positions[faceVerts[i]] - origin;
        polygon[0].push_back(Point{double(dot(d, t)), double(dot(d, b))});
    }

    const std::vector<uint32_t> local = mapbox::earcut<uint32_t>(polygon);
    if (local.size() < 3) {
        // Fallback fan (rare earcut failure on degenerate projected faces).
        for (size_t i = 1; i + 1 < faceVertCount; ++i) {
            outTriangles.push_back(faceVerts[0]);
            outTriangles.push_back(faceVerts[i]);
            outTriangles.push_back(faceVerts[i + 1]);
        }
        return outTriangles.size() >= 3;
    }

    outTriangles.reserve(outTriangles.size() + local.size());
    for (uint32_t li : local) {
        if (li >= faceVertCount) return false;
        outTriangles.push_back(faceVerts[li]);
    }
    return true;
}

bool triangulateMeshFaces(const std::vector<Vec3>& positions, const std::vector<uint32_t>& faceVertexCounts,
                          const std::vector<uint32_t>& faceVertexIndices, std::vector<uint32_t>& outIndices) {
    outIndices.clear();
    if (faceVertexCounts.empty() || faceVertexIndices.empty()) return false;
    size_t cursor = 0;
    for (uint32_t count : faceVertexCounts) {
        if (count < 3 || cursor + size_t(count) > faceVertexIndices.size()) {
            cursor += size_t(count);
            continue;
        }
        triangulatePolygon(positions, faceVertexIndices.data() + cursor, size_t(count), outIndices);
        cursor += size_t(count);
    }
    return !outIndices.empty();
}

}  // namespace sol
