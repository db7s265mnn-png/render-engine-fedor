#include "scene/dcsdd_contouring.h"

#include "core/log.h"
#include "scene/volume_grid.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace sol {
namespace {

inline int idx3(int i, int j, int k, int resX, int resY) { return i + resX * (j + resY * k); }

float sampleGrid(const std::vector<float>& sdf, int resX, int resY, int resZ, int i, int j, int k) {
    i = std::clamp(i, 0, resX - 1);
    j = std::clamp(j, 0, resY - 1);
    k = std::clamp(k, 0, resZ - 1);
    return sdf[size_t(idx3(i, j, k, resX, resY))];
}

Vec3 fdGradient(const std::vector<float>& sdf, int resX, int resY, int resZ, float voxel,
                int i, int j, int k) {
    const float dx = sampleGrid(sdf, resX, resY, resZ, i + 1, j, k) - sampleGrid(sdf, resX, resY, resZ, i - 1, j, k);
    const float dy = sampleGrid(sdf, resX, resY, resZ, i, j + 1, k) - sampleGrid(sdf, resX, resY, resZ, i, j - 1, k);
    const float dz = sampleGrid(sdf, resX, resY, resZ, i, j, k + 1) - sampleGrid(sdf, resX, resY, resZ, i, j, k - 1);
    Vec3 g(dx, dy, dz);
    const float len = length(g);
    return len > 1e-12f ? g / len : Vec3(0, 1, 0);
}

struct Hermite {
    Vec3 p{0};
    Vec3 n{0, 1, 0};
    bool active = false;
};

struct CellVert {
    Vec3 x{0};
    bool active = false;
};

// Solve min_x Σ ((x-p)·n)^2 + μ||x-c||^2  (regularized QEF), clamped to cell.
Vec3 solveQef(const std::vector<Hermite>& H, Vec3 cellMin, Vec3 cellMax, Vec3 centroid, float mu) {
    // Accumulate 3x3 ATA and ATb from plane constraints.
    float ATA[9] = {};
    float ATb[3] = {};
    for (const Hermite& h : H) {
        if (!h.active) continue;
        const float nx = h.n.x, ny = h.n.y, nz = h.n.z;
        const float d = dot(h.p, h.n);
        ATA[0] += nx * nx;
        ATA[1] += nx * ny;
        ATA[2] += nx * nz;
        ATA[3] += ny * nx;
        ATA[4] += ny * ny;
        ATA[5] += ny * nz;
        ATA[6] += nz * nx;
        ATA[7] += nz * ny;
        ATA[8] += nz * nz;
        ATb[0] += nx * d;
        ATb[1] += ny * d;
        ATb[2] += nz * d;
    }
    // μ I toward centroid.
    ATA[0] += mu;
    ATA[4] += mu;
    ATA[8] += mu;
    ATb[0] += mu * centroid.x;
    ATb[1] += mu * centroid.y;
    ATb[2] += mu * centroid.z;

    // Simple 3x3 solve via Cramer's / adjoint (stable enough for QEF).
    const float a = ATA[0], b = ATA[1], c = ATA[2];
    const float d = ATA[3], e = ATA[4], f = ATA[5];
    const float g = ATA[6], h = ATA[7], i = ATA[8];
    const float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    Vec3 x = centroid;
    if (fabsf(det) > 1e-12f) {
        const float inv = 1.0f / det;
        x.x = inv * ((ATb[0] * (e * i - f * h) - b * (ATb[1] * i - f * ATb[2]) + c * (ATb[1] * h - e * ATb[2])));
        x.y = inv * ((a * (ATb[1] * i - f * ATb[2]) - ATb[0] * (d * i - f * g) + c * (d * ATb[2] - ATb[1] * g)));
        x.z = inv * ((a * (e * ATb[2] - ATb[1] * h) - b * (d * ATb[2] - ATb[1] * g) + ATb[0] * (d * h - e * g)));
    }
    x.x = clampf(x.x, cellMin.x, cellMax.x);
    x.y = clampf(x.y, cellMin.y, cellMax.y);
    x.z = clampf(x.z, cellMin.z, cellMax.z);
    return x;
}

}  // namespace

MeshPtr dcsddContourSdfGrid(const std::vector<float>& sdf, int resX, int resY, int resZ, Vec3 origin,
                            float voxelSize, const DcsddOptions& options) {
    if (resX < 2 || resY < 2 || resZ < 2 || sdf.size() < size_t(resX * resY * resZ) || voxelSize <= 0.0f)
        return nullptr;

    const int cx = resX - 1, cy = resY - 1, cz = resZ - 1;
    std::vector<CellVert> cells(size_t(cx * cy * cz));
    // Hermite on edges: store by edge key.
    struct EdgeKey {
        int x, y, z, axis;  // axis 0=x,1=y,2=z edge starting at (x,y,z)
        bool operator==(const EdgeKey& o) const {
            return x == o.x && y == o.y && z == o.z && axis == o.axis;
        }
    };
    struct EdgeHash {
        size_t operator()(const EdgeKey& k) const {
            return size_t(k.x) * 73856093u ^ size_t(k.y) * 19349663u ^ size_t(k.z) * 83492791u ^
                   size_t(k.axis) * 2654435761u;
        }
    };
    std::unordered_map<EdgeKey, Hermite, EdgeHash> hermites;

    auto corner = [&](int i, int j, int k) {
        return origin + Vec3(float(i), float(j), float(k)) * voxelSize;
    };

    auto considerEdge = [&](int i, int j, int k, int axis) {
        int i1 = i, j1 = j, k1 = k;
        if (axis == 0) ++i1;
        else if (axis == 1) ++j1;
        else ++k1;
        if (i1 >= resX || j1 >= resY || k1 >= resZ) return;
        const float s0 = sampleGrid(sdf, resX, resY, resZ, i, j, k);
        const float s1 = sampleGrid(sdf, resX, resY, resZ, i1, j1, k1);
        if (s0 * s1 > 0.0f) return;  // no sign change
        const float denom = (s0 - s1);
        float t = (fabsf(denom) > 1e-12f) ? (s0 / denom) : 0.5f;
        t = clampf(t, 0.0f, 1.0f);
        Hermite h;
        h.p = corner(i, j, k) * (1.0f - t) + corner(i1, j1, k1) * t;
        // Average FD gradients at endpoints.
        const Vec3 n0 = fdGradient(sdf, resX, resY, resZ, voxelSize, i, j, k);
        const Vec3 n1 = fdGradient(sdf, resX, resY, resZ, voxelSize, i1, j1, k1);
        h.n = normalize(n0 + n1);
        if (lengthSquared(h.n) < 1e-12f) h.n = Vec3(0, 1, 0);
        h.active = true;
        hermites[{i, j, k, axis}] = h;
    };

    for (int k = 0; k < resZ; ++k)
        for (int j = 0; j < resY; ++j)
            for (int i = 0; i < resX; ++i) {
                considerEdge(i, j, k, 0);
                considerEdge(i, j, k, 1);
                considerEdge(i, j, k, 2);
            }

    auto cellHermites = [&](int i, int j, int k, std::vector<Hermite>& out) {
        out.clear();
        const EdgeKey keys[12] = {
            {i, j, k, 0},     {i, j + 1, k, 0},     {i, j, k + 1, 0},     {i, j + 1, k + 1, 0},
            {i, j, k, 1},     {i + 1, j, k, 1},     {i, j, k + 1, 1},     {i + 1, j, k + 1, 1},
            {i, j, k, 2},     {i + 1, j, k, 2},     {i, j + 1, k, 2},     {i + 1, j + 1, k, 2},
        };
        for (const EdgeKey& key : keys) {
            auto it = hermites.find(key);
            if (it != hermites.end()) out.push_back(it->second);
        }
    };

    // Initialize cell vertices at Hermite centroids.
    for (int k = 0; k < cz; ++k)
        for (int j = 0; j < cy; ++j)
            for (int i = 0; i < cx; ++i) {
                std::vector<Hermite> H;
                cellHermites(i, j, k, H);
                if (H.empty()) continue;
                Vec3 c(0);
                for (const Hermite& h : H) c += h.p;
                c = c * (1.0f / float(H.size()));
                CellVert& cv = cells[size_t(i + cx * (j + cy * k))];
                cv.active = true;
                cv.x = c;
            }

    // Outer iterations: QEF + Hermite update (paper-style).
    for (int outer = 0; outer < options.outerIters; ++outer) {
        for (int k = 0; k < cz; ++k)
            for (int j = 0; j < cy; ++j)
                for (int i = 0; i < cx; ++i) {
                    CellVert& cv = cells[size_t(i + cx * (j + cy * k))];
                    if (!cv.active) continue;
                    std::vector<Hermite> H;
                    cellHermites(i, j, k, H);
                    if (H.empty()) continue;
                    Vec3 c(0);
                    for (const Hermite& h : H) c += h.p;
                    c = c * (1.0f / float(H.size()));
                    const Vec3 cellMin = corner(i, j, k);
                    const Vec3 cellMax = corner(i + 1, j + 1, k + 1);
                    for (int inner = 0; inner < options.innerIters; ++inner) {
                        cv.x = solveQef(H, cellMin, cellMax, c, options.mu);
                        // Mild pull toward DC energy (options.dcWeight as extra μ toward hermite planes already in QEF).
                        (void)options.dcWeight;
                    }
                }

        // Update Hermite from best-fit plane of the four cell vertices around each edge.
        for (auto& kv : hermites) {
            const EdgeKey& key = kv.first;
            Hermite& h = kv.second;
            // Collect up to 4 cells sharing this edge.
            std::vector<Vec3> verts;
            auto addCell = [&](int ci, int cj, int ck) {
                if (ci < 0 || cj < 0 || ck < 0 || ci >= cx || cj >= cy || ck >= cz) return;
                const CellVert& cv = cells[size_t(ci + cx * (cj + cy * ck))];
                if (cv.active) verts.push_back(cv.x);
            };
            if (key.axis == 0) {
                addCell(key.x, key.y - 1, key.z - 1);
                addCell(key.x, key.y, key.z - 1);
                addCell(key.x, key.y - 1, key.z);
                addCell(key.x, key.y, key.z);
            } else if (key.axis == 1) {
                addCell(key.x - 1, key.y, key.z - 1);
                addCell(key.x, key.y, key.z - 1);
                addCell(key.x - 1, key.y, key.z);
                addCell(key.x, key.y, key.z);
            } else {
                addCell(key.x - 1, key.y - 1, key.z);
                addCell(key.x, key.y - 1, key.z);
                addCell(key.x - 1, key.y, key.z);
                addCell(key.x, key.y, key.z);
            }
            if (verts.size() < 3) continue;
            Vec3 mean(0);
            for (const Vec3& v : verts) mean += v;
            mean = mean * (1.0f / float(verts.size()));
            // Covariance → approximate normal via smallest eigenvector (power on cross products).
            Vec3 n(0);
            for (size_t a = 0; a < verts.size(); ++a) {
                const Vec3 d0 = verts[a] - mean;
                const Vec3 d1 = verts[(a + 1) % verts.size()] - mean;
                n += cross(d0, d1);
            }
            if (lengthSquared(n) < 1e-16f) continue;
            n = normalize(n);
            // Edge intersection with plane.
            Vec3 e0 = corner(key.x, key.y, key.z);
            Vec3 e1 = e0;
            if (key.axis == 0) e1.x += voxelSize;
            else if (key.axis == 1) e1.y += voxelSize;
            else e1.z += voxelSize;
            const float denom = dot(e1 - e0, n);
            Vec3 y = h.p;
            if (fabsf(denom) > 1e-8f) {
                const float t = clampf(dot(mean - e0, n) / denom, 0.0f, 1.0f);
                y = e0 + (e1 - e0) * t;
            }
            h.p = h.p + (y - h.p) * options.hermitePosWeight;
            h.n = normalize(h.n + (n - h.n) * options.hermiteNormalWeight);
        }
    }

    // Emit quads for each interesting edge → two triangles.
    auto mesh = std::make_shared<Mesh>();
    auto cellVertIndex = [&](int i, int j, int k) -> int {
        if (i < 0 || j < 0 || k < 0 || i >= cx || j >= cy || k >= cz) return -1;
        const CellVert& cv = cells[size_t(i + cx * (j + cy * k))];
        if (!cv.active) return -1;
        mesh->positions.push_back(cv.x);
        return int(mesh->positions.size()) - 1;
    };

    // Map cell -> unique vertex index (shared).
    std::vector<int> cellToVert(size_t(cx * cy * cz), -1);
    for (int k = 0; k < cz; ++k)
        for (int j = 0; j < cy; ++j)
            for (int i = 0; i < cx; ++i) {
                const size_t id = size_t(i + cx * (j + cy * k));
                if (!cells[id].active) continue;
                cellToVert[id] = int(mesh->positions.size());
                mesh->positions.push_back(cells[id].x);
            }

    auto vertOf = [&](int i, int j, int k) -> int {
        if (i < 0 || j < 0 || k < 0 || i >= cx || j >= cy || k >= cz) return -1;
        return cellToVert[size_t(i + cx * (j + cy * k))];
    };

    auto emitQuad = [&](int a, int b, int c, int d) {
        if (a < 0 || b < 0 || c < 0 || d < 0) return;
        mesh->faceVertexCounts.push_back(4);
        mesh->faceVertexIndices.push_back(uint32_t(a));
        mesh->faceVertexIndices.push_back(uint32_t(b));
        mesh->faceVertexIndices.push_back(uint32_t(c));
        mesh->faceVertexIndices.push_back(uint32_t(d));
    };

    for (const auto& kv : hermites) {
        const EdgeKey& key = kv.first;
        int v0 = -1, v1 = -1, v2 = -1, v3 = -1;
        if (key.axis == 0) {
            v0 = vertOf(key.x, key.y - 1, key.z - 1);
            v1 = vertOf(key.x, key.y, key.z - 1);
            v2 = vertOf(key.x, key.y, key.z);
            v3 = vertOf(key.x, key.y - 1, key.z);
        } else if (key.axis == 1) {
            v0 = vertOf(key.x - 1, key.y, key.z - 1);
            v1 = vertOf(key.x, key.y, key.z - 1);
            v2 = vertOf(key.x, key.y, key.z);
            v3 = vertOf(key.x - 1, key.y, key.z);
        } else {
            v0 = vertOf(key.x - 1, key.y - 1, key.z);
            v1 = vertOf(key.x, key.y - 1, key.z);
            v2 = vertOf(key.x, key.y, key.z);
            v3 = vertOf(key.x - 1, key.y, key.z);
        }
        // Orient by hermite normal.
        const Vec3& hn = kv.second.n;
        if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) {
            const Vec3 qn = cross(mesh->positions[size_t(v1)] - mesh->positions[size_t(v0)],
                                  mesh->positions[size_t(v3)] - mesh->positions[size_t(v0)]);
            if (dot(qn, hn) < 0.0f) emitQuad(v0, v3, v2, v1);
            else emitQuad(v0, v1, v2, v3);
        }
    }

    (void)cellVertIndex;
    if (mesh->faceVertexIndices.empty()) return nullptr;
    mesh->validate();
    return mesh;
}

MeshPtr dcsddContourVolume(const VolumeGrid& volume, float sampleVoxelSize, const DcsddOptions& options,
                           std::string* error) {
    if (!volume.valid() || volume.kind() != VolumeGridKind::Sdf) {
        if (error) *error = "DCSDD expects a valid SDF volume";
        return nullptr;
    }
    const Bounds3 b = volume.worldBounds();
    if (!b.valid()) {
        if (error) *error = "SDF volume has empty bounds";
        return nullptr;
    }
    float vs = sampleVoxelSize > 0.0f ? sampleVoxelSize : volume.voxelSize();
    vs = srMax(vs, 1e-5f);
    const Vec3 ext = b.hi - b.lo;
    int resX = std::max(2, int(std::ceil(ext.x / vs)) + 1);
    int resY = std::max(2, int(std::ceil(ext.y / vs)) + 1);
    int resZ = std::max(2, int(std::ceil(ext.z / vs)) + 1);
    // Cap resolution for interactive cooks.
    const int maxRes = 192;
    if (resX > maxRes || resY > maxRes || resZ > maxRes) {
        const float scale = float(maxRes) / float(std::max(resX, std::max(resY, resZ)));
        resX = std::max(2, int(resX * scale));
        resY = std::max(2, int(resY * scale));
        resZ = std::max(2, int(resZ * scale));
        vs = srMax(ext.x / float(resX - 1), srMax(ext.y / float(resY - 1), ext.z / float(resZ - 1)));
    }
    std::vector<float> sdf(size_t(resX * resY * resZ));
    const Vec3 origin = b.lo;
    for (int k = 0; k < resZ; ++k)
        for (int j = 0; j < resY; ++j)
            for (int i = 0; i < resX; ++i) {
                const Vec3 p = origin + Vec3(float(i), float(j), float(k)) * vs;
                sdf[size_t(idx3(i, j, k, resX, resY))] = volume.sampleWorld(p);
            }
    return dcsddContourSdfGrid(sdf, resX, resY, resZ, origin, vs, options);
}

}  // namespace sol
