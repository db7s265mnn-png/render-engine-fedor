// Render-time tessellation for displacement: none / linear / catclark,
// frustum cull, screen-space dicing.
#include "scene/tessellate.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/log.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPENSUBDIV
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/topologyRefinerFactory.h>
#endif

namespace sol {
namespace {

// Safety ceiling only — prefer freeing previous-render memory over clamping.
// 200M tris ≈ heavy displace mesh; real limiter is host RAM / BVH build.
constexpr size_t kMaxTessTriangles = 200000000ull;

int clampSubdivLevelsForBudget(int requested, size_t triCount) {
    const int want = std::max(0, requested);
    if (want == 0 || triCount == 0) return 0;
    size_t est = triCount;
    int levels = 0;
    while (levels < want) {
        if (est > kMaxTessTriangles / 4) break;
        est *= 4;
        ++levels;
    }
    if (levels < want) {
        logWarning("tessellate: clamped subdiv iterations " + std::to_string(want) + " → " +
                   std::to_string(levels) + " (triangle budget " +
                   std::to_string(kMaxTessTriangles) + ")");
    }
    return levels;
}

bool indexInRange(uint32_t idx, size_t count) { return size_t(idx) < count; }

bool meshIsTriangleOnly(const Mesh& mesh) {
    // Our Mesh stores triangles only (3 indices per face). Catclark needs quads;
    // triangle-only cages fall back to linear.
    return !mesh.indices.empty();
}

bool meshHasQuadsHint(const Mesh& mesh) {
    // Current importer always triangulates — treat all cages as tris for catclark
    // fallback. Reserved for future ngon cages.
    (void)mesh;
    return false;
}

Mat4 worldToCamera(const CameraData& cam) {
    return inverse(cam.cameraToWorld);
}

// Project object-space point through instance xform into NDC (xy in ~[-1,1]).
bool projectToNdc(Vec3 pObject, const Mat4& objToWorld, const Mat4& worldToCam, const CameraData& cam,
                  float aspect, float& ndcX, float& ndcY, float& viewZ) {
    const Vec3 pWorld = transformPoint(objToWorld, pObject);
    const Vec3 pCam = transformPoint(worldToCam, pWorld);
    viewZ = -pCam.z;  // camera looks down -Z
    if (!(viewZ > 1.0e-5f)) return false;
    const float focal = std::max(1.0e-3f, cam.focalLength);
    const float sensorW = std::max(1.0e-3f, cam.sensorWidth);
    const float sensorH = sensorW / std::max(0.01f, aspect);
    // Match integrator film mapping: dir ~ (sx, sy, -focal) with sx in ±sensorW/2.
    const float halfW = 0.5f * sensorW * (viewZ / focal);
    const float halfH = 0.5f * sensorH * (viewZ / focal);
    if (!(halfW > 1e-8f) || !(halfH > 1e-8f)) return false;
    ndcX = pCam.x / halfW;
    ndcY = pCam.y / halfH;
    return true;
}

bool pointInPaddedFrustum(float ndcX, float ndcY, float padFrac) {
    const float lim = 1.0f + padFrac;
    return ndcX >= -lim && ndcX <= lim && ndcY >= -lim && ndcY <= lim;
}

float screenEdgePixels(Vec3 aObj, Vec3 bObj, const Mat4& objToWorld, const Mat4& worldToCam,
                       const CameraData& cam, float aspect, int resX, int resY) {
    float ax, ay, az, bx, by, bz;
    if (!projectToNdc(aObj, objToWorld, worldToCam, cam, aspect, ax, ay, az)) return 0.0f;
    if (!projectToNdc(bObj, objToWorld, worldToCam, cam, aspect, bx, by, bz)) return 0.0f;
    const float dx = (bx - ax) * 0.5f * float(resX);
    const float dy = (by - ay) * 0.5f * float(resY);
    return std::sqrt(dx * dx + dy * dy);
}

// Slab test: ray (origin + t*dir, t>=0) vs AABB. dir need not be unit.
bool rayHitsAabb(Vec3 origin, Vec3 dir, const Bounds3& b) {
    if (!b.valid()) return false;
    float tMin = 0.0f;
    float tMax = 1.0e30f;
    for (int a = 0; a < 3; ++a) {
        const float o = a == 0 ? origin.x : (a == 1 ? origin.y : origin.z);
        const float d = a == 0 ? dir.x : (a == 1 ? dir.y : dir.z);
        const float lo = a == 0 ? b.lo.x : (a == 1 ? b.lo.y : b.lo.z);
        const float hi = a == 0 ? b.hi.x : (a == 1 ? b.hi.y : b.hi.z);
        if (std::fabs(d) < 1.0e-12f) {
            if (o < lo || o > hi) return false;
            continue;
        }
        float t0 = (lo - o) / d;
        float t1 = (hi - o) / d;
        if (t0 > t1) std::swap(t0, t1);
        tMin = std::max(tMin, t0);
        tMax = std::min(tMax, t1);
        if (tMin > tMax) return false;
    }
    return tMax >= 0.0f;
}

bool pointInTriangle2D(float px, float py, float ax, float ay, float bx, float by, float cx, float cy) {
    const float v0x = cx - ax, v0y = cy - ay;
    const float v1x = bx - ax, v1y = by - ay;
    const float v2x = px - ax, v2y = py - ay;
    const float dot00 = v0x * v0x + v0y * v0y;
    const float dot01 = v0x * v1x + v0y * v1y;
    const float dot02 = v0x * v2x + v0y * v2y;
    const float dot11 = v1x * v1x + v1y * v1y;
    const float dot12 = v1x * v2x + v1y * v2y;
    const float den = dot00 * dot11 - dot01 * dot01;
    if (std::fabs(den) < 1.0e-20f) return false;
    const float inv = 1.0f / den;
    const float u = (dot11 * dot02 - dot01 * dot12) * inv;
    const float v = (dot00 * dot12 - dot01 * dot02) * inv;
    return u >= -1.0e-4f && v >= -1.0e-4f && (u + v) <= 1.0f + 1.0e-4f;
}

bool meshVisibleInFrustum(const Mesh& mesh, const std::vector<const InstanceData*>& instances,
                          const CameraData& cam, const Mat4& worldToCam, float aspect, float padFrac) {
    if (instances.empty()) return false;
    const size_t nPos = mesh.positions.size();
    const float focal = std::max(1.0e-3f, cam.focalLength);
    const float sensorW = std::max(1.0e-3f, cam.sensorWidth);
    const float sensorH = sensorW / std::max(0.01f, aspect);
    const float lim = 1.0f + padFrac;
    const Vec3 eye = transformPoint(cam.cameraToWorld, Vec3(0.0f));

    Bounds3 localBounds = mesh.bounds;
    if (!localBounds.valid()) {
        for (const Vec3& p : mesh.positions) localBounds.extend(p);
    }

    // Frustum sample directions in NDC (center, corners, edge mids) — catches the
    // common close-up case where a huge triangle covers the screen but all cage
    // vertices sit outside the frame (so point sampling alone returns false).
    const float ndcSamples[][2] = {
        {0.0f, 0.0f},   {lim, lim},    {-lim, lim},   {lim, -lim}, {-lim, -lim},
        {lim, 0.0f},    {-lim, 0.0f},  {0.0f, lim},   {0.0f, -lim},
    };

    for (const InstanceData* inst : instances) {
        if (!inst) continue;
        const Bounds3 worldBounds = localBounds.valid()
                                        ? transformBounds(inst->xform, localBounds)
                                        : Bounds3{};

        // Camera inside / very near the object → always dice (close-up on a plane).
        if (worldBounds.valid()) {
            const Vec3 pad(worldBounds.radius() * 0.05f + 1.0e-3f);
            if (eye.x >= worldBounds.lo.x - pad.x && eye.x <= worldBounds.hi.x + pad.x &&
                eye.y >= worldBounds.lo.y - pad.y && eye.y <= worldBounds.hi.y + pad.y &&
                eye.z >= worldBounds.lo.z - pad.z && eye.z <= worldBounds.hi.z + pad.z)
                return true;
        }

        if (worldBounds.valid()) {
            for (const auto& s : ndcSamples) {
                const Vec3 dirCam =
                    normalize(Vec3(s[0] * 0.5f * sensorW, s[1] * 0.5f * sensorH, -focal));
                const Vec3 dirWorld = transformVector(cam.cameraToWorld, dirCam);
                if (rayHitsAabb(eye, dirWorld, worldBounds)) return true;
            }
        }

        // Projected-triangle overlap: verts may be off-screen while the face covers NDC.
        const size_t step = std::max<size_t>(1, mesh.indices.size() / 3 / 256);
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3 * step) {
            const uint32_t ia = mesh.indices[t + 0];
            const uint32_t ib = mesh.indices[t + 1];
            const uint32_t ic = mesh.indices[t + 2];
            if (!indexInRange(ia, nPos) || !indexInRange(ib, nPos) || !indexInRange(ic, nPos)) continue;
            const Vec3& a = mesh.positions[ia];
            const Vec3& b = mesh.positions[ib];
            const Vec3& c = mesh.positions[ic];

            float nx[3], ny[3], nz[3];
            const bool ok0 = projectToNdc(a, inst->xform, worldToCam, cam, aspect, nx[0], ny[0], nz[0]);
            const bool ok1 = projectToNdc(b, inst->xform, worldToCam, cam, aspect, nx[1], ny[1], nz[1]);
            const bool ok2 = projectToNdc(c, inst->xform, worldToCam, cam, aspect, nx[2], ny[2], nz[2]);
            if (ok0 && pointInPaddedFrustum(nx[0], ny[0], padFrac)) return true;
            if (ok1 && pointInPaddedFrustum(nx[1], ny[1], padFrac)) return true;
            if (ok2 && pointInPaddedFrustum(nx[2], ny[2], padFrac)) return true;

            if (ok0 && ok1 && ok2) {
                for (const auto& s : ndcSamples) {
                    if (pointInTriangle2D(s[0], s[1], nx[0], ny[0], nx[1], ny[1], nx[2], ny[2]))
                        return true;
                }
            }

            const Vec3 mid = (a + b + c) * (1.0f / 3.0f);
            float mx, my, mz;
            if (projectToNdc(mid, inst->xform, worldToCam, cam, aspect, mx, my, mz) &&
                pointInPaddedFrustum(mx, my, padFrac))
                return true;
        }
    }
    return false;
}

const InstanceData* nearestInstance(const std::vector<const InstanceData*>& instances,
                                    const CameraData& cam) {
    if (instances.empty()) return nullptr;
    const Vec3 eye = transformPoint(cam.cameraToWorld, Vec3(0.0f));
    const InstanceData* best = instances[0];
    float bestD = 1.0e30f;
    for (const InstanceData* inst : instances) {
        if (!inst) continue;
        const Vec3 o = transformPoint(inst->xform, Vec3(0.0f));
        const float d = lengthSquared(o - eye);
        if (d < bestD) {
            bestD = d;
            best = inst;
        }
    }
    return best;
}

void subdivideLinearOnce(Mesh& mesh) {
    const size_t triCount = mesh.indices.size() / 3;
    if (triCount == 0) return;
    const size_t nPos = mesh.positions.size();
    const bool hasN = mesh.normals.size() == nPos;
    const bool hasUv = mesh.uvs.size() == nPos;

    struct EdgeKey {
        uint32_t a, b;
        bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
    };
    struct EdgeHash {
        size_t operator()(const EdgeKey& e) const {
            return (size_t(e.a) * 0x9e3779b97f4a7c15ULL) ^ size_t(e.b);
        }
    };
    std::unordered_map<EdgeKey, uint32_t, EdgeHash> mid;
    mid.reserve(triCount * 2);

    auto midpoint = [&](uint32_t i0, uint32_t i1) -> uint32_t {
        EdgeKey key{std::min(i0, i1), std::max(i0, i1)};
        const auto it = mid.find(key);
        if (it != mid.end()) return it->second;
        const Vec3 p = (mesh.positions[i0] + mesh.positions[i1]) * 0.5f;
        const uint32_t id = uint32_t(mesh.positions.size());
        mesh.positions.push_back(p);
        if (hasN) {
            Vec3 n = mesh.normals[i0] + mesh.normals[i1];
            const float len = length(n);
            mesh.normals.push_back(len > 1e-8f ? n / len : mesh.normals[i0]);
        }
        if (hasUv) mesh.uvs.push_back((mesh.uvs[i0] + mesh.uvs[i1]) * 0.5f);
        mid.emplace(key, id);
        return id;
    };

    std::vector<uint32_t> newIdx;
    newIdx.reserve(triCount * 12);
    for (size_t t = 0; t < triCount; ++t) {
        const uint32_t i0 = mesh.indices[t * 3 + 0];
        const uint32_t i1 = mesh.indices[t * 3 + 1];
        const uint32_t i2 = mesh.indices[t * 3 + 2];
        if (!indexInRange(i0, nPos) || !indexInRange(i1, nPos) || !indexInRange(i2, nPos)) {
            throw std::runtime_error("tessellate: triangle index out of range");
        }
        const uint32_t m01 = midpoint(i0, i1);
        const uint32_t m12 = midpoint(i1, i2);
        const uint32_t m20 = midpoint(i2, i0);
        newIdx.insert(newIdx.end(), {i0, m01, m20, i1, m12, m01, i2, m20, m12, m01, m12, m20});
    }
    mesh.indices.swap(newIdx);
    mesh.restPositions.clear();
    mesh.restNormals.clear();
    mesh.motionPositions.clear();
    mesh.motionPositionsPacked_.clear();
}

// Split only edges that exceed `targetPx` in screen space (and force midpoints
// on shared edges so neighbors stay watertight). True spatial dicing — unlike
// uniform 1→4 on the whole mesh when only the longest edge is large.
bool subdivideLinearScreenAdaptiveOnce(Mesh& mesh, const Mat4& objToWorld, const Mat4& worldToCam,
                                       const CameraData& cam, float aspect, int resX, int resY,
                                       float targetPx) {
    const size_t triCount = mesh.indices.size() / 3;
    if (triCount == 0) return false;
    const size_t nPos = mesh.positions.size();
    const bool hasN = mesh.normals.size() == nPos;
    const bool hasUv = mesh.uvs.size() == nPos;

    struct EdgeKey {
        uint32_t a, b;
        bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
    };
    struct EdgeHash {
        size_t operator()(const EdgeKey& e) const {
            return (size_t(e.a) * 0x9e3779b97f4a7c15ULL) ^ size_t(e.b);
        }
    };

    std::unordered_map<EdgeKey, uint32_t, EdgeHash> mid;
    mid.reserve(triCount);
    auto edgeKey = [](uint32_t a, uint32_t b) {
        return EdgeKey{std::min(a, b), std::max(a, b)};
    };

    // Mark edges that are too long on screen.
    std::unordered_map<EdgeKey, char, EdgeHash> longEdge;
    longEdge.reserve(triCount * 2);
    for (size_t t = 0; t < triCount; ++t) {
        const uint32_t i0 = mesh.indices[t * 3 + 0];
        const uint32_t i1 = mesh.indices[t * 3 + 1];
        const uint32_t i2 = mesh.indices[t * 3 + 2];
        if (!indexInRange(i0, nPos) || !indexInRange(i1, nPos) || !indexInRange(i2, nPos)) continue;
        const float e01 =
            screenEdgePixels(mesh.positions[i0], mesh.positions[i1], objToWorld, worldToCam, cam, aspect,
                             resX, resY);
        const float e12 =
            screenEdgePixels(mesh.positions[i1], mesh.positions[i2], objToWorld, worldToCam, cam, aspect,
                             resX, resY);
        const float e20 =
            screenEdgePixels(mesh.positions[i2], mesh.positions[i0], objToWorld, worldToCam, cam, aspect,
                             resX, resY);
        if (e01 > targetPx) longEdge[edgeKey(i0, i1)] = 1;
        if (e12 > targetPx) longEdge[edgeKey(i1, i2)] = 1;
        if (e20 > targetPx) longEdge[edgeKey(i2, i0)] = 1;
    }
    if (longEdge.empty()) return false;

    auto midpoint = [&](uint32_t i0, uint32_t i1) -> uint32_t {
        EdgeKey key = edgeKey(i0, i1);
        const auto it = mid.find(key);
        if (it != mid.end()) return it->second;
        const Vec3 p = (mesh.positions[i0] + mesh.positions[i1]) * 0.5f;
        const uint32_t id = uint32_t(mesh.positions.size());
        mesh.positions.push_back(p);
        if (hasN) {
            Vec3 n = mesh.normals[i0] + mesh.normals[i1];
            const float len = length(n);
            mesh.normals.push_back(len > 1e-8f ? n / len : mesh.normals[i0]);
        }
        if (hasUv) mesh.uvs.push_back((mesh.uvs[i0] + mesh.uvs[i1]) * 0.5f);
        mid.emplace(key, id);
        return id;
    };

    auto isLong = [&](uint32_t a, uint32_t b) { return longEdge.find(edgeKey(a, b)) != longEdge.end(); };

    std::vector<uint32_t> newIdx;
    newIdx.reserve(triCount * 12);
    for (size_t t = 0; t < triCount; ++t) {
        const uint32_t i0 = mesh.indices[t * 3 + 0];
        const uint32_t i1 = mesh.indices[t * 3 + 1];
        const uint32_t i2 = mesh.indices[t * 3 + 2];
        if (!indexInRange(i0, nPos) || !indexInRange(i1, nPos) || !indexInRange(i2, nPos)) continue;
        const bool b01 = isLong(i0, i1);
        const bool b12 = isLong(i1, i2);
        const bool b20 = isLong(i2, i0);
        const int bits = (b01 ? 1 : 0) | (b12 ? 2 : 0) | (b20 ? 4 : 0);
        if (bits == 0) {
            newIdx.insert(newIdx.end(), {i0, i1, i2});
            continue;
        }
        const uint32_t m01 = b01 ? midpoint(i0, i1) : 0;
        const uint32_t m12 = b12 ? midpoint(i1, i2) : 0;
        const uint32_t m20 = b20 ? midpoint(i2, i0) : 0;
        switch (bits) {
            case 1:  // split i0-i1
                newIdx.insert(newIdx.end(), {i0, m01, i2, m01, i1, i2});
                break;
            case 2:  // split i1-i2
                newIdx.insert(newIdx.end(), {i0, i1, m12, i0, m12, i2});
                break;
            case 4:  // split i2-i0
                newIdx.insert(newIdx.end(), {i0, i1, m20, m20, i1, i2});
                break;
            case 3:  // i0-i1, i1-i2
                newIdx.insert(newIdx.end(), {i0, m01, i2, m01, i1, m12, m01, m12, i2});
                break;
            case 5:  // i0-i1, i2-i0
                newIdx.insert(newIdx.end(), {i0, m01, m20, m01, i1, i2, m20, m01, i2});
                break;
            case 6:  // i1-i2, i2-i0
                newIdx.insert(newIdx.end(), {i0, i1, m12, i0, m12, m20, m20, m12, i2});
                break;
            default:  // all three — classic 1→4
                newIdx.insert(newIdx.end(),
                              {i0, m01, m20, i1, m12, m01, i2, m20, m12, m01, m12, m20});
                break;
        }
    }
    mesh.indices.swap(newIdx);
    mesh.restPositions.clear();
    mesh.restNormals.clear();
    mesh.motionPositions.clear();
    mesh.motionPositionsPacked_.clear();
    return true;
}

float maxScreenEdge(const Mesh& mesh, const Mat4& objToWorld, const Mat4& worldToCam,
                    const CameraData& cam, float aspect, int resX, int resY) {
    float m = 0.0f;
    const size_t nPos = mesh.positions.size();
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        const uint32_t ia = mesh.indices[t + 0];
        const uint32_t ib = mesh.indices[t + 1];
        const uint32_t ic = mesh.indices[t + 2];
        if (!indexInRange(ia, nPos) || !indexInRange(ib, nPos) || !indexInRange(ic, nPos)) continue;
        const Vec3& a = mesh.positions[ia];
        const Vec3& b = mesh.positions[ib];
        const Vec3& c = mesh.positions[ic];
        m = std::max(m, screenEdgePixels(a, b, objToWorld, worldToCam, cam, aspect, resX, resY));
        m = std::max(m, screenEdgePixels(b, c, objToWorld, worldToCam, cam, aspect, resX, resY));
        m = std::max(m, screenEdgePixels(c, a, objToWorld, worldToCam, cam, aspect, resX, resY));
    }
    return m;
}

void refineLinear(Mesh& mesh, int maxIter, bool screenAdaptive, float dicingQuality,
                  const InstanceData* diceInst, const CameraData& cam, const Mat4& worldToCam,
                  float aspect, int resX, int resY) {
    maxIter = clampSubdivLevelsForBudget(maxIter, mesh.triangleCount());
    if (maxIter <= 0) return;
    if (mesh.normals.size() != mesh.positions.size()) {
        mesh.normals.clear();
        mesh.computeNormalsIfMissing();
    }
    // Karma: dicing_quality 1 ≈ 1 micropolygon per pixel → target edge ≈ 1/quality px.
    const float quality = std::max(1.0e-3f, dicingQuality);
    const float targetPx = 1.0f / quality;

    if (screenAdaptive && diceInst) {
        const float probe =
            maxScreenEdge(mesh, diceInst->xform, worldToCam, cam, aspect, resX, resY);
        if (!(probe > 0.0f)) {
            logWarning("tessellate: screen adaptive projection failed — falling back to uniform");
            screenAdaptive = false;
        }
    }

    for (int i = 0; i < maxIter; ++i) {
        if (mesh.triangleCount() * 4 > kMaxTessTriangles) {
            logWarning("tessellate: linear subdiv stopped at budget");
            break;
        }
        if (screenAdaptive && diceInst) {
            if (!subdivideLinearScreenAdaptiveOnce(mesh, diceInst->xform, worldToCam, cam, aspect, resX,
                                                   resY, targetPx))
                break;
        } else {
            subdivideLinearOnce(mesh);
        }
    }
}

#if SOLSTICE_HAVE_OPENSUBDIV
struct OsdVertex {
    OsdVertex() = default;
    explicit OsdVertex(const Vec3& p) : x(p.x), y(p.y), z(p.z) {}
    void Clear(void* = nullptr) { x = y = z = 0.0f; }
    void AddWithWeight(const OsdVertex& src, float w) {
        x += w * src.x;
        y += w * src.y;
        z += w * src.z;
    }
    Vec3 asVec3() const { return Vec3(x, y, z); }
    float x = 0, y = 0, z = 0;
};

bool refineCatclarkOpenSubdiv(Mesh& mesh, int levels) {
    if (levels <= 0 || mesh.indices.size() < 3) return false;
    levels = clampSubdivLevelsForBudget(levels, mesh.triangleCount());
    if (levels <= 0) return false;
    namespace Far = OpenSubdiv::Far;
    // Build triangle faces (3 verts each). OSD catclark on tris is supported but
    // callers should prefer linear for pure-tri cages.
    const int nFaces = int(mesh.indices.size() / 3);
    std::vector<int> vertsPerFace(size_t(nFaces), 3);
    std::vector<int> faceVerts(mesh.indices.begin(), mesh.indices.end());

    Far::TopologyDescriptor desc;
    desc.numVertices = int(mesh.positions.size());
    desc.numFaces = nFaces;
    desc.numVertsPerFace = vertsPerFace.data();
    desc.vertIndicesPerFace = faceVerts.data();

    Far::TopologyRefiner* refiner =
        Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(
            desc, Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Options(
                      OpenSubdiv::Sdc::SCHEME_CATMARK,
                      OpenSubdiv::Sdc::Options()));
    if (!refiner) return false;

    Far::TopologyRefiner::UniformOptions uopt(levels);
    uopt.fullTopologyInLastLevel = true;
    refiner->RefineUniform(uopt);

    const int nRefined = refiner->GetNumVerticesTotal();
    std::vector<OsdVertex> refined(static_cast<size_t>(nRefined));
    for (size_t i = 0; i < mesh.positions.size(); ++i) refined[i] = OsdVertex(mesh.positions[i]);

    Far::PrimvarRefiner primvar(*refiner);
    OsdVertex* src = refined.data();
    for (int lvl = 1; lvl <= levels; ++lvl) {
        OsdVertex* dst = src + refiner->GetLevel(lvl - 1).GetNumVertices();
        primvar.Interpolate(lvl, src, dst);
        src = dst;
    }

    const Far::TopologyLevel& last = refiner->GetLevel(levels);
    const int nVerts = last.GetNumVertices();
    const int vertOffset = refiner->GetNumVerticesTotal() - nVerts;
    Mesh out;
    out.name = mesh.name;
    out.subdivType = mesh.subdivType;
    out.subdivIterations = mesh.subdivIterations;
    out.dicingQuality = mesh.dicingQuality;
    out.boundsPadding = mesh.boundsPadding;
    out.positions.resize(size_t(nVerts));
    for (int i = 0; i < nVerts; ++i) out.positions[size_t(i)] = refined[size_t(vertOffset + i)].asVec3();

    // Emit triangles from refined faces (catmark may produce quads).
    for (int f = 0; f < last.GetNumFaces(); ++f) {
        Far::ConstIndexArray fv = last.GetFaceVertices(f);
        if (fv.size() < 3) continue;
        for (int i = 1; i + 1 < fv.size(); ++i) {
            out.indices.push_back(uint32_t(fv[0]));
            out.indices.push_back(uint32_t(fv[i]));
            out.indices.push_back(uint32_t(fv[i + 1]));
        }
    }
    out.computeNormalsIfMissing();
    out.computeBounds();
    mesh = std::move(out);
    delete refiner;
    return true;
}
#endif

void refineCatclark(Mesh& mesh, int levels, bool screenAdaptive, float dicingQuality,
                    const InstanceData* diceInst, const CameraData& cam, const Mat4& worldToCam,
                    float aspect, int resX, int resY) {
    int useLevels = clampSubdivLevelsForBudget(levels, mesh.triangleCount());
    if (screenAdaptive && diceInst && useLevels > 0) {
        // Prefer true screen-adaptive linear splits for dicing. Catclark OSD is
        // uniform-only; estimate a level count from screen size as a fallback.
        const float quality = std::max(1.0e-3f, dicingQuality);
        const float targetPx = 1.0f / quality;
        float maxPx = maxScreenEdge(mesh, diceInst->xform, worldToCam, cam, aspect, resX, resY);
        if (!(maxPx > 0.0f)) {
            logWarning("tessellate: screen adaptive projection failed — using uniform catclark");
        } else {
            // Triangle cages already fall back to linear adaptive in tessellateOne.
            // For rare quad cages keep a level estimate for OSD.
            int est = 0;
            while (est < useLevels && maxPx > targetPx) {
                maxPx *= 0.5f;
                ++est;
            }
            useLevels = est;
        }
    }
    if (useLevels <= 0) return;

#if SOLSTICE_HAVE_OPENSUBDIV
    if (refineCatclarkOpenSubdiv(mesh, useLevels)) return;
    logWarning("tessellate: OpenSubdiv catclark failed — falling back to linear");
#else
    logWarning("tessellate: OpenSubdiv not in this build — catclark falls back to linear");
#endif
    refineLinear(mesh, useLevels, false, dicingQuality, diceInst, cam, worldToCam, aspect, resX, resY);
}

MeshPtr tessellateOne(Mesh cage, const Material& mat, const Scene& scene,
                      const std::vector<const InstanceData*>& instances, const CameraData& cam) {
    const RenderSettingsData& rs = scene.settings;
    const Mat4 w2c = worldToCamera(cam);
    const float padFrac = std::max(0.0f, rs.frustumPadding) * 0.01f;
    const int resX = std::max(1, rs.resolutionX);
    const int resY = std::max(1, rs.resolutionY);
    const float aspect = float(resX) / float(resY);

    const bool needDisp = materialHasGeometricDisplacement(mat);
    // Tessellation exists to feed displacement (and Pref). Without a displace
    // shader, leave the authored cage alone — default catclark/3 must not explode
    // every primitive in the default scene.
    if (!needDisp) {
        return std::make_shared<Mesh>(std::move(cage));
    }

    bool inFrustum = true;
    if (rs.frustumCull) {
        inFrustum = meshVisibleInFrustum(cage, instances, cam, w2c, aspect, padFrac);
    }

    // Outside frustum: displace cage only (no subdiv), even if subdiv authored.
    if (rs.frustumCull && !inFrustum) {
        return displaceMeshOnly(std::move(cage), mat, scene);
    }

    const int authoredSubdivType = cage.subdivType;
    const int authoredSubdivIterations = cage.subdivIterations;
    const float authoredDicingQuality = cage.dicingQuality;
    const float authoredPad = cage.boundsPadding;

    // Steal cage storage — no parallel cage+work copy while refining.
    auto work = std::make_shared<Mesh>(std::move(cage));
    work->motionPositionsPacked_.clear();
    const InstanceData* diceInst = nearestInstance(instances, cam);
    const bool adaptive = rs.screenAdaptive != 0;
    int type = work->subdivType;
    if (type == kSubdivCatclark && !meshHasQuadsHint(*work) && meshIsTriangleOnly(*work)) {
        type = kSubdivLinear;  // whole-mesh tris → linear
    }

    const int cap = clampSubdivLevelsForBudget(work->subdivIterations, work->triangleCount());
    try {
        if (type == kSubdivNone || cap == 0) {
            // no subdiv
        } else if (type == kSubdivCatclark) {
            refineCatclark(*work, cap, adaptive, work->dicingQuality, diceInst, cam, w2c, aspect, resX,
                           resY);
        } else {
            refineLinear(*work, cap, adaptive, work->dicingQuality, diceInst, cam, w2c, aspect, resX, resY);
        }
    } catch (const std::bad_alloc&) {
        logWarning("tessellate: subdiv ran out of memory — displacing densest level kept so far");
    }

    if (needDisp) {
        Material m = mat;
        m.subdivIterations = cap;
        const float pad = std::max(work->boundsPadding, authoredPad);
        try {
            MeshPtr displaced = displaceMeshOnly(std::move(*work), m, scene);
            displaced->boundsPadding = pad;
            if (displaced->boundsPadding > 0.0f) displaced->computeBounds();
            displaced->subdivType = authoredSubdivType;
            displaced->subdivIterations = authoredSubdivIterations;
            displaced->dicingQuality = authoredDicingQuality;
            return displaced;
        } catch (const std::bad_alloc&) {
            logWarning("tessellate: displace OOM — returning densified cage without Pref lock");
            work->computeBounds();
            return work;
        }
    }
    work->computeBounds();
    return work;
}

}  // namespace

void tessellateSceneForRender(Scene& scene, const CameraData& dicingCamera) {
    if (scene.meshes.empty() || scene.instances.empty()) return;

    std::vector<std::vector<const InstanceData*>> byMesh(scene.meshes.size());
    for (const InstanceData& inst : scene.instances) {
        if (inst.meshIndex < 0 || size_t(inst.meshIndex) >= scene.meshes.size()) continue;
        byMesh[size_t(inst.meshIndex)].push_back(&inst);
    }

    for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
        MeshPtr& mesh = scene.meshes[mi];
        if (!mesh) continue;
        const auto& insts = byMesh[mi];
        if (insts.empty()) continue;

        // Same shader on all instances of a mesh (spec). Use first.
        const int matIndex = insts[0]->materialIndex;
        Material mat;
        if (matIndex >= 0 && size_t(matIndex) < scene.materials.size())
            mat = scene.materials[size_t(matIndex)];

        const bool need = materialHasGeometricDisplacement(mat);
        if (!need) continue;

        const int authoredIterations = mesh->subdivIterations;
        const std::string meshName = mesh->name;
        try {
            // Steal cage buffers into a local Mesh while keeping the shared_ptr
            // slot non-null (empty shell). Avoids a parallel cage+work copy and
            // never leaves a nullptr mesh index for Embree/OptiX.
            Mesh working = std::move(*mesh);
            MeshPtr tess = tessellateOne(std::move(working), mat, scene, insts, dicingCamera);
            if (matIndex >= 0 && size_t(matIndex) < scene.materials.size())
                scene.materials[size_t(matIndex)].subdivIterations = authoredIterations;
            if (tess) mesh = std::move(tess);
        } catch (const std::bad_alloc&) {
            logError("tessellate: out of memory on mesh '" + meshName +
                     "' — keeping emptied cage slot (lower Subdiv Iterations)");
        } catch (const std::exception& ex) {
            logError(std::string("tessellate: ") + ex.what() + " on mesh '" + meshName + "'");
        }
    }

    // Critical: finalize() rebuilt meshViews_ from pre-displace cages. After we
    // replace meshes, those views dangle (UAF crash) and restPositions stay
    // nullptr in SceneView → triplanar samples displaced P (seam artifacts).
    scene.finalize();
    logInfo("tessellate: render tessellation complete");
}

std::string dicingCameraFingerprint(const CameraData& cam) {
    // Quantize so tiny orbit noise does not thrash re-tess; still reacts to
    // meaningful dolly / tumble that changes projected edge length.
    const Vec3 eye = transformPoint(cam.cameraToWorld, Vec3(0.0f));
    const Vec3 fwd(-cam.cameraToWorld.at(0, 2), -cam.cameraToWorld.at(1, 2), -cam.cameraToWorld.at(2, 2));
    auto q = [](float v, float step) {
        return int(std::lround(double(v) / double(step)));
    };
    // ~2cm / ~2° steps — coarse enough for IPR, fine enough for close-up densify.
    std::string key;
    key += "e=";
    key += std::to_string(q(eye.x, 0.02f));
    key += ",";
    key += std::to_string(q(eye.y, 0.02f));
    key += ",";
    key += std::to_string(q(eye.z, 0.02f));
    key += ";f=";
    key += std::to_string(q(fwd.x, 0.03f));
    key += ",";
    key += std::to_string(q(fwd.y, 0.03f));
    key += ",";
    key += std::to_string(q(fwd.z, 0.03f));
    key += ";fl=";
    key += std::to_string(q(cam.focalLength, 0.5f));
    key += ";sw=";
    key += std::to_string(q(cam.sensorWidth, 0.5f));
    return key;
}

std::string tessellationFingerprint(const Scene& scene) {
    const RenderSettingsData& rs = scene.settings;
    std::string key;
    key.reserve(256);
    key += "fc=";
    key += std::to_string(rs.frustumCull);
    key += ";fp=";
    key += std::to_string(rs.frustumPadding);
    key += ";sa=";
    key += std::to_string(rs.screenAdaptive);
    key += ";dc=";
    key += std::to_string(rs.dicingCameraMode);
    key += ";rx=";
    key += std::to_string(rs.resolutionX);
    key += ";ry=";
    key += std::to_string(rs.resolutionY);
    key += ";";

    for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
        const MeshPtr& mesh = scene.meshes[mi];
        if (!mesh) continue;
        // Find a material that drives displacement for this mesh.
        const Material* mat = nullptr;
        for (const InstanceData& inst : scene.instances) {
            if (inst.meshIndex != int(mi)) continue;
            if (inst.materialIndex < 0 || size_t(inst.materialIndex) >= scene.materials.size()) continue;
            mat = &scene.materials[size_t(inst.materialIndex)];
            break;
        }
        if (!mat || !materialHasGeometricDisplacement(*mat)) continue;
        key += "m";
        key += std::to_string(mi);
        key += ":st=";
        key += std::to_string(mesh->subdivType);
        key += ";si=";
        key += std::to_string(mesh->subdivIterations);
        key += ";dq=";
        key += std::to_string(mesh->dicingQuality);
        key += ";bp=";
        key += std::to_string(mesh->boundsPadding);
        key += ";dt=";
        key += std::to_string(mat->displacementTex);
        key += ";dp=";
        key += std::to_string(mat->displacementProc);
        key += ";dh=";
        key += std::to_string(mat->displacementHeight);
        key += ";ds=";
        key += std::to_string(mat->displacementScale);
        key += ";dz=";
        key += std::to_string(mat->displacementZeroValue);
        key += ";";
    }
    return key;
}

}  // namespace sol
