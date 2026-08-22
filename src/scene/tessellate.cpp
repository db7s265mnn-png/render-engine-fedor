// Render-time tessellation for displacement: none / linear / catclark,
// frustum cull, screen-space dicing.
#include "scene/tessellate.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/log.h"
#include "core/thread_pool.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPENSUBDIV
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/topologyRefinerFactory.h>
#endif

namespace sol {
namespace {

// Safety ceiling only — prefer freeing previous-render memory over clamping.
// Overridden per-scene by Render Settings → Dicing Poly Limit (M).
constexpr size_t kDefaultTessTriangles = 10000000ull;  // 10M
constexpr size_t kHardMaxTessTriangles = 200000000ull;  // absolute ceiling

size_t tessTriangleBudget(const RenderSettingsData& rs) {
    const int millions = std::clamp(rs.dicingPolyLimitM, 1, 200);
    const size_t want = size_t(millions) * 1000000ull;
    return std::min(want, kHardMaxTessTriangles);
}

// Leave 2 cores free for UI / OS while densifying (user request).
// Returns the desired *total* concurrent chunk runners (including the caller).
int diceParallelism(int settingsThreads) {
    int hw = int(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 4;
    const int want = settingsThreads > 0 ? settingsThreads : hw;
    return std::max(1, want - 2);
}

// ThreadPool(N) spawns N workers and also runs chunks on the calling thread
// (N+1 runners). Size the ctor arg so total concurrency ≈ diceParallelism.
int dicePoolThreadCount(int parallelism) {
    if (parallelism <= 1) return 1;
    return std::max(1, parallelism - 1);
}

struct TessRuntime {
    ThreadPool* pool = nullptr;
    TessProgressFn progress;
    int meshIndex = 0;
    int meshCount = 1;
    std::string meshName;

    int globalPct(float meshFrac) const {
        const float n = float(std::max(1, meshCount));
        const float g = (float(meshIndex) + std::clamp(meshFrac, 0.0f, 1.0f)) / n;
        return int(std::clamp(g * 100.0f, 0.0f, 99.0f));
    }

    void reportPct(int pct, size_t tris) const {
        if (!progress) return;
        char buf[192];
        std::snprintf(buf, sizeof(buf), "Dicing %d%%  ·  %.2fM tris", std::clamp(pct, 0, 100),
                      double(tris) / 1.0e6);
        progress(std::string(buf));
    }

    void reportMeshFrac(float meshFrac, size_t tris) const { reportPct(globalPct(meshFrac), tris); }
};

int clampSubdivLevelsForBudget(int requested, size_t triCount, size_t budget) {
    const int want = std::max(0, requested);
    if (want == 0 || triCount == 0) return 0;
    const size_t lim = std::max<size_t>(1, budget);
    size_t est = triCount;
    int levels = 0;
    while (levels < want) {
        if (est > lim / 4) break;
        est *= 4;
        ++levels;
    }
    if (levels < want) {
        logWarning("tessellate: clamped subdiv iterations " + std::to_string(want) + " → " +
                   std::to_string(levels) + " (triangle budget " + std::to_string(lim) + ")");
    }
    return levels;
}

int clampSubdivLevelsForBudget(int requested, size_t triCount) {
    return clampSubdivLevelsForBudget(requested, triCount, kDefaultTessTriangles);
}

bool indexInRange(uint32_t idx, size_t count) { return size_t(idx) < count; }

struct EdgeKey {
    uint32_t a, b;
    bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
};
struct EdgeHash {
    size_t operator()(const EdgeKey& e) const {
        return (size_t(e.a) * 0x9e3779b97f4a7c15ULL) ^ size_t(e.b);
    }
};
EdgeKey makeEdgeKey(uint32_t a, uint32_t b) { return EdgeKey{std::min(a, b), std::max(a, b)}; }

bool meshIsTriangleOnly(const Mesh& mesh) {
    // True when every authored face is a triangle (or there is no polygon cage).
    if (!mesh.hasPolygonCage()) return !mesh.indices.empty();
    for (uint32_t c : mesh.faceVertexCounts) {
        if (c != 3) return false;
    }
    return !mesh.faceVertexCounts.empty();
}

bool meshHasQuadsHint(const Mesh& mesh) {
    if (!mesh.hasPolygonCage()) return false;
    for (uint32_t c : mesh.faceVertexCounts) {
        if (c == 4) return true;
    }
    return false;
}

Mat4 worldToCamera(const CameraData& cam) {
    return inverse(cam.cameraToWorld);
}

constexpr float kDiceNearViewZ = 1.0e-4f;

// Project object-space point through instance xform into NDC (xy in ~[-1,1]).
bool projectToNdc(Vec3 pObject, const Mat4& objToWorld, const Mat4& worldToCam, const CameraData& cam,
                  float aspect, float& ndcX, float& ndcY, float& viewZ) {
    const Vec3 pWorld = transformPoint(objToWorld, pObject);
    const Vec3 pCam = transformPoint(worldToCam, pWorld);
    viewZ = -pCam.z;  // camera looks down -Z
    if (!(viewZ > 0.0f)) return false;
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

bool projectCamToNdc(Vec3 pCam, const CameraData& cam, float aspect, float& ndcX, float& ndcY) {
    const float viewZ = -pCam.z;
    // Any point in front of the camera. Clip hits sit at viewZ == kDiceNearViewZ;
    // allow a little FP slack below that.
    if (!(viewZ > 0.0f)) return false;
    const float focal = std::max(1.0e-3f, cam.focalLength);
    const float sensorW = std::max(1.0e-3f, cam.sensorWidth);
    const float sensorH = sensorW / std::max(0.01f, aspect);
    const float halfW = 0.5f * sensorW * (viewZ / focal);
    const float halfH = 0.5f * sensorH * (viewZ / focal);
    if (!(halfW > 1e-8f) || !(halfH > 1e-8f)) return false;
    ndcX = pCam.x / halfW;
    ndcY = pCam.y / halfH;
    return true;
}

// Clip camera-space segment to viewZ >= near. Returns false if fully behind.
bool clipEdgeToNearCam(Vec3 aCam, Vec3 bCam, Vec3& aOut, Vec3& bOut) {
    const float za = -aCam.z;
    const float zb = -bCam.z;
    const bool aFront = za >= kDiceNearViewZ;
    const bool bFront = zb >= kDiceNearViewZ;
    if (aFront && bFront) {
        aOut = aCam;
        bOut = bCam;
        return true;
    }
    if (!aFront && !bFront) return false;
    // z(t) = a.z + t*(b.z-a.z) = -near → viewZ(t) = near
    const float denom = bCam.z - aCam.z;
    if (std::fabs(denom) < 1.0e-20f) return false;
    const float t = std::clamp((-kDiceNearViewZ - aCam.z) / denom, 0.0f, 1.0f);
    Vec3 hit = aCam + (bCam - aCam) * t;
    // Nudge onto the near plane so FP error cannot push the hit behind.
    hit.z = -kDiceNearViewZ;
    if (aFront) {
        aOut = aCam;
        bOut = hit;
    } else {
        aOut = hit;
        bOut = bCam;
    }
    return true;
}

float ndcEdgePixels(float ax, float ay, float bx, float by, int resX, int resY) {
    const float dx = (bx - ax) * 0.5f * float(resX);
    const float dy = (by - ay) * 0.5f * float(resY);
    return std::sqrt(dx * dx + dy * dy);
}

bool pointInPaddedFrustum(float ndcX, float ndcY, float padFrac) {
    const float lim = 1.0f + padFrac;
    return ndcX >= -lim && ndcX <= lim && ndcY >= -lim && ndcY <= lim;
}

float screenEdgePixels(Vec3 aObj, Vec3 bObj, const Mat4& objToWorld, const Mat4& worldToCam,
                       const CameraData& cam, float aspect, int resX, int resY) {
    const Vec3 aCam = transformPoint(worldToCam, transformPoint(objToWorld, aObj));
    const Vec3 bCam = transformPoint(worldToCam, transformPoint(objToWorld, bObj));
    Vec3 ca, cb;
    if (!clipEdgeToNearCam(aCam, bCam, ca, cb)) return 0.0f;
    float ax, ay, bx, by;
    if (!projectCamToNdc(ca, cam, aspect, ax, ay) || !projectCamToNdc(cb, cam, aspect, bx, by))
        return 0.0f;
    return ndcEdgePixels(ax, ay, bx, by, resX, resY);
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

// Split only the marked edges (red-green style). Shared midpoints keep seams
// watertight when a neighbor also marks the same edge.
bool subdivideMarkedEdgesOnce(
    Mesh& mesh, const std::unordered_map<EdgeKey, char, EdgeHash>& splitEdge) {
    if (splitEdge.empty()) return false;
    const size_t triCount = mesh.indices.size() / 3;
    if (triCount == 0) return false;
    const size_t nPos = mesh.positions.size();
    const bool hasN = mesh.normals.size() == nPos;
    const bool hasUv = mesh.uvs.size() == nPos;

    std::unordered_map<EdgeKey, uint32_t, EdgeHash> mid;
    mid.reserve(splitEdge.size() * 2);

    auto midpoint = [&](uint32_t i0, uint32_t i1) -> uint32_t {
        EdgeKey key = makeEdgeKey(i0, i1);
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

    auto isSplit = [&](uint32_t a, uint32_t b) {
        return splitEdge.find(makeEdgeKey(a, b)) != splitEdge.end();
    };

    std::vector<uint32_t> newIdx;
    newIdx.reserve(triCount * 12);
    auto emit3 = [&](uint32_t a, uint32_t b, uint32_t c) {
        newIdx.push_back(a);
        newIdx.push_back(b);
        newIdx.push_back(c);
    };
    for (size_t t = 0; t < triCount; ++t) {
        const uint32_t i0 = mesh.indices[t * 3 + 0];
        const uint32_t i1 = mesh.indices[t * 3 + 1];
        const uint32_t i2 = mesh.indices[t * 3 + 2];
        if (!indexInRange(i0, nPos) || !indexInRange(i1, nPos) || !indexInRange(i2, nPos)) continue;
        const bool b01 = isSplit(i0, i1);
        const bool b12 = isSplit(i1, i2);
        const bool b20 = isSplit(i2, i0);
        const int bits = (b01 ? 1 : 0) | (b12 ? 2 : 0) | (b20 ? 4 : 0);
        if (bits == 0) {
            emit3(i0, i1, i2);
            continue;
        }
        const uint32_t m01 = b01 ? midpoint(i0, i1) : 0;
        const uint32_t m12 = b12 ? midpoint(i1, i2) : 0;
        const uint32_t m20 = b20 ? midpoint(i2, i0) : 0;
        switch (bits) {
            case 1:
                emit3(i0, m01, i2);
                emit3(m01, i1, i2);
                break;
            case 2:
                emit3(i0, i1, m12);
                emit3(i0, m12, i2);
                break;
            case 4:
                emit3(i0, i1, m20);
                emit3(m20, i1, i2);
                break;
            case 3:
                emit3(i0, m01, i2);
                emit3(m01, i1, m12);
                emit3(m01, m12, i2);
                break;
            case 5:
                emit3(i0, m01, m20);
                emit3(m01, i1, i2);
                emit3(m20, m01, i2);
                break;
            case 6:
                emit3(i0, i1, m12);
                emit3(i0, m12, m20);
                emit3(m20, m12, i2);
                break;
            default:
                emit3(i0, m01, m20);
                emit3(i1, m12, m01);
                emit3(i2, m20, m12);
                emit3(m01, m12, m20);
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

void subdivideLinearOnce(Mesh& mesh) {
    const size_t triCount = mesh.indices.size() / 3;
    if (triCount == 0) return;
    const size_t nPos = mesh.positions.size();

    std::unordered_map<EdgeKey, char, EdgeHash> allEdges;
    allEdges.reserve(triCount * 2);
    for (size_t t = 0; t < triCount; ++t) {
        const uint32_t i0 = mesh.indices[t * 3 + 0];
        const uint32_t i1 = mesh.indices[t * 3 + 1];
        const uint32_t i2 = mesh.indices[t * 3 + 2];
        if (!indexInRange(i0, nPos) || !indexInRange(i1, nPos) || !indexInRange(i2, nPos)) {
            throw std::runtime_error("tessellate: triangle index out of range");
        }
        allEdges[makeEdgeKey(i0, i1)] = 1;
        allEdges[makeEdgeKey(i1, i2)] = 1;
        allEdges[makeEdgeKey(i2, i0)] = 1;
    }
    subdivideMarkedEdgesOnce(mesh, allEdges);
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
        if (e01 > targetPx) longEdge[makeEdgeKey(i0, i1)] = 1;
        if (e12 > targetPx) longEdge[makeEdgeKey(i1, i2)] = 1;
        if (e20 > targetPx) longEdge[makeEdgeKey(i2, i0)] = 1;
    }
    return subdivideMarkedEdgesOnce(mesh, longEdge);
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

struct VertProj {
    Vec3 cam{0.0f, 0.0f, 0.0f};  // camera-space position (for near-plane clip)
    float nx = 0.0f;
    float ny = 0.0f;
    uint8_t ok = 0;  // 1 if in front of near plane and projectable
};

void projectAllVertices(const Mesh& mesh, const Mat4& objToWorld, const Mat4& worldToCam,
                        const CameraData& cam, float aspect, std::vector<VertProj>& out,
                        ThreadPool* pool) {
    const size_t nPos = mesh.positions.size();
    out.resize(nPos);
    if (nPos == 0) return;
    auto one = [&](size_t i) {
        const Vec3 pCam = transformPoint(worldToCam, transformPoint(objToWorld, mesh.positions[i]));
        out[i].cam = pCam;
        float nx, ny;
        if (projectCamToNdc(pCam, cam, aspect, nx, ny)) {
            out[i].nx = nx;
            out[i].ny = ny;
            out[i].ok = 1;
        } else {
            out[i].nx = 0.0f;
            out[i].ny = 0.0f;
            out[i].ok = 0;
        }
    };
    constexpr size_t kParallelMin = 256;
    if (pool && pool->threadCount() > 1 && nPos > kParallelMin) {
        pool->parallelFor(int(nPos), [&](int i, int) { one(size_t(i)); });
    } else {
        for (size_t i = 0; i < nPos; ++i) one(i);
    }
}

// Screen length of the *visible* portion (clipped to the near plane). Never invent
// a fake "infinite" length — that used to drive Screen Adaptive to the poly limit
// on close-ups where many cage edges straddled the camera.
float screenEdgePixelsFromProj(const VertProj& a, const VertProj& b, const CameraData& cam, float aspect,
                               int resX, int resY) {
    Vec3 ca, cb;
    if (!clipEdgeToNearCam(a.cam, b.cam, ca, cb)) return 0.0f;
    float ax, ay, bx, by;
    if (!projectCamToNdc(ca, cam, aspect, ax, ay) || !projectCamToNdc(cb, cam, aspect, bx, by))
        return 0.0f;
    // Cap pathological near-camera projections (NDC blows up as viewZ → 0).
    const float cap = 4.0f * float(std::max(resX, resY));
    return std::min(ndcEdgePixels(ax, ay, bx, by, resX, resY), cap);
}

bool ndcPolyHitsFrustum(const float* xs, const float* ys, int n, float padFrac) {
    if (n <= 0) return false;
    const float lim = 1.0f + padFrac;
    for (int i = 0; i < n; ++i) {
        if (pointInPaddedFrustum(xs[i], ys[i], padFrac)) return true;
    }
    if (n < 2) return false;
    float minx = xs[0], maxx = xs[0], miny = ys[0], maxy = ys[0];
    for (int i = 1; i < n; ++i) {
        minx = std::min(minx, xs[i]);
        maxx = std::max(maxx, xs[i]);
        miny = std::min(miny, ys[i]);
        maxy = std::max(maxy, ys[i]);
    }
    // Fully outside the padded NDC box.
    if (maxx < -lim || minx > lim || maxy < -lim || miny > lim) return false;
    const float samples[][2] = {
        {0.0f, 0.0f}, {lim, lim},  {-lim, lim}, {lim, -lim}, {-lim, -lim},
        {lim, 0.0f},  {-lim, 0.0f}, {0.0f, lim}, {0.0f, -lim},
    };
    if (n >= 3) {
        // Fan from first vertex — clipped near-plane polys are convex.
        for (const auto& s : samples) {
            for (int i = 1; i + 1 < n; ++i) {
                if (pointInTriangle2D(s[0], s[1], xs[0], ys[0], xs[i], ys[i], xs[i + 1], ys[i + 1]))
                    return true;
            }
        }
    }
    // AABB overlaps the frustum. Keep this conservative fallback: huge faces that
    // cover the frame often have extreme NDC verts where point-in-triangle loses
    // precision. Off-screen faces are still rejected by the AABB test above.
    return true;
}

// Clip triangle to the near plane → NDC polygon (0..4 verts).
int clipFaceToNearNdc(const VertProj& a, const VertProj& b, const VertProj& c, const CameraData& cam,
                      float aspect, float* xs, float* ys) {
    const VertProj* v[3] = {&a, &b, &c};
    Vec3 poly[4];
    int n = 0;
    for (int i = 0; i < 3; ++i) {
        const VertProj& p0 = *v[i];
        const VertProj& p1 = *v[(i + 1) % 3];
        const bool in0 = p0.ok != 0;
        const bool in1 = p1.ok != 0;
        if (in0) {
            poly[n++] = p0.cam;
        }
        if (in0 != in1) {
            Vec3 ca, cb;
            if (clipEdgeToNearCam(p0.cam, p1.cam, ca, cb)) {
                // The new vertex is the one that lies on the near plane.
                const Vec3 hit = in0 ? cb : ca;
                if (n == 0 || lengthSquared(poly[n - 1] - hit) > 1.0e-16f) poly[n++] = hit;
            }
        }
        if (n >= 4) break;
    }
    int outN = 0;
    for (int i = 0; i < n; ++i) {
        float nx, ny;
        if (!projectCamToNdc(poly[i], cam, aspect, nx, ny)) continue;
        xs[outN] = nx;
        ys[outN] = ny;
        ++outN;
    }
    return outN;
}

bool faceInPaddedFrustumProj(const VertProj& a, const VertProj& b, const VertProj& c, float padFrac,
                             const CameraData& cam, float aspect) {
    float xs[4], ys[4];
    const int n = clipFaceToNearNdc(a, b, c, cam, aspect, xs, ys);
    return ndcPolyHitsFrustum(xs, ys, n, padFrac);
}

void markFacesInFrustumProjected(const Mesh& mesh, const std::vector<VertProj>& proj, float padFrac,
                                 const CameraData& cam, float aspect, std::vector<uint8_t>& faceMark,
                                 size_t& markedCount, ThreadPool* pool) {
    const size_t triCount = mesh.indices.size() / 3;
    const size_t nPos = proj.size();
    faceMark.assign(triCount, 0);
    markedCount = 0;
    if (triCount == 0) return;

    auto markOne = [&](size_t t) {
        const uint32_t i0 = mesh.indices[t * 3 + 0];
        const uint32_t i1 = mesh.indices[t * 3 + 1];
        const uint32_t i2 = mesh.indices[t * 3 + 2];
        if (!indexInRange(i0, nPos) || !indexInRange(i1, nPos) || !indexInRange(i2, nPos)) return;
        if (faceInPaddedFrustumProj(proj[i0], proj[i1], proj[i2], padFrac, cam, aspect)) faceMark[t] = 1;
    };

    constexpr size_t kParallelMin = 256;
    if (pool && pool->threadCount() > 1 && triCount > kParallelMin) {
        pool->parallelFor(int(triCount), [&](int t, int) { markOne(size_t(t)); });
    } else {
        for (size_t t = 0; t < triCount; ++t) markOne(t);
    }
    for (uint8_t m : faceMark) markedCount += m ? 1u : 0u;
}

void expandFaceMarksOneRing(const Mesh& mesh, std::vector<uint8_t>& faceMark) {
    const size_t triCount = mesh.indices.size() / 3;
    if (faceMark.size() != triCount) return;
    struct EdgeFaces {
        int f0 = -1, f1 = -1;
    };
    std::unordered_map<EdgeKey, EdgeFaces, EdgeHash> edgeFaces;
    edgeFaces.reserve(triCount * 2);
    auto addEdge = [&](uint32_t a, uint32_t b, int fi) {
        EdgeFaces& ef = edgeFaces[makeEdgeKey(a, b)];
        if (ef.f0 < 0)
            ef.f0 = fi;
        else if (ef.f1 < 0 && ef.f0 != fi)
            ef.f1 = fi;
    };
    for (size_t t = 0; t < triCount; ++t) {
        addEdge(mesh.indices[t * 3 + 0], mesh.indices[t * 3 + 1], int(t));
        addEdge(mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2], int(t));
        addEdge(mesh.indices[t * 3 + 2], mesh.indices[t * 3 + 0], int(t));
    }
    std::vector<uint8_t> next = faceMark;
    for (const auto& kv : edgeFaces) {
        const EdgeFaces& ef = kv.second;
        if (ef.f0 < 0 || ef.f1 < 0) continue;
        if (faceMark[size_t(ef.f0)] && !faceMark[size_t(ef.f1)]) next[size_t(ef.f1)] = 1;
        if (faceMark[size_t(ef.f1)] && !faceMark[size_t(ef.f0)]) next[size_t(ef.f0)] = 1;
    }
    faceMark.swap(next);
}

// Uniform face-depth dice (frustum-local generations). No cross-face rate flood.
bool diceMeshByFaceLevels(Mesh& mesh, const std::vector<uint8_t>& faceLevel, size_t polyBudget) {
    const size_t triCount = mesh.indices.size() / 3;
    if (triCount == 0 || faceLevel.size() != triCount) return false;

    bool any = false;
    uint8_t maxL = 0;
    for (uint8_t L : faceLevel) {
        if (L) any = true;
        maxL = std::max(maxL, L);
    }
    if (!any) return false;

    // Budget: 4^L per seeded face.
    std::vector<uint8_t> levels = faceLevel;
    auto estimate = [&]() {
        size_t est = 0;
        for (uint8_t L : levels) est += size_t(1u << (2 * std::min(int(L), 15)));
        return est;
    };
    while (estimate() > polyBudget) {
        if (maxL == 0) break;
        for (uint8_t& L : levels) {
            if (L >= maxL) --L;
        }
        --maxL;
        maxL = 0;
        for (uint8_t L : levels) maxL = std::max(maxL, L);
    }

    const size_t nPos = mesh.positions.size();
    const bool hasN = mesh.normals.size() == nPos;
    const bool hasUv = mesh.uvs.size() == nPos;
    std::unordered_map<EdgeKey, uint32_t, EdgeHash> mid;
    mid.reserve(triCount * 4);

    auto ensureMid = [&](uint32_t i0, uint32_t i1) -> uint32_t {
        const EdgeKey key = makeEdgeKey(i0, i1);
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
    newIdx.reserve(std::min(estimate(), polyBudget) * 3 + 16);

    std::function<void(uint32_t, uint32_t, uint32_t, int)> emitRec;
    emitRec = [&](uint32_t i0, uint32_t i1, uint32_t i2, int L) {
        if (L <= 0) {
            newIdx.push_back(i0);
            newIdx.push_back(i1);
            newIdx.push_back(i2);
            return;
        }
        const uint32_t m01 = ensureMid(i0, i1);
        const uint32_t m12 = ensureMid(i1, i2);
        const uint32_t m20 = ensureMid(i2, i0);
        emitRec(i0, m01, m20, L - 1);
        emitRec(m01, i1, m12, L - 1);
        emitRec(m20, m12, i2, L - 1);
        emitRec(m01, m12, m20, L - 1);
    };

    for (size_t t = 0; t < triCount; ++t) {
        if (!levels[t]) continue;
        const uint32_t i0 = mesh.indices[t * 3 + 0];
        const uint32_t i1 = mesh.indices[t * 3 + 1];
        const uint32_t i2 = mesh.indices[t * 3 + 2];
        if (!indexInRange(i0, nPos) || !indexInRange(i1, nPos) || !indexInRange(i2, nPos)) continue;
        emitRec(i0, i1, i2, int(levels[t]));
    }

    auto hasMid = [&](uint32_t a, uint32_t b) {
        return mid.find(makeEdgeKey(a, b)) != mid.end();
    };
    auto midOf = [&](uint32_t a, uint32_t b) { return mid.find(makeEdgeKey(a, b))->second; };
    auto emit3 = [&](uint32_t a, uint32_t b, uint32_t c) {
        newIdx.push_back(a);
        newIdx.push_back(b);
        newIdx.push_back(c);
    };
    for (size_t t = 0; t < triCount; ++t) {
        if (levels[t]) continue;
        const uint32_t i0 = mesh.indices[t * 3 + 0];
        const uint32_t i1 = mesh.indices[t * 3 + 1];
        const uint32_t i2 = mesh.indices[t * 3 + 2];
        if (!indexInRange(i0, nPos) || !indexInRange(i1, nPos) || !indexInRange(i2, nPos)) continue;
        const bool b01 = hasMid(i0, i1);
        const bool b12 = hasMid(i1, i2);
        const bool b20 = hasMid(i2, i0);
        const int bits = (b01 ? 1 : 0) | (b12 ? 2 : 0) | (b20 ? 4 : 0);
        if (bits == 0) {
            emit3(i0, i1, i2);
            continue;
        }
        const uint32_t m01 = b01 ? midOf(i0, i1) : 0;
        const uint32_t m12 = b12 ? midOf(i1, i2) : 0;
        const uint32_t m20 = b20 ? midOf(i2, i0) : 0;
        switch (bits) {
            case 1: emit3(i0, m01, i2); emit3(m01, i1, i2); break;
            case 2: emit3(i0, i1, m12); emit3(i0, m12, i2); break;
            case 4: emit3(i0, i1, m20); emit3(m20, i1, i2); break;
            case 3: emit3(i0, m01, i2); emit3(m01, i1, m12); emit3(m01, m12, i2); break;
            case 5: emit3(i0, m01, m20); emit3(m01, i1, i2); emit3(m20, m01, i2); break;
            case 6: emit3(i0, i1, m12); emit3(i0, m12, m20); emit3(m20, m12, i2); break;
            default:
                emit3(i0, m01, m20); emit3(i1, m12, m01); emit3(i2, m20, m12); emit3(m01, m12, m20);
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

// Collect edges of marked faces whose screen length exceeds targetPx.
float collectLongEdgesForAdaptive(const Mesh& mesh, const std::vector<uint8_t>& faceMark,
                                  const std::vector<VertProj>& proj, float targetPx,
                                  const CameraData& cam, float aspect, int resX, int resY,
                                  ThreadPool* pool, std::unordered_map<EdgeKey, char, EdgeHash>& splitEdge) {
    const size_t triCount = mesh.indices.size() / 3;
    const size_t nPos = proj.size();
    splitEdge.clear();

    const int workers = pool ? std::max(1, pool->threadCount()) : 1;
    std::vector<std::vector<EdgeKey>> local(size_t(workers + 1));
    std::vector<float> localMax(size_t(workers + 1), 0.0f);
    for (auto& v : local) v.reserve(triCount / size_t(std::max(1, workers)) * 2 + 16);

    auto consider = [&](size_t t, int tid) {
        if (!faceMark[t]) return;
        const uint32_t i0 = mesh.indices[t * 3 + 0];
        const uint32_t i1 = mesh.indices[t * 3 + 1];
        const uint32_t i2 = mesh.indices[t * 3 + 2];
        if (!indexInRange(i0, nPos) || !indexInRange(i1, nPos) || !indexInRange(i2, nPos)) return;
        const int slot = std::clamp(tid, 0, workers);
        // Underfoot bias: binary edge residuals land coarser in screen space near the
        // camera than in mid-field. Cheap fix — only shrink the split threshold for
        // near viewZ faces (no screen-area tests).
        float minViewZ = 1.0e30f;
        int nFront = 0;
        bool anyBehind = false;
        auto account = [&](const VertProj& v) {
            if (!v.ok) {
                anyBehind = true;
                return;
            }
            ++nFront;
            minViewZ = std::min(minViewZ, -v.cam.z);
        };
        account(proj[i0]);
        account(proj[i1]);
        account(proj[i2]);
        // Straddling near-plane faces count as very near.
        if (anyBehind && nFront > 0) minViewZ = std::min(minViewZ, 0.2f);

        float localTarget = targetPx;
        constexpr float kNearBoostEnd = 3.0f;   // metres — full target beyond this
        constexpr float kNearBoostMin = 0.35f;  // ×target at the camera
        if (nFront > 0 && minViewZ < kNearBoostEnd) {
            const float w = std::clamp(minViewZ / kNearBoostEnd, 0.0f, 1.0f);
            const float s = w * w * (3.0f - 2.0f * w);  // smoothstep
            localTarget = targetPx * (kNearBoostMin + (1.0f - kNearBoostMin) * s);
        }
        auto pushIfLong = [&](uint32_t a, uint32_t b) {
            const float px = screenEdgePixelsFromProj(proj[a], proj[b], cam, aspect, resX, resY);
            localMax[size_t(slot)] = std::max(localMax[size_t(slot)], px);
            if (px > localTarget) local[size_t(slot)].push_back(makeEdgeKey(a, b));
        };
        pushIfLong(i0, i1);
        pushIfLong(i1, i2);
        pushIfLong(i2, i0);
    };

    constexpr size_t kParallelMin = 256;
    if (pool && pool->threadCount() > 1 && triCount > kParallelMin) {
        pool->parallelFor(int(triCount), [&](int t, int tid) { consider(size_t(t), tid); });
    } else {
        for (size_t t = 0; t < triCount; ++t) consider(t, 0);
    }

    float maxPx = 0.0f;
    size_t total = 0;
    for (size_t i = 0; i < local.size(); ++i) {
        maxPx = std::max(maxPx, localMax[i]);
        total += local[i].size();
    }
    splitEdge.reserve(total / 2 + 8);
    for (const auto& v : local)
        for (const EdgeKey& k : v) splitEdge[k] = 1;
    return maxPx;
}

float adaptiveMeshFrac(float startMaxPx, float curMaxPx, float targetPx) {
    if (!(startMaxPx > targetPx * 1.01f)) return 1.0f;
    const float cur = std::max(curMaxPx, targetPx);
    const float num = std::log2(std::max(1.0e-3f, cur / targetPx));
    const float den = std::log2(std::max(1.0e-3f, startMaxPx / targetPx));
    if (!(den > 1.0e-6f)) return 1.0f;
    return 1.0f - std::clamp(num / den, 0.0f, 1.0f);
}

// Frustum-local: iterative binary dice with rematch — far corners stay coarse.
void refineLinearFrustumLocal(Mesh& mesh, int maxIter, float dicingQuality, size_t polyBudget,
                              const Mat4& objToWorld, const CameraData& cam, const Mat4& worldToCam,
                              float aspect, int resX, int resY, float padFrac, TessRuntime* rt) {
    maxIter = std::max(0, std::min(maxIter, 16));
    if (maxIter <= 0) return;
    if (mesh.normals.size() != mesh.positions.size()) {
        mesh.normals.clear();
        mesh.computeNormalsIfMissing();
    }
    (void)dicingQuality;
    (void)resX;
    (void)resY;
    constexpr int kTailIters = 2;
    ThreadPool* pool = rt ? rt->pool : nullptr;
    std::vector<VertProj> proj;

    for (int iter = 0; iter < maxIter; ++iter) {
        if (mesh.triangleCount() * 4 > polyBudget) {
            logWarning("tessellate: frustum-local subdiv stopped at triangle budget (" +
                       std::to_string(polyBudget) + ") after " + std::to_string(iter) +
                       " iteration(s)");
            break;
        }
        if (rt) rt->reportMeshFrac(float(iter) / float(std::max(1, maxIter)), mesh.triangleCount());

        projectAllVertices(mesh, objToWorld, worldToCam, cam, aspect, proj, pool);
        std::vector<uint8_t> faceMark;
        size_t markedCount = 0;
        markFacesInFrustumProjected(mesh, proj, padFrac, cam, aspect, faceMark, markedCount, pool);
        if (markedCount == 0) break;
        if (iter < kTailIters && markedCount < faceMark.size()) expandFaceMarksOneRing(mesh, faceMark);

        if (markedCount == faceMark.size() && faceMark.size() >= 1024u) {
            const int remain = maxIter - iter;
            const int allowed = clampSubdivLevelsForBudget(remain, mesh.triangleCount(), polyBudget);
            if (allowed <= 0) break;
            std::vector<uint8_t> faceLevel(faceMark.size(), uint8_t(allowed));
            diceMeshByFaceLevels(mesh, faceLevel, polyBudget);
            break;
        }

        std::vector<uint8_t> faceLevel(faceMark.size(), 0);
        for (size_t t = 0; t < faceMark.size(); ++t) {
            if (faceMark[t]) faceLevel[t] = 1;
        }
        if (!diceMeshByFaceLevels(mesh, faceLevel, polyBudget)) break;
    }
    if (rt) rt->reportMeshFrac(1.0f, mesh.triangleCount());
}

// Screen Adaptive: split only edges longer than targetPx (red-green, shared mids).
// Far edges fall below target first → world density drops with distance.
// No centroid fans (those cracked displace / Pref / reflections).
void refineScreenAdaptiveDice(Mesh& mesh, float dicingQuality, size_t polyBudget,
                              const Mat4& objToWorld, const CameraData& cam, const Mat4& worldToCam,
                              float aspect, int resX, int resY, float padFrac, TessRuntime* rt) {
    if (mesh.normals.size() != mesh.positions.size()) {
        mesh.normals.clear();
        mesh.computeNormalsIfMissing();
    }
    const float quality = std::max(1.0e-3f, dicingQuality);
    const float targetPx = 1.0f / quality;
    constexpr int kSafetyPasses = 48;
    constexpr int kTailPasses = 2;
    ThreadPool* pool = rt ? rt->pool : nullptr;
    std::vector<VertProj> proj;
    float startMaxPx = -1.0f;

    for (int pass = 0; pass < kSafetyPasses; ++pass) {
        if (mesh.triangleCount() >= polyBudget) {
            logWarning("tessellate: Screen Adaptive stopped at Dicing Poly Limit (" +
                       std::to_string(polyBudget) + " tris)");
            break;
        }

        projectAllVertices(mesh, objToWorld, worldToCam, cam, aspect, proj, pool);
        std::vector<uint8_t> faceMark;
        size_t markedCount = 0;
        markFacesInFrustumProjected(mesh, proj, padFrac, cam, aspect, faceMark, markedCount, pool);
        if (markedCount == 0) break;
        if (pass < kTailPasses && markedCount < faceMark.size()) {
            expandFaceMarksOneRing(mesh, faceMark);
            if (pass == 0) expandFaceMarksOneRing(mesh, faceMark);
        }

        std::unordered_map<EdgeKey, char, EdgeHash> splitEdge;
        const float maxPx = collectLongEdgesForAdaptive(mesh, faceMark, proj, targetPx, cam, aspect, resX,
                                                        resY, pool, splitEdge);
        if (startMaxPx < 0.0f) startMaxPx = std::max(maxPx, targetPx);
        if (rt) rt->reportMeshFrac(adaptiveMeshFrac(startMaxPx, maxPx, targetPx), mesh.triangleCount());
        if (splitEdge.empty()) break;

        const size_t before = mesh.triangleCount();
        if (!subdivideMarkedEdgesOnce(mesh, splitEdge)) break;
        if (mesh.triangleCount() <= before) break;
        if (mesh.triangleCount() >= polyBudget) {
            logWarning("tessellate: Screen Adaptive stopped at Dicing Poly Limit (" +
                       std::to_string(polyBudget) + " tris)");
            break;
        }
    }
    if (rt) rt->reportMeshFrac(1.0f, mesh.triangleCount());
}

void refineLinear(Mesh& mesh, int maxIter, bool screenAdaptive, float dicingQuality, size_t polyBudget,
                  const InstanceData* diceInst, const CameraData& cam, const Mat4& worldToCam,
                  float aspect, int resX, int resY) {
    maxIter = clampSubdivLevelsForBudget(maxIter, mesh.triangleCount(), polyBudget);
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
        if (mesh.triangleCount() * 4 > polyBudget) {
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

void refineCatclark(Mesh& mesh, int levels, float dicingQuality, size_t polyBudget,
                    const InstanceData* diceInst, const CameraData& cam, const Mat4& worldToCam,
                    float aspect, int resX, int resY) {
    int useLevels = clampSubdivLevelsForBudget(levels, mesh.triangleCount(), polyBudget);
    if (useLevels <= 0) return;
    (void)dicingQuality;
    (void)diceInst;
    (void)cam;
    (void)worldToCam;
    (void)aspect;
    (void)resX;
    (void)resY;

#if SOLSTICE_HAVE_OPENSUBDIV
    if (refineCatclarkOpenSubdiv(mesh, useLevels)) return;
    logWarning("tessellate: OpenSubdiv catclark failed — falling back to linear");
#else
    logWarning("tessellate: OpenSubdiv not in this build — catclark falls back to linear");
#endif
    refineLinear(mesh, useLevels, false, dicingQuality, polyBudget, diceInst, cam, worldToCam, aspect, resX,
                 resY);
}

MeshPtr tessellateOne(Mesh cage, const Material& mat, const Scene& scene,
                      const std::vector<const InstanceData*>& instances, const CameraData& cam,
                      TessRuntime* rt) {
    const RenderSettingsData& rs = scene.settings;
    const Mat4 w2c = worldToCamera(cam);
    const float padFrac = std::max(0.0f, rs.frustumPadding) * 0.01f;
    const int resX = std::max(1, rs.resolutionX);
    const int resY = std::max(1, rs.resolutionY);
    const float aspect = float(resX) / float(resY);
    const size_t polyBudget = tessTriangleBudget(rs);

    // Wireframe overlay stays at the authored cage — capture before densify/displace.
    if (cage.wireIndices.empty()) cage.captureWireCage();
    // Densify rebuilds triangle indices; the cage edge mask would go stale.
    cage.triEdgeMask.clear();

    const bool needDisp = materialHasGeometricDisplacement(mat);
    // Master switch: skip subdiv + displace entirely (cages only).
    if (rs.enableDisplacement == 0 || !needDisp) {
        return std::make_shared<Mesh>(std::move(cage));
    }

    const bool adaptive = rs.screenAdaptive != 0;
    // Screen Adaptive always implies camera-visible dicing (Karma-style).
    // Frustum Cull checkbox still gates the non-adaptive path.
    const bool useFrustumGate = adaptive || rs.frustumCull != 0;

    bool inFrustum = true;
    if (useFrustumGate) {
        inFrustum = meshVisibleInFrustum(cage, instances, cam, w2c, aspect, padFrac);
    }
    if (useFrustumGate && !inFrustum) {
        if (rt) rt->reportMeshFrac(1.0f, cage.triangleCount());
        return displaceMeshOnly(std::move(cage), mat, scene);
    }

    const int authoredSubdivType = cage.subdivType;
    const int authoredSubdivIterations = cage.subdivIterations;
    const float authoredDicingQuality = cage.dicingQuality;
    const float authoredPad = cage.boundsPadding;
    const bool authoredTimeDependent = cage.timeDependent;

    // Steal cage storage — no parallel cage+work copy while refining.
    auto work = std::make_shared<Mesh>(std::move(cage));
    work->motionPositionsPacked_.clear();
    const InstanceData* diceInst = nearestInstance(instances, cam);
    int type = work->subdivType;
    if (type == kSubdivCatclark && !meshHasQuadsHint(*work) && meshIsTriangleOnly(*work)) {
        type = kSubdivLinear;  // whole-mesh tris → linear
    }

    const int wantIters = std::max(0, work->subdivIterations);
    try {
        if (!diceInst) {
            // no instance — leave cage
        } else if (adaptive) {
            // Karma/Mantra/PRMan: Quality-driven raster dicing; Iterations ignored.
            refineScreenAdaptiveDice(*work, work->dicingQuality, polyBudget, diceInst->xform, cam, w2c,
                                     aspect, resX, resY, padFrac, rt);
        } else if (type == kSubdivNone || wantIters == 0) {
            // no subdiv
        } else if (rs.frustumCull) {
            refineLinearFrustumLocal(*work, wantIters, work->dicingQuality, polyBudget, diceInst->xform, cam,
                                     w2c, aspect, resX, resY, padFrac, rt);
        } else {
            const int uniformCap =
                clampSubdivLevelsForBudget(wantIters, work->triangleCount(), polyBudget);
            if (uniformCap == 0) {
                // cage already past budget
            } else if (type == kSubdivCatclark) {
                if (rt) rt->reportMeshFrac(0.0f, work->triangleCount());
                refineCatclark(*work, uniformCap, work->dicingQuality, polyBudget, diceInst, cam, w2c,
                               aspect, resX, resY);
                if (rt) rt->reportMeshFrac(1.0f, work->triangleCount());
            } else {
                if (rt) rt->reportMeshFrac(0.0f, work->triangleCount());
                refineLinear(*work, uniformCap, false, work->dicingQuality, polyBudget, diceInst, cam, w2c,
                             aspect, resX, resY);
                if (rt) rt->reportMeshFrac(1.0f, work->triangleCount());
            }
        }
    } catch (const std::bad_alloc&) {
        logWarning("tessellate: subdiv ran out of memory — displacing densest level kept so far");
    }

    if (needDisp) {
        Material m = mat;
        m.subdivIterations = wantIters;
        const float pad = std::max(work->boundsPadding, authoredPad);
        try {
            MeshPtr displaced = displaceMeshOnly(std::move(*work), m, scene);
            displaced->boundsPadding = pad;
            if (displaced->boundsPadding > 0.0f) displaced->computeBounds();
            displaced->subdivType = authoredSubdivType;
            displaced->subdivIterations = authoredSubdivIterations;
            displaced->dicingQuality = authoredDicingQuality;
            displaced->timeDependent = authoredTimeDependent;
            return displaced;
        } catch (const std::bad_alloc&) {
            logWarning("tessellate: displace OOM — returning densified cage without Pref lock");
            work->timeDependent = authoredTimeDependent;
            work->computeBounds();
            return work;
        }
    }
    work->timeDependent = authoredTimeDependent;
    work->computeBounds();
    return work;
}

}  // namespace

void tessellateSceneForRender(Scene& scene, const CameraData& dicingCamera,
                              const TessProgressFn& progress, bool onlyTimeDependent) {
    if (scene.meshes.empty() || scene.instances.empty()) return;
    if (scene.settings.enableDisplacement == 0) {
        scene.finalize();
        return;
    }

    std::vector<std::vector<const InstanceData*>> byMesh(scene.meshes.size());
    for (const InstanceData& inst : scene.instances) {
        if (inst.meshIndex < 0 || size_t(inst.meshIndex) >= scene.meshes.size()) continue;
        byMesh[size_t(inst.meshIndex)].push_back(&inst);
    }

    // Pre-count displace meshes for "Dicing i/n" progress.
    std::vector<size_t> diceMeshes;
    diceMeshes.reserve(scene.meshes.size());
    for (size_t mi = 0; mi < scene.meshes.size(); ++mi) {
        MeshPtr& mesh = scene.meshes[mi];
        if (!mesh || byMesh[mi].empty()) continue;
        if (onlyTimeDependent && !mesh->timeDependent) continue;
        const int matIndex = byMesh[mi][0]->materialIndex;
        Material mat;
        if (matIndex >= 0 && size_t(matIndex) < scene.materials.size())
            mat = scene.materials[size_t(matIndex)];
        if (materialHasGeometricDisplacement(mat)) diceMeshes.push_back(mi);
    }
    if (diceMeshes.empty()) {
        scene.finalize();
        return;
    }

    // Leave 2 cores free for UI/OS. Destroy the pool before Embree starts so
    // render workers are not oversubscribed against leftover dice threads.
    // ThreadPool(N) = N workers + caller → size ctor so total ≈ parallelism.
    const int parallelism = diceParallelism(scene.settings.threads);
    const int poolThreads = dicePoolThreadCount(parallelism);
    std::unique_ptr<ThreadPool> dicePool;
    if (poolThreads > 1) dicePool = std::make_unique<ThreadPool>(poolThreads);
    TessRuntime rt;
    rt.pool = dicePool.get();
    rt.progress = progress;
    rt.meshCount = int(diceMeshes.size());
    logInfo("tessellate: dicing with " + std::to_string(parallelism) +
            " concurrent thread(s) (2 cores reserved)");

    const auto diceStart = std::chrono::steady_clock::now();
    for (int di = 0; di < int(diceMeshes.size()); ++di) {
        const size_t mi = diceMeshes[size_t(di)];
        MeshPtr& mesh = scene.meshes[mi];
        if (!mesh) continue;
        const auto& insts = byMesh[mi];

        const int matIndex = insts[0]->materialIndex;
        Material mat;
        if (matIndex >= 0 && size_t(matIndex) < scene.materials.size())
            mat = scene.materials[size_t(matIndex)];

        const int authoredIterations = mesh->subdivIterations;
        const std::string meshName = mesh->name;
        rt.meshIndex = di;
        rt.meshName = meshName;
        try {
            // Steal cage buffers into a local Mesh while keeping the shared_ptr
            // slot non-null (empty shell). Avoids a parallel cage+work copy and
            // never leaves a nullptr mesh index for Embree/OptiX.
            Mesh working = std::move(*mesh);
            MeshPtr tess = tessellateOne(std::move(working), mat, scene, insts, dicingCamera, &rt);
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

    dicePool.reset();  // join dice workers before Embree/OptiX thread pool starts
    {
        const double sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - diceStart).count();
        logInfo("tessellate: densify wall time " + std::to_string(sec) + " s");
        if (progress) progress("Dicing 100%");
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
    key += ";ed=";
    key += std::to_string(rs.enableDisplacement);
    key += ";dpl=";
    key += std::to_string(rs.dicingPolyLimitM);
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
