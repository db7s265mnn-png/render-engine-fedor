// Render-time tessellation for displacement: none / linear / catclark,
// frustum cull, screen-space dicing.
#include "scene/tessellate.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <stdexcept>
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

// Soft cap before Render/BVH — 64M was effectively no cap and OOMed on
// Subdiv Iterations ≥6 for dense cages. 4M tris is still a heavy displace mesh.
constexpr size_t kMaxTessTriangles = 4000000ull;

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

bool meshVisibleInFrustum(const Mesh& mesh, const std::vector<const InstanceData*>& instances,
                          const CameraData& cam, const Mat4& worldToCam, float aspect, float padFrac) {
    if (instances.empty()) return false;
    const size_t nPos = mesh.positions.size();
    const size_t step = std::max<size_t>(1, mesh.indices.size() / 3 / 256);
    for (const InstanceData* inst : instances) {
        if (!inst) continue;
        for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3 * step) {
            const uint32_t ia = mesh.indices[t + 0];
            const uint32_t ib = mesh.indices[t + 1];
            const uint32_t ic = mesh.indices[t + 2];
            if (!indexInRange(ia, nPos) || !indexInRange(ib, nPos) || !indexInRange(ic, nPos)) continue;
            const Vec3& a = mesh.positions[ia];
            const Vec3& b = mesh.positions[ib];
            const Vec3& c = mesh.positions[ic];
            const Vec3 mid = (a + b + c) * (1.0f / 3.0f);
            float nx, ny, nz;
            if (projectToNdc(mid, inst->xform, worldToCam, cam, aspect, nx, ny, nz) &&
                pointInPaddedFrustum(nx, ny, padFrac))
                return true;
            for (const Vec3& p : {a, b, c}) {
                if (projectToNdc(p, inst->xform, worldToCam, cam, aspect, nx, ny, nz) &&
                    pointInPaddedFrustum(nx, ny, padFrac))
                    return true;
            }
        }
    }
    Bounds3 b = mesh.bounds;
    if (!b.valid()) {
        Bounds3 tmp;
        for (const Vec3& p : mesh.positions) tmp.extend(p);
        b = tmp;
    }
    if (b.valid()) {
        const Vec3 corners[8] = {
            {b.lo.x, b.lo.y, b.lo.z}, {b.hi.x, b.lo.y, b.lo.z}, {b.lo.x, b.hi.y, b.lo.z},
            {b.hi.x, b.hi.y, b.lo.z}, {b.lo.x, b.lo.y, b.hi.z}, {b.hi.x, b.lo.y, b.hi.z},
            {b.lo.x, b.hi.y, b.hi.z}, {b.hi.x, b.hi.y, b.hi.z},
        };
        for (const InstanceData* inst : instances) {
            if (!inst) continue;
            for (const Vec3& c : corners) {
                float nx, ny, nz;
                if (projectToNdc(c, inst->xform, worldToCam, cam, aspect, nx, ny, nz) &&
                    pointInPaddedFrustum(nx, ny, padFrac))
                    return true;
            }
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

    for (int i = 0; i < maxIter; ++i) {
        if (mesh.triangleCount() * 4 > kMaxTessTriangles) {
            logWarning("tessellate: linear subdiv stopped at budget");
            break;
        }
        if (screenAdaptive && diceInst) {
            const float maxPx =
                maxScreenEdge(mesh, diceInst->xform, worldToCam, cam, aspect, resX, resY);
            if (maxPx <= targetPx) break;
        }
        subdivideLinearOnce(mesh);
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
        // Estimate levels from bbox screen size vs dicing quality.
        const float quality = std::max(1.0e-3f, dicingQuality);
        const float targetPx = 1.0f / quality;
        float maxPx = maxScreenEdge(mesh, diceInst->xform, worldToCam, cam, aspect, resX, resY);
        int est = 0;
        // Each level ≈ halves edges.
        while (est < useLevels && maxPx > targetPx) {
            maxPx *= 0.5f;
            ++est;
        }
        useLevels = est;
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

MeshPtr tessellateOne(const Mesh& cage, const Material& mat, const Scene& scene,
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
        return std::make_shared<Mesh>(cage);
    }

    bool inFrustum = true;
    if (rs.frustumCull) {
        inFrustum = meshVisibleInFrustum(cage, instances, cam, w2c, aspect, padFrac);
    }

    // Outside frustum: displace cage only (no subdiv), even if subdiv authored.
    if (rs.frustumCull && !inFrustum) {
        if (!needDisp) return std::make_shared<Mesh>(cage);
        MeshPtr out = displaceMeshOnly(cage, mat, scene);
        return out;
    }

    auto work = std::make_shared<Mesh>(cage);
    work->motionPositionsPacked_.clear();
    const InstanceData* diceInst = nearestInstance(instances, cam);
    const bool adaptive = rs.screenAdaptive != 0;
    int type = work->subdivType;
    if (type == kSubdivCatclark && !meshHasQuadsHint(*work) && meshIsTriangleOnly(*work)) {
        type = kSubdivLinear;  // whole-mesh tris → linear
    }

    const int cap = clampSubdivLevelsForBudget(work->subdivIterations, work->triangleCount());
    if (type == kSubdivNone || cap == 0) {
        // no subdiv
    } else if (type == kSubdivCatclark) {
        refineCatclark(*work, cap, adaptive, work->dicingQuality, diceInst, cam, w2c, aspect, resX, resY);
    } else {
        refineLinear(*work, cap, adaptive, work->dicingQuality, diceInst, cam, w2c, aspect, resX, resY);
    }

    // Copy applied levels onto material residual via caller; displace now.
    if (needDisp) {
        Material m = mat;
        m.subdivIterations = cap;
        const int subdivType = cage.subdivType;
        const int subdivIterations = cage.subdivIterations;
        const float dicingQuality = cage.dicingQuality;
        const float pad = std::max(work->boundsPadding, cage.boundsPadding);
        MeshPtr displaced = displaceMeshOnly(std::move(*work), m, scene);
        displaced->boundsPadding = pad;
        if (displaced->boundsPadding > 0.0f) displaced->computeBounds();
        displaced->subdivType = subdivType;
        displaced->subdivIterations = subdivIterations;
        displaced->dicingQuality = dicingQuality;
        return displaced;
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

        try {
            MeshPtr tess = tessellateOne(*mesh, mat, scene, insts, dicingCamera);
            // Propagate residual levels into the shared material for autobump weight.
            if (matIndex >= 0 && size_t(matIndex) < scene.materials.size())
                scene.materials[size_t(matIndex)].subdivIterations = mesh->subdivIterations;
            mesh = std::move(tess);
        } catch (const std::bad_alloc&) {
            logError("tessellate: out of memory on mesh '" + mesh->name +
                     "' — leaving cage (lower Subdiv Iterations)");
        } catch (const std::exception& ex) {
            logError(std::string("tessellate: ") + ex.what() + " on mesh '" + mesh->name +
                     "' — leaving cage");
        }
    }

    logInfo("tessellate: render tessellation complete");
}

}  // namespace sol
