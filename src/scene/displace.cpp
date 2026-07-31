#include "scene/displace.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/log.h"
#include "render/procedural.h"
#include "render/shading.h"

namespace sol {
namespace {

// Soft safety only — subdiv_iterations is no longer hard-capped. Mid-edge
// splits are 4× tris per level, so this budget is what stops runaway values.
constexpr size_t kMaxDisplaceTriangles = 64000000;

struct EdgeKey {
    uint32_t a = 0;
    uint32_t b = 0;
    bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& e) const {
        return (size_t(e.a) * 0x9e3779b97f4a7c15ULL) ^ size_t(e.b);
    }
};

EdgeKey makeEdge(uint32_t i0, uint32_t i1) {
    return i0 < i1 ? EdgeKey{i0, i1} : EdgeKey{i1, i0};
}

// Mid-edge UV. True wrap seams (cylinder / open UV cut) have a short 3D edge
// with |Δu| or |Δv| > 0.5 — unwrap those so the midpoint stays on the seam.
// Coarse grids/planes often span the whole UV island with ONE long edge
// (Δu≈1 across the quad); treating that as a seam collapses every midpoint
// toward 0 and displacement samples a single texel (flat floor).
Vec2 midUvSeamSafe(Vec2 a, Vec2 b, Vec3 pa, Vec3 pb, float meshSize) {
    float u0 = a.x, u1 = b.x;
    float v0 = a.y, v1 = b.y;
    const bool bigJump = (std::fabs(u1 - u0) > 0.5f) || (std::fabs(v1 - v0) > 0.5f);
    const float edgeLen = length(pb - pa);
    // Half the bbox size: long floor edges stay as plain averages; fine wrap
    // edges on cylinders/spheres still unwrap.
    const bool looksLikeWrapSeam =
        bigJump && meshSize > 1e-8f && edgeLen <= 0.5f * meshSize;
    if (looksLikeWrapSeam) {
        if (u1 - u0 > 0.5f) u0 += 1.0f;
        else if (u0 - u1 > 0.5f) u1 += 1.0f;
        if (v1 - v0 > 0.5f) v0 += 1.0f;
        else if (v0 - v1 > 0.5f) v1 += 1.0f;
    }
    float u = 0.5f * (u0 + u1);
    float v = 0.5f * (v0 + v1);
    if (looksLikeWrapSeam) {
        u -= std::floor(u);
        v -= std::floor(v);
    }
    return Vec2(u, v);
}

float meshMaxEdgeLength(const Mesh& mesh) {
    float maxLen = 0.0f;
    const size_t n = mesh.positions.size();
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        const uint32_t i0 = mesh.indices[t + 0];
        const uint32_t i1 = mesh.indices[t + 1];
        const uint32_t i2 = mesh.indices[t + 2];
        if (i0 >= n || i1 >= n || i2 >= n) continue;
        maxLen = std::max(maxLen, length(mesh.positions[i1] - mesh.positions[i0]));
        maxLen = std::max(maxLen, length(mesh.positions[i2] - mesh.positions[i1]));
        maxLen = std::max(maxLen, length(mesh.positions[i0] - mesh.positions[i2]));
    }
    return maxLen;
}

void subdivideOnce(Mesh& mesh, float meshSize) {
    if (mesh.indices.size() < 3 || mesh.positions.empty()) return;

    const size_t oldVertCount = mesh.positions.size();
    const bool hasNormals = mesh.normals.size() == oldVertCount;
    const bool hasUvs = mesh.uvs.size() == oldVertCount;

    // Drop motion keys that don't match cage vertex count before topology change.
    mesh.motionPositions.erase(
        std::remove_if(mesh.motionPositions.begin(), mesh.motionPositions.end(),
                       [oldVertCount](const std::vector<Vec3>& k) { return k.size() != oldVertCount; }),
        mesh.motionPositions.end());

    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> midpoint;
    midpoint.reserve(mesh.indices.size());

    auto getMid = [&](uint32_t i0, uint32_t i1) -> uint32_t {
        const EdgeKey key = makeEdge(i0, i1);
        auto it = midpoint.find(key);
        if (it != midpoint.end()) return it->second;
        const uint32_t mid = uint32_t(mesh.positions.size());
        mesh.positions.push_back((mesh.positions[i0] + mesh.positions[i1]) * 0.5f);
        if (hasNormals) {
            Vec3 n = mesh.normals[i0] + mesh.normals[i1];
            const float len = length(n);
            mesh.normals.push_back(len > 1e-8f ? n / len : mesh.normals[i0]);
        }
        if (hasUvs)
            mesh.uvs.push_back(midUvSeamSafe(mesh.uvs[i0], mesh.uvs[i1], mesh.positions[i0],
                                             mesh.positions[i1], meshSize));
        for (std::vector<Vec3>& keyPositions : mesh.motionPositions)
            keyPositions.push_back((keyPositions[i0] + keyPositions[i1]) * 0.5f);
        midpoint.emplace(key, mid);
        return mid;
    };

    std::vector<uint32_t> newIndices;
    newIndices.reserve(mesh.indices.size() * 4);
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        const uint32_t i0 = mesh.indices[t + 0];
        const uint32_t i1 = mesh.indices[t + 1];
        const uint32_t i2 = mesh.indices[t + 2];
        if (i0 >= oldVertCount || i1 >= oldVertCount || i2 >= oldVertCount) continue;
        const uint32_t m01 = getMid(i0, i1);
        const uint32_t m12 = getMid(i1, i2);
        const uint32_t m20 = getMid(i2, i0);
        newIndices.insert(newIndices.end(), {i0, m01, m20});
        newIndices.insert(newIndices.end(), {i1, m12, m01});
        newIndices.insert(newIndices.end(), {i2, m20, m12});
        newIndices.insert(newIndices.end(), {m01, m12, m20});
    }
    mesh.indices = std::move(newIndices);
}

TextureView textureViewFromImage(const Image& image) {
    TextureView view;
    if (image.empty()) return view;
    view.pixels = image.data();
    view.width = image.width();
    view.height = image.height();
    view.mipCount = image.mipCount() > 0 ? image.mipCount() : 1;
    if (image.isUdimAtlas()) {
        view.udimGridU = image.udimGridU();
        view.udimGridV = image.udimGridV();
    }
    return view;
}

SceneView makeDisplaceSceneView(const Scene& scene, std::vector<TextureView>& textureViews) {
    textureViews.clear();
    textureViews.reserve(scene.textures.size());
    for (const std::shared_ptr<Image>& image : scene.textures)
        textureViews.push_back(image ? textureViewFromImage(*image) : TextureView{});

    SceneView view;
    view.textures = textureViews.data();
    view.textureCount = int(textureViews.size());
    view.procedurals = scene.procedurals.data();
    view.proceduralCount = int(scene.procedurals.size());
    return view;
}

Vec3 sampleDisplacementVector(const SceneView& scene, const Material& mat, Vec2 uv, Vec3 p, Vec3 n) {
    ProceduralCtx ctx;
    ctx.uv = uv;
    ctx.pObject = p;
    ctx.nObject = n;
    // Cage is Pref during vertex displace — mark it so shade-time Pref matches.
    ctx.pRef = p;
    ctx.nRef = n;
    ctx.hasPref = 1;
    ctx.filterWidth = 0.0f;
    ctx.forDisplacement = 1;

    Vec3 amount(0.0f);
    if (mat.displacementProc >= 0) {
        const Vec4 c = evalProceduralRoot(scene, mat.displacementProc, ctx);
        if (mat.displacementVector)
            amount = Vec3(c.x, c.y, c.z);
        else
            // Arnold/MaterialX mono height lives in R. Do NOT use luminance — many
            // EXR displacement maps store height only in R (G=B=0); luminance
            // would shrink relief by ~0.21x and look almost flat with banding.
            amount = Vec3(c.x);
    } else if (mat.displacementTex >= 0 && mat.displacementTex < scene.textureCount && scene.textures) {
        const TextureView& tex = scene.textures[mat.displacementTex];
        const Vec4 c = sampleTextureRGBA(tex, uv);
        if (mat.displacementVector)
            amount = Vec3(c.x, c.y, c.z);
        else
            amount = Vec3(c.x);  // height maps: use red (Arnold / MaterialX convention)
    } else {
        amount = Vec3(mat.displacementHeight);
    }

    amount -= Vec3(mat.displacementZeroValue);
    return amount * mat.displacementScale;
}

float sampleHeightScalar(const SceneView& scene, const Material& mat, Vec2 uv, Vec3 p, Vec3 n) {
    const float scale = std::fabs(mat.displacementScale) > 1e-12f ? mat.displacementScale : 1.0f;
    return sampleDisplacementVector(scene, mat, uv, p, n).x / scale;
}

// Largest |Δheight| across mesh edges — drives extra subdiv for 8K maps / strata.
float meshMaxEdgeHeightDelta(const Mesh& mesh, const Material& mat, const SceneView& scene) {
    // Procedural displace (triplanar / noise3d) keys off P/N; UV maps need UVs.
    const bool needsUv = mat.displacementProc < 0;
    if (needsUv && mesh.uvs.size() != mesh.positions.size()) return 0.0f;
    float maxDh = 0.0f;
    const size_t n = mesh.positions.size();
    const bool hasN = mesh.normals.size() == n;
    const bool hasU = mesh.uvs.size() == n;
    // Sparse sample: every Nth triangle keeps this O(tris) but cheaper for huge meshes.
    const size_t stride = mesh.indices.size() > 300000 ? 9 : 3;
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += stride) {
        const uint32_t idx[3] = {mesh.indices[t + 0], mesh.indices[t + 1], mesh.indices[t + 2]};
        float h[3];
        for (int k = 0; k < 3; ++k) {
            if (idx[k] >= n) {
                h[k] = 0.0f;
                continue;
            }
            const Vec3 nrm = hasN ? mesh.normals[idx[k]] : Vec3(0.0f, 1.0f, 0.0f);
            const Vec2 uv = hasU ? mesh.uvs[idx[k]] : Vec2(0.0f, 0.0f);
            h[k] = sampleHeightScalar(scene, mat, uv, mesh.positions[idx[k]], nrm);
        }
        maxDh = std::max(maxDh, std::fabs(h[0] - h[1]));
        maxDh = std::max(maxDh, std::fabs(h[1] - h[2]));
        maxDh = std::max(maxDh, std::fabs(h[2] - h[0]));
    }
    return maxDh;
}

void displaceVertices(Mesh& mesh, const Material& mat, const SceneView& scene) {
    if (mesh.positions.empty()) return;
    if (mesh.normals.size() != mesh.positions.size()) {
        mesh.normals.clear();
        mesh.computeNormalsIfMissing();
    }
    const bool hasUvs = mesh.uvs.size() == mesh.positions.size();
    const size_t n = mesh.positions.size();
    // Keep pre-displace normals to reorient after recompute — winding often
    // disagrees with authored / interpolated outward normals (e.g. grid +Y vs
    // cross-product -Y), which makes the whole surface shade nearly black.
    const std::vector<Vec3> orientRef = mesh.normals;

    // Weld by position so sphere poles / UV seams share one offset. Otherwise
    // different UVs on coincident verts explode into floating shards.
    struct PositionHash {
        size_t operator()(const Vec3& p) const {
            auto q = [](float v) { return int64_t(std::llround(double(v) * 100000.0)); };
            const size_t h1 = std::hash<int64_t>()(q(p.x));
            const size_t h2 = std::hash<int64_t>()(q(p.y));
            const size_t h3 = std::hash<int64_t>()(q(p.z));
            return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL) ^ (h3 * 0xc2b2ae3d27d4eb4fULL);
        }
    };
    struct PositionEqual {
        bool operator()(const Vec3& a, const Vec3& b) const {
            auto q = [](float v) { return int64_t(std::llround(double(v) * 100000.0)); };
            return q(a.x) == q(b.x) && q(a.y) == q(b.y) && q(a.z) == q(b.z);
        }
    };
    std::unordered_map<Vec3, std::vector<uint32_t>, PositionHash, PositionEqual> groups;
    groups.reserve(n);
    for (size_t i = 0; i < n; ++i) groups[mesh.positions[i]].push_back(uint32_t(i));

    std::vector<Vec3> offsets(n, Vec3(0.0f));
    for (const auto& entry : groups) {
        const std::vector<uint32_t>& ids = entry.second;
        Vec3 avgN(0.0f);
        Vec2 avgUv(0.0f);
        Vec3 avgP(0.0f);
        for (uint32_t id : ids) {
            avgN = avgN + mesh.normals[id];
            avgP = avgP + mesh.positions[id];
            if (hasUvs) avgUv = avgUv + mesh.uvs[id];
        }
        const float inv = 1.0f / float(ids.size());
        avgP = avgP * inv;
        avgUv = avgUv * inv;
        const float nlen = length(avgN);
        avgN = nlen > 1e-8f ? avgN / nlen : Vec3(0.0f, 1.0f, 0.0f);
        const Vec3 sample = sampleDisplacementVector(scene, mat, avgUv, avgP, avgN);
        const Vec3 off = mat.displacementVector ? sample : avgN * sample.x;
        for (uint32_t id : ids) offsets[id] = off;
    }

    for (size_t i = 0; i < n; ++i) mesh.positions[i] += offsets[i];
    for (std::vector<Vec3>& keyPositions : mesh.motionPositions) {
        if (keyPositions.size() != n) continue;
        for (size_t i = 0; i < n; ++i) keyPositions[i] += offsets[i];
    }

    // Rebuild smooth normals on the displaced surface, then flip any that ended
    // up on the opposite hemisphere from the pre-displace reference.
    mesh.normals.clear();
    mesh.computeNormalsIfMissing();
    if (mesh.normals.size() == orientRef.size()) {
        for (size_t i = 0; i < mesh.normals.size(); ++i) {
            if (dot(mesh.normals[i], orientRef[i]) < 0.0f) mesh.normals[i] = -mesh.normals[i];
        }
    }
}

}  // namespace

MeshPtr displaceMeshOnly(Mesh&& src, const Material& mat, const Scene& scene) {
    auto out = std::make_shared<Mesh>(std::move(src));
    out->name = out->name.empty() ? "displaced" : out->name + "_disp";
    out->motionPositionsPacked_.clear();

    if (out->normals.size() != out->positions.size()) {
        out->normals.clear();
        out->computeNormalsIfMissing();
    }

    std::vector<TextureView> textureViews;
    const SceneView view = makeDisplaceSceneView(scene, textureViews);
    if (mat.displacementTex >= 0 &&
        (mat.displacementTex >= view.textureCount || !view.textures ||
         !view.textures[mat.displacementTex].valid())) {
        logWarning("displacement: texture index " + std::to_string(mat.displacementTex) +
                   " is missing/empty — vertices will not pick up the height map");
    }

    // Pref / Nref: lock triplanar & autobump to the cage before vertex offset.
    out->restPositions = out->positions;
    if (out->normals.size() == out->positions.size())
        out->restNormals = out->normals;
    else {
        out->restNormals.clear();
        out->computeNormalsIfMissing();
        out->restNormals = out->normals;
    }

    displaceVertices(*out, mat, view);

    out->computeBounds();
    const float pad = out->boundsPadding;
    if (pad > 0.0f) {
        out->boundsPadding = pad;
        out->computeBounds();
    }
    return out;
}

MeshPtr displaceMeshOnly(const Mesh& src, const Material& mat, const Scene& scene) {
    return displaceMeshOnly(Mesh(src), mat, scene);
}

}  // namespace sol
