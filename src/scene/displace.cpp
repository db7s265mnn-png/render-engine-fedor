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

constexpr int kMaxSubdivIterations = 5;

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

void subdivideOnce(Mesh& mesh) {
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
        if (hasUvs) mesh.uvs.push_back((mesh.uvs[i0] + mesh.uvs[i1]) * 0.5f);
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
    ctx.filterWidth = 0.0f;

    Vec3 amount(0.0f);
    if (mat.displacementProc >= 0) {
        const Vec4 c = evalProceduralRoot(scene, mat.displacementProc, ctx);
        if (mat.displacementVector)
            amount = Vec3(c.x, c.y, c.z);
        else
            amount = Vec3(0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z);
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

    std::vector<Vec3> offsets(n, Vec3(0.0f));
    for (size_t i = 0; i < n; ++i) {
        const Vec2 uv = hasUvs ? mesh.uvs[i] : Vec2(0.0f, 0.0f);
        const Vec3 nrm = mesh.normals[i];
        const Vec3 sample = sampleDisplacementVector(scene, mat, uv, mesh.positions[i], nrm);
        if (mat.displacementVector)
            offsets[i] = sample;
        else
            offsets[i] = nrm * sample.x;
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

MeshPtr applyArnoldDisplacement(const Mesh& src, const Material& mat, const Scene& scene) {
    auto out = std::make_shared<Mesh>(src);
    out->name = src.name.empty() ? "displaced" : src.name + "_disp";
    out->motionPositionsPacked_.clear();

    int iterations = mat.subdivIterations;
    if (iterations < 0) iterations = 0;
    if (iterations > kMaxSubdivIterations) {
        logWarning("displacement: clamping subdiv_iterations " + std::to_string(iterations) + " → " +
                   std::to_string(kMaxSubdivIterations));
        iterations = kMaxSubdivIterations;
    }

    // Ensure cage normals before subdiv so midpoints interpolate something sensible.
    if (out->normals.size() != out->positions.size()) {
        out->normals.clear();
        out->computeNormalsIfMissing();
    }

    for (int i = 0; i < iterations; ++i) subdivideOnce(*out);

    std::vector<TextureView> textureViews;
    const SceneView view = makeDisplaceSceneView(scene, textureViews);
    displaceVertices(*out, mat, view);

    out->computeBounds();
    const float pad = mat.displacementBoundsPadding;
    if (pad > 0.0f) {
        out->boundsPadding = pad;
        out->computeBounds();
    }
    return out;
}

}  // namespace sol
