#include "scene/scene.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>

#include "core/log.h"

namespace sol {

void Mesh::computeBounds() {
    bounds = Bounds3();
    for (const Vec3& p : positions) bounds.extend(p);
}

void Mesh::computeNormalsIfMissing() {
    if (normals.size() == positions.size() && !normals.empty()) return;
    normals.assign(positions.size(), Vec3(0.0f));

    // Face varying attributes force meshes to be split per corner, which would
    // give faceted shading. Welding by position recovers smooth normals.
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

    std::unordered_map<Vec3, uint32_t, PositionHash, PositionEqual> weld;
    weld.reserve(positions.size());
    std::vector<uint32_t> representative(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        auto inserted = weld.emplace(positions[i], uint32_t(i));
        representative[i] = inserted.first->second;
    }

    std::vector<Vec3> accumulated(positions.size(), Vec3(0.0f));
    for (size_t t = 0; t + 2 < indices.size(); t += 3) {
        const uint32_t i0 = indices[t + 0], i1 = indices[t + 1], i2 = indices[t + 2];
        if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) continue;
        const Vec3 faceN = cross(positions[i1] - positions[i0], positions[i2] - positions[i0]);  // area weighted
        accumulated[representative[i0]] += faceN;
        accumulated[representative[i1]] += faceN;
        accumulated[representative[i2]] += faceN;
    }
    for (size_t i = 0; i < positions.size(); ++i) {
        const Vec3 n = accumulated[representative[i]];
        const float len = length(n);
        normals[i] = len > 0.0f ? n / len : Vec3(0.0f, 1.0f, 0.0f);
    }
}

void Mesh::validate() {
    // Drop degenerate or out of range triangles so the BVH builders stay happy.
    std::vector<uint32_t> cleaned;
    cleaned.reserve(indices.size());
    const uint32_t vertexCount = static_cast<uint32_t>(positions.size());
    for (size_t t = 0; t + 2 < indices.size(); t += 3) {
        const uint32_t i0 = indices[t + 0], i1 = indices[t + 1], i2 = indices[t + 2];
        if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) continue;
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;
        const Vec3 n = cross(positions[i1] - positions[i0], positions[i2] - positions[i0]);
        if (lengthSquared(n) <= 0.0f) continue;
        cleaned.push_back(i0);
        cleaned.push_back(i1);
        cleaned.push_back(i2);
    }
    if (cleaned.size() != indices.size()) indices.swap(cleaned);
    if (!uvs.empty() && uvs.size() != positions.size()) uvs.clear();
    computeNormalsIfMissing();
    computeBounds();
}

MeshView Mesh::view() const {
    MeshView v;
    v.positions = positions.data();
    v.normals = normals.size() == positions.size() ? normals.data() : nullptr;
    v.uvs = uvs.size() == positions.size() ? uvs.data() : nullptr;
    v.indices = indices.data();
    v.triangleCount = static_cast<uint32_t>(indices.size() / 3);
    v.vertexCount = static_cast<uint32_t>(positions.size());
    return v;
}

// ---------------------------------------------------------------------------

void EnvironmentMap::buildSamplingTables() {
    if (image.empty()) {
        distribution = Distribution2D();
        luminanceFunc.clear();
        return;
    }
    const int w = image.width();
    const int h = image.height();
    luminanceFunc.resize(size_t(w) * size_t(h));
    for (int y = 0; y < h; ++y) {
        // sin(theta) accounts for the equirectangular solid angle distortion.
        const float theta = (float(y) + 0.5f) / float(h) * kPi;
        const float sinTheta = std::sin(theta);
        for (int x = 0; x < w; ++x) {
            const Vec3 c = image.rgb(x, y);
            luminanceFunc[size_t(y) * w + x] = std::max(0.0f, luminance(c)) * sinTheta;
        }
    }
    distribution.build(luminanceFunc, w, h);
}

EnvMapView EnvironmentMap::view() const {
    EnvMapView v;
    if (image.empty()) return v;
    v.pixels = image.data();
    v.width = image.width();
    v.height = image.height();
    if (distribution.valid()) {
        v.condCdf = distribution.conditionalCdfData();
        v.condIntegral = distribution.conditionalIntegralData();
        v.margCdf = distribution.marginalCdfData();
        v.margFunc = distribution.marginalFuncData();
        v.func = distribution.funcData();
        v.integral = distribution.integral();
    }
    return v;
}

// ---------------------------------------------------------------------------

int Scene::addMesh(MeshPtr mesh) {
    meshes.push_back(std::move(mesh));
    return static_cast<int>(meshes.size()) - 1;
}

int Scene::addMaterial(const Material& material) {
    materials.push_back(material);
    return static_cast<int>(materials.size()) - 1;
}

int Scene::addEnvMap(std::shared_ptr<EnvironmentMap> env) {
    envMaps.push_back(std::move(env));
    return static_cast<int>(envMaps.size()) - 1;
}

int Scene::addTexture(std::shared_ptr<Image> image) {
    if (!image || image->empty()) return -1;
    textures.push_back(std::move(image));
    return static_cast<int>(textures.size()) - 1;
}

size_t Scene::totalTriangles() const {
    size_t total = 0;
    for (const InstanceData& inst : instances) {
        if (inst.meshIndex >= 0 && inst.meshIndex < int(meshes.size()) && meshes[inst.meshIndex])
            total += meshes[inst.meshIndex]->triangleCount();
    }
    return total;
}

void Scene::buildLightProxies() {
    // Area lights are represented analytically for sampling but also need real
    // geometry so that BSDF rays can hit them (needed for MIS and for the
    // "visible to camera" flag).
    const size_t lightCount = lights.size();
    for (size_t i = 0; i < lightCount; ++i) {
        LightData& light = lights[i];
        MeshPtr proxy;
        switch (light.type) {
            case kLightRect: proxy = makeRectMesh(light.width, light.height); break;
            case kLightDisk: proxy = makeDiskMesh(light.radius); break;
            case kLightSphere: proxy = makeSphereMesh(light.radius, 64, 32); break;
            default: break;
        }
        if (!proxy) continue;
        proxy->name = "light_proxy";
        proxy->validate();
        InstanceData inst;
        inst.xform = light.xform;
        inst.xformInv = inverse(light.xform);
        inst.meshIndex = addMesh(proxy);
        inst.materialIndex = -1;
        inst.lightIndex = static_cast<int>(i);
        inst.visibleCamera = light.visibleCamera;
        // Area lights are visible to camera/MIS, but only cast shadows when self-shadow is on.
        inst.visibilityMask = light.selfShadowEnable ? kVisAll : kVisPrimary;
        instances.push_back(inst);
    }
}

void Scene::finalize() {
    for (MeshPtr& mesh : meshes) {
        if (mesh) mesh->validate();
    }
    buildLightProxies();

    for (std::shared_ptr<EnvironmentMap>& env : envMaps) {
        if (env && env->luminanceFunc.empty()) env->buildSamplingTables();
    }

    bounds_ = Bounds3();
    for (InstanceData& inst : instances) {
        inst.xformInv = inverse(inst.xform);
        if (inst.meshIndex < 0 || inst.meshIndex >= int(meshes.size()) || !meshes[inst.meshIndex]) continue;
        bounds_.extend(transformBounds(inst.xform, meshes[inst.meshIndex]->bounds));
    }
    if (!bounds_.valid()) {
        bounds_.extend(Vec3(-1.0f));
        bounds_.extend(Vec3(1.0f));
    }

    for (LightData& light : lights) light.xformInv = inverse(light.xform);

    domeLightIndex_ = -1;
    for (size_t i = 0; i < lights.size(); ++i) {
        if (lights[i].type == kLightDome) {
            domeLightIndex_ = static_cast<int>(i);
            break;
        }
    }

    meshViews_.clear();
    meshViews_.reserve(meshes.size());
    for (const MeshPtr& mesh : meshes) meshViews_.push_back(mesh ? mesh->view() : MeshView());

    envViews_.clear();
    envViews_.reserve(envMaps.size());
    for (const std::shared_ptr<EnvironmentMap>& env : envMaps)
        envViews_.push_back(env ? env->view() : EnvMapView());

    textureViews_.clear();
    textureViews_.reserve(textures.size());
    for (const std::shared_ptr<Image>& image : textures) {
        TextureView view;
        if (image && !image->empty()) {
            view.pixels = image->data();
            view.width = image->width();
            view.height = image->height();
            if (image->isUdimAtlas()) {
                view.udimGridU = image->udimGridU();
                view.udimGridV = image->udimGridV();
            }
        }
        textureViews_.push_back(view);
    }
}

SceneView Scene::view() const {
    SceneView v;
    v.meshes = meshViews_.data();
    v.instances = instances.data();
    v.materials = materials.data();
    v.lights = lights.data();
    v.envMaps = envViews_.data();
    v.textures = textureViews_.data();
    v.meshCount = static_cast<int>(meshViews_.size());
    v.instanceCount = static_cast<int>(instances.size());
    v.materialCount = static_cast<int>(materials.size());
    v.lightCount = static_cast<int>(lights.size());
    v.envMapCount = static_cast<int>(envViews_.size());
    v.textureCount = static_cast<int>(textureViews_.size());
    v.domeLightIndex = domeLightIndex_;
    v.camera = camera;
    v.settings = settings;
    v.worldBounds = bounds_;
    return v;
}

void Scene::frameCameraOnContents() {
    Bounds3 b = bounds_;
    if (!b.valid()) {
        b.extend(Vec3(-1.0f));
        b.extend(Vec3(1.0f));
    }
    const Vec3 center = b.center();
    const float radius = std::max(0.25f, b.radius());
    const float aspect = float(std::max(1, settings.resolutionX)) / float(std::max(1, settings.resolutionY));
    const float sensorHeight = camera.sensorWidth / std::max(0.01f, aspect);
    const float fovX = 2.0f * std::atan(0.5f * camera.sensorWidth / std::max(1.0f, camera.focalLength));
    const float fovY = 2.0f * std::atan(0.5f * sensorHeight / std::max(1.0f, camera.focalLength));
    const float fov = std::max(0.05f, std::min(fovX, fovY));
    // Fit the bounding sphere into the narrower field of view with a small margin.
    const float distance = radius / std::sin(fov * 0.5f) * 0.9f;
    const Vec3 dir = normalize(Vec3(0.55f, 0.35f, 1.0f));
    const Vec3 eye = center + dir * distance;
    camera.cameraToWorld = lookAtMatrix(eye, center, Vec3(0.0f, 1.0f, 0.0f));
    camera.focusDistance = length(eye - center);
}

// ---------------------------------------------------------------------------
// Primitive builders
// ---------------------------------------------------------------------------

MeshPtr makeSphereMesh(float radius, int segmentsU, int segmentsV) {
    auto mesh = std::make_shared<Mesh>();
    mesh->name = "sphere";
    segmentsU = std::max(3, segmentsU);
    segmentsV = std::max(2, segmentsV);
    for (int v = 0; v <= segmentsV; ++v) {
        const float tv = float(v) / float(segmentsV);
        const float theta = tv * kPi;
        const float sinTheta = std::sin(theta);
        const float cosTheta = std::cos(theta);
        for (int u = 0; u <= segmentsU; ++u) {
            const float tu = float(u) / float(segmentsU);
            const float phi = tu * kTwoPi;
            const Vec3 n(sinTheta * std::cos(phi), cosTheta, sinTheta * std::sin(phi));
            mesh->positions.push_back(n * radius);
            mesh->normals.push_back(n);
            mesh->uvs.emplace_back(tu, 1.0f - tv);
        }
    }
    const int rowStride = segmentsU + 1;
    for (int v = 0; v < segmentsV; ++v) {
        for (int u = 0; u < segmentsU; ++u) {
            const uint32_t i0 = uint32_t(v * rowStride + u);
            const uint32_t i1 = uint32_t(v * rowStride + u + 1);
            const uint32_t i2 = uint32_t((v + 1) * rowStride + u + 1);
            const uint32_t i3 = uint32_t((v + 1) * rowStride + u);
            mesh->indices.insert(mesh->indices.end(), {i0, i2, i1});
            mesh->indices.insert(mesh->indices.end(), {i0, i3, i2});
        }
    }
    mesh->validate();
    return mesh;
}

MeshPtr makeGridMesh(float sizeX, float sizeZ, int divisionsX, int divisionsZ) {
    auto mesh = std::make_shared<Mesh>();
    mesh->name = "grid";
    divisionsX = std::max(1, divisionsX);
    divisionsZ = std::max(1, divisionsZ);
    for (int z = 0; z <= divisionsZ; ++z) {
        const float tz = float(z) / float(divisionsZ);
        for (int x = 0; x <= divisionsX; ++x) {
            const float tx = float(x) / float(divisionsX);
            mesh->positions.emplace_back((tx - 0.5f) * sizeX, 0.0f, (tz - 0.5f) * sizeZ);
            mesh->normals.emplace_back(0.0f, 1.0f, 0.0f);
            mesh->uvs.emplace_back(tx, tz);
        }
    }
    const int stride = divisionsX + 1;
    for (int z = 0; z < divisionsZ; ++z) {
        for (int x = 0; x < divisionsX; ++x) {
            const uint32_t i0 = uint32_t(z * stride + x);
            const uint32_t i1 = uint32_t(z * stride + x + 1);
            const uint32_t i2 = uint32_t((z + 1) * stride + x + 1);
            const uint32_t i3 = uint32_t((z + 1) * stride + x);
            mesh->indices.insert(mesh->indices.end(), {i0, i1, i2});
            mesh->indices.insert(mesh->indices.end(), {i0, i2, i3});
        }
    }
    mesh->validate();
    return mesh;
}

MeshPtr makeBoxMesh(Vec3 size) {
    auto mesh = std::make_shared<Mesh>();
    mesh->name = "box";
    const Vec3 h = size * 0.5f;
    const Vec3 faceNormals[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    const Vec3 faceTangents[6] = {{0, 0, -1}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {-1, 0, 0}};
    for (int f = 0; f < 6; ++f) {
        const Vec3 n = faceNormals[f];
        const Vec3 t = faceTangents[f];
        const Vec3 b = cross(n, t);
        const Vec3 center(n.x * h.x, n.y * h.y, n.z * h.z);
        const Vec3 du = Vec3(t.x * h.x, t.y * h.y, t.z * h.z);
        const Vec3 dv = Vec3(b.x * h.x, b.y * h.y, b.z * h.z);
        const uint32_t base = uint32_t(mesh->positions.size());
        mesh->positions.push_back(center - du - dv);
        mesh->positions.push_back(center + du - dv);
        mesh->positions.push_back(center + du + dv);
        mesh->positions.push_back(center - du + dv);
        for (int i = 0; i < 4; ++i) mesh->normals.push_back(n);
        mesh->uvs.insert(mesh->uvs.end(), {Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)});
        mesh->indices.insert(mesh->indices.end(), {base, base + 1, base + 2});
        mesh->indices.insert(mesh->indices.end(), {base, base + 2, base + 3});
    }
    mesh->validate();
    return mesh;
}

MeshPtr makeRectMesh(float width, float height) {
    auto mesh = std::make_shared<Mesh>();
    mesh->name = "rect";
    const float hw = width * 0.5f;
    const float hh = height * 0.5f;
    mesh->positions = {Vec3(-hw, -hh, 0.0f), Vec3(hw, -hh, 0.0f), Vec3(hw, hh, 0.0f), Vec3(-hw, hh, 0.0f)};
    mesh->normals.assign(4, Vec3(0.0f, 0.0f, 1.0f));
    mesh->uvs = {Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)};
    mesh->indices = {0, 1, 2, 0, 2, 3};
    mesh->validate();
    return mesh;
}

MeshPtr makeDiskMesh(float radius, int segments) {
    auto mesh = std::make_shared<Mesh>();
    mesh->name = "disk";
    segments = std::max(3, segments);
    mesh->positions.emplace_back(0.0f, 0.0f, 0.0f);
    mesh->normals.emplace_back(0.0f, 0.0f, 1.0f);
    mesh->uvs.emplace_back(0.5f, 0.5f);
    for (int i = 0; i <= segments; ++i) {
        const float t = float(i) / float(segments);
        const float a = t * kTwoPi;
        mesh->positions.emplace_back(std::cos(a) * radius, std::sin(a) * radius, 0.0f);
        mesh->normals.emplace_back(0.0f, 0.0f, 1.0f);
        mesh->uvs.emplace_back(0.5f + 0.5f * std::cos(a), 0.5f + 0.5f * std::sin(a));
    }
    for (int i = 1; i <= segments; ++i) {
        mesh->indices.insert(mesh->indices.end(), {0u, uint32_t(i), uint32_t(i + 1)});
    }
    mesh->validate();
    return mesh;
}

MeshPtr makeTubeMesh(float radius, float height, int segments) {
    auto mesh = std::make_shared<Mesh>();
    mesh->name = "tube";
    segments = std::max(3, segments);
    for (int i = 0; i <= segments; ++i) {
        const float t = float(i) / float(segments);
        const float a = t * kTwoPi;
        const Vec3 n(std::cos(a), 0.0f, std::sin(a));
        mesh->positions.push_back(Vec3(n.x * radius, -height * 0.5f, n.z * radius));
        mesh->normals.push_back(n);
        mesh->uvs.emplace_back(t, 0.0f);
        mesh->positions.push_back(Vec3(n.x * radius, height * 0.5f, n.z * radius));
        mesh->normals.push_back(n);
        mesh->uvs.emplace_back(t, 1.0f);
    }
    for (int i = 0; i < segments; ++i) {
        const uint32_t b = uint32_t(i * 2);
        mesh->indices.insert(mesh->indices.end(), {b, b + 1, b + 3});
        mesh->indices.insert(mesh->indices.end(), {b, b + 3, b + 2});
    }
    mesh->validate();
    return mesh;
}

}  // namespace sol
