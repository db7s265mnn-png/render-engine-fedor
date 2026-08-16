#include "scene/scene.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>

#include "core/log.h"
#include "scene/triangulate.h"
#include "scene/volume_grid.h"
#include "solstice_config.h"

namespace sol {

void Mesh::ensureRenderTriangles() {
    if (!indices.empty()) return;
    if (!hasPolygonCage()) return;
    triangulateMeshFaces(positions, faceVertexCounts, faceVertexIndices, indices, &triEdgeMask);
}

void Mesh::ensurePolygonCageFromTriangles() {
    if (hasPolygonCage()) return;
    if (indices.size() < 3) return;
    faceVertexCounts.clear();
    faceVertexIndices.clear();
    faceVertexCounts.reserve(indices.size() / 3);
    faceVertexIndices.reserve(indices.size());
    for (size_t t = 0; t + 2 < indices.size(); t += 3) {
        faceVertexCounts.push_back(3);
        faceVertexIndices.push_back(indices[t + 0]);
        faceVertexIndices.push_back(indices[t + 1]);
        faceVertexIndices.push_back(indices[t + 2]);
    }
    // Every edge of a triangle face is a boundary edge.
    triEdgeMask.assign(indices.size() / 3, uint8_t(7));
}

void Mesh::captureWireCage() {
    wireIndices.clear();
    wirePositions.clear();
    wireNormals.clear();
    if (positions.empty()) return;

    // Compact unique undirected edges from the authored polygon cage (preferred)
    // or from render triangles when no cage exists.
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    if (hasPolygonCage()) {
        size_t cursor = 0;
        for (uint32_t count : faceVertexCounts) {
            if (count < 2 || cursor + size_t(count) > faceVertexIndices.size()) {
                cursor += size_t(count);
                continue;
            }
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t a = faceVertexIndices[cursor + i];
                uint32_t b = faceVertexIndices[cursor + ((i + 1) % count)];
                if (a > b) std::swap(a, b);
                if (a != b) edges.emplace_back(a, b);
            }
            cursor += size_t(count);
        }
    } else if (indices.size() >= 3) {
        for (size_t t = 0; t + 2 < indices.size(); t += 3) {
            const uint32_t v[3] = {indices[t], indices[t + 1], indices[t + 2]};
            for (int e = 0; e < 3; ++e) {
                uint32_t a = v[e];
                uint32_t b = v[(e + 1) % 3];
                if (a > b) std::swap(a, b);
                if (a != b) edges.emplace_back(a, b);
            }
        }
    }
    if (edges.empty()) return;
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    if (normals.size() != positions.size()) computeNormalsIfMissing();

    // Remap referenced verts into a compact wirePositions buffer (cage-sized).
    std::unordered_map<uint32_t, uint32_t> remap;
    remap.reserve(edges.size() * 2);
    auto mapVert = [&](uint32_t old) -> uint32_t {
        auto it = remap.find(old);
        if (it != remap.end()) return it->second;
        const uint32_t neu = uint32_t(wirePositions.size());
        if (old < positions.size()) {
            wirePositions.push_back(positions[old]);
            if (old < normals.size()) wireNormals.push_back(normals[old]);
            else wireNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
        } else {
            wirePositions.push_back(Vec3(0.0f));
            wireNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
        }
        remap.emplace(old, neu);
        return neu;
    };
    wireIndices.reserve(edges.size() * 2);
    for (const auto& e : edges) {
        wireIndices.push_back(mapVert(e.first));
        wireIndices.push_back(mapVert(e.second));
    }
}

void Mesh::computeBounds() {
    bounds = Bounds3();
    for (const Vec3& p : positions) bounds.extend(p);
    if (boundsPadding > 0.0f && bounds.valid()) {
        bounds.lo = bounds.lo - Vec3(boundsPadding);
        bounds.hi = bounds.hi + Vec3(boundsPadding);
    }
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
    // Densify from the polygon cage only when render triangles are missing.
    // Never rebuild over an existing triangulation (subdiv / dicing would be wiped).
    if (indices.empty() && hasPolygonCage()) {
        triangulateMeshFaces(positions, faceVertexCounts, faceVertexIndices, indices, &triEdgeMask);
    }
    // Drop degenerate or out of range triangles so the BVH builders stay happy.
    std::vector<uint32_t> cleaned;
    cleaned.reserve(indices.size());
    const uint32_t vertexCount = static_cast<uint32_t>(positions.size());
    std::vector<uint8_t> cleanedMask;
    const bool keepMask = triEdgeMask.size() == indices.size() / 3 && !triEdgeMask.empty();
    if (keepMask) cleanedMask.reserve(triEdgeMask.size());
    for (size_t t = 0; t + 2 < indices.size(); t += 3) {
        const uint32_t i0 = indices[t + 0], i1 = indices[t + 1], i2 = indices[t + 2];
        if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) continue;
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;
        const Vec3 n = cross(positions[i1] - positions[i0], positions[i2] - positions[i0]);
        if (lengthSquared(n) <= 0.0f) continue;
        cleaned.push_back(i0);
        cleaned.push_back(i1);
        cleaned.push_back(i2);
        if (keepMask) cleanedMask.push_back(triEdgeMask[t / 3]);
    }
    if (cleaned.size() != indices.size()) {
        indices.swap(cleaned);
        if (keepMask && cleanedMask.size() == indices.size() / 3) triEdgeMask.swap(cleanedMask);
        else triEdgeMask.clear();
    }
    if (!uvs.empty() && uvs.size() != positions.size()) uvs.clear();
    computeNormalsIfMissing();
    if (wireIndices.empty()) captureWireCage();
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
    if (bounds.valid()) {
        v.boundsLo = bounds.lo;
        v.boundsHi = bounds.hi;
    }
    v.restPositions =
        (restPositions.size() == positions.size() && !restPositions.empty()) ? restPositions.data() : nullptr;
    v.restNormals =
        (restNormals.size() == positions.size() && !restNormals.empty()) ? restNormals.data() : nullptr;
    v.triEdgeMask =
        (triEdgeMask.size() == indices.size() / 3 && !triEdgeMask.empty()) ? triEdgeMask.data() : nullptr;
    if (!wireIndices.empty() && !wirePositions.empty() && (wireIndices.size() % 2) == 0) {
        v.wireIndices = wireIndices.data();
        v.wirePositions = wirePositions.data();
        v.wireNormals =
            (wireNormals.size() == wirePositions.size()) ? wireNormals.data() : nullptr;
        v.wireEdgeCount = uint32_t(wireIndices.size() / 2);
        v.wireVertexCount = uint32_t(wirePositions.size());
    }
    v.motionKeyCount = 1;
    v.motionPositions = nullptr;
    if (!motionPositions.empty() && !positions.empty()) {
        const size_t keyCount = motionPositions.size() + 1;
        const size_t vertexCount = positions.size();
        bool ok = true;
        for (const auto& key : motionPositions) {
            if (key.size() != vertexCount) {
                ok = false;
                break;
            }
        }
        if (ok) {
            motionPositionsPacked_.resize(keyCount * vertexCount);
            std::copy(positions.begin(), positions.end(), motionPositionsPacked_.begin());
            for (size_t k = 0; k < motionPositions.size(); ++k) {
                std::copy(motionPositions[k].begin(), motionPositions[k].end(),
                          motionPositionsPacked_.begin() + (k + 1) * vertexCount);
            }
            v.motionPositions = motionPositionsPacked_.data();
            v.motionKeyCount = int(keyCount);
        }
    }
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
    double sum = 0.0;
    for (int y = 0; y < h; ++y) {
        // sin(theta) accounts for the equirectangular solid angle distortion.
        const float theta = (float(y) + 0.5f) / float(h) * kPi;
        const float sinTheta = std::sin(theta);
        for (int x = 0; x < w; ++x) {
            const Vec3 c = image.rgb(x, y);
            const float v = std::max(0.0f, luminance(c)) * sinTheta;
            luminanceFunc[size_t(y) * w + x] = v;
            sum += double(v);
        }
    }
    // PBRT MIS compensation (Ch.2 / §12.5): sharpen the env sampling PDF by
    // subtracting the mean so bright regions are not undersampled relative to
    // BSDF sampling. Uniform leftover mass is recovered via BSDF→env MIS.
    const size_t n = luminanceFunc.size();
    if (n > 0 && sum > 0.0) {
        const float c = float(sum / double(n));
        double sum2 = 0.0;
        for (float& v : luminanceFunc) {
            v = std::max(0.0f, v - c);
            sum2 += double(v);
        }
        // Degenerate HDRI (flat): keep the original distribution.
        if (sum2 <= 1e-20) {
            for (int y = 0; y < h; ++y) {
                const float theta = (float(y) + 0.5f) / float(h) * kPi;
                const float sinTheta = std::sin(theta);
                for (int x = 0; x < w; ++x) {
                    const Vec3 col = image.rgb(x, y);
                    luminanceFunc[size_t(y) * w + x] = std::max(0.0f, luminance(col)) * sinTheta;
                }
            }
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

int Scene::addVolume(VolumeGridPtr volume) {
    volumes.push_back(std::move(volume));
    return static_cast<int>(volumes.size()) - 1;
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

int Scene::addMedium(const MediumData& medium, const std::string& vdbPath) {
    MediumData m = medium;
    if (!vdbPath.empty()) {
        // Assign a volumeIndex pointing at the vdbPath entry.
        // Deduplicate by path so the same VDB file is not loaded twice.
        int vi = -1;
        for (int i = 0; i < int(volumePaths.size()); ++i) {
            if (volumePaths[size_t(i)] == vdbPath) {
                vi = i;
                break;
            }
        }
        if (vi < 0) {
            vi = int(volumePaths.size());
            volumePaths.push_back(vdbPath);
        }
        m.volumeIndex = vi;
#if !SOLSTICE_HAVE_OPENVDB
        // Without OpenVDB, type-2 media still use the authored homogeneous σa/σs.
        logWarning("OpenVDB not in this build — volume '" + vdbPath +
                   "' uses homogeneous σa/σs/density fallback");
#endif
    }
    media.push_back(m);
    return static_cast<int>(media.size()) - 1;
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
    // Idempotent: drop proxies from a previous finalize() so re-finalizing a
    // mutated scene never leaves phantom light geometry behind.
    instances.erase(std::remove_if(instances.begin(), instances.end(),
                                   [](const InstanceData& inst) { return inst.lightIndex >= 0; }),
                    instances.end());
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

// ---------------------------------------------------------------------------
// Light BVH construction helpers (host-only, scene.cpp internal)
// ---------------------------------------------------------------------------

// Approximate emitted power for one light (same formula as lightFluxWeight in lights.h).
static float lightPowerForBvh(const LightData& l,
                               const std::vector<std::shared_ptr<EnvironmentMap>>& envMaps) {
    const Vec3 emitted = l.emittedRadiance();
    const float intens = std::max(1e-8f, (emitted.x + emitted.y + emitted.z) * (1.0f / 3.0f));
    switch (l.type) {
        case kLightDome:
            if (l.envIndex >= 0 && l.envIndex < int(envMaps.size()) && envMaps[l.envIndex] &&
                envMaps[l.envIndex]->distribution.valid())
                return intens * std::max(1e-4f, envMaps[l.envIndex]->distribution.integral());
            return intens * 4.0f;
        case kLightRect: {
            const Vec3 ax = transformVector(l.xform, Vec3(l.width,  0.0f, 0.0f));
            const Vec3 ay = transformVector(l.xform, Vec3(0.0f, l.height, 0.0f));
            return intens * std::max(1e-6f, length(cross(ax, ay)));
        }
        case kLightDisk: {
            const Vec3 ax = transformVector(l.xform, Vec3(l.radius, 0.0f, 0.0f));
            const Vec3 ay = transformVector(l.xform, Vec3(0.0f, l.radius, 0.0f));
            return intens * std::max(1e-6f, kPi * length(cross(ax, ay)));
        }
        case kLightSphere: {
            const float sx = length(transformVector(l.xform, Vec3(1.0f, 0.0f, 0.0f)));
            const float sy = length(transformVector(l.xform, Vec3(0.0f, 1.0f, 0.0f)));
            const float sz = length(transformVector(l.xform, Vec3(0.0f, 0.0f, 1.0f)));
            const float r = l.radius * (sx + sy + sz) * (1.0f / 3.0f);
            return intens * std::max(1e-6f, 4.0f * kPi * r * r);
        }
        case kLightDistant: {
            const float halfAngle = (l.angle * kPi / 180.0f) * 0.5f;
            if (halfAngle < 1e-8f) return intens;
            // Keep in sync with lightFluxWeight in lights.h.
            if (l.normalize) return intens;
            return intens * std::max(1e-6f, kTwoPi * (1.0f - std::cos(halfAngle)));
        }
        case kLightPoint:
        default:
            return intens;
    }
}

// World-space AABB for a finite light (point, sphere, rect, disk).
static Bounds3 lightWorldBounds(const LightData& l) {
    Bounds3 b;
    const Vec3 origin = transformPoint(l.xform, Vec3(0.0f, 0.0f, 0.0f));
    switch (l.type) {
        case kLightSphere: {
            const float sx = length(transformVector(l.xform, Vec3(1.0f, 0.0f, 0.0f)));
            const float sy = length(transformVector(l.xform, Vec3(0.0f, 1.0f, 0.0f)));
            const float sz = length(transformVector(l.xform, Vec3(0.0f, 0.0f, 1.0f)));
            const float r = l.radius * (sx + sy + sz) * (1.0f / 3.0f);
            b.extend(origin - Vec3(r));
            b.extend(origin + Vec3(r));
            break;
        }
        case kLightRect: {
            const float hw = l.width  * 0.5f;
            const float hh = l.height * 0.5f;
            b.extend(transformPoint(l.xform, Vec3(-hw, -hh, 0.0f)));
            b.extend(transformPoint(l.xform, Vec3( hw, -hh, 0.0f)));
            b.extend(transformPoint(l.xform, Vec3( hw,  hh, 0.0f)));
            b.extend(transformPoint(l.xform, Vec3(-hw,  hh, 0.0f)));
            break;
        }
        case kLightDisk: {
            const float r = l.radius;
            b.extend(transformPoint(l.xform, Vec3(-r, -r, 0.0f)));
            b.extend(transformPoint(l.xform, Vec3( r, -r, 0.0f)));
            b.extend(transformPoint(l.xform, Vec3( r,  r, 0.0f)));
            b.extend(transformPoint(l.xform, Vec3(-r,  r, 0.0f)));
            break;
        }
        default: // point and anything else: tiny box around origin
            break;
    }
    if (!b.valid()) {
        // Point light or degenerate: a small box around the origin
        b.extend(origin - Vec3(1e-3f));
        b.extend(origin + Vec3(1e-3f));
    }
    return b;
}

void Scene::buildLightBvh() {
    lightBvhNodes_.clear();
    infiniteLightIndices_.clear();
    infiniteLightPower_ = 0.f;
    finiteLightPower_   = 0.f;

    struct BuildEntry {
        int     lightIdx;
        Bounds3 bounds;
        Vec3    center;
        float   power;
    };
    std::vector<BuildEntry> finites;
    finites.reserve(lights.size());

    for (int i = 0; i < int(lights.size()); ++i) {
        const LightData& l = lights[i];
        const float pw = lightPowerForBvh(l, envMaps);
        if (l.type == kLightDome || l.type == kLightDistant) {
            infiniteLightIndices_.push_back(i);
            infiniteLightPower_ += pw;
        } else {
            BuildEntry e;
            e.lightIdx = i;
            e.bounds   = lightWorldBounds(l);
            e.center   = (e.bounds.lo + e.bounds.hi) * 0.5f;
            e.power    = pw;
            finiteLightPower_ += pw;
            finites.push_back(e);
        }
    }

    if (finites.empty()) return;

    // Few finite lights: skip the BVH. Position-aware AABB importance
    // (power / dist²_to_box) imprints axis-aligned "buckets" on diffuse floors
    // under area lights — visible as square caustic sampling structure on every
    // integrator. Flux-only selection is unbiased and cheaper for small N.
    constexpr int kLightBvhMinFinites = 32;
    if (int(finites.size()) < kLightBvhMinFinites) return;

    // Pre-allocate: a full binary tree with N leaves has at most 2*N - 1 nodes.
    lightBvhNodes_.reserve(2 * finites.size());

    // Recursive median-split builder (iterative via std::function to avoid ABI issues).
    std::function<int(int, int)> build = [&](int first, int count) -> int {
        const int nodeIdx = int(lightBvhNodes_.size());
        lightBvhNodes_.push_back(LightBvhNode{});

        // Compute subtree AABB and total power.
        Bounds3 bbox;
        float   totalPow = 0.f;
        for (int i = first; i < first + count; ++i) {
            bbox.extend(finites[i].bounds);
            totalPow += finites[i].power;
        }
        lightBvhNodes_[nodeIdx].bMin  = bbox.valid() ? bbox.lo : Vec3(-1.f);
        lightBvhNodes_[nodeIdx].bMax  = bbox.valid() ? bbox.hi : Vec3( 1.f);
        lightBvhNodes_[nodeIdx].power = totalPow;

        if (count == 1) {
            lightBvhNodes_[nodeIdx].childOrLight = finites[first].lightIdx;
            lightBvhNodes_[nodeIdx].rightChild   = -1;
            lightBvhNodes_[nodeIdx].isLeaf       = 1;
            return nodeIdx;
        }

        // Split along the longest axis of light centers.
        Bounds3 centerBounds;
        for (int i = first; i < first + count; ++i)
            centerBounds.extend(finites[i].center);
        const Vec3 ext = centerBounds.valid() ? centerBounds.extent() : Vec3(1.f);
        int axis = 0;
        if (ext.y > ext.x) axis = 1;
        if (ext.z > (axis == 0 ? ext.x : ext.y)) axis = 2;

        const int mid = first + count / 2;
        std::nth_element(finites.begin() + first,
                         finites.begin() + mid,
                         finites.begin() + first + count,
                         [axis](const BuildEntry& a, const BuildEntry& b) {
                             return a.center[axis] < b.center[axis];
                         });

        lightBvhNodes_[nodeIdx].isLeaf = 0;
        // Recurse; use nodeIdx to write back afterwards (vector may grow but
        // reserve above ensures no reallocation, so indices stay valid).
        const int leftIdx  = build(first,       mid - first);
        const int rightIdx = build(mid,   (first + count) - mid);
        lightBvhNodes_[nodeIdx].childOrLight = leftIdx;
        lightBvhNodes_[nodeIdx].rightChild   = rightIdx;
        return nodeIdx;
    };

    build(0, int(finites.size()));
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
            view.mipCount = image->mipCount() > 0 ? image->mipCount() : 1;
            if (image->isUdimAtlas()) {
                view.udimGridU = image->udimGridU();
                view.udimGridV = image->udimGridV();
            }
        }
        textureViews_.push_back(view);
    }

    // Build light BVH after env views are ready (lightPowerForBvh uses them).
    buildLightBvh();
}

void Scene::refreshMeshViews() {
    meshViews_.clear();
    meshViews_.reserve(meshes.size());
    for (const MeshPtr& mesh : meshes) meshViews_.push_back(mesh ? mesh->view() : MeshView());
}

SceneView Scene::view() const {
    SceneView v;
    v.meshes = meshViews_.data();
    v.instances = instances.data();
    v.materials = materials.data();
    v.lights = lights.data();
    v.envMaps = envViews_.data();
    v.textures = textureViews_.data();
    v.procedurals = procedurals.data();
    v.media = media.empty() ? nullptr : media.data();
    volumePtrs_.clear();
    volumePtrs_.reserve(volumes.size());
    for (const VolumeGridPtr& g : volumes) volumePtrs_.push_back(g.get());
    v.volumes = volumePtrs_.empty() ? nullptr : volumePtrs_.data();
    v.meshCount = static_cast<int>(meshViews_.size());
    v.instanceCount = static_cast<int>(instances.size());
    v.materialCount = static_cast<int>(materials.size());
    v.lightCount = static_cast<int>(lights.size());
    v.envMapCount = static_cast<int>(envViews_.size());
    v.textureCount = static_cast<int>(textureViews_.size());
    v.proceduralCount = static_cast<int>(procedurals.size());
    v.mediumCount = static_cast<int>(media.size());
    v.volumeCount = static_cast<int>(volumePtrs_.size());
    v.domeLightIndex = domeLightIndex_;
    v.hasDispersion = 0;
    for (const Material& m : materials) {
        if (m.dispersionAbbe > 0.0f && m.transmission > 1e-4f) {
            v.hasDispersion = 1;
            break;
        }
    }
    v.camera = camera;
    v.settings = settings;
    v.worldBounds = bounds_;
    v.motionXforms = motionXforms.empty() ? nullptr : motionXforms.data();
    v.cameraMotionXforms = cameraMotionXforms.empty() ? nullptr : cameraMotionXforms.data();
    v.cameraMotionKeyCount = cameraMotionXforms.empty() ? 1 : int(cameraMotionXforms.size());
    v.lightBvh           = lightBvhNodes_.empty() ? nullptr : lightBvhNodes_.data();
    v.lightBvhNodeCount  = int(lightBvhNodes_.size());
    v.infiniteLightIndices = infiniteLightIndices_.empty() ? nullptr : infiniteLightIndices_.data();
    v.infiniteLightCount   = int(infiniteLightIndices_.size());
    v.infiniteLightPower   = infiniteLightPower_;
    v.finiteLightPower     = finiteLightPower_;
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
            mesh->indices.insert(mesh->indices.end(), {i0, i2, i1});
            mesh->indices.insert(mesh->indices.end(), {i0, i3, i2});
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
