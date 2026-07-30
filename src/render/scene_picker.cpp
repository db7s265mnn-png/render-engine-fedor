#include "render/scene_picker.h"

#include <embree4/rtcore.h>
#include <cstring>
#include <mutex>
#include <vector>

#include "core/rng.h"
#include "render/cpu/polynomial_optics.h"
#include "render/integrator.h"

namespace sol {
namespace {

struct PickCache {
    std::mutex mutex;
    const Scene* sceneKey = nullptr;
    ScenePickMode mode = ScenePickMode::Interactive;
    RTCDevice device = nullptr;
    RTCScene topScene = nullptr;
    std::vector<RTCScene> meshScenes;

    void clear() {
        if (topScene) {
            rtcReleaseScene(topScene);
            topScene = nullptr;
        }
        for (RTCScene meshScene : meshScenes) {
            if (meshScene) rtcReleaseScene(meshScene);
        }
        meshScenes.clear();
        if (device) {
            rtcReleaseDevice(device);
            device = nullptr;
        }
        sceneKey = nullptr;
    }

    ~PickCache() { clear(); }
};

PickCache& pickCache() {
    static PickCache cache;
    return cache;
}

const Mat4& instancePickXform(const Scene& scene, size_t instanceIndex) {
    if (instanceIndex < scene.pickXforms.size()) return scene.pickXforms[instanceIndex];
    return scene.instances[instanceIndex].xform;
}

// Interactive picks use shutter-center geometry: lerp open→close when MB keys exist.
void fillPickVertices(const Mesh& mesh, float* vertices) {
    const size_t vertexCount = mesh.positions.size();
    if (mesh.motionPositions.empty()) {
        std::memcpy(vertices, mesh.positions.data(), vertexCount * 3 * sizeof(float));
        return;
    }
    const std::vector<Vec3>& last = mesh.motionPositions.back();
    if (last.size() != vertexCount) {
        std::memcpy(vertices, mesh.positions.data(), vertexCount * 3 * sizeof(float));
        return;
    }
    for (size_t i = 0; i < vertexCount; ++i) {
        const Vec3 p = lerp(mesh.positions[i], last[i], 0.5f);
        vertices[i * 3 + 0] = p.x;
        vertices[i * 3 + 1] = p.y;
        vertices[i * 3 + 2] = p.z;
    }
}

bool ensurePickScene(const ScenePtr& scene, ScenePickMode mode) {
    PickCache& cache = pickCache();
    if (cache.sceneKey == scene.get() && cache.topScene && cache.mode == mode) return true;
    cache.clear();
    if (!scene || scene->instances.empty()) return false;

    cache.device = rtcNewDevice("verbose=0");
    if (!cache.device) return false;
    cache.mode = mode;

    cache.meshScenes.assign(scene->meshes.size(), nullptr);
    for (size_t i = 0; i < scene->meshes.size(); ++i) {
        const MeshPtr& mesh = scene->meshes[i];
        if (!mesh || mesh->indices.empty()) continue;
        RTCScene meshScene = rtcNewScene(cache.device);
        RTCGeometry geom = rtcNewGeometry(cache.device, RTC_GEOMETRY_TYPE_TRIANGLE);
        const size_t vertexCount = mesh->positions.size();
        const size_t triCount = mesh->indices.size() / 3;
        float* vertices = static_cast<float*>(rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), vertexCount));
        uint32_t* indices = static_cast<uint32_t*>(rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(uint32_t), triCount));
        if (!vertices || !indices) {
            rtcReleaseGeometry(geom);
            rtcReleaseScene(meshScene);
            continue;
        }
        if (mode == ScenePickMode::Interactive)
            fillPickVertices(*mesh, vertices);
        else
            std::memcpy(vertices, mesh->positions.data(), vertexCount * 3 * sizeof(float));
        std::memcpy(indices, mesh->indices.data(), triCount * 3 * sizeof(uint32_t));
        rtcCommitGeometry(geom);
        rtcAttachGeometry(meshScene, geom);
        rtcReleaseGeometry(geom);
        rtcCommitScene(meshScene);
        cache.meshScenes[i] = meshScene;
    }

    cache.topScene = rtcNewScene(cache.device);
    rtcSetSceneFlags(cache.topScene, RTC_SCENE_FLAG_ROBUST);
    for (size_t i = 0; i < scene->instances.size(); ++i) {
        const InstanceData& inst = scene->instances[i];
        if (inst.meshIndex < 0 || inst.meshIndex >= int(cache.meshScenes.size())) continue;
        RTCScene meshScene = cache.meshScenes[size_t(inst.meshIndex)];
        if (!meshScene) continue;
        RTCGeometry instGeom = rtcNewGeometry(cache.device, RTC_GEOMETRY_TYPE_INSTANCE);
        rtcSetGeometryInstancedScene(instGeom, meshScene);
        rtcSetGeometryTimeStepCount(instGeom, 1);
        const Mat4& xform =
            mode == ScenePickMode::Interactive ? instancePickXform(*scene, i) : inst.xform;
        rtcSetGeometryTransform(instGeom, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, xform.m);
        rtcCommitGeometry(instGeom);
        rtcAttachGeometryByID(cache.topScene, instGeom, static_cast<unsigned int>(i));
        rtcReleaseGeometry(instGeom);
    }
    rtcCommitScene(cache.topScene);
    cache.sceneKey = scene.get();
    return cache.topScene != nullptr;
}

bool generatePickRay(const CameraData& camera, float px, float py, int resolutionX, int resolutionY,
                     ScenePickMode mode, Vec3& origin, Vec3& direction) {
    CameraData cam = camera;
    // Interactive navigation / select / focus always use a pinhole ray so lens
    // effects and shutter sampling cannot skew the hit.
    if (mode == ScenePickMode::Interactive) {
        cam.opticalModel = 0;
        cam.fStop = 0.0f;
    }

    SceneView view{};
    view.camera = cam;
    view.settings.resolutionX = std::max(1, resolutionX);
    view.settings.resolutionY = std::max(1, resolutionY);

    if (mode == ScenePickMode::Beauty && cam.opticalModel == 1) {
        PolynomialOpticsCamera lens;
        lens.prepare(cam);
        if (!lens.active) {
            generateCameraRay(view, px, py, 0.5f, 0.5f, origin, direction);
            return true;
        }
        Rng rng(0xC0FFEEu, 0xF0CALu);
        float tau = 1.0f;
        if (!generatePolynomialOpticsRay(lens, cam, px, py, resolutionX, resolutionY, 0.5f, 0.5f, rng,
                                         origin, direction, -1.0f, &tau)) {
            generateCameraRay(view, px, py, 0.5f, 0.5f, origin, direction);
        }
        return true;
    }
    generateCameraRay(view, px, py, 0.5f, 0.5f, origin, direction);
    return true;
}

}  // namespace

bool pickSceneSurface(const ScenePtr& scene, const CameraData& camera, int resolutionX, int resolutionY,
                      float u, float v, Vec3& hitPoint, int* instanceIndexOut, ScenePickMode mode) {
    if (instanceIndexOut) *instanceIndexOut = -1;
    if (!scene || scene->instances.empty()) return false;

    PickCache& cache = pickCache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (!ensurePickScene(scene, mode)) return false;

    const float px = u * float(std::max(1, resolutionX));
    const float py = v * float(std::max(1, resolutionY));
    Vec3 origin, direction;
    generatePickRay(camera, px, py, resolutionX, resolutionY, mode, origin, direction);

    RTCRayHit rayhit{};
    rayhit.ray.org_x = origin.x;
    rayhit.ray.org_y = origin.y;
    rayhit.ray.org_z = origin.z;
    rayhit.ray.dir_x = direction.x;
    rayhit.ray.dir_y = direction.y;
    rayhit.ray.dir_z = direction.z;
    rayhit.ray.tnear = 0.0001f;
    rayhit.ray.tfar = 1e7f;
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(cache.topScene, &rayhit, nullptr);

    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) return false;
    hitPoint = origin + direction * rayhit.ray.tfar;
    if (instanceIndexOut) {
        const unsigned int instID = rayhit.hit.instID[0];
        *instanceIndexOut =
            instID == RTC_INVALID_GEOMETRY_ID ? int(rayhit.hit.geomID) : int(instID);
    }
    return true;
}

}  // namespace sol
