#include "render/scene_picker.h"

#include <embree4/rtcore.h>
#include <cstring>
#include <vector>

#include "render/integrator.h"

namespace sol {
namespace {

RTCScene buildPickScene(RTCDevice device, const ScenePtr& scene) {
    if (!device || !scene) return nullptr;

    std::vector<RTCScene> meshScenes(scene->meshes.size(), nullptr);
    for (size_t i = 0; i < scene->meshes.size(); ++i) {
        const MeshPtr& mesh = scene->meshes[i];
        if (!mesh || mesh->indices.empty()) continue;
        RTCScene meshScene = rtcNewScene(device);
        RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
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
        std::memcpy(vertices, mesh->positions.data(), vertexCount * 3 * sizeof(float));
        std::memcpy(indices, mesh->indices.data(), triCount * 3 * sizeof(uint32_t));
        rtcCommitGeometry(geom);
        rtcAttachGeometry(meshScene, geom);
        rtcReleaseGeometry(geom);
        rtcCommitScene(meshScene);
        meshScenes[i] = meshScene;
    }

    RTCScene topScene = rtcNewScene(device);
    rtcSetSceneFlags(topScene, RTC_SCENE_FLAG_ROBUST);
    for (size_t i = 0; i < scene->instances.size(); ++i) {
        const InstanceData& inst = scene->instances[i];
        if (inst.meshIndex < 0 || inst.meshIndex >= int(meshScenes.size())) continue;
        RTCScene meshScene = meshScenes[size_t(inst.meshIndex)];
        if (!meshScene) continue;
        RTCGeometry instGeom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
        rtcSetGeometryInstancedScene(instGeom, meshScene);
        rtcSetGeometryTimeStepCount(instGeom, 1);
        rtcSetGeometryTransform(instGeom, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, inst.xform.m);
        rtcCommitGeometry(instGeom);
        rtcAttachGeometryByID(topScene, instGeom, static_cast<unsigned int>(i));
        rtcReleaseGeometry(instGeom);
    }
    rtcCommitScene(topScene);

    for (RTCScene meshScene : meshScenes) {
        if (meshScene) rtcReleaseScene(meshScene);
    }
    return topScene;
}

}  // namespace

bool pickSceneSurface(const ScenePtr& scene, const Mat4& cameraToWorld, float aspectRatio, float u, float v,
                      Vec3& hitPoint) {
    if (!scene || scene->instances.empty()) return false;

    RTCDevice device = rtcNewDevice("verbose=0");
    if (!device) return false;
    RTCScene pickScene = buildPickScene(device, scene);
    if (!pickScene) {
        rtcReleaseDevice(device);
        return false;
    }

    SceneView view = scene->view();
    view.camera.cameraToWorld = cameraToWorld;

    const float px = u * float(view.settings.resolutionX);
    const float py = v * float(view.settings.resolutionY);
    Vec3 origin, direction;
    generateCameraRay(view, px, py, 0.5f, 0.5f, origin, direction);

    RTCRayHit rayhit{};
    rayhit.ray.org_x = origin.x;
    rayhit.ray.org_y = origin.y;
    rayhit.ray.org_z = origin.z;
    rayhit.ray.dir_x = direction.x;
    rayhit.ray.dir_y = direction.y;
    rayhit.ray.dir_z = direction.z;
    rayhit.ray.tnear = 0.001f;
    rayhit.ray.tfar = 1e6f;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(pickScene, &rayhit, nullptr);

    rtcReleaseScene(pickScene);
    rtcReleaseDevice(device);

    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) return false;
    hitPoint = origin + direction * rayhit.ray.tfar;
    return true;
}

}  // namespace sol
