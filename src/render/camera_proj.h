// Pinhole / thin-lens-center camera importance for light-tracing splats.
// Shared by CPU BDPT and OptiX (device-safe, no STL).
#pragma once

#include "core/math.h"
#include "scene/types.h"

namespace sol {

struct CameraProj {
    Mat4 worldToCam;
    Vec3 camPos{0.0f};
    float focal = 50.0f;    // mm
    float sensorW = 36.0f;  // mm
    float sensorH = 24.0f;
    float resX = 1.0f;
    float resY = 1.0f;
    float pixelArea = 1.0f;  // sensor mm² per pixel
    bool valid = false;
};

SR_INL SR_HD CameraProj buildCameraProj(const SceneView& scene) {
    CameraProj c;
    const CameraData& cam = scene.camera;
    c.worldToCam = inverse(cam.cameraToWorld);
    c.camPos = transformPoint(cam.cameraToWorld, Vec3(0.0f, 0.0f, 0.0f));
    c.focal = srMax(1e-3f, cam.focalLength);
    c.resX = float(srMax(1, scene.settings.resolutionX));
    c.resY = float(srMax(1, scene.settings.resolutionY));
    c.sensorW = cam.sensorWidth;
    c.sensorH = cam.sensorWidth * (c.resY / c.resX);
    c.pixelArea = (c.sensorW / c.resX) * (c.sensorH / c.resY);
    c.valid = c.pixelArea > 1e-12f;
    return c;
}

// Projects a world point onto the raster. Returns false when outside the
// frustum. cosTheta is measured against the optical axis.
SR_INL SR_HD bool projectToPixel(const CameraProj& proj, Vec3 pWorld, float& px, float& py, float& cosTheta,
                                 float& dist2) {
    const Vec3 pc = transformPoint(proj.worldToCam, pWorld);
    if (pc.z >= -1e-5f) return false;  // behind the pinhole
    dist2 = lengthSquared(pc);
    cosTheta = -pc.z / sqrtf(srMax(1e-12f, dist2));
    const float sx = pc.x * (proj.focal / -pc.z);
    const float sy = pc.y * (proj.focal / -pc.z);
    px = (sx / proj.sensorW + 0.5f) * proj.resX;
    py = (0.5f - sy / proj.sensorH) * proj.resY;
    return px >= 0.0f && px < proj.resX && py >= 0.0f && py < proj.resY;
}

// Solid-angle pdf of the camera generating a ray toward direction with cosTheta
// (uniform sampling over the pixel; matches generateCameraRay's density).
SR_INL SR_HD float cameraPdfOmega(const CameraProj& proj, float cosTheta) {
    const float c = srMax(1e-4f, cosTheta);
    return (proj.focal * proj.focal) / (proj.pixelArea * c * c * c);
}

}  // namespace sol
