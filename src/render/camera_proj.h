// Pinhole / thin-lens-center camera importance for light-tracing splats.
// Shared by CPU BDPT and OptiX (device-safe, no STL).
#pragma once

#include "core/math.h"
#include "scene/types.h"

namespace sol {

struct CameraProj {
    Mat4 worldToCam;
    Mat4 cameraToWorld;
    Vec3 camPos{0.0f};
    float focal = 50.0f;    // mm
    float sensorW = 36.0f;  // mm
    float sensorH = 24.0f;
    float resX = 1.0f;
    float resY = 1.0f;
    float pixelArea = 1.0f;  // sensor mm² per pixel
    float lensRadius = 0.0f;  // metres; 0 = pinhole
    bool valid = false;
};

SR_INL SR_HD CameraProj buildCameraProj(const SceneView& scene) {
    CameraProj c;
    const CameraData& cam = scene.camera;
    c.cameraToWorld = cam.cameraToWorld;
    c.worldToCam = inverse(cam.cameraToWorld);
    c.camPos = transformPoint(cam.cameraToWorld, Vec3(0.0f, 0.0f, 0.0f));
    c.focal = srMax(1e-3f, cam.focalLength);
    c.resX = float(srMax(1, scene.settings.resolutionX));
    c.resY = float(srMax(1, scene.settings.resolutionY));
    c.sensorW = cam.sensorWidth;
    c.sensorH = cam.sensorWidth * (c.resY / c.resX);
    c.pixelArea = (c.sensorW / c.resX) * (c.sensorH / c.resY);
    c.lensRadius = 0.0f;
    if (cam.fStop > 0.0f) c.lensRadius = (cam.focalLength * 0.001f) / (2.0f * cam.fStop);
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

// Half-angle of the pinhole accept cone for L S* C. Two pixels, floored at 5e-4 rad
// so a true Dirac still has finite measure on a pinhole camera.
SR_INL SR_HD float pinholeAcceptHalfAngle(const CameraProj& proj) {
    const float pixel = (proj.sensorW / srMax(1.0f, proj.resX)) / srMax(1e-6f, proj.focal);
    return srMax(2.0f * pixel, 5.0e-4f);
}

// True when a sampled outgoing direction from pWorld continues into the camera
// and lands on the film. dest pixel is the pinhole projection of pWorld.
// Pinhole: dir must align with (camPos - p) within pinholeAcceptHalfAngle.
// Thin lens: the ray must hit the z=0 aperture disk (lensRadius in metres).
SR_INL SR_HD bool cameraContinuesToFilm(const CameraProj& proj, Vec3 pWorld, Vec3 dirWorld, float& destPx,
                                        float& destPy, float& cosTheta, float& dist2) {
    destPx = 0.0f;
    destPy = 0.0f;
    cosTheta = 0.0f;
    dist2 = 0.0f;
    const Vec3 d = normalize(dirWorld);
    if (lengthSquared(d) < 1e-20f) return false;

    if (proj.lensRadius > 1e-8f) {
        const Vec3 oCam = transformPoint(proj.worldToCam, pWorld);
        const Vec3 dCam = transformVector(proj.worldToCam, d);
        if (dCam.z <= 1e-8f) return false;
        const float t = -oCam.z / dCam.z;
        if (t <= 1e-6f) return false;
        const float lx = oCam.x + t * dCam.x;
        const float ly = oCam.y + t * dCam.y;
        if (lx * lx + ly * ly > proj.lensRadius * proj.lensRadius) return false;
        if (!projectToPixel(proj, pWorld, destPx, destPy, cosTheta, dist2)) return false;
        const Vec3 lensW = transformPoint(proj.cameraToWorld, Vec3(lx, ly, 0.0f));
        dist2 = lengthSquared(lensW - pWorld);
        if (dist2 < 1e-20f) return false;
        cosTheta = srMax(1e-4f, dCam.z);
        return true;
    }

    const Vec3 toCam = proj.camPos - pWorld;
    const float d2 = lengthSquared(toCam);
    if (d2 < 1e-20f) return false;
    const Vec3 toCamN = toCam * (1.0f / sqrtf(d2));
    if (dot(d, toCamN) < cosf(pinholeAcceptHalfAngle(proj))) return false;
    if (!projectToPixel(proj, pWorld, destPx, destPy, cosTheta, dist2)) return false;
    return dist2 >= 1e-8f;
}

}  // namespace sol
