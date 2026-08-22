// Hosek–Wilkie Physical Sky (SIGGRAPH 2012 RGB data + 2013 solar chromaticity).
// Karma-style: sky is a dome env map with no solar disc; the sun is a distant
// light (cone NEE, Angular Size). One node emits both prims.
#pragma once

#include "core/image.h"
#include "core/math.h"

namespace sol {

struct PhysicalSkyParams {
    float turbidity = 3.0f;  // 2 arctic, 3 clear, 6 humid, 10 hazy (Hosek / Karma)
    Vec3 groundAlbedo{0.1f, 0.1f, 0.1f};
    Vec3 skyTint{1.0f, 1.0f, 1.0f};
    float elevationDeg = 45.0f;  // Solar altitude: 0 = horizon, 90 = zenith
    float azimuthDeg = 90.0f;    // 0 = −Z, 90 = +X (Y-up, Karma/Solaris)
    float intensity = 1.0f;      // overall scale for sky and sun (LightData.intensity)
    float skyIntensity = 1.0f;   // extra sky-dome multiplier (baked into the map)
    float exposure = 0.0f;
    bool enableSky = true;
    bool enableSun = true;
    float sunIntensity = 1.0f;   // extra sun multiplier (folded into distant intensity)
    Vec3 sunTint{1.0f, 1.0f, 1.0f};
    float sunSizeDeg = 0.53f;  // Karma Angular Size; brightness independent (normalize)
    bool computeGroundColor = true;
    Vec3 groundColor{0.1f, 0.1f, 0.1f};
    float horizonBlurDeg = 5.0f;  // blend sky→ground this many degrees below the horizon
};

// Unit direction toward the sun in local Y-up space.
Vec3 physicalSkySunDirection(const PhysicalSkyParams& p);

// Rotation whose +Z axis equals the sun direction (distant NEE / camera disc).
// Translation is zero — distant lights only use orientation.
Mat4 physicalSkySunLookAt(const PhysicalSkyParams& p);

// Sky / ground radiance in ACEScg for a local-space direction (no solar disc).
// Sky/Sun Intensity: sky tint and skyIntensity are included; overall intensity
// and exposure stay on LightData. Sun disc is the distant light, not this.
Vec3 physicalSkyRadianceAceScg(const PhysicalSkyParams& p, Vec3 dirLocal);

// Distant-light colour in ACEScg (Hosek solar chromaticity × sun tint).
Vec3 physicalSkySunColorAceScg(const PhysicalSkyParams& p);

// Distant-light intensity: overall intensity × Hosek RGB irradiance × sunIntensity.
// LightData.exposure is applied on top. Angular Size is not in this value
// (distant normalize divides by solid angle at sample time).
float physicalSkySunIntensity(const PhysicalSkyParams& p);

void bakePhysicalSkyEnv(Image& image, const PhysicalSkyParams& p, int width, int height);

}  // namespace sol
