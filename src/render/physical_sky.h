// Hosek–Wilkie Physical Sky (SIGGRAPH 2012 RGB data + 2013 solar disc on the dome).
// One env map: sky from the tabulated RGB model, sun disc from the matching solar
// radiance function (chromaticity) with irradiance in the same RGB units.
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
    float intensity = 1.0f;      // overall scale for sky and sun
    float skyIntensity = 1.0f;   // extra sky-dome multiplier
    float exposure = 0.0f;
    bool enableSky = true;
    bool enableSun = true;
    float sunIntensity = 1.0f;   // extra sun multiplier
    Vec3 sunTint{1.0f, 1.0f, 1.0f};
    float sunSizeDeg = 0.53f;  // Karma Angular Size; brightness independent (normalize)
    bool computeGroundColor = true;
    Vec3 groundColor{0.1f, 0.1f, 0.1f};
    float horizonBlurDeg = 5.0f;  // blend sky→ground this many degrees below the horizon
};

// Unit direction toward the sun in local Y-up space.
Vec3 physicalSkySunDirection(const PhysicalSkyParams& p);

// Light-to-world rotation whose +Z axis equals the sun direction (distant NEE).
Mat4 physicalSkySunLookAt(const PhysicalSkyParams& p);

// Sky radiance in ACEScg for a local-space direction, including Sky/Sun Intensity
// and tints. Overall intensity / exposure stay on LightData.
Vec3 physicalSkyRadianceAceScg(const PhysicalSkyParams& p, Vec3 dirLocal);

// Sun disc chromaticity in ACEScg (Hosek solar radiance × sun tint).
Vec3 physicalSkySunColorAceScg(const PhysicalSkyParams& p);

// Direct sun irradiance in RGB-Hosek units (overall intensity × sunIntensity).
float physicalSkySunIntensity(const PhysicalSkyParams& p);

void bakePhysicalSkyEnv(Image& image, const PhysicalSkyParams& p, int width, int height);

}  // namespace sol
