// Hosek–Wilkie Physical Sky (SIGGRAPH 2012) with Karma Physical Sky controls.
// Baked dome is sky only — the solar disc is a distant light (sharp shadows, no HDRI spike).
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
    float intensity = 1.0f;
    float exposure = 0.0f;
    bool enableSky = true;
    bool enableSun = true;
    float sunIntensity = 1.0f;
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

// Sky radiance in ACEScg for a local-space direction.
// Does not include intensity / exposure / sky tint (those live on LightData).
Vec3 physicalSkyRadianceAceScg(const PhysicalSkyParams& p, Vec3 dirLocal);

// Sun disc chromaticity in ACEScg (Hosek solar radiance × sun tint).
Vec3 physicalSkySunColorAceScg(const PhysicalSkyParams& p);

// Distant-light intensity (irradiance-like when normalize=1). Includes the
// shared intensity and sunIntensity; exposure stays on LightData.
float physicalSkySunIntensity(const PhysicalSkyParams& p);

void bakePhysicalSkyEnv(Image& image, const PhysicalSkyParams& p, int width, int height);

}  // namespace sol
