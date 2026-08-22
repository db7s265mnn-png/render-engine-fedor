#include "render/physical_sky.h"

#include <cmath>

#include "core/thread_pool.h"
#include "render/color_space.h"
#include "render/spectrum.h"

#include "ArHosekSkyModel.h"

namespace sol {
namespace {

// Hosek RGB is linear sRGB. A small scale keeps intensity=1 in the same
// ballpark as area lights; the sun/sky ratio still comes from the model.
constexpr float kHosekSkyScale = 1.0f;

struct HosekEval {
    ArHosekSkyModelState* rgb[3] = {nullptr, nullptr, nullptr};
    ArHosekSkyModelState* spec = nullptr;
    Vec3 sunDir{0.0f, 1.0f, 0.0f};
    float turbidity = 3.0f;
    float elevationRad = 0.0f;
    PhysicalSkyParams params;
    Vec3 groundColor{0.0f};
    float sunIrradianceValue = 0.0f;
    Vec3 sunChroma{1.0f};

    // `needSunDisc` loads the 2013 spectral solar model (chromaticity + irradiance).
    // Sky bake does not: the disc is a distant light, not texels on the map.
    explicit HosekEval(const PhysicalSkyParams& p, bool needSunDisc) : params(p) {
        turbidity = clampf(p.turbidity, 1.0f, 10.0f);
        const float elevDeg = clampf(p.elevationDeg, 0.0f, 90.0f);
        elevationRad = radians(elevDeg);
        sunDir = physicalSkySunDirection(p);
        const Vec3 alb(clampf(p.groundAlbedo.x, 0.0f, 1.0f), clampf(p.groundAlbedo.y, 0.0f, 1.0f),
                       clampf(p.groundAlbedo.z, 0.0f, 1.0f));
        for (int c = 0; c < 3; ++c) {
            rgb[c] = arhosek_rgb_skymodelstate_alloc_init(double(turbidity), double(alb[c]),
                                                          double(elevationRad));
        }
        if (needSunDisc && p.enableSun) {
            const float albY = clampf(luminance(alb), 0.0f, 1.0f);
            spec = arhosekskymodelstate_alloc_init(double(elevationRad), double(turbidity), double(albY));
        }
        cacheConstants(needSunDisc && p.enableSun);
    }

    HosekEval(const HosekEval&) = delete;
    HosekEval& operator=(const HosekEval&) = delete;

    ~HosekEval() {
        for (int c = 0; c < 3; ++c) {
            if (rgb[c]) arhosekskymodelstate_free(rgb[c]);
        }
        if (spec) arhosekskymodelstate_free(spec);
    }

    Vec3 skyLinearSrgb(Vec3 dir) const {
        dir = normalize(dir);
        const float cosTheta = clampf(dir.y, 0.0f, 1.0f);
        const float theta = std::acos(cosTheta);
        const float gamma = std::acos(clampf(dot(dir, sunDir), -1.0f, 1.0f));
        Vec3 rgbOut(0.0f);
        for (int c = 0; c < 3; ++c) {
            if (!rgb[c]) continue;
            const double v = arhosek_tristim_skymodel_radiance(rgb[c], double(theta), double(gamma), c);
            rgbOut[c] = srMax(0.0f, float(v) * kHosekSkyScale);
        }
        if (!isFinite(rgbOut)) return Vec3(0.0f);
        return rgbOut;
    }

    Vec3 sampleAverageSky() const {
        Vec3 acc(0.0f);
        int n = 0;
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 16; ++x) {
                const float u = (float(x) + 0.5f) / 16.0f;
                const float v = (float(y) + 0.5f) / 16.0f * 0.5f;
                const Vec3 dir = equirectToDirection(u, v);
                if (dir.y <= 0.0f) continue;
                acc += skyLinearSrgb(dir);
                ++n;
            }
        }
        return n > 0 ? acc / float(n) : Vec3(0.05f);
    }

    // Disc-only radiance (Hosek solar minus sky in-scatter) → linear sRGB.
    Vec3 sampleSunDiscLinearSrgb() const {
        if (!spec) return Vec3(1.0f);
        const float thetaSun = std::acos(clampf(sunDir.y, 0.0f, 1.0f));
        const float solarRadius = float(spec->solar_radius);
        // A few samples across the disc (limb darkening — Hosek caveat #3).
        const float gammas[5] = {0.0f, 0.45f * solarRadius, 0.45f * solarRadius, 0.85f * solarRadius,
                                 0.85f * solarRadius};
        Xyz xyz(0.0f, 0.0f, 0.0f);
        int n = 0;
        const float dw = 40.0f;
        for (int s = 0; s < 5; ++s) {
            const float gamma = gammas[s];
            for (int wl = 1; wl <= 10; ++wl) {
                const float lambda = 320.0f + 40.0f * float(wl);
                const double sunAndSky =
                    arhosekskymodel_solar_radiance(spec, double(thetaSun), double(gamma), double(lambda));
                const double skyOnly =
                    arhosekskymodel_radiance(spec, double(thetaSun), double(gamma), double(lambda));
                const float S = srMax(0.0f, float(sunAndSky - skyOnly));
                float cx = 0.0f, cy = 0.0f, cz = 0.0f;
                cieXyzAtLambda(lambda, cx, cy, cz);
                xyz.x += S * cx * dw;
                xyz.y += S * cy * dw;
                xyz.z += S * cz * dw;
            }
            ++n;
        }
        if (n > 0) {
            const float inv = 1.0f / float(n);
            xyz.x *= inv;
            xyz.y *= inv;
            xyz.z *= inv;
        }
        Vec3 rgbOut = colorSpaceSrgb().toRgb(xyz);
        rgbOut.x = srMax(0.0f, rgbOut.x);
        rgbOut.y = srMax(0.0f, rgbOut.y);
        rgbOut.z = srMax(0.0f, rgbOut.z);
        if (!isFinite(rgbOut) || luminance(rgbOut) < 1e-12f) return Vec3(1.0f);
        return rgbOut;
    }

    void cacheConstants(bool needSunDisc) {
        const Vec3 avgSky = sampleAverageSky();
        if (!params.computeGroundColor) {
            // Authored ground colour is working-space ACEScg like other light colours.
            groundColor = acescgToLinearSrgb(vmax(Vec3(0.0f), params.groundColor));
        } else {
            groundColor = vmax(Vec3(0.0f), params.groundAlbedo) * avgSky;
        }
        if (!needSunDisc) {
            sunIrradianceValue = 0.0f;
            sunChroma = Vec3(1.0f);
            return;
        }
        const float Lsky = srMax(1e-4f, luminance(vmax(Vec3(0.0f), avgSky)));
        const float t = clampf((turbidity - 1.0f) / 9.0f, 0.0f, 1.0f);
        const float k = lerpf(8.0f, 2.0f, t);
        sunIrradianceValue = k * kPi * Lsky * srMax(0.0f, params.sunIntensity);
        const Vec3 sunDiscRgb = sampleSunDiscLinearSrgb();
        const float lum = luminance(sunDiscRgb);
        sunChroma = lum > 1e-8f ? sunDiscRgb / lum : Vec3(1.0f);
    }

    Vec3 visibleGroundLinearSrgb() const { return groundColor; }
    float sunIrradiance() const { return sunIrradianceValue; }
    Vec3 sunDiscChromaLinearSrgb() const { return sunChroma; }
};

Vec3 shadeSkyOnly(const HosekEval& eval, Vec3 dir) {
    if (!eval.params.enableSky) return Vec3(0.0f);
    dir = normalize(dir);
    const float elevDeg = degrees(std::asin(clampf(dir.y, -1.0f, 1.0f)));
    const float blur = srMax(0.05f, eval.params.horizonBlurDeg);
    Vec3 sky;
    if (elevDeg >= 0.0f) {
        sky = eval.skyLinearSrgb(dir);
    } else {
        const Vec3 ground = eval.visibleGroundLinearSrgb();
        const float t = clampf(1.0f + elevDeg / blur, 0.0f, 1.0f);
        if (t <= 0.0f) {
            sky = ground;
        } else {
            Vec3 horizonDir = dir;
            horizonDir.y = 0.0f;
            if (lengthSquared(horizonDir) < 1e-8f) horizonDir = Vec3(0.0f, 0.0f, -1.0f);
            horizonDir = normalize(horizonDir);
            // Tiny lift so Hosek theta stays just above the horizon singularity.
            horizonDir = normalize(horizonDir + Vec3(0.0f, 1e-3f, 0.0f));
            sky = lerp(ground, eval.skyLinearSrgb(horizonDir), t);
        }
    }
    return sky * vmax(Vec3(0.0f), eval.params.skyTint) * srMax(0.0f, eval.params.skyIntensity);
}

}  // namespace

Vec3 physicalSkySunDirection(const PhysicalSkyParams& p) {
    const float elev = radians(clampf(p.elevationDeg, -12.0f, 90.0f));
    const float azim = radians(p.azimuthDeg);
    const float ce = std::cos(elev);
    const Vec3 d(ce * std::sin(azim), std::sin(elev), -ce * std::cos(azim));
    const float len2 = lengthSquared(d);
    return len2 > 1e-12f ? d / std::sqrt(len2) : Vec3(0.0f, 1.0f, 0.0f);
}

Mat4 physicalSkySunLookAt(const PhysicalSkyParams& p) {
    const Vec3 sun = physicalSkySunDirection(p);
    const Vec3 up = std::fabs(sun.y) > 0.99f ? Vec3(0.0f, 0.0f, 1.0f) : Vec3(0.0f, 1.0f, 0.0f);
    Mat4 m = lookAtMatrix(sun, Vec3(0.0f), up);
    m.at(0, 3) = 0.0f;
    m.at(1, 3) = 0.0f;
    m.at(2, 3) = 0.0f;
    return m;
}

Vec3 physicalSkyRadianceAceScg(const PhysicalSkyParams& p, Vec3 dirLocal) {
    HosekEval eval(p, /*needSunDisc=*/false);
    return linearSrgbToAcescg(vmax(Vec3(0.0f), shadeSkyOnly(eval, dirLocal)));
}

Vec3 physicalSkySunColorAceScg(const PhysicalSkyParams& p) {
    HosekEval eval(p, /*needSunDisc=*/true);
    return linearSrgbToAcescg(eval.sunDiscChromaLinearSrgb()) * vmax(Vec3(0.0f), p.sunTint);
}

float physicalSkySunIntensity(const PhysicalSkyParams& p) {
    if (!p.enableSun) return 0.0f;
    HosekEval eval(p, /*needSunDisc=*/true);
    return srMax(0.0f, p.intensity * eval.sunIrradiance());
}

void bakePhysicalSkyEnv(Image& image, const PhysicalSkyParams& p, int width, int height) {
    width = srMax(32, width);
    height = srMax(16, height);
    image.resize(width, height);
    HosekEval eval(p, /*needSunDisc=*/false);
    auto writeRow = [&](int y) {
        const float v = (float(y) + 0.5f) / float(height);
        for (int x = 0; x < width; ++x) {
            const float u = (float(x) + 0.5f) / float(width);
            const Vec3 dir = equirectToDirection(u, v);
            const Vec3 c = linearSrgbToAcescg(vmax(Vec3(0.0f), shadeSkyOnly(eval, dir)));
            image.setRgb(x, y, vmax(Vec3(0.0f), c));
        }
    };
    if (width * height >= 256 * 128) {
        ThreadPool pool;
        pool.parallelFor(height, [&](int y, int) { writeRow(y); });
    } else {
        for (int y = 0; y < height; ++y) writeRow(y);
    }
}

}  // namespace sol
