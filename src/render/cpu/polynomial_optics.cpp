// CPU polynomial-optics camera — Lentil forward path without Arnold.
#include "render/cpu/polynomial_optics.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "core/log.h"

namespace sol {
namespace {

// Minimal Eigen subset for generated pt_sample_aperture Newton solves.
namespace Eigen {

struct Vector2d {
    double v[2] = {0.0, 0.0};
    Vector2d() = default;
    Vector2d(double a, double b) {
        v[0] = a;
        v[1] = b;
    }
    double& operator()(int i) { return v[i]; }
    double operator()(int i) const { return v[i]; }
    double& operator[](int i) { return v[i]; }
    double operator[](int i) const { return v[i]; }
};

struct Vector3d {
    double v[3] = {0.0, 0.0, 0.0};
    Vector3d() = default;
    Vector3d(double a, double b, double c) {
        v[0] = a;
        v[1] = b;
        v[2] = c;
    }
    double& operator()(int i) { return v[i]; }
    double operator()(int i) const { return v[i]; }
    double& operator[](int i) { return v[i]; }
    double operator[](int i) const { return v[i]; }
    void normalize() {
        const double len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (len > 0.0) {
            v[0] /= len;
            v[1] /= len;
            v[2] /= len;
        }
    }
    double dot(const Vector3d& o) const { return v[0] * o.v[0] + v[1] * o.v[1] + v[2] * o.v[2]; }
};

struct Matrix2d {
    double m[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    double& operator()(int r, int c) { return m[r][c]; }
    double operator()(int r, int c) const { return m[r][c]; }
};

}  // namespace Eigen

static inline double lens_ipow(const double x, const int exp) {
    if (exp == 0) return 1.0;
    if (exp == 1) return x;
    if (exp == 2) return x * x;
    const double p2 = lens_ipow(x, exp / 2);
    if (exp & 1) return x * p2 * p2;
    return p2 * p2;
}

static inline void raytrace_cross(Eigen::Vector3d& r, const Eigen::Vector3d& u, const Eigen::Vector3d& v) {
    r(0) = u(1) * v(2) - u(2) * v(1);
    r(1) = u(2) * v(0) - u(0) * v(2);
    r(2) = u(0) * v(1) - u(1) * v(0);
}

static inline double raytrace_dot(const Eigen::Vector3d& u, const Eigen::Vector3d& v) {
    return u(0) * v(0) + u(1) * v(1) + u(2) * v(2);
}

static inline void raytrace_normalise(Eigen::Vector3d& v) {
    const double ilen = 1.0 / std::sqrt(std::max(1e-30, raytrace_dot(v, v)));
    v(0) *= ilen;
    v(1) *= ilen;
    v(2) *= ilen;
}

static inline void sphereToCs(const Eigen::Vector2d& inpos, const Eigen::Vector2d& indir, Eigen::Vector3d& outpos,
                              Eigen::Vector3d& outdir, const double center, const double sphereRad) {
    const Eigen::Vector3d normal(
        inpos(0) / sphereRad, inpos(1) / sphereRad,
        std::sqrt(std::max(0.0, sphereRad * sphereRad - inpos(0) * inpos(0) - inpos(1) * inpos(1))) /
            std::abs(sphereRad));

    const Eigen::Vector3d tempDir(indir(0), indir(1),
                                  std::sqrt(std::max(0.0, 1.0 - indir(0) * indir(0) - indir(1) * indir(1))));

    Eigen::Vector3d ex(normal(2), 0.0, -normal(0));
    raytrace_normalise(ex);
    Eigen::Vector3d ey(0.0, 0.0, 0.0);
    raytrace_cross(ey, normal, ex);

    outdir(0) = tempDir(0) * ex(0) + tempDir(1) * ey(0) + tempDir(2) * normal(0);
    outdir(1) = tempDir(0) * ex(1) + tempDir(1) * ey(1) + tempDir(2) * normal(1);
    outdir(2) = tempDir(0) * ex(2) + tempDir(1) * ey(2) + tempDir(2) * normal(2);

    outpos(0) = inpos(0);
    outpos(1) = inpos(1);
    outpos(2) = normal(2) * sphereRad + center;
}

static inline Eigen::Vector3d linePlaneIntersectionY0(Eigen::Vector3d rayOrigin, Eigen::Vector3d rayDirection) {
    rayDirection.normalize();
    const Eigen::Vector3d planeNormal(0.0, 1.0, 0.0);
    const double denom = planeNormal.dot(rayDirection);
    if (std::abs(denom) < 1e-12) return rayOrigin;
    const double t = -planeNormal.dot(rayOrigin) / denom;
    return Eigen::Vector3d(rayOrigin(0) + rayDirection(0) * t, rayOrigin(1) + rayDirection(1) * t,
                           rayOrigin(2) + rayDirection(2) * t);
}

enum LensModel {
#include "includes/pota_h_lenses.h"
    kLensModelCount
};

static const char* kLensModelLabels[] = {
#include "includes/pota_cpp_lenses.h"
};

struct LensState {
    LensModel lensModel = cooke__speed_panchro__1920__50mm;
    const char* lens_name = "";
    double lens_outer_pupil_radius = 0.0;
    double lens_inner_pupil_radius = 0.0;
    double lens_inner_pupil_curvature_radius = 0.0;
    double lens_outer_pupil_curvature_radius = 0.0;
    double lens_length = 0.0;
    double lens_back_focal_length = 0.0;
    double lens_effective_focal_length = 0.0;
    double lens_aperture_pos = 0.0;
    double lens_aperture_housing_radius = 0.0;
    const char* lens_outer_pupil_geometry = "spherical";
    const char* lens_inner_pupil_geometry = "spherical";
    double lens_fstop = 0.0;
    double lens_aperture_radius_at_fstop = 0.0;
    double lens_field_of_view = 0.0;
    double lambda = 0.55;
    double aperture_radius = 1.0;
    double sensor_shift = 0.0;
};

static void loadLensConstants(LensState& s) {
    const char* lens_name = "";
    double lens_outer_pupil_radius = 0.0;
    double lens_inner_pupil_radius = 0.0;
    double lens_inner_pupil_curvature_radius = 0.0;
    double lens_outer_pupil_curvature_radius = 0.0;
    double lens_length = 0.0;
    double lens_back_focal_length = 0.0;
    double lens_effective_focal_length = 0.0;
    double lens_aperture_pos = 0.0;
    double lens_aperture_housing_radius = 0.0;
    std::string lens_outer_pupil_geometry = "spherical";
    std::string lens_inner_pupil_geometry = "spherical";
    double lens_fstop = 0.0;
    double lens_aperture_radius_at_fstop = 0.0;
    double lens_field_of_view = 0.0;

    switch (s.lensModel) {
#include "includes/load_lens_constants.h"
    }

    static thread_local std::string outerGeom;
    static thread_local std::string innerGeom;
    outerGeom = lens_outer_pupil_geometry;
    innerGeom = lens_inner_pupil_geometry;

    s.lens_name = lens_name;
    s.lens_outer_pupil_radius = lens_outer_pupil_radius;
    s.lens_inner_pupil_radius = lens_inner_pupil_radius;
    s.lens_inner_pupil_curvature_radius = lens_inner_pupil_curvature_radius;
    s.lens_outer_pupil_curvature_radius = lens_outer_pupil_curvature_radius;
    s.lens_length = lens_length;
    s.lens_back_focal_length = lens_back_focal_length;
    s.lens_effective_focal_length = lens_effective_focal_length;
    s.lens_aperture_pos = lens_aperture_pos;
    s.lens_aperture_housing_radius = lens_aperture_housing_radius;
    s.lens_outer_pupil_geometry = outerGeom.c_str();
    s.lens_inner_pupil_geometry = innerGeom.c_str();
    s.lens_fstop = lens_fstop;
    s.lens_aperture_radius_at_fstop = lens_aperture_radius_at_fstop;
    s.lens_field_of_view = lens_field_of_view;
}

static double lensEvaluate(const LensState& s, const double in[5], double out[5]) {
    const double x = in[0], y = in[1], dx = in[2], dy = in[3], lambda = in[4];
    double out_transmittance = 0.0;
    switch (s.lensModel) {
#include "includes/load_pt_evaluate.h"
    }
    out[4] = lambda;
    return std::max(0.0, out_transmittance);
}

static void lensPtSampleAperture(const LensState& s, double in[5], double out[5], double dist) {
    double out_x = out[0], out_y = out[1], out_dx = out[2], out_dy = out[3], out_transmittance = 1.0;
    double x = in[0], y = in[1], dx = in[2], dy = in[3], lambda = in[4];
    (void)out_transmittance;

    switch (s.lensModel) {
#include "includes/load_pt_sample_aperture.h"
    }

    out[0] = out_x;
    out[1] = out_y;
    out[2] = out_dx;
    out[3] = out_dy;
    in[0] = x;
    in[1] = y;
    in[2] = dx;
    in[3] = dy;
}

static double y0IntersectionDistance(const LensState& s, double sensorShift) {
    double sensor[5] = {0, 0, 0, 0, s.lambda};
    double aperture[5] = {0, s.lens_aperture_housing_radius * 0.25, 0, 0, 0};
    double out[5] = {0, 0, 0, 0, 0};

    lensPtSampleAperture(s, sensor, aperture, sensorShift);
    sensor[0] += sensor[2] * sensorShift;
    sensor[1] += sensor[3] * sensorShift;
    lensEvaluate(s, sensor, out);

    Eigen::Vector2d outpos(out[0], out[1]);
    Eigen::Vector2d outdir(out[2], out[3]);
    Eigen::Vector3d cameraSpacePos(0, 0, 0);
    Eigen::Vector3d cameraSpaceOmega(0, 0, 0);
    sphereToCs(outpos, outdir, cameraSpacePos, cameraSpaceOmega, -s.lens_outer_pupil_curvature_radius,
               s.lens_outer_pupil_curvature_radius);
    return linePlaneIntersectionY0(cameraSpacePos, cameraSpaceOmega)(2);
}

static void concentricDisk(double u, double v, double& dx, double& dy) {
    const double a = 2.0 * u - 1.0;
    const double b = 2.0 * v - 1.0;
    if (a == 0.0 && b == 0.0) {
        dx = dy = 0.0;
        return;
    }
    double r, phi;
    if (a * a > b * b) {
        r = a;
        phi = (0.78539816339) * (b / a);
    } else {
        r = b;
        phi = 1.57079632679 - (0.78539816339) * (a / b);
    }
    dx = r * std::cos(phi);
    dy = r * std::sin(phi);
}

static double sampleBundleCoCAtDistance(const LensState& s, double sensorShift, double apertureRadius,
                                        double focusDistanceMm) {
    // Approximate circle-of-confusion diameter (mm) of an on-axis pixel at focusDistanceMm.
    double minX = 1e30, maxX = -1e30, minY = 1e30, maxY = -1e30;
    int ok = 0;
    static const double kSamples[12][2] = {
        {0.0, 0.0}, {0.5, 0.5}, {0.15, 0.5}, {0.85, 0.5}, {0.5, 0.15}, {0.5, 0.85},
        {0.2, 0.2}, {0.8, 0.2}, {0.2, 0.8}, {0.8, 0.8}, {0.35, 0.65}, {0.65, 0.35},
    };
    for (const auto& uv : kSamples) {
        double sensor[5] = {0.0, 0.0, 0.0, 0.0, s.lambda};
        double aperture[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        double diskX = 0.0, diskY = 0.0;
        concentricDisk(uv[0], uv[1], diskX, diskY);
        aperture[0] = diskX * apertureRadius;
        aperture[1] = diskY * apertureRadius;
        lensPtSampleAperture(s, sensor, aperture, sensorShift);
        sensor[0] += sensor[2] * sensorShift;
        sensor[1] += sensor[3] * sensorShift;
        double out[5] = {0, 0, 0, 0, 0};
        if (lensEvaluate(s, sensor, out) <= 0.0) continue;
        if (out[0] * out[0] + out[1] * out[1] > s.lens_outer_pupil_radius * s.lens_outer_pupil_radius) continue;
        Eigen::Vector2d outpos(out[0], out[1]);
        Eigen::Vector2d outdir(out[2], out[3]);
        Eigen::Vector3d pos(0, 0, 0), dir(0, 0, 0);
        sphereToCs(outpos, outdir, pos, dir, -s.lens_outer_pupil_curvature_radius,
                   s.lens_outer_pupil_curvature_radius);
        if (std::abs(dir(2)) < 1e-12) continue;
        // Lentil camera space: +Z into the scene. Intersect plane z = focusDistanceMm.
        const double t = (focusDistanceMm - pos(2)) / dir(2);
        if (!(t > 0.0)) continue;
        const double hx = pos(0) + dir(0) * t;
        const double hy = pos(1) + dir(1) * t;
        minX = std::min(minX, hx);
        maxX = std::max(maxX, hx);
        minY = std::min(minY, hy);
        maxY = std::max(maxY, hy);
        ++ok;
    }
    if (ok < 3) return 1e30;
    const double dx = maxX - minX;
    const double dy = maxY - minY;
    return std::sqrt(dx * dx + dy * dy);
}

static double logarithmicFocusSearch(const LensState& s, double focalDistanceMm, double apertureRadius) {
    // 1) Paraxial y0 search (Lentil) for a good starting shift.
    double bestErr = 1e30;
    double bestSensorShift = 0.0;
    for (double i = -1.0; i <= 1.0; i += 0.0005) {
        const double sensorshift = (i < 0.0 ? -1.0 : 1.0) * (i * i) * 45.0;
        const double intersectionDistance = y0IntersectionDistance(s, sensorshift);
        if (!(intersectionDistance > 0.0) || !std::isfinite(intersectionDistance)) continue;
        const double err = std::abs(intersectionDistance - focalDistanceMm);
        if (err < bestErr) {
            bestErr = err;
            bestSensorShift = sensorshift;
        }
    }

    // 2) Refine for the actual wide-open / stopped-down bundle. Vintage lenses have
    //    spherical aberration, so the paraxial focus plane is soft at low f-stops —
    //    pick the sensor shift that minimises CoC at the requested focus distance.
    double bestCoC = sampleBundleCoCAtDistance(s, bestSensorShift, apertureRadius, focalDistanceMm);
    double span = 2.0;
    for (int iter = 0; iter < 20; ++iter) {
        bool improved = false;
        for (double delta : {-span, -0.5 * span, 0.5 * span, span}) {
            const double sensorshift = bestSensorShift + delta;
            const double coc = sampleBundleCoCAtDistance(s, sensorshift, apertureRadius, focalDistanceMm);
            if (coc < bestCoC) {
                bestCoC = coc;
                bestSensorShift = sensorshift;
                improved = true;
            }
        }
        span *= improved ? 0.6 : 0.5;
        if (span < 1e-4) break;
    }

    logInfo(std::string("Polynomial optics focus: target=") + std::to_string(focalDistanceMm) +
            "mm  shift=" + std::to_string(bestSensorShift) + "mm  CoC=" + std::to_string(bestCoC) + "mm");
    return bestSensorShift;
}

}  // namespace

const std::vector<std::string>& polynomialOpticsLensNames() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> out;
        out.reserve(size_t(kLensModelCount));
        for (int i = 0; i < int(kLensModelCount); ++i) out.emplace_back(kLensModelLabels[i]);
        return out;
    }();
    return names;
}

void PolynomialOpticsCamera::prepare(const CameraData& camera) {
    active = false;
    if (camera.opticalModel != 1) return;

    LensState state;
    const int count = int(kLensModelCount);
    state.lensModel = LensModel(std::clamp(camera.lensModel, 0, count - 1));
    state.lambda = double(std::max(1.0f, camera.opticalWavelengthNm)) * 0.001;
    loadLensConstants(state);

    // Focus distance is authored in metres → mm for the polynomial model.
    const double focusMm = double(std::max(0.01f, camera.focusDistance)) * 1000.0;

    // Aperture first — focus refinement uses the real bundle size (SA depends on it).
    if (camera.fStop <= 0.0f) {
        state.aperture_radius = state.lens_aperture_radius_at_fstop;
    } else {
        // Map f-stop relative to the lens's native wide-open stop. This matches
        // thin-lens intuition (lower f → more bokeh) while never exceeding the
        // physical iris. Requesting f/1 on an f/1.1 lens → wide open.
        const double userF = double(std::max(0.05f, camera.fStop));
        const double nativeF = std::max(0.05, state.lens_fstop);
        state.aperture_radius =
            std::min(state.lens_aperture_radius_at_fstop,
                     state.lens_aperture_radius_at_fstop * (nativeF / userF));
    }

    state.sensor_shift = logarithmicFocusSearch(state, focusMm, state.aperture_radius);

    active = true;
    lensModel = int(state.lensModel);
    sensorWidthMm = double(std::max(1.0f, camera.sensorWidth));
    apertureRadiusMm = state.aperture_radius;
    sensorShiftMm = state.sensor_shift;
    lambda = state.lambda;
    lensName = state.lens_name;
    lensOuterPupilRadius = state.lens_outer_pupil_radius;
    lensInnerPupilRadius = state.lens_inner_pupil_radius;
    lensLength = state.lens_length;
    lensBackFocalLength = state.lens_back_focal_length;
    lensEffectiveFocalLength = state.lens_effective_focal_length;
    lensAperturePos = state.lens_aperture_pos;
    lensApertureHousingRadius = state.lens_aperture_housing_radius;
    lensInnerPupilCurvatureRadius = state.lens_inner_pupil_curvature_radius;
    lensOuterPupilCurvatureRadius = state.lens_outer_pupil_curvature_radius;
    lensFieldOfView = state.lens_field_of_view;
    lensFstop = state.lens_fstop;
    lensApertureRadiusAtFstop = state.lens_aperture_radius_at_fstop;

    lensOuterPupilGeometry = state.lens_outer_pupil_geometry ? state.lens_outer_pupil_geometry : "spherical";
    lensInnerPupilGeometry = state.lens_inner_pupil_geometry ? state.lens_inner_pupil_geometry : "spherical";

    logInfo(std::string("Polynomial optics: ") + (lensName ? lensName : "?") +
            "  f/" + std::to_string(lensFstop) + "  aperture=" + std::to_string(apertureRadiusMm) +
            "mm  sensor_shift=" + std::to_string(sensorShiftMm) + "mm  focus=" +
            std::to_string(camera.focusDistance) + "m");
}

bool PolynomialOpticsCamera::generateRayCameraSpace(double sensorXmm, double sensorYmm, float lensU, float lensV,
                                                    Rng& rng, Vec3& originCam, Vec3& dirCam) const {
    if (!active) return false;

    LensState state;
    state.lensModel = LensModel(lensModel);
    state.lambda = lambda;
    state.aperture_radius = apertureRadiusMm;
    state.sensor_shift = sensorShiftMm;
    state.lens_outer_pupil_radius = lensOuterPupilRadius;
    state.lens_inner_pupil_radius = lensInnerPupilRadius;
    state.lens_back_focal_length = lensBackFocalLength;
    state.lens_outer_pupil_curvature_radius = lensOuterPupilCurvatureRadius;
    state.lens_inner_pupil_curvature_radius = lensInnerPupilCurvatureRadius;
    state.lens_aperture_housing_radius = lensApertureHousingRadius;
    state.lens_outer_pupil_geometry = lensOuterPupilGeometry.c_str();
    state.lens_inner_pupil_geometry = lensInnerPupilGeometry.c_str();

    float r1 = lensU;
    float r2 = lensV;
    bool success = false;
    double out[5] = {0, 0, 0, 0, 0};
    double sensor[5] = {0, 0, 0, 0, lambda};

    for (int tries = 0; tries <= vignettingRetries && !success; ++tries) {
        if (tries > 0) {
            r1 = rng.nextFloat();
            r2 = rng.nextFloat();
        }

        sensor[0] = sensorXmm;
        sensor[1] = sensorYmm;
        sensor[2] = 0.0;
        sensor[3] = 0.0;
        sensor[4] = lambda;

        double aperture[5] = {0, 0, 0, 0, 0};
        double diskX = 0.0, diskY = 0.0;
        concentricDisk(double(r1), double(r2), diskX, diskY);
        aperture[0] = diskX * apertureRadiusMm;
        aperture[1] = diskY * apertureRadiusMm;

        // Always sample aperture so the direction through the lens is consistent.
        lensPtSampleAperture(state, sensor, aperture, sensorShiftMm);

        sensor[0] += sensor[2] * sensorShiftMm;
        sensor[1] += sensor[3] * sensorShiftMm;

        const double transmittance = lensEvaluate(state, sensor, out);
        if (transmittance <= 0.0) continue;

        if (out[0] * out[0] + out[1] * out[1] > lensOuterPupilRadius * lensOuterPupilRadius) continue;

        const double px = sensor[0] + sensor[2] * lensBackFocalLength;
        const double py = sensor[1] + sensor[3] * lensBackFocalLength;
        if (px * px + py * py > lensInnerPupilRadius * lensInnerPupilRadius) continue;

        success = true;
    }

    if (!success) return false;

    Eigen::Vector2d outpos(out[0], out[1]);
    Eigen::Vector2d outdir(out[2], out[3]);
    Eigen::Vector3d csOrigin(0, 0, 0);
    Eigen::Vector3d csDirection(0, 0, 0);
    sphereToCs(outpos, outdir, csOrigin, csDirection, -lensOuterPupilCurvatureRadius,
               lensOuterPupilCurvatureRadius);

    // Polynomials are in millimetres; scene is metres. Flip to look down -Z.
    originCam = Vec3(float(-csOrigin(0) * 0.001), float(-csOrigin(1) * 0.001), float(-csOrigin(2) * 0.001));
    dirCam = normalize(Vec3(float(-csDirection(0)), float(-csDirection(1)), float(-csDirection(2))));

    if (!std::isfinite(originCam.x) || !std::isfinite(originCam.y) || !std::isfinite(originCam.z) ||
        !std::isfinite(dirCam.x) || !std::isfinite(dirCam.y) || !std::isfinite(dirCam.z)) {
        return false;
    }
    return true;
}

bool generatePolynomialOpticsRay(const PolynomialOpticsCamera& lens, const CameraData& camera, float pixelX,
                                 float pixelY, int resolutionX, int resolutionY, float lensU, float lensV,
                                 Rng& rng, Vec3& origin, Vec3& direction) {
    const float resX = float(std::max(1, resolutionX));
    const float resY = float(std::max(1, resolutionY));
    const float sensorHeight = camera.sensorWidth * (resY / resX);
    const double sx = double((pixelX / resX - 0.5f) * camera.sensorWidth);
    const double sy = double((0.5f - pixelY / resY) * sensorHeight);

    Vec3 originCam, dirCam;
    if (!lens.generateRayCameraSpace(sx, sy, lensU, lensV, rng, originCam, dirCam)) return false;

    origin = transformPoint(camera.cameraToWorld, originCam);
    direction = normalize(transformVector(camera.cameraToWorld, dirCam));
    return true;
}

}  // namespace sol
