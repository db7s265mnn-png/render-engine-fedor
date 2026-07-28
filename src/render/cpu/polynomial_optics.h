// Embree-only polynomial optics camera (Lentil / Hanika et al. 2016).
// Maps sensor + aperture samples through fitted lens polynomials so real
// objectives produce optical vignetting and high-order aberrations.
// OptiX keeps the thin-lens path for now.
#pragma once

#include <string>
#include <vector>

#include "core/math.h"
#include "core/rng.h"
#include "scene/types.h"

namespace sol {

// Menu labels matching third_party/lentil lens enum order.
const std::vector<std::string>& polynomialOpticsLensNames();

struct PolynomialOpticsCamera {
    bool active = false;
    int lensModel = 0;
    double sensorWidthMm = 36.0;
    double apertureRadiusMm = 1.0;
    double sensorShiftMm = 0.0;
    double lambda = 0.55;  // micrometres (550 nm)
    int vignettingRetries = 16;

    // Lens constants (mm) loaded from the selected model.
    const char* lensName = "";
    double lensOuterPupilRadius = 0.0;
    double lensInnerPupilRadius = 0.0;
    double lensLength = 0.0;
    double lensBackFocalLength = 0.0;
    double lensEffectiveFocalLength = 0.0;
    double lensAperturePos = 0.0;
    double lensApertureHousingRadius = 0.0;
    double lensInnerPupilCurvatureRadius = 0.0;
    double lensOuterPupilCurvatureRadius = 0.0;
    double lensFieldOfView = 0.0;
    double lensFstop = 0.0;
    double lensApertureRadiusAtFstop = 0.0;
    std::string lensInnerPupilGeometry = "spherical";
    std::string lensOuterPupilGeometry = "spherical";

    // Prepare from authored CameraData. No-op / inactive when opticalModel==0.
    void prepare(const CameraData& camera);

    // Sensor position in millimetres (same convention as generateCameraRay sx/sy).
    // lensU/V in [0,1]. wavelengthNm <= 0 uses the prepared monochromatic lambda.
    // On success returns camera-space origin/direction in metres (already flipped to
    // look down -Z like the thin-lens path). Caller applies cameraToWorld.
    // outTransmittance receives the polynomial Fresnel/vignetting weight when non-null.
    bool generateRayCameraSpace(double sensorXmm, double sensorYmm, float lensU, float lensV, Rng& rng,
                                Vec3& originCam, Vec3& dirCam, float wavelengthNm = -1.0f,
                                float* outTransmittance = nullptr) const;
};

// Convenience: full world-space ray. Returns false if the sample vignettes out.
// wavelengthNm <= 0 uses the prepared monochromatic lambda.
bool generatePolynomialOpticsRay(const PolynomialOpticsCamera& lens, const CameraData& camera, float pixelX,
                                 float pixelY, int resolutionX, int resolutionY, float lensU, float lensV,
                                 Rng& rng, Vec3& origin, Vec3& direction, float wavelengthNm = -1.0f,
                                 float* outTransmittance = nullptr);

// Canonical RGB sample wavelengths (nm) for chromatic aberration.
inline float chromaticWavelengthNm(int channel) {
    static const float kNm[3] = {630.0f, 550.0f, 450.0f};
    return kNm[channel < 0 ? 0 : (channel > 2 ? 2 : channel)];
}

}  // namespace sol
