// RGB color spaces for spectral → XYZ → RGB (pbrt-v4 style).
#pragma once

#include <algorithm>
#include <cmath>

#include "core/math.h"
#include "render/cie_tables.h"

namespace sol {

enum SpectralColorSpace : int {
    kSpectralColorSpaceSrgb = 0,     // linear sRGB / Rec.709, D65 (default beauty)
    kSpectralColorSpaceAcesCg = 1,   // ACEScg (AP1), D60
    kSpectralColorSpaceRec2020 = 2,  // Rec.2020 linear, D65
    kSpectralColorSpaceDisplayP3 = 3,
};

struct Xyz {
    float x = 0, y = 0, z = 0;
    Xyz() = default;
    Xyz(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
};

struct RGBColorSpace {
    const char* name = "sRGB";
    // Row-major 3×3: rgb = M * xyz
    float rgbFromXyz[9]{};
    float xyzFromRgb[9]{};

    Vec3 toRgb(Xyz xyz) const {
        const float* m = rgbFromXyz;
        return Vec3(m[0] * xyz.x + m[1] * xyz.y + m[2] * xyz.z,
                    m[3] * xyz.x + m[4] * xyz.y + m[5] * xyz.z,
                    m[6] * xyz.x + m[7] * xyz.y + m[8] * xyz.z);
    }
    Xyz toXyz(Vec3 rgb) const {
        const float* m = xyzFromRgb;
        return Xyz(m[0] * rgb.x + m[1] * rgb.y + m[2] * rgb.z,
                   m[3] * rgb.x + m[4] * rgb.y + m[5] * rgb.z,
                   m[6] * rgb.x + m[7] * rgb.y + m[8] * rgb.z);
    }
};

inline void mulMat3(const float* a, const float* b, float* out) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out[r * 3 + c] = a[r * 3 + 0] * b[0 * 3 + c] + a[r * 3 + 1] * b[1 * 3 + c] +
                             a[r * 3 + 2] * b[2 * 3 + c];
        }
    }
}

inline bool invertMat3(const float* m, float* out) {
    const float det =
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::fabs(det) < 1e-12f) return false;
    const float inv = 1.0f / det;
    out[0] = (m[4] * m[8] - m[5] * m[7]) * inv;
    out[1] = (m[2] * m[7] - m[1] * m[8]) * inv;
    out[2] = (m[1] * m[5] - m[2] * m[4]) * inv;
    out[3] = (m[5] * m[6] - m[3] * m[8]) * inv;
    out[4] = (m[0] * m[8] - m[2] * m[6]) * inv;
    out[5] = (m[2] * m[3] - m[0] * m[5]) * inv;
    out[6] = (m[3] * m[7] - m[4] * m[6]) * inv;
    out[7] = (m[1] * m[6] - m[0] * m[7]) * inv;
    out[8] = (m[0] * m[4] - m[1] * m[3]) * inv;
    return true;
}

inline RGBColorSpace makeColorSpace(const char* name, const float* rgbFromXyz9) {
    RGBColorSpace cs;
    cs.name = name;
    for (int i = 0; i < 9; ++i) cs.rgbFromXyz[i] = rgbFromXyz9[i];
    invertMat3(cs.rgbFromXyz, cs.xyzFromRgb);
    return cs;
}

// Linear sRGB / Rec.709, D65 (IEC 61966-2-1).
inline const RGBColorSpace& colorSpaceSrgb() {
    static const float m[9] = {3.2404542f, -1.5371385f, -0.4985314f, -0.9692660f, 1.8760108f,
                               0.0415560f, 0.0556434f,  -0.2040259f, 1.0572252f};
    static const RGBColorSpace cs = makeColorSpace("sRGB", m);
    return cs;
}

// ACEScg (AP1), D60 — XYZ→AP1 matrix from ACES.
inline const RGBColorSpace& colorSpaceAcesCg() {
    static const float m[9] = {1.6410233797f, -0.3248032942f, -0.2364246952f, -0.6636628587f,
                               1.6153315917f,  0.0167563477f,  0.0117218943f,  -0.0082844420f,
                               0.9883948585f};
    static const RGBColorSpace cs = makeColorSpace("ACEScg", m);
    return cs;
}

// Rec.2020 linear, D65.
inline const RGBColorSpace& colorSpaceRec2020() {
    static const float m[9] = {1.716651187971268f, -0.355670783776392f, -0.253366281373660f,
                               -0.666684351832019f, 1.616481236634939f,  0.015768545813911f,
                               0.017639857445311f, -0.042770613257809f, 0.942103121235474f};
    static const RGBColorSpace cs = makeColorSpace("Rec.2020", m);
    return cs;
}

// Display P3 linear, D65.
inline const RGBColorSpace& colorSpaceDisplayP3() {
    static const float m[9] = {2.493496911941425f, -0.931383617919123f, -0.402710784450717f,
                               -0.829488969561575f, 1.762664060318346f,  0.023624685841943f,
                               0.035845830243784f, -0.076172389268042f, 0.956884524007687f};
    static const RGBColorSpace cs = makeColorSpace("Display P3", m);
    return cs;
}

inline const RGBColorSpace& colorSpaceById(int id) {
    switch (id) {
        case kSpectralColorSpaceAcesCg: return colorSpaceAcesCg();
        case kSpectralColorSpaceRec2020: return colorSpaceRec2020();
        case kSpectralColorSpaceDisplayP3: return colorSpaceDisplayP3();
        default: return colorSpaceSrgb();
    }
}

// Tabulated CIE 1931 2° CMFs with linear interpolation (5 nm).
inline void cieXyzAtLambdaTabulated(float lambda, float& x, float& y, float& z) {
    using namespace cie_tab;
    if (lambda <= kCieLambdaMin) {
        x = kCieX[0];
        y = kCieY[0];
        z = kCieZ[0];
        return;
    }
    if (lambda >= kCieLambdaMax) {
        x = kCieX[kCieTabSamples - 1];
        y = kCieY[kCieTabSamples - 1];
        z = kCieZ[kCieTabSamples - 1];
        return;
    }
    const float t = (lambda - kCieLambdaMin) / kCieLambdaStep;
    const int i0 = int(t);
    const int i1 = std::min(i0 + 1, kCieTabSamples - 1);
    const float f = t - float(i0);
    x = kCieX[i0] * (1.0f - f) + kCieX[i1] * f;
    y = kCieY[i0] * (1.0f - f) + kCieY[i1] * f;
    z = kCieZ[i0] * (1.0f - f) + kCieZ[i1] * f;
}

}  // namespace sol
