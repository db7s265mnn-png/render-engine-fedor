// pbrt-v4 RGBColorSpace illuminants (Y-normalized).
// CIE D65 and ACES D60, 5 nm, 300–830 nm. Scale is ∫ S ȳ dλ = CIE_Y so
// SampledSpectrum::ToXYZ / CIE_Y maps the white illuminant to the colour
// space white point (sRGB/Rec.2020/P3 ≈ (1,1,1), ACEScg ≈ (1,1,1) under D60).
#pragma once

#include "render/color_space.h"

namespace sol {
namespace illum_tab {

constexpr int kIllumTabSamples = 107;
constexpr float kIllumLambdaMin = 300.f;
constexpr float kIllumLambdaMax = 830.f;
constexpr float kIllumLambdaStep = 5.f;

// CIE Illuminant D6500 × (CIE_Y / InnerProduct(S, ȳ)), pbrt stdillum-D65.
static const float kIllumD65[kIllumTabSamples] = {
    0.00034483f, 0.01683013f, 0.03331542f, 0.11897484f, 0.20463527f, 0.28966770f, 0.37470118f, 0.38933992f,
    0.40397970f, 0.42907270f, 0.45416673f, 0.46289681f, 0.47162687f, 0.49918730f, 0.52674772f, 0.51606089f,
    0.50537406f, 0.52899974f, 0.55262647f, 0.69473956f, 0.83685366f, 0.88099951f, 0.92514638f, 0.93498477f,
    0.94482317f, 0.91069569f, 0.87656923f, 0.96850444f, 1.06044065f, 1.12183324f, 1.18323601f, 1.18730122f,
    1.19136634f, 1.17644042f, 1.16152458f, 1.16689427f, 1.17226396f, 1.13630414f, 1.10034431f, 1.10308482f,
    1.10583535f, 1.09798817f, 1.09014091f, 1.07491154f, 1.05968224f, 1.07433513f, 1.08899822f, 1.07239354f,
    1.05578894f, 1.05396869f, 1.05215856f, 1.03170115f, 1.01124366f, 0.99270857f, 0.97417347f, 0.97141175f,
    0.96865010f, 0.93273880f, 0.89682751f, 0.90350476f, 0.91018202f, 0.90812307f, 0.90606520f, 0.89645641f,
    0.88684755f, 0.86454860f, 0.84225066f, 0.84432673f, 0.84640289f, 0.82783439f, 0.80926596f, 0.81021547f,
    0.81116506f, 0.82159706f, 0.83202906f, 0.81183650f, 0.79164404f, 0.74834762f, 0.70505221f, 0.71459735f,
    0.72414249f, 0.73799547f, 0.75184954f, 0.68740806f, 0.62296655f, 0.66484009f, 0.70671367f, 0.73301309f,
    0.75931251f, 0.70119431f, 0.64307716f, 0.55623864f, 0.46940112f, 0.57248325f, 0.67556534f, 0.65825997f,
    0.64095456f, 0.64561233f, 0.65027013f, 0.62573633f, 0.60120357f, 0.56331732f, 0.52543209f, 0.55314825f,
    0.58086444f, 0.59538486f, 0.60990633f};

// ACES D60 × (CIE_Y / InnerProduct(S, ȳ)), pbrt illum-acesD60.
static const float kIllumD60[kIllumTabSamples] = {
    0.00029704f, 0.01308300f, 0.02586896f, 0.09164510f, 0.15742124f, 0.22264602f, 0.28787079f, 0.30366455f,
    0.31945831f, 0.34246089f, 0.36546347f, 0.37741493f, 0.38936639f, 0.41232585f, 0.43528531f, 0.42665877f,
    0.41803224f, 0.44446017f, 0.47088810f, 0.60120324f, 0.73151837f, 0.77277784f, 0.81403731f, 0.82666846f,
    0.83929962f, 0.81294523f, 0.78659085f, 0.87810103f, 0.96961120f, 1.03192242f, 1.09423364f, 1.10243512f,
    1.11063659f, 1.10158449f, 1.09253238f, 1.10231998f, 1.11210757f, 1.08185611f, 1.05160464f, 1.05938917f,
    1.06717370f, 1.06313713f, 1.05910055f, 1.04953259f, 1.03996463f, 1.05790145f, 1.07583828f, 1.06193447f,
    1.04803067f, 1.04919224f, 1.05035380f, 1.03241140f, 1.01446899f, 0.99800314f, 0.98153730f, 0.98134810f,
    0.98115890f, 0.94651276f, 0.91186661f, 0.92232528f, 0.93278395f, 0.93320647f, 0.93362900f, 0.92621830f,
    0.91880761f, 0.89788317f, 0.87695873f, 0.88213253f, 0.88730632f, 0.86895153f, 0.85059674f, 0.85425948f,
    0.85792222f, 0.87186711f, 0.88581200f, 0.86541103f, 0.84501006f, 0.79798182f, 0.75095357f, 0.76323930f,
    0.77552502f, 0.78798524f, 0.80044545f, 0.73169337f, 0.66294128f, 0.70668823f, 0.75043518f, 0.77789685f,
    0.80535853f, 0.74348961f, 0.68162070f, 0.59027842f, 0.49893614f, 0.60840039f, 0.71786463f, 0.69900870f,
    0.68015276f, 0.68524235f, 0.69033195f, 0.66410285f, 0.63787375f, 0.59737970f, 0.55688565f, 0.58643003f,
    0.61597441f, 0.63149782f, 0.64702122f};

inline float sampleTable(const float* v, float lambdaNm) {
    if (lambdaNm <= kIllumLambdaMin) return v[0];
    if (lambdaNm >= kIllumLambdaMax) return v[kIllumTabSamples - 1];
    const float t = (lambdaNm - kIllumLambdaMin) / kIllumLambdaStep;
    const int i0 = int(t);
    const int i1 = (i0 + 1 < kIllumTabSamples) ? i0 + 1 : kIllumTabSamples - 1;
    const float f = t - float(i0);
    return v[i0] * (1.0f - f) + v[i1] * f;
}

}  // namespace illum_tab

inline float sampleWhiteIlluminant(int whiteIlluminant, float lambdaNm) {
    using namespace illum_tab;
    const float* tab = (whiteIlluminant == kWhiteIlluminantD60) ? kIllumD60 : kIllumD65;
    return sampleTable(tab, lambdaNm);
}

inline float sampleColorSpaceIlluminant(const RGBColorSpace& cs, float lambdaNm) {
    return sampleWhiteIlluminant(cs.whiteIlluminant, lambdaNm);
}

}  // namespace sol
