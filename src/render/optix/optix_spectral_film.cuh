// Spectral film / wavelengths for OptiX wavefront PT. No BSDF, no optixTrace.
#pragma once

#include "render/optix/optix_wavefront.cuh"
#include "render/spectrum_device.h"

namespace sol {

__device__ inline const GpuSpectralTables& gpuSpec() { return launchParams().spec; }

__device__ inline void addPathRadianceS(GpuPath& path, const float* add, float scale, float clampValue) {
    const int n = path.nLambda;
    float tmp[kMaxSpectrumSamples];
    for (int i = 0; i < n; ++i) tmp[i] = path.throughputS[i] * add[i] * scale;
    specClampIndirect(tmp, n, clampValue);
    if (!specIsFinite(tmp, n)) return;
    for (int i = 0; i < n; ++i) path.radianceS[i] += tmp[i];
}

// NEE already multiplied by vertex throughput. Do not use live path.throughputS
// (shade_surface / shade_volume step the BSDF or RR before shade_shadow).
__device__ inline void addBakedRadianceS(GpuPath& path, const float* add) {
    const int n = path.nLambda;
    if (!specIsFinite(add, n)) return;
    for (int i = 0; i < n; ++i) path.radianceS[i] += add[i];
}

__device__ inline void bakeNeeAtVertexS(const GpuPath& path, const float* neeS, float clampValue,
                                        float* out) {
    const int n = path.nLambda;
    for (int i = 0; i < n; ++i) out[i] = path.throughputS[i] * neeS[i];
    specClampIndirect(out, n, clampValue);
}

__device__ inline void addPathEmissionRgb(GpuPath& path, Vec3 rgb, float scale, float clampValue) {
    if (isBlack(rgb)) return;
    float s[kMaxSpectrumSamples];
    specUpsampleEmission(gpuSpec(), rgb, path.lambda, path.nLambda, s);
    addPathRadianceS(path, s, scale, clampValue);
}

__device__ inline void flushPathFilm(int pixel) {
    const LaunchParams& p = launchParams();
    if (!p.paths || !p.shadows) return;
    GpuPath& path = p.paths[pixel];
    const GpuShadow& shadow = p.shadows[pixel];
    if (!path.filmOpen) return;
    if (path.queue != kQueueDead) return;
    if (shadow.queue != kShadowIdle) return;
    Vec3 rgb = specToRgb(p.spec, path.radianceS, path.lambda, path.pdf, path.nLambda);
    path.filmOpen = 0;
    specZero(path.radianceS, path.nLambda);
    addRadiance(pixel, rgb);
}

__device__ inline void killPath(int pixel, GpuPath& path) {
    path.queue = kQueueDead;
    flushPathFilm(pixel);
}

__device__ inline void samplePathWavelengths(GpuPath& path, const GpuSpectralTables& tab) {
    int n = tab.samples;
    if (n < 2) n = 2;
    if (n > kMaxSpectrumSamples) n = kMaxSpectrumSamples;
    const float u = path.rng.nextFloat();
    specSampleVisible(n, u, path.lambda, path.pdf, path.nLambda);
    const int hero = int(path.rng.nextFloat() * float(path.nLambda));
    specPromoteHero(path.lambda, path.pdf, path.nLambda, hero);
    specFill(path.throughputS, path.nLambda, 1.0f);
    specZero(path.radianceS, path.nLambda);
    path.filmOpen = 1;
}

__device__ inline void specLightEmission(const LightData& light, const GpuPath& path, float* out) {
    const GpuSpectralTables& tab = gpuSpec();
    const int n = path.nLambda;
    const Vec3 rgb = light.emittedRadiance();
    if (light.colorTemperatureK > 50.0f) {
        specBlackbodyNorm(light.colorTemperatureK, path.lambda, n, out);
        Vec3 bbRgb = specToRgb(tab, out, path.lambda, path.pdf, n);
        const float bbLum = 0.2126f * bbRgb.x + 0.7152f * bbRgb.y + 0.0722f * bbRgb.z;
        const float wantLum = 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
        specMulS(out, wantLum / srMax(bbLum, 1e-8f), n);
        const Vec3 tint = light.color;
        if (fabsf(tint.x - 1.0f) > 1e-4f || fabsf(tint.y - 1.0f) > 1e-4f ||
            fabsf(tint.z - 1.0f) > 1e-4f) {
            float t[kMaxSpectrumSamples];
            specUpsampleEmission(tab, tint, path.lambda, n, t);
            float tAvg = 0.0f;
            for (int i = 0; i < n; ++i) tAvg += t[i];
            tAvg = srMax(tAvg / float(srMax(1, n)), 1e-8f);
            for (int i = 0; i < n; ++i) out[i] *= t[i] / tAvg;
        }
        return;
    }
    specUpsampleEmission(tab, rgb, path.lambda, n, out);
}

// Working-space RGB radiance (NEE Le, cosine-scaled area, env texel).
__device__ inline void specAuthoredRadiance(const LightData& light, Vec3 rgbLe, const GpuPath& path,
                                           float* out) {
    if (light.colorTemperatureK > 50.0f) {
        specLightEmission(light, path, out);
        const float have = length(light.emittedRadiance());
        const float want = length(rgbLe);
        if (have > 1e-8f) specMulS(out, want / have, path.nLambda);
        return;
    }
    specUpsampleEmission(gpuSpec(), rgbLe, path.lambda, path.nLambda, out);
}

__device__ inline void evalSurfaceNeeS(const LightData& light, Vec3 rgbLe, Vec3 rgbF, float scale,
                                       const GpuPath& path, float* out) {
    float Le[kMaxSpectrumSamples];
    float fS[kMaxSpectrumSamples];
    specAuthoredRadiance(light, rgbLe, path, Le);
    specUpsampleReflectance(gpuSpec(), rgbF, path.lambda, path.nLambda, fS);
    for (int i = 0; i < path.nLambda; ++i) out[i] = Le[i] * fS[i] * scale;
}

__device__ inline void enqueueShadowS(GpuShadow& shadow, Vec3 origin, Vec3 dir, float tMax,
                                      const float* contribS, int n, int mediumIndex,
                                      int eyeBounceNee = 0) {
    if (!specIsFinite(contribS, n) || specIsBlack(contribS, n)) {
        shadow.queue = kShadowIdle;
        shadow.splatPixel = -1;
        shadow.specContrib = 0;
        shadow.eyeBounceNee = 0;
        return;
    }
    shadow.origin = origin;
    shadow.direction = dir;
    shadow.tMax = tMax;
    shadow.contrib = Vec3(0.0f);
    shadow.specContrib = 1;
    for (int i = 0; i < n && i < kMaxSpectrumSamples; ++i) shadow.contribS[i] = contribS[i];
    shadow.occluded = 0;
    shadow.volumeTr = 1;
    shadow.mediumIndex = mediumIndex;
    shadow.splatPixel = -1;
    shadow.mneeCaster = -1;
    shadow.eyeBounceNee = eyeBounceNee;
    shadow.queue = kShadowTrace;
}

__device__ inline void enqueueOrAddVertexNeeS(GpuPath& path, GpuShadow& shadow, Vec3 origin, Vec3 dir,
                                              float tMax, const float* neeS, int mediumIndex,
                                              int shadowEnable, float clampValue, int eyeBounceNee = 0) {
    float baked[kMaxSpectrumSamples];
    bakeNeeAtVertexS(path, neeS, clampValue, baked);
    if (shadowEnable) {
        enqueueShadowS(shadow, origin, dir, tMax, baked, path.nLambda, mediumIndex, eyeBounceNee);
    } else {
        addBakedRadianceS(path, baked);
    }
}

}  // namespace sol
