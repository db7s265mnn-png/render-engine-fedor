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

__device__ inline void addPathEmissionRgb(GpuPath& path, Vec3 rgb, float scale, float clampValue) {
    if (isBlack(rgb)) return;
    float s[kMaxSpectrumSamples];
    specUpsampleEmission(gpuSpec(), rgb, path.lambda, path.nLambda, s);
    addPathRadianceS(path, s, scale, clampValue);
}

__device__ inline void addPathLinearRgb(GpuPath& path, Vec3 rgb, float scale, float clampValue) {
    if (isBlack(rgb)) return;
    float s[kMaxSpectrumSamples];
    specUpsampleLinear(rgb, path.lambda, path.nLambda, s);
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

}  // namespace sol
