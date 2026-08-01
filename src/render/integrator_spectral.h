// PT Spectral — hero-wavelength unidirectional path tracer (CPU / Embree).
// v1: surface PT + NEE/MIS, no SSS / MNEE / photons / guiding.
// RGB BSDF sampling reused; colours lifted via smooth RGB→spectrum upsample.
#pragma once

#include <algorithm>
#include <vector>

#include "render/integrator_base.h"
#include "render/metal_spectra.h"
#include "render/spectrum.h"
#include "render/spectrum_rgb.h"

namespace sol {

struct SpectralBinBuffer {
    int width = 0;
    int height = 0;
    int bins = 0;
    std::vector<float> accum;

    void resize(int w, int h, int b) {
        width = std::max(0, w);
        height = std::max(0, h);
        bins = std::clamp(b, 0, 64);
        accum.assign(size_t(std::max(width, 0)) * size_t(std::max(height, 0)) * size_t(std::max(bins, 1)),
                     0.0f);
    }
    void clear() { std::fill(accum.begin(), accum.end(), 0.0f); }

    void addSample(int x, int y, const SampledSpectrum& s, const SampledWavelengths& w) {
        if (bins <= 0 || width <= 0 || height <= 0 || s.n <= 0) return;
        if (x < 0 || y < 0 || x >= width || y >= height) return;
        const size_t base = (size_t(y) * size_t(width) + size_t(x)) * size_t(bins);
        const float span = kSpectrumLambdaMax - kSpectrumLambdaMin;
        const int n = std::min(s.n, w.n);
        for (int i = 0; i < n; ++i) {
            float t = (w.lambda[i] - kSpectrumLambdaMin) / span;
            int bin = int(t * float(bins));
            bin = std::clamp(bin, 0, bins - 1);
            const float pdf = srMax(w.pdf[i], 1e-8f);
            accum[base + size_t(bin)] += s.values[i] / pdf;
        }
    }

    float binValue(int x, int y, int bin) const {
        if (bins <= 0 || !width || bin < 0 || bin >= bins) return 0.0f;
        if (x < 0 || y < 0 || x >= width || y >= height) return 0.0f;
        return accum[(size_t(y) * size_t(width) + size_t(x)) * size_t(bins) + size_t(bin)];
    }
};

inline const char* spectralMetalPresetName(int id) {
    switch (id) {
        case 1: return "Au";
        case 2: return "Ag";
        case 3: return "Cu";
        case 4: return "Al";
        default: return "";
    }
}

inline SampledSpectrum upsampleRgb(Vec3 rgb, const SampledWavelengths& w) {
    return rgbToSpectrumEmission(rgb, w);
}

// Lift an RGB BSDF weight with optional spectral metal Fresnel.
inline SampledSpectrum liftBsdfWeight(const Material& mat, const Frame& frame, Vec3 wo, Vec3 wi,
                                      Vec3 rgbWeight, const SampledWavelengths& w) {
    SampledSpectrum base = upsampleRgb(rgbWeight, w);
    if (mat.spectralMetalPreset <= 0 || mat.metallic < 0.5f) return base;
    const Vec3 wh = normalize(wo + wi);
    if (length(wh) < 1e-6f) return base;
    SampledSpectrum s(w.n);
    for (int i = 0; i < w.n; ++i) {
        const SpectralNk nk = metalNk(spectralMetalPresetName(mat.spectralMetalPreset), w.lambda[i]);
        const float F = conductorFresnel(dot(wh, wo), nk.eta, nk.k);
        const float mag = (rgbWeight.x + rgbWeight.y + rgbWeight.z) * (1.0f / 3.0f);
        s.values[i] = mag * (0.25f + 0.75f * F);
        (void)frame;
    }
    return s;
}

template <typename Tracer>
class SpectralPathIntegrator final : public Integrator<Tracer> {
public:
    const char* name() const override { return "PT Spectral"; }

    Vec3 Li(IntegratorSampleContext<Tracer>& ctx) const override {
        return LiPixel(ctx, -1, -1, nullptr);
    }

    Vec3 LiPixel(IntegratorSampleContext<Tracer>& ctx, int x, int y, SpectralBinBuffer* bins) const {
        const SceneView& scene = *ctx.scene;
        Tracer& tracer = *ctx.tracer;
        Rng& rng = *ctx.rng;
        const RenderSettingsData& settings = scene.settings;

        const int nLambda = std::clamp(settings.spectralSamples, 2, kMaxSpectrumSamples);
        const SampledWavelengths waves = SampledWavelengths::sampleUniform(nLambda, rng.nextFloat());

        SampledSpectrum radiance = SampledSpectrum::zero(waves.n);
        SampledSpectrum throughput = SampledSpectrum::constant(waves.n, 1.0f);
        float bsdfPdf = 0.0f;
        bool specularBounce = true;
        int depth = 0;
        int passThrough = 0;
        Vec3 origin = ctx.origin;
        Vec3 direction = ctx.direction;
        RayShadeKind rayKind = RayShadeKind::Camera;
        const int maxDepth = srMax(1, settings.maxDepth);

        while (depth <= maxDepth) {
            RayHit hit;
            if (!tracer.intersect(origin, direction, kFloatMax, hit)) {
                if (scene.domeLightIndex >= 0) {
                    const LightData& dome = scene.lights[scene.domeLightIndex];
                    const bool primary = depth == 0 && passThrough == 0;
                    if (!(primary && (!settings.envVisibleCamera || !dome.visibleCamera))) {
                        Vec3 envL = domeRadiance(scene, dome, direction);
                        if (!isBlack(envL)) {
                            float weight = 1.0f;
                            if (!specularBounce) {
                                const float lp =
                                    lightPdfDirection(scene, scene.domeLightIndex, origin, direction, origin,
                                                      direction) *
                                    lightSelectionPdfIndex(scene, scene.domeLightIndex);
                                weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                            }
                            SampledSpectrum contrib = throughput * upsampleRgb(envL, waves) * weight;
                            if (depth > 0 && !specularBounce && settings.clampIndirect > 0.0f) {
                                for (int i = 0; i < contrib.n; ++i)
                                    contrib.values[i] = srMin(contrib.values[i], settings.clampIndirect);
                            }
                            radiance += contrib;
                        }
                    }
                }
                break;
            }

            SurfaceInteraction si;
            if (!buildSurfaceInteraction(scene, hit, origin, direction, si)) break;
            const InstanceData& inst = scene.instances[si.instanceIndex];

            if (si.lightIndex >= 0 && depth == 0 && !inst.visibleCamera) {
                origin = offsetRayOrigin(si.p, si.ng, direction);
                ++passThrough;
                if (passThrough > 16) break;
                continue;
            }

            if (si.lightIndex >= 0) {
                const LightData& light = scene.lights[si.lightIndex];
                const Vec3 lightN = light.type == kLightSphere ? si.ng : areaLightNormal(light);
                Vec3 emitted = areaLightEmission(scene, light, direction, lightN);
                if (!isBlack(emitted)) {
                    float weight = 1.0f;
                    if (!specularBounce) {
                        const float lp =
                            lightPdfDirection(scene, si.lightIndex, origin, direction, si.p, lightN) *
                            lightSelectionPdfIndex(scene, si.lightIndex);
                        weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                    }
                    SampledSpectrum contrib = throughput * upsampleRgb(emitted, waves) * weight;
                    if (depth > 0 && !specularBounce && settings.clampIndirect > 0.0f) {
                        for (int i = 0; i < contrib.n; ++i)
                            contrib.values[i] = srMin(contrib.values[i], settings.clampIndirect);
                    }
                    radiance += contrib;
                }
                break;
            }

            Material baseMat = materialForRay(scene, si.materialIndex, rayKind);
            Material mat = evaluateTexturedMaterial(scene, baseMat, si.uv, si.ns, si.pObject, si.nObject,
                                                    si.uvFilterWidth, si.pRef, si.nRef, si.hasPref);

            if (mat.transmission <= 0.0f && mat.doubleSided && dot(si.ns, -direction) < 0.0f) {
                si.ns = -si.ns;
                si.ng = -si.ng;
            }

            if (mat.emissionStrength > 0.0f && !isBlack(mat.emissionColor)) {
                const bool frontFacing = dot(si.ns, -direction) > 0.0f;
                if (frontFacing || mat.doubleSided)
                    radiance +=
                        throughput * upsampleRgb(mat.emissionColor * mat.emissionStrength, waves);
            }

            if (mat.opacity <= 1e-6f || (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity)) {
                origin = offsetRayOrigin(si.p, si.ng, direction);
                ++passThrough;
                if (passThrough > 32) break;
                continue;
            }

            if (depth >= maxDepth) break;

            const Vec3 wo = -direction;
            const Frame frame(si.ns);

            // Apply spectral IOR for dispersing glass at this hero set (average λ).
            if (mat.dispersionAbbe > 0.0f && mat.transmission > 1e-4f && waves.n > 0) {
                float avgLam = 0.0f;
                for (int i = 0; i < waves.n; ++i) avgLam += waves.lambda[i];
                avgLam /= float(waves.n);
                mat.ior = dielectricIorFromAbbe(mat.ior, mat.dispersionAbbe, avgLam);
            }

            const Vec3 nee = nextEventEstimation(scene, tracer, si, mat, frame, wo, rng);
            if (!isBlack(nee)) {
                SampledSpectrum contrib = throughput * upsampleRgb(nee, waves);
                if (depth > 0 && settings.clampIndirect > 0.0f) {
                    for (int i = 0; i < contrib.n; ++i)
                        contrib.values[i] = srMin(contrib.values[i], settings.clampIndirect);
                }
                radiance += contrib;
            }

            const Vec3 woLocal = frame.toLocal(wo);
            BsdfSample bs =
                bsdfSampleLocal(mat, woLocal, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                rng.nextFloat());
            if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;

            const Vec3 wiWorld = normalize(frame.toWorld(bs.wi));
            SampledSpectrum wSpec = liftBsdfWeight(mat, frame, wo, wiWorld, bs.weight, waves);
            throughput *= wSpec;
            bsdfPdf = bs.pdf;
            specularBounce = bs.specular;
            rayKind = nextRayShadeKind(bs, computeLobes(mat));

            origin = offsetRayOrigin(si.p, si.ng, wiWorld);
            direction = wiWorld;
            ++depth;

            if (depth >= settings.rrStartDepth) {
                const float q = srMin(0.95f, spectrumMaxComponent(throughput));
                if (rng.nextFloat() > q) break;
                throughput *= (1.0f / srMax(q, 1e-4f));
            }
            if (spectrumMaxComponent(throughput) < 1e-6f) break;
        }

        if (bins) bins->addSample(x, y, radiance, waves);
        Vec3 rgb = spectrumToRgb(radiance, waves);
        return isFinite(rgb) ? rgb : Vec3(0.0f);
    }
};

}  // namespace sol
