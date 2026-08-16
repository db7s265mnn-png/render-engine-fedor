// PT Spectral — hero-wavelength unidirectional path tracer (CPU / Embree).
// RGB BSDF sampling reused; authored colours via Jakob; MC weights linear.
// Wavelength PDF + TerminateSecondary after first scattering (pbrt-v4).
#pragma once

#include <algorithm>

#include "render/integrator_base.h"
#include "render/spectral_common.h"

namespace sol {

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
        SampledWavelengths waves =
            (settings.spectralWavelengthSampling == 1)
                ? SampledWavelengths::sampleUniform(nLambda, rng.nextFloat())
                : SampledWavelengths::sampleVisible(nLambda, rng.nextFloat());
        // Hero λ for geometric dispersion — promote to slot 0 before TerminateSecondary.
        const int heroPick = std::clamp(int(rng.nextFloat() * float(waves.n)), 0, waves.n - 1);
        waves.promoteHero(heroPick);
        const int heroIdx = 0;

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
                                    lightSelectionPdfIndex(scene, origin, scene.domeLightIndex);
                                weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                            }
                            // Env: RGB texture → Jakob illuminant (or blackbody if CCT set).
                            SampledSpectrum envS =
                                (dome.colorTemperatureK > 50.0f)
                                    ? lightEmissionSpectrum(dome, waves) *
                                          (length(envL) /
                                           srMax(1e-6f, length(dome.emittedRadiance())))
                                    : upsampleEmission(envL, waves);
                            SampledSpectrum contrib = throughput * envS * weight;
                            radiance += contrib;
                        }
                    }
                }
                {
                    const bool primarySun = depth == 0 && passThrough == 0;
                    if (!(primarySun && !settings.envVisibleCamera)) {
                        const Vec3 sunL =
                            cameraSunDiscRadiance(scene, origin, direction, bsdfPdf, specularBounce,
                                                  primarySun, false);
                        if (!isBlack(sunL)) {
                            SampledSpectrum contrib = throughput * upsampleEmission(sunL, waves);
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
                            lightSelectionPdfIndex(scene, origin, si.lightIndex);
                        weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                    }
                    SampledSpectrum Le = lightEmissionSpectrum(light, waves);
                    // areaLightEmission may include cosine/visibility scaling vs emittedRadiance.
                    const float rgbScale =
                        length(emitted) / srMax(1e-6f, length(light.emittedRadiance()));
                    SampledSpectrum contrib = throughput * Le * (weight * rgbScale);
                    if (depth > 0 && !specularBounce)
                        contrib = clampSpectrumIndirect(contrib, settings.clampDirect);
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
                    radiance += throughput *
                                upsampleEmission(mat.emissionColor * mat.emissionStrength, waves);
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

            const float baseIor = mat.ior;
            // NEE stays RGB (glass specular contributes ~0); upsample aggregate.
            const Vec3 nee = nextEventEstimation(scene, tracer, si, mat, frame, wo, rng);
            if (!isBlack(nee)) {
                SampledSpectrum contrib = throughput * upsampleRgb(nee, waves);
                if (depth > 0) contrib = clampSpectrumIndirect(contrib, settings.clampDirect);
                radiance += contrib;
            }

            const Vec3 woLocal = frame.toLocal(wo);
            const float uLobe = rng.nextFloat();
            const float u1 = rng.nextFloat();
            const float u2 = rng.nextFloat();
            const float uChoice = rng.nextFloat();
            BsdfSampleSpectral ss =
                bsdfSampleSpectral(mat, woLocal, uLobe, u1, u2, uChoice, waves, baseIor, heroIdx);
            if (!ss.valid || ss.pdf <= 0.0f) break;

            const Vec3 wiWorld = normalize(frame.toWorld(ss.wi));
            throughput *= ss.weight;
            bsdfPdf = ss.pdf;
            specularBounce = ss.specular;
            BsdfSample bs{};
            bs.wi = ss.wi;
            bs.pdf = ss.pdf;
            bs.specular = ss.specular;
            bs.transmitted = ss.transmitted;
            bs.weight = Vec3(1.0f);  // unused after spectral weight
            rayKind = nextRayShadeKind(bs, computeLobes(mat));

            const LobeWeights lw = computeLobes(mat);
            if (shouldTerminateSecondaryWavelengths(bs, lw) && !waves.secondaryTerminated())
                waves.terminateSecondary();

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
        Vec3 rgb = spectrumToRgb(radiance, waves, settings.spectralColorSpace);
        return isFinite(rgb) ? rgb : Vec3(0.0f);
    }
};

}  // namespace sol
