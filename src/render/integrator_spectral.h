// PT Spectral — hero-wavelength unidirectional path tracer (CPU / Embree).
// RGB BSDF sampling reused; authored colours via Jakob; MC weights linear.
// Wavelength PDF + TerminateSecondary after first scattering (pbrt-v4).
#pragma once

#include <algorithm>

#include "render/integrator_base.h"
#include "render/spectral_common.h"
#include "render/volume_vdb.h"

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
        bool suppressCausticLight = false;
        bool sawNonSpecular = false;
        bool causticSuffix = false;
        int depth = 0;
        int passThrough = 0;
        int volumeScatterCount = 0;
        Vec3 origin = ctx.origin;
        Vec3 direction = ctx.direction;
        int currentMedium = -1;
        if (settings.integrator != kIntegratorWireframe)
            currentMedium = startingFogMediumIndex(scene, origin);
        RayShadeKind rayKind = RayShadeKind::Camera;
        const int maxDepth = srMax(1, settings.maxDepth);

        while (depth <= maxDepth) {
            RayHit hit;
            const bool didHit = tracer.intersect(origin, direction, kFloatMax, hit);
            if (const MediumData* med = getMedium(scene, currentMedium)) {
                const float tMax = didHit ? hit.t : 1.0e6f;
                MediumData medWalk = *med;
                if (settings.volumeSimilarity != 0)
                    medWalk = mediumWithVolumeSimilarity(*med, volumeScatterCount);
                MediumSample ms;
                if (medWalk.type == 2 && medWalk.volumeIndex >= 0 &&
                    medWalk.volumeIndex < scene.volumeCount && scene.volumes &&
                    scene.volumes[medWalk.volumeIndex]) {
                    const VolumeGrid& fogVol = *scene.volumes[medWalk.volumeIndex];
                    ms = sampleMediumVdbFogSpectral(fogVol, medWalk, origin, direction, tMax, rng,
                                                    throughput, waves);
                } else {
                    ms = sampleMediumHomogeneousSpectral(medWalk, tMax, rng, throughput, waves);
                }
                if (ms.absorbed || spectrumMaxComponent(throughput) < 1e-20f) break;
                if (ms.scattered) {
                    origin = origin + direction * ms.t;
                    const Vec3 woVol = -direction;
                    if (scene.lightCount > 0 && depth < maxDepth) {
                        const Vec3 volDirect =
                            nextEventEstimationVolumeOnce(scene, tracer, origin, woVol, medWalk, rng);
                        SampledSpectrum contrib = throughput * upsampleRgb(volDirect, waves);
                        if (depth > 0) contrib = clampSpectrumIndirect(contrib, settings.clampDirect);
                        radiance += contrib;
                    }
                    float phasePdf = 0.0f;
                    direction = sampleHenyeyGreenstein(woVol, medWalk.g, rng.nextFloat(), rng.nextFloat(),
                                                       phasePdf);
                    bsdfPdf = phasePdf;
                    specularBounce = false;
                    sawNonSpecular = true;
                    causticSuffix = false;
                    ++depth;
                    ++volumeScatterCount;
                    continue;
                }
                // Exited the fog AABB (analytical) before hitting any surface — leave the medium.
                if (medWalk.type == 2 && (!didHit || ms.t + 1e-4f < hit.t)) {
                    origin = origin + direction * ms.t;
                    currentMedium = -1;
                    ++passThrough;
                    continue;
                }
            }
            if (!didHit) {
                if (!suppressCausticLight && scene.domeLightIndex >= 0) {
                    const LightData& dome = scene.lights[scene.domeLightIndex];
                    if (!(causticSuffix && !lightContributesCaustics(dome))) {
                    const bool primary = depth == 0 && passThrough == 0;
                    if (!(primary && (!settings.envVisibleCamera || !dome.visibleCamera))) {
                        Vec3 envL = domeRadiance(scene, dome, direction, /*nearestTexel=*/depth > 0);
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
                            if (depth > 0 && !specularBounce)
                                contrib = clampSpectrumIndirect(contrib, settings.clampDirect);
                            radiance += contrib;
                        }
                    }
                    }
                }
                if (!suppressCausticLight) {
                    const bool primarySun = depth == 0 && passThrough == 0;
                    if (!(primarySun && !settings.envVisibleCamera)) {
                        const Vec3 sunL =
                            cameraSunDiscRadiance(scene, origin, direction, bsdfPdf, specularBounce,
                                                  primarySun, causticSuffix);
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

            if (consumeVolumeProxyHit(scene, settings.integrator, inst, hit, si, origin, direction,
                                      currentMedium)) {
                ++passThrough;
                continue;
            }

            if (si.lightIndex >= 0 && depth == 0 && !inst.visibleCamera) {
                origin = offsetRayOrigin(si.p, si.ng, direction);
                ++passThrough;
                continue;
            }

            if (si.lightIndex >= 0) {
                if (suppressCausticLight) break;
                const LightData& light = scene.lights[si.lightIndex];
                if (causticSuffix && !lightContributesCaustics(light)) break;
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
                continue;
            }

            if (depth >= maxDepth) break;

            const Vec3 wo = -direction;
            const Frame frame(si.ns);
            const float baseIor = mat.ior;

            // Arnold Standard Surface mix: stochastic SSS vs complementary BRDF.
            const float sssWeight = saturatef(mat.subsurface);
            if (materialSupportsSss(mat) && rng.nextFloat() < sssWeight) {
                Material specMat = sssSpecularEntryMaterial(mat);
                const Vec3 woLocalEntry = frame.toLocal(wo);
                const LobeWeights specLw = computeLobes(specMat, woLocalEntry);
                const float pSpec = sssEntrySpecularProb(specMat, woLocalEntry);

                if (pSpec > 0.0f && !(suppressCausticLight && !specularBounce)) {
                    const Vec3 nee =
                        nextEventEstimation(scene, tracer, si, specMat, frame, wo, rng, currentMedium);
                    if (!isBlack(nee)) {
                        SampledSpectrum contrib = throughput * upsampleRgb(nee, waves);
                        if (depth > 0) contrib = clampSpectrumIndirect(contrib, settings.clampDirect);
                        radiance += contrib;
                    }
                }

                if (pSpec > 0.0f && rng.nextFloat() < pSpec) {
                    throughput *= (1.0f / pSpec);
                    const float uSpec = specLw.diffuse + specLw.specular * rng.nextFloat();
                    BsdfSampleSpectral specBs =
                        bsdfSampleSpectral(specMat, woLocalEntry, uSpec, rng.nextFloat(), rng.nextFloat(),
                                           rng.nextFloat(), waves, specMat.ior, heroIdx);
                    if (specBs.valid && specBs.pdf > 0.0f) {
                        const Vec3 wiWorld = normalize(frame.toWorld(specBs.wi));
                        throughput *= specBs.weight;
                        origin = offsetRayOrigin(si.p, si.ng, wiWorld);
                        direction = wiWorld;
                        bsdfPdf = specBs.pdf;
                        specularBounce = specBs.specular;
                        BsdfSample bs{};
                        bs.wi = specBs.wi;
                        bs.pdf = specBs.pdf;
                        bs.specular = specBs.specular;
                        bs.transmitted = specBs.transmitted;
                        bs.weight = Vec3(1.0f);
                        rayKind = nextRayShadeKind(bs, specLw);
                        if (shouldTerminateSecondaryWavelengths(bs, specLw) &&
                            !waves.secondaryTerminated())
                            waves.terminateSecondary();
                        const bool causticBounce = specBs.specular || isNearSpecularLobe(specLw);
                        if (settings.caustics == 0 && causticBounce && sawNonSpecular)
                            suppressCausticLight = true;
                        if (causticBounce && sawNonSpecular) causticSuffix = true;
                        if (!causticBounce) {
                            sawNonSpecular = true;
                            causticSuffix = false;
                        }
                        ++depth;
                        continue;
                    }
                    break;
                }
                if (pSpec > 0.0f && pSpec < 0.999f) throughput *= (1.0f / (1.0f - pSpec));

                const SssWalkResult walk = sampleSssRandomWalk(scene, tracer, si, wo, mat, rng);
                if (!walk.escaped || isBlack(walk.pathWeight) || !isFinite(walk.pathWeight)) break;
                Material lambert = sssExitLambertMaterial();
                SurfaceInteraction ssSi = si;
                ssSi.p = walk.exitP;
                ssSi.ns = walk.exitN;
                ssSi.ng = walk.exitN;
                const Frame ssFrame(walk.exitN);
                if (!(suppressCausticLight && !specularBounce)) {
                    const Vec3 nee = nextEventEstimation(scene, tracer, ssSi, lambert, ssFrame,
                                                         walk.exitWo, rng, currentMedium);
                    if (!isBlack(nee)) {
                        SampledSpectrum contrib =
                            throughput * upsampleRgb(walk.pathWeight * nee, waves);
                        if (depth > 0) contrib = clampSpectrumIndirect(contrib, settings.clampDirect);
                        radiance += contrib;
                    }
                }
                const BsdfSample ssBs =
                    bsdfSampleLocal(lambert, ssFrame.toLocal(walk.exitWo), rng.nextFloat(),
                                    rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
                if (ssBs.pdf > 0.0f && !isBlack(ssBs.weight)) {
                    const Vec3 wiWorld = normalize(ssFrame.toWorld(ssBs.wi));
                    throughput *= upsampleRgb(walk.pathWeight * ssBs.weight, waves);
                    origin = offsetRayOrigin(walk.exitP, walk.exitN, wiWorld);
                    direction = wiWorld;
                    bsdfPdf = ssBs.pdf;
                    specularBounce = false;
                    rayKind = nextRayShadeKind(ssBs, computeLobes(lambert, ssFrame.toLocal(walk.exitWo)));
                    if (!waves.secondaryTerminated()) waves.terminateSecondary();
                    sawNonSpecular = true;
                    causticSuffix = false;
                    ++depth;
                    continue;
                }
                break;
            }

            if (!(suppressCausticLight && !specularBounce)) {
                // NEE stays RGB (glass specular contributes ~0); upsample aggregate.
                const Vec3 nee =
                    nextEventEstimation(scene, tracer, si, mat, frame, wo, rng, currentMedium);
                if (!isBlack(nee)) {
                    SampledSpectrum contrib = throughput * upsampleRgb(nee, waves);
                    if (depth > 0) contrib = clampSpectrumIndirect(contrib, settings.clampDirect);
                    radiance += contrib;
                }
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
            const LobeWeights lw = computeLobes(mat, woLocal);
            rayKind = nextRayShadeKind(bs, lw);
            if (shouldTerminateSecondaryWavelengths(bs, lw) && !waves.secondaryTerminated())
                waves.terminateSecondary();

            origin = offsetRayOrigin(si.p, si.ng, wiWorld);
            direction = wiWorld;
            if (ss.transmitted && inst.mediumIndex >= 0 && mediumIsActive(scene, inst.mediumIndex)) {
                const bool entering = dot(si.ng, wiWorld) < 0.0f;
                currentMedium = entering ? inst.mediumIndex : -1;
            }
            const bool causticBounce = ss.specular || isNearSpecularLobe(lw);
            if (settings.caustics == 0 && causticBounce && sawNonSpecular) suppressCausticLight = true;
            if (causticBounce && sawNonSpecular) causticSuffix = true;
            if (!causticBounce) {
                sawNonSpecular = true;
                causticSuffix = false;
            }
            ++depth;

            if (depth >= settings.rrStartDepth) {
                const float q = clampf(spectrumMaxComponent(throughput), 0.05f, 1.0f);
                if (rng.nextFloat() > q) break;
                throughput *= (1.0f / q);
            }
            if (spectrumMaxComponent(throughput) < 1e-6f) break;
        }

        if (bins) bins->addSample(x, y, radiance, waves);
        Vec3 rgb = spectrumToRgb(radiance, waves, settings.spectralColorSpace);
        return isFinite(rgb) ? rgb : Vec3(0.0f);
    }
};

}  // namespace sol
