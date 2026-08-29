// Path Tracer — hero-wavelength unidirectional path tracer (CPU / Embree).
// pbrt-v4 PathIntegrator: textures filter in RGB; albedo → RGBAlbedoSpectrum,
// lights/env → RGBIlluminantSpectrum (Jakob × D65/D60). NEE is SampleLd:
// Illuminant(Le) × Albedo(f) × geom. Film is pbrt ToXYZ → working-space RGB.
#pragma once

#include <algorithm>

#include "render/integrator_base.h"
#include "render/spectral_common.h"
#include "render/sss_spectral.h"
#include "render/volume_vdb.h"

namespace sol {

// Eye-path contribution clamp. Specular / SDS (look-through-glass-at-light) uses
// the existing caustic firefly floor (10, or causticClamp); other indirect uses
// Direct Clamp. Primary hits (depth 0) stay unclamped, matching pbrt.
inline SampledSpectrum clampPathContribution(SampledSpectrum contrib, const RenderSettingsData& settings,
                                             int depth, bool specularBounce, bool causticSuffix) {
    return clampSpectrumIndirect(contrib, pathContributionClamp(settings, depth, specularBounce,
                                                               causticSuffix));
}

// pbrt SampleLd: RGB light sample + MIS pdf, then Illuminant(Le) × Albedo(f) × geom
// (not RGBIlluminantSpectrum of the RGB product). Matches OptiX evalSurfaceNeeSpectral.
template <typename Tracer>
inline SampledSpectrum nextEventEstimationSpectralOnce(const SceneView& scene, const Tracer& tracer,
                                                       const SurfaceInteraction& si, const Material& mat,
                                                       const Frame& frame, Vec3 wo, Rng& rng,
                                                       const SampledWavelengths& waves,
                                                       const RGBColorSpace& cs, int mediumIndex,
                                                       int eyeBounceNee = 0) {
    SampledSpectrum result = SampledSpectrum::zero(waves.n);
    if (scene.lightCount <= 0) return result;

    const Vec3 woLocal = frame.toLocal(wo);
    if (!eyePathNeeConnectable(mat, woLocal)) return result;
    float selectPdf = 0.0f;
    const int lightIndex = sampleLightIndex(scene, si.p, rng.nextFloat(), selectPdf);
    if (lightIndex < 0 || selectPdf <= 0.0f) return result;
    LightSample ls;
    if (!sampleLight(scene, lightIndex, si.p, rng.nextFloat(), rng.nextFloat(), ls)) return result;
    if (ls.pdf <= 0.0f || isBlack(ls.radiance)) return result;
    if (!shadingNormalConsistent(si.ng, si.ns, wo, ls.wi)) return result;
    const Vec3 wiLocal = frame.toLocal(ls.wi);
    const BsdfEval be = bsdfEvalLocal(mat, woLocal, wiLocal);
    if (be.pdf <= 0.0f || isBlack(be.f)) return result;

    const float lightPdf = ls.pdf * selectPdf;
    float visibility = 1.0f;
    Vec3 shadowOrigin = si.p;
    float tMax = 1.0e8f;
    if (scene.lights[lightIndex].shadowEnable) {
        shadowOrigin = offsetRayOrigin(si.p, si.ng, ls.wi);
        if (ls.distance < 1.0e7f) tMax = ls.distance * (1.0f - 1e-3f);
        visibility = shadowVisibility(scene, tracer, shadowOrigin, ls.wi, tMax, eyeBounceNee);
        if (visibility <= 1e-5f) return result;
    }

    const float misWeight = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, be.pdf);
    const float geom = fabsf(wiLocal.z) * visibility * misWeight / lightPdf;
    const SampledSpectrum Le =
        upsampleLightRadiance(ls.radiance, waves, cs, &scene.lights[lightIndex]);
    const SampledSpectrum f = bsdfEvalSpectral(mat, si.ng, si.ns, wo, ls.wi, waves, mat.ior, cs);
    result = Le * f * geom;

    if (scene.lights[lightIndex].shadowEnable)
        result *= rgbToSpectrumLinear(shadowTransmittanceFogVolumes(scene, shadowOrigin, ls.wi, tMax, rng),
                                      waves);
    if (const MediumData* med = getMedium(scene, mediumIndex)) {
        if (med->type != 2 && ls.distance < 1.0e7f)
            result *= rgbToSpectrumLinear(mediumShadowTr(*med, ls.distance), waves);
    }
    return result;
}

template <typename Tracer>
inline SampledSpectrum nextEventEstimationVolumeSpectralOnce(const SceneView& scene, const Tracer& tracer,
                                                             Vec3 origin, Vec3 woVol, const MediumData& med,
                                                             Rng& rng, const SampledWavelengths& waves,
                                                             const RGBColorSpace& cs, int eyeBounceNee = 0) {
    SampledSpectrum result = SampledSpectrum::zero(waves.n);
    if (scene.lightCount <= 0) return result;

    float selectPdf = 0.0f;
    const int li = sampleVolumeLightIndex(scene, origin, woVol, med.g, rng.nextFloat(), selectPdf);
    if (li < 0 || selectPdf <= 0.0f) return result;
    LightSample ls;
    if (!sampleLight(scene, li, origin, rng.nextFloat(), rng.nextFloat(), ls) || ls.pdf <= 0.0f ||
        isBlack(ls.radiance))
        return result;
    const float cosTheta = clampf(dot(woVol, ls.wi), -1.0f, 1.0f);
    const float phasePdfL = henyeyGreenstein(cosTheta, med.g);
    if (phasePdfL <= 0.0f) return result;
    const float lightPdf = ls.pdf * selectPdf;
    float vis = 1.0f;
    float tShadow = 1.0e8f;
    if (scene.lights[li].shadowEnable) {
        if (ls.distance < 1.0e7f) tShadow = ls.distance * (1.0f - 1e-3f);
        vis = shadowVisibility(scene, tracer, origin, ls.wi, tShadow, eyeBounceNee);
        if (vis <= 1e-5f) return result;
    }
    const float misW = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, phasePdfL);
    const float geom = phasePdfL * vis * misW / lightPdf;
    result = upsampleLightRadiance(ls.radiance, waves, cs, &scene.lights[li]) * geom;

    if (scene.lights[li].shadowEnable)
        result *= rgbToSpectrumLinear(shadowTransmittanceFogVolumes(scene, origin, ls.wi, tShadow, rng),
                                      waves);
    if (med.type != 2 && ls.distance < 1.0e7f)
        result *= rgbToSpectrumLinear(mediumShadowTr(med, ls.distance), waves);
    return result;
}

template <typename Tracer>
class SpectralPathIntegrator final : public Integrator<Tracer> {
public:
    const char* name() const override { return "Path Tracer"; }

    Vec3 Li(IntegratorSampleContext<Tracer>& ctx) const override {
        const SceneView& scene = *ctx.scene;
        Tracer& tracer = *ctx.tracer;
        Rng& rng = *ctx.rng;
        const RenderSettingsData& settings = scene.settings;
        const RGBColorSpace& filmCs = pathColorSpace(settings);

        const int nLambda = kMaxSpectrumSamples;
        SampledWavelengths waves = SampledWavelengths::sampleVisible(nLambda, rng.nextFloat());
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
        bool throughGlass = false;
        int depth = 0;
        int passThrough = 0;
        int exitEscapeMat = -1;
        int exitEscapeSkips = 0;
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
                float tMax = didHit ? hit.t : 1.0e6f;
                MediumData medWalk = *med;
                if (settings.volumeSimilarity != 0)
                    medWalk = mediumWithVolumeSimilarity(*med, volumeScatterCount);
                MediumSample ms;
                if (const VolumeGrid* fogVol =
                        (medWalk.type == 2) ? fogGridForMedium(scene, medWalk) : nullptr) {
                    tMax = clipTMaxToFogAabb(*fogVol, origin, direction, tMax);
                    ms = sampleMediumVdbFogSpectral(*fogVol, medWalk, origin, direction, tMax, rng,
                                                    throughput, waves);
                } else {
                    ms = sampleMediumHomogeneousSpectral(medWalk, tMax, rng, throughput, waves);
                }
                if (ms.absorbed || spectrumMaxComponent(throughput) < 1e-20f) break;
                        if (ms.scattered) {
                    origin = origin + direction * ms.t;
                    if (!isBlack(med->emission))
                        radiance += throughput * upsampleEmission(med->emission, waves, filmCs);
                    const Vec3 woVol = -direction;
                    if (scene.lightCount > 0 && depth < maxDepth) {
                        SampledSpectrum contrib =
                            throughput * nextEventEstimationVolumeSpectralOnce(
                                             scene, tracer, origin, woVol, medWalk, rng, waves, filmCs,
                                             depth > 0 ? 1 : 0);
                        contrib = clampPathContribution(contrib, settings, depth, false, false);
                        radiance += contrib;
                    }
                    float phasePdf = 0.0f;
                    direction = sampleHenyeyGreenstein(woVol, medWalk.g, rng.nextFloat(), rng.nextFloat(),
                                                       phasePdf);
                    bsdfPdf = phasePdf;
                    specularBounce = false;
                    sawNonSpecular = true;
                    causticSuffix = false;
                    rayKind = RayShadeKind::Volume;
                    ++depth;
                    ++volumeScatterCount;
                    if (depth >= settings.rrStartDepth) {
                        const float q = clampf(spectrumMaxComponent(throughput), 0.05f, 1.0f);
                        if (rng.nextFloat() > q) break;
                        throughput *= (1.0f / q);
                    }
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
                const bool exitEscaping = exitEscapeMat >= 0;
                if ((exitEscaping ||
                     (!suppressCausticLight &&
                      !cpuAimedSkipCameraSds(settings, causticSuffix ? 1 : 0, throughGlass ? 1 : 0))) &&
                    scene.domeLightIndex >= 0) {
                    const LightData& dome = scene.lights[scene.domeLightIndex];
                    if (exitEscaping || !(causticSuffix && !lightContributesCaustics(dome))) {
                    const bool primary = depth == 0 && passThrough == 0 && !exitEscaping;
                    if (exitEscaping ||
                        !(primary && (!settings.envVisibleCamera || !dome.visibleCamera))) {
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
                            // Env map: RGB texture → RGBIlluminantSpectrum (Jakob × D65/D60).
                            SampledSpectrum envS =
                                (dome.colorTemperatureK > 50.0f)
                                    ? lightEmissionSpectrum(dome, waves, filmCs) *
                                          (length(envL) /
                                           srMax(1e-6f, length(dome.emittedRadiance())))
                                    : upsampleEmission(envL, waves, filmCs);
                            SampledSpectrum contrib = throughput * envS * weight;
                            contrib = clampPathContribution(contrib, settings, depth, specularBounce,
                                                            causticSuffix);
                            radiance += contrib;
                        }
                    }
                    }
                }
                if (exitEscaping ||
                    (!suppressCausticLight &&
                     !cpuAimedSkipCameraSds(settings, causticSuffix ? 1 : 0, throughGlass ? 1 : 0))) {
                    const bool primarySun = depth == 0 && passThrough == 0 && !exitEscaping;
                    if (!(primarySun && !settings.envVisibleCamera)) {
                        const Vec3 sunL =
                            cameraSunDiscRadiance(scene, origin, direction, bsdfPdf, specularBounce,
                                                  primarySun, causticSuffix);
                        if (!isBlack(sunL)) {
                            SampledSpectrum contrib = throughput * upsampleEmission(sunL, waves, filmCs);
                            contrib = clampPathContribution(contrib, settings, depth, specularBounce,
                                                            causticSuffix);
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
                if (cpuAimedSkipCameraSds(settings, causticSuffix ? 1 : 0, throughGlass ? 1 : 0)) break;
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
                    SampledSpectrum Le = lightEmissionSpectrum(light, waves, filmCs);
                    // areaLightEmission may include cosine/visibility scaling vs emittedRadiance.
                    const float rgbScale =
                        length(emitted) / srMax(1e-6f, length(light.emittedRadiance()));
                    SampledSpectrum contrib = throughput * Le * (weight * rgbScale);
                    contrib = clampPathContribution(contrib, settings, depth, specularBounce, causticSuffix);
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
                                upsampleEmission(mat.emissionColor * mat.emissionStrength, waves,
                                                 filmCs);
            }

            if (mat.opacity <= 1e-6f || (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity)) {
                origin = offsetRayOrigin(si.p, si.ng, direction);
                ++passThrough;
                continue;
            }

            if (exitEscapeMat >= 0) {
                if (exitToDiffuseSkipSelf(exitEscapeMat, si.materialIndex, exitEscapeSkips)) {
                    origin = offsetRayOrigin(si.p, si.ng, direction);
                    ++exitEscapeSkips;
                    ++passThrough;
                    continue;
                }
                SampledSpectrum contrib =
                    throughput * nextEventEstimationSpectralOnce(
                                     scene, tracer, si, exitToDiffuseLambert(mat), Frame(si.ns),
                                     -direction, rng, waves, filmCs, currentMedium, 1);
                contrib = clampPathContribution(contrib, settings, depth, false, causticSuffix);
                radiance += contrib;
                break;
            }
            if (depth >= maxDepth) {
                if (exitToDiffuseShouldStart(mat, depth)) {
                    exitEscapeMat = si.materialIndex;
                    origin = offsetRayOrigin(si.p, si.ng, direction);
                    ++exitEscapeSkips;
                    ++passThrough;
                    continue;
                }
                break;
            }

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
                    SampledSpectrum contrib =
                        throughput * nextEventEstimationSpectralOnce(scene, tracer, si, specMat, frame,
                                                                     wo, rng, waves, filmCs, currentMedium,
                                                                     depth > 0 ? 1 : 0);
                    contrib = clampPathContribution(contrib, settings, depth, specularBounce, causticSuffix);
                    radiance += contrib;
                }

                if (pSpec > 0.0f && rng.nextFloat() < pSpec) {
                    throughput *= (1.0f / pSpec);
                    const float uSpec = specLw.diffuse + specLw.specular * rng.nextFloat();
                    terminateSecondaryIfSpectralEta(specMat, waves);
                    BsdfSampleSpectral specBs =
                        bsdfSampleSpectral(specMat, woLocalEntry, uSpec, rng.nextFloat(), rng.nextFloat(),
                                           rng.nextFloat(), waves, specMat.ior, heroIdx, filmCs);
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
                        if (shouldTerminateSecondaryWavelengths(bs, specLw, specMat) &&
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

                const Material sssBody = sssBodyMaterial(scene, si, mat);
                const SssWalkResultSpectral walk =
                    sampleSssRandomWalkSpectral(scene, tracer, si, wo, sssBody, rng, waves);
                if (!walk.escaped || !sssSpectrumWeightValid(walk.pathWeight)) break;
                Material lambert = sssExitLambertMaterial();
                SurfaceInteraction ssSi = si;
                ssSi.p = walk.exitP;
                ssSi.ns = walk.exitN;
                ssSi.ng = walk.exitN;
                const Frame ssFrame(walk.exitN);
                if (!(suppressCausticLight && !specularBounce)) {
                    SampledSpectrum contrib =
                        throughput * walk.pathWeight *
                        nextEventEstimationSpectralOnce(scene, tracer, ssSi, lambert, ssFrame,
                                                        walk.exitWo, rng, waves, filmCs, currentMedium,
                                                        depth > 0 ? 1 : 0);
                    contrib = clampPathContribution(contrib, settings, depth, false, causticSuffix);
                    radiance += contrib;
                }
                const BsdfSample ssBs =
                    bsdfSampleLocal(lambert, ssFrame.toLocal(walk.exitWo), rng.nextFloat(),
                                    rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
                if (ssBs.pdf > 0.0f && !isBlack(ssBs.weight)) {
                    const Vec3 wiWorld = normalize(ssFrame.toWorld(ssBs.wi));
                    throughput *= walk.pathWeight * upsampleAlbedo(ssBs.weight, waves, filmCs);
                    origin = offsetRayOrigin(walk.exitP, walk.exitN, wiWorld);
                    direction = wiWorld;
                    bsdfPdf = ssBs.pdf;
                    specularBounce = false;
                    rayKind = nextRayShadeKind(ssBs, computeLobes(lambert, ssFrame.toLocal(walk.exitWo)));
                    sawNonSpecular = true;
                    causticSuffix = false;
                    ++depth;
                    continue;
                }
                break;
            }

            if (!(suppressCausticLight && !specularBounce) &&
                !cpuAimedSkipCameraSds(settings, causticSuffix ? 1 : 0, throughGlass ? 1 : 0)) {
                // pbrt SampleLd: Illuminant(Le) × Albedo(f) × geom at this vertex.
                SampledSpectrum contrib =
                    throughput * nextEventEstimationSpectralOnce(scene, tracer, si, mat, frame, wo, rng,
                                                                 waves, filmCs, currentMedium,
                                                                 depth > 0 ? 1 : 0);
                contrib = clampPathContribution(contrib, settings, depth, specularBounce, causticSuffix);
                radiance += contrib;
            }

            const Vec3 woLocal = frame.toLocal(wo);
            const float uLobe = rng.nextFloat();
            const float u1 = rng.nextFloat();
            const float u2 = rng.nextFloat();
            const float uChoice = rng.nextFloat();
            terminateSecondaryIfSpectralEta(mat, waves);
            BsdfSampleSpectral ss =
                bsdfSampleSpectral(mat, woLocal, uLobe, u1, u2, uChoice, waves, baseIor, heroIdx,
                                   filmCs);
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
            if (shouldTerminateSecondaryWavelengths(bs, lw, mat) && !waves.secondaryTerminated())
                waves.terminateSecondary();

            origin = offsetRayOrigin(si.p, si.ng, wiWorld);
            direction = wiWorld;
            if (ss.transmitted && inst.mediumIndex >= 0 && mediumIsActive(scene, inst.mediumIndex)) {
                const bool entering = dot(si.ng, wiWorld) < 0.0f;
                currentMedium = entering ? inst.mediumIndex : -1;
            }
            const bool causticBounce = ss.specular || isNearSpecularLobe(lw);
            if (ss.transmitted && materialContributesCaustics(mat)) throughGlass = true;
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

        Vec3 rgb = spectrumToRgb(radiance, waves, filmCs);
        return isFinite(rgb) ? rgb : Vec3(0.0f);
    }
};

}  // namespace sol
