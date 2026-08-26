// Spectral bidirectional path tracing. Geometry, MIS, camera projection and
// caustic-family partitioning are shared with the RGB BDPT implementation.
// Each subpath keeps its own SampledWavelengths; vertices snapshot arrival λ
// so TerminateSecondary on a continuation bounce cannot recolour SDS splats.
// Lights are RGBIlluminantSpectrum, opaque BSDF is RGBAlbedoSpectrum, film is
// spectrumToRgb (pbrt PathIntegrator — no per-sample von Kries).
#pragma once

#include <algorithm>

#include "render/integrator_base.h"
#include "render/integrator_bdpt.h"
#include "render/spectral_common.h"
#include "render/sss_spectral.h"

namespace sol {

inline bool spectrumNearBlack(const SampledSpectrum& s) {
    return spectrumMaxComponent(s) < 1e-8f;
}

inline bool spectrumIsFinite(const SampledSpectrum& s) {
    for (int i = 0; i < s.n; ++i)
        if (!srIsFinite(s.values[i])) return false;
    return true;
}

inline SampledSpectrum vertBsdfFSpectral(const bdpt::Vert& v, Vec3 woW, Vec3 wiW,
                                         const SampledWavelengths& w, const RGBColorSpace& cs) {
    if (v.mediumScatter) {
        const float cosTheta = clampf(dot(woW, wiW), -1.0f, 1.0f);
        const float p = henyeyGreenstein(cosTheta, v.mediumG);
        return SampledSpectrum::constant(w.n, p);
    }
    return bsdfEvalSpectral(v.mat, v.ng, v.ns, woW, wiW, w, v.mat.ior, cs);
}

namespace spectral_bdpt {

struct WalkConfig {
    bool eyePath = false;
    const RGBColorSpace* colorSpace = nullptr;
#if SOLSTICE_HAVE_OPENPGL
    PathGuiding::ThreadState* guiding = nullptr;
#endif
};

// Extend a BDPT subpath while carrying wavelength-sampled throughput in a
// parallel array. Eye-path SSS uses the same Chiang random-walk + Christensen–
// Burley fallback as the path tracer (light subpaths keep the specular entry).
template <typename Tracer>
SR_INL int randomWalk(const SceneView& scene, const Tracer& tracer, Rng& rng, bdpt::Vert* path,
                      SampledSpectrum* betaPath, SampledWavelengths* wavePath, int count,
                      Vec3 origin, Vec3 dir, float pdfDirSa, int maxVerts, const WalkConfig& cfg,
                      SampledWavelengths& waves, int heroIdx) {
    using namespace bdpt;
    const RGBColorSpace& cs = cfg.colorSpace ? *cfg.colorSpace : colorSpaceSrgb();
    SampledSpectrum beta = betaPath[count - 1];
    float pdfSaFwd = pdfDirSa;
    int passThrough = 0;
    RayShadeKind rayKind = RayShadeKind::Camera;
    int currentMedium = -1;
    const float heroLambda = waves.lambda[std::clamp(heroIdx, 0, waves.n - 1)];

    while (count < maxVerts) {
        Vert& prev = path[count - 1];
        RayHit hit;
        const bool didHit = tracer.intersect(origin, dir, kFloatMax, hit);

        // Keep the RGB medium sampler's event distribution, but apply its
        // transmittance/albedo as a wavelength-sampled transport factor.
        if (const MediumData* medium = getMedium(scene, currentMedium)) {
            const float tMax = didHit ? hit.t : 1.0e6f;
            const MediumSample ms =
                sampleMediumHomogeneousSpectral(*medium, tMax, rng, beta, waves);
            if (ms.absorbed || !spectrumIsFinite(beta) || spectrumNearBlack(beta)) break;
            if (ms.scattered) {
                Vert v{};
                v.type = VType::Surface;
                v.p = origin + dir * ms.t;
                v.ng = v.ns = -dir;
                v.wo = -dir;
                v.beta = spectrumToRgb(beta, waves, cs);
                v.mediumScatter = true;
                v.mediumG = medium->g;
                v.mediumIndex = currentMedium;
                v.connectable = true;
                const float majorant = mediumMajorant(*medium);
                const float distancePdf = majorant * expf(-majorant * ms.t);
                v.pdfFwd = pdfSaFwd * distancePdf;
                path[count] = v;
                betaPath[count] = beta;
                wavePath[count] = waves;
                ++count;
                if (count >= maxVerts) break;

                float phasePdf = 0.0f;
                const Vec3 wi = sampleHenyeyGreenstein(
                    -dir, medium->g, rng.nextFloat(), rng.nextFloat(), phasePdf);
                Vert& current = path[count - 1];
                Vert& previous = path[count - 2];
                const float reverseSa = henyeyGreensteinPdf(
                    clampf(dot(wi, -dir), -1.0f, 1.0f), medium->g);
                previous.pdfRev = toAreaPdf(reverseSa, current.p, previous.p, previous.ns);
                origin = v.p;
                dir = wi;
                pdfSaFwd = phasePdf;
                continue;
            }
        }

        if (!didHit) {
            if (cfg.eyePath && scene.domeLightIndex >= 0) {
                Vert v{};
                v.type = VType::Light;
                v.lightIndex = scene.domeLightIndex;
                v.p = origin + dir * 1.0e6f;
                v.ng = v.ns = -dir;
                v.wo = -dir;
                v.beta = spectrumToRgb(beta, waves, cs);
                v.pdfFwd = pdfSaFwd;
                v.connectable = false;
                path[count] = v;
                betaPath[count] = beta;
                wavePath[count] = waves;
                ++count;
            }
            break;
        }

        SurfaceInteraction si;
        if (!buildSurfaceInteraction(scene, hit, origin, dir, si)) break;

        if (si.lightIndex >= 0) {
            if (cfg.eyePath) {
                const InstanceData& inst = scene.instances[si.instanceIndex];
                if (count == 1 && !inst.visibleCamera) {
                    origin = offsetRayOrigin(si.p, si.ng, dir);
                    ++passThrough;
                    continue;
                }
                Vert v{};
                v.type = VType::Light;
                v.lightIndex = si.lightIndex;
                v.p = si.p;
                v.ng = v.ns = si.ng;
                v.wo = -dir;
                v.beta = spectrumToRgb(beta, waves, cs);
                v.pdfFwd = toAreaPdf(pdfSaFwd, prev.p, si.p, si.ng);
                v.connectable = false;
                path[count] = v;
                betaPath[count] = beta;
                wavePath[count] = waves;
                ++count;
            }
            break;
        }

        Material mat = cfg.eyePath ? materialForRay(scene, si.materialIndex, rayKind)
                                   : materialForCausticTransport(scene, si.materialIndex);
        mat = evaluateTexturedMaterial(scene, mat, si.uv, si.ns, si.pObject, si.nObject,
                                       si.uvFilterWidth, si.pRef, si.nRef, si.hasPref);
        const float baseIor = mat.ior;

        if (mat.opacity <= 1e-6f || (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity)) {
            origin = offsetRayOrigin(si.p, si.ng, dir);
            ++passThrough;
            continue;
        }

        Vert v{};
        v.type = VType::Surface;
        v.p = si.p;
        v.ng = si.ng;
        v.ns = si.ns;
        v.mat = mat;  // keep nd (Abbe) — hero η only inside bsdfSampleSpectral
        v.wo = -dir;
        v.beta = spectrumToRgb(beta, waves, cs);
        v.mediumIndex = scene.instances[si.instanceIndex].mediumIndex;
        v.pdfFwd = toAreaPdf(pdfSaFwd, prev.p, si.p, si.ns);
        {
            Material matHero = mat;
            matHero.ior = spectralAbsoluteIor(baseIor, mat.dispersionAbbe, heroLambda);
            const LobeWeights lw = computeLobes(matHero, Frame(si.ns).toLocal(-dir));
            v.delta = lw.delta && lw.diffuse < 1e-4f;
            v.connectable = !v.delta;
            v.nearSpec = v.delta || isNearSpecularLobe(lw) || isPhotonCausticCasterLobe(lw);
        }
        path[count] = v;
        betaPath[count] = beta;
        wavePath[count] = waves;
        ++count;
        if (count >= maxVerts) break;
        Vert& cur = path[count - 1];

        const Frame frame(cur.ns);
        const Vec3 woLocal = frame.toLocal(cur.wo);
        BsdfSample bs{};
        bool haveSample = false;
        Vec3 wiWorld;
        SampledSpectrum weight = SampledSpectrum::zero(waves.n);

#if SOLSTICE_HAVE_OPENPGL
        if (cfg.eyePath && cfg.guiding && cfg.guiding->active()) {
            cfg.guiding->beginSegment(cur.p, cur.wo);
            cur.guideSeg = cfg.guiding->segmentHandle();
        }
#endif

        if (materialSupportsSss(cur.mat) && rng.nextFloat() < saturatef(cur.mat.subsurface)) {
            Material specMat = sssSpecularEntryMaterial(cur.mat);
            const LobeWeights specLw = computeLobes(specMat, woLocal);
            const float pSpec = sssEntrySpecularProb(specMat, woLocal);

            if (pSpec > 0.0f && rng.nextFloat() < pSpec) {
                beta *= 1.0f / pSpec;
                cur.mat = specMat;
                cur.delta = specLw.delta && specLw.diffuse < 1e-4f;
                cur.connectable = !cur.delta;
                cur.nearSpec = cur.delta || isNearSpecularLobe(specLw) || isPhotonCausticCasterLobe(specLw);
                cur.beta = spectrumToRgb(beta, waves, cs);
                betaPath[count - 1] = beta;
                const float uSpec = specLw.diffuse + specLw.specular * rng.nextFloat();
                terminateSecondaryIfSpectralEta(specMat, waves);
                BsdfSampleSpectral ss =
                    bsdfSampleSpectral(specMat, woLocal, uSpec, rng.nextFloat(), rng.nextFloat(),
                                       rng.nextFloat(), waves, specMat.ior, heroIdx, cs);
                if (!ss.valid) break;
                bs.wi = ss.wi;
                bs.pdf = ss.pdf;
                bs.specular = ss.specular;
                bs.transmitted = ss.transmitted;
                bs.weight = Vec3(1.0f);
                weight = ss.weight;
                wiWorld = normalize(frame.toWorld(bs.wi));
                haveSample = true;
            } else if (!cfg.eyePath) {
                // Light subpath: no volume walk — stop diffuse transport into SSS.
                break;
            } else {
                if (pSpec > 0.0f && pSpec < 0.999f) beta *= 1.0f / (1.0f - pSpec);
                const SssWalkResultSpectral walk =
                    sampleSssRandomWalkSpectral(scene, tracer, si, -dir, mat, rng, waves);
                if (!walk.escaped || !sssSpectrumWeightValid(walk.pathWeight)) break;
                cur.p = walk.exitP;
                cur.ng = walk.exitN;
                cur.ns = walk.exitN;
                cur.wo = walk.exitWo;
                cur.mat = sssExitLambertMaterial();
                cur.delta = false;
                cur.connectable = true;
                cur.nearSpec = false;
                beta *= walk.pathWeight;
                cur.beta = spectrumToRgb(beta, waves, cs);
                betaPath[count - 1] = beta;

                const Frame ssFrame(cur.ns);
                const Vec3 ssWoLocal = ssFrame.toLocal(cur.wo);
                bs = bsdfSampleLocal(cur.mat, ssWoLocal, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                     rng.nextFloat());
                if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;
                wiWorld = normalize(ssFrame.toWorld(bs.wi));
                weight = upsampleAlbedo(bs.weight, waves, cs);
                bs.weight = Vec3(1.0f);
                haveSample = true;
            }
        }

#if SOLSTICE_HAVE_OPENPGL
        bool guideReady = false;
        const LobeWeights curLw = computeLobes(cur.mat, woLocal);
        if (!haveSample && cfg.eyePath && cfg.guiding && cfg.guiding->active() && !cur.delta &&
            !cur.nearSpec && curLw.diffuse > 1e-4f) {
            guideReady = cfg.guiding->prepare(cur.p, cur.ns, rng);
            if (guideReady && rng.nextFloat() < cfg.guiding->guideProbability()) {
                float guidePdf = 0.0f;
                if (cfg.guiding->sample(rng.nextFloat(), rng.nextFloat(), wiWorld, guidePdf) &&
                    guidePdf > 0.0f) {
                    const Vec3 wiLocal = frame.toLocal(wiWorld);
                    SampledSpectrum fSpec =
                        bsdfEvalSpectral(cur.mat, cur.ng, cur.ns, cur.wo, wiWorld, waves, baseIor, cs);
                    const BsdfEval ev = bsdfEvalLocal(cur.mat, woLocal, wiLocal);
                    if (ev.pdf > 0.0f && spectrumMaxComponent(fSpec) > 0.0f) {
                        const float pg = cfg.guiding->guideProbability();
                        const float mixPdf = pg * guidePdf + (1.0f - pg) * ev.pdf;
                        bs.wi = wiLocal;
                        bs.pdf = mixPdf;
                        bs.specular = false;
                        bs.transmitted = wiLocal.z < 0.0f;
                        bs.weight = Vec3(1.0f);
                        weight = fSpec * (fabsf(wiLocal.z) / mixPdf);
                        haveSample = true;
                    }
                }
            }
        }
#endif

        if (!haveSample) {
            const float uLobe = rng.nextFloat();
            const float u1 = rng.nextFloat();
            const float u2 = rng.nextFloat();
            const float uChoice = rng.nextFloat();
            terminateSecondaryIfSpectralEta(cur.mat, waves);
            BsdfSampleSpectral ss = bsdfSampleSpectral(cur.mat, woLocal, uLobe, u1, u2, uChoice,
                                                       waves, baseIor, heroIdx, cs);
            if (!ss.valid) break;
            bs.wi = ss.wi;
            bs.pdf = ss.pdf;
            bs.specular = ss.specular;
            bs.transmitted = ss.transmitted;
            bs.weight = Vec3(1.0f);
            weight = ss.weight;
            wiWorld = normalize(frame.toWorld(bs.wi));
#if SOLSTICE_HAVE_OPENPGL
            if (cfg.eyePath && cfg.guiding && guideReady && !bs.specular) {
                const float pg = cfg.guiding->guideProbability();
                const float guidePdf = cfg.guiding->pdf(wiWorld);
                const float mixPdf = pg * guidePdf + (1.0f - pg) * bs.pdf;
                if (mixPdf > 0.0f) {
                    weight *= bs.pdf / mixPdf;
                    bs.pdf = mixPdf;
                }
            }
#endif
        }

        if (!shadingNormalConsistent(cur.ng, cur.ns, cur.wo, wiWorld)) break;
        {
            const float revSa = bs.specular ? 0.0f : bsdfPdfSa(cur, wiWorld, cur.wo);
            prev.pdfRev =
                toAreaPdf(revSa, cur.p, prev.p, prev.type == VType::Surface ? prev.ns : prev.ng);
        }

#if SOLSTICE_HAVE_OPENPGL
        if (cfg.eyePath && cfg.guiding && cfg.guiding->active()) {
            const bool deltaSegment = bs.specular || cur.nearSpec;
            cfg.guiding->recordBounce(cur.ns, wiWorld, bs.pdf, spectrumToRgb(weight, waves, cs),
                                      deltaSegment, cur.mat.roughness, computeLobes(cur.mat).eta,
                                      1.0f);
        }
#endif

        beta *= weight;
        if (!spectrumIsFinite(beta) || spectrumNearBlack(beta)) break;
        {
            const LobeWeights lwTerm = computeLobes(cur.mat, woLocal);
            if (shouldTerminateSecondaryWavelengths(bs, lwTerm, cur.mat) &&
                !waves.secondaryTerminated())
                waves.terminateSecondary();
        }
        pdfSaFwd = bs.specular ? 0.0f : bs.pdf;
        origin = offsetRayOrigin(cur.p, cur.ng, wiWorld);
        dir = wiWorld;
        if (bs.transmitted && cur.mediumIndex >= 0 &&
            mediumIsActive(scene, cur.mediumIndex)) {
            const bool entering = dot(cur.ng, wiWorld) < 0.0f;
            currentMedium = entering ? cur.mediumIndex : -1;
        }
        if (cfg.eyePath) rayKind = nextRayShadeKind(bs, computeLobes(cur.mat, woLocal));
        passThrough = 0;
    }
    return count;
}

}  // namespace spectral_bdpt

// Spectral counterpart of traceRadianceBdpt. The RGB Vert arrays retain all
// geometric and MIS state; beta arrays carry transport at sampled wavelengths.
template <typename Tracer>
inline Vec3 traceRadianceBdptSpectral(
    const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction, Rng& rng,
    SampledWavelengths& waves, int heroIdx,
#if SOLSTICE_HAVE_OPENPGL
    PathGuiding::ThreadState* guiding,
#endif
    Framebuffer* splatFb, DispersionContext* dispersion, const CausticPhotonMap* photons,
    SampledSpectrum* outSpectrum = nullptr) {
    using namespace bdpt;
    const RenderSettingsData& settings = scene.settings;
    const RGBColorSpace& filmCs = pathColorSpace(settings);
    int maxVerts = std::clamp(settings.maxDepth + 1, 2, kMaxVerts);
    const bool causticsOn = settings.caustics != 0;
    const bool photonEngine = photons != nullptr;
    const bool photonCaustics = photonEngine && !photons->empty();
    const float photonRadius = photonCaustics ? photons->gatherRadius(settings) : 0.0f;

    const CameraProj camProj = buildCameraProj(scene);
    const bool doSplats = splatFb != nullptr && camProj.valid;
    if (doSplats) splatFb->addSplatPath();

    Vert eye[kMaxVerts];
    Vert light[kMaxVerts];
    SampledSpectrum eyeBeta[kMaxVerts];
    SampledSpectrum lightBeta[kMaxVerts];
    SampledWavelengths lightWaves = waves;
    SampledWavelengths eyeWaves = waves;
    SampledWavelengths lightWavePath[kMaxVerts];
    SampledWavelengths eyeWavePath[kMaxVerts];
    lightWavePath[0] = lightWaves;
    eyeWavePath[0] = eyeWaves;

    int nLight = 0;
    bool lightOriginDelta = false;
    if (scene.lightCount > 0) {
        Vec3 emitDir;
        float pdfDirSa = 0.0f;
        if (startLightPath(scene, rng, light[0], emitDir, pdfDirSa)) {
            nLight = 1;
            if (light[0].lightIndex >= 0) {
                lightBeta[0] = lightEmissionSpectrum(scene.lights[light[0].lightIndex], lightWaves, filmCs);
                const float rgbLum = length(light[0].beta);
                const Vec3 sRgb = spectrumToRgb(lightBeta[0], lightWaves, filmCs);
                const float sLum = length(sRgb);
                if (sLum > 1e-8f && rgbLum > 0.0f) lightBeta[0] *= rgbLum / sLum;
            } else {
                lightBeta[0] = upsampleEmission(light[0].beta, lightWaves, filmCs);
            }
            light[0].beta = spectrumToRgb(lightBeta[0], lightWaves, filmCs);
            lightOriginDelta = lightOriginIsDelta(scene, light[0]);
            spectral_bdpt::WalkConfig cfg;
            cfg.colorSpace = &filmCs;
            const bool inf = lightIsInfinite(scene.lights[light[0].lightIndex]);
            const Vec3 o = inf ? light[0].p : offsetRayOrigin(light[0].p, light[0].ng, emitDir);
            nLight = spectral_bdpt::randomWalk(scene, tracer, rng, light, lightBeta, lightWavePath,
                                               nLight, o, emitDir, pdfDirSa, maxVerts, cfg,
                                               lightWaves, heroIdx);
            correctInfiniteLightSubpathPdfs(scene, light, nLight, emitDir);
        }
    }

    eye[0] = Vert{};
    eye[0].type = VType::Camera;
    eye[0].p = origin;
    eye[0].ng = eye[0].ns = direction;
    eye[0].wo = -direction;
    eye[0].beta = Vec3(1.0f);
    eye[0].pdfFwd = 1.0f;
    eye[0].connectable = false;
    eyeBeta[0] = SampledSpectrum::constant(waves.n, 1.0f);
    spectral_bdpt::WalkConfig eyeCfg;
    eyeCfg.eyePath = true;
    eyeCfg.colorSpace = &filmCs;
#if SOLSTICE_HAVE_OPENPGL
    eyeCfg.guiding = guiding;
#endif
    float camPdfSa = 1.0f;
    if (camProj.valid) {
        const Vec3 dc = transformVector(camProj.worldToCam, direction);
        camPdfSa = cameraPdfOmega(camProj, srMax(1e-4f, -dc.z));
    }
    const int nEye =
        spectral_bdpt::randomWalk(scene, tracer, rng, eye, eyeBeta, eyeWavePath, 1, origin,
                                  direction, camPdfSa, maxVerts, eyeCfg, eyeWaves, heroIdx);

    SampledSpectrum radiance = SampledSpectrum::zero(waves.n);
    Vec3 filmRgb(0.0f);
    auto addFilm = [&](const SampledSpectrum& c, const SampledWavelengths& wConn) {
        if (!spectrumIsFinite(c)) return;
        radiance += c;
        const Vec3 rgb = spectrumToRgb(c, wConn, filmCs);
        if (isFinite(rgb)) filmRgb += rgb;
    };

    // Caustic photon gather.
    if (photonCaustics) {
        for (int t = 2; t <= nEye; ++t) {
            const Vert& E = eye[t - 1];
            if (E.type != VType::Surface || !E.connectable || E.nearSpec) continue;
            const Vec3 gather = photons->gather(E.p, E.ns, E.wo, E.mat, photonRadius);
            if (isBlack(gather) || !isFinite(gather)) continue;
            const SampledWavelengths& wE = eyeWavePath[t - 1];
            SampledSpectrum c = eyeBeta[t - 1] * upsampleLightRadiance(gather, wE, filmCs);
            if (t > 2) c = clampSpectrumIndirect(c, settings.clampDirect);
            addFilm(c, wE);
#if SOLSTICE_HAVE_OPENPGL
            if (guiding && guiding->active() && E.guideSeg)
                guiding->addScatteredAt(E.guideSeg,
                                        spectrumToRgb(upsampleLightRadiance(gather, wE, filmCs), wE,
                                                      filmCs));
#endif
        }
    }

    // t = 1 light-tracing splats.
    if (doSplats && nLight >= 2) {
        for (int s = 2; s <= nLight; ++s) {
            const Vert& v = light[s - 1];
            if (v.type != VType::Surface || !v.connectable) continue;
            bool lightPrefixCaustic = false;
            for (int i = 1; i < s - 1; ++i)
                if (light[i].nearSpec && materialContributesCaustics(light[i].mat))
                    lightPrefixCaustic = true;
            if (!causticsOn) {
                if (lightPrefixCaustic) continue;
            } else if (photonEngine && lightPrefixCaustic) {
                continue;
            } else if (lightPrefixCaustic && light[0].lightIndex >= 0 &&
                       !lightContributesCaustics(scene.lights[light[0].lightIndex])) {
                continue;
            }

            float px = 0.0f, py = 0.0f, cosTheta = 0.0f, dist2 = 0.0f;
            if (!projectToPixel(camProj, v.p, px, py, cosTheta, dist2) || dist2 < 1e-8f)
                continue;
            const Vec3 toCam = normalize(camProj.camPos - v.p);
            const SampledWavelengths& wL = lightWavePath[s - 1];
            const SampledSpectrum f = vertBsdfFSpectral(v, v.wo, toCam, wL, filmCs);
            if (spectrumNearBlack(f) ||
                !connectionVisible(scene, tracer, v.p, v.ng, camProj.camPos, -1))
                continue;

            const float pdfOmega = cameraPdfOmega(camProj, cosTheta);
            const float cosV = fabsf(dot(v.ns, toCam));
            SampledSpectrum c =
                lightBeta[s - 1] * f * (cosV * pdfOmega / dist2);
            MisOverride ov;
            ov.splatStrategy = true;
            ov.s0Sampled = !(causticsOn && lightPrefixCaustic);
            ov.lightOriginDelta = lightOriginDelta;
            ov.lightLastRev = toAreaPdf(pdfOmega, camProj.camPos, v.p, v.ns);
            if (s >= 2)
                ov.lightPrevRev =
                    toAreaPdf(bsdfPdfSa(v, toCam, normalize(light[s - 2].p - v.p)), v.p,
                              light[s - 2].p,
                              light[s - 2].type == VType::Surface ? light[s - 2].ns
                                                                  : light[s - 2].ng);
            Vert cameraVert = eye[0];
            cameraVert.p = camProj.camPos;
            c *= misWeight(&cameraVert, 1, light, s, ov);
            // Indirect Clamp (LT): radiance-scaled via W·H (see lightTraceSplatClamp).
            if (s >= 2) c = clampSpectrumIndirect(c, lightTraceSplatClamp(settings));
            if (!spectrumIsFinite(c)) continue;
            // Arrival λ at this vertex — not the walk's post-bounce TerminateSecondary.
            const Vec3 rgb = spectrumToRgb(c, wL, filmCs);
            if (isFinite(rgb)) splatFb->addSplat(int(px), int(py), rgb);
        }
    }

    // s = 0: eye path emission and light hits.
    for (int t = 2; t <= nEye; ++t) {
        const Vert& v = eye[t - 1];
        if (v.type == VType::Surface) {
            if (v.mat.emissionStrength > 0.0f && !isBlack(v.mat.emissionColor)) {
                const bool front = dot(v.ns, v.wo) > 0.0f;
                if (front || v.mat.doubleSided) {
                    const SampledWavelengths& wE = eyeWavePath[t - 1];
                    SampledSpectrum c =
                        eyeBeta[t - 1] *
                        upsampleEmission(v.mat.emissionColor * v.mat.emissionStrength, wE, filmCs);
                    if (t > 2) c = clampSpectrumIndirect(c, settings.clampDirect);
                    addFilm(c, wE);
                }
            }
            continue;
        }
        if (v.type != VType::Light || v.lightIndex < 0) continue;
        const LightData& l = scene.lights[v.lightIndex];

        if (t >= 4) {
            bool sawDiffuseThenSpec = false;
            bool diffuseSeen = false;
            for (int i = 1; i < t - 1; ++i) {
                if (!eye[i].nearSpec)
                    diffuseSeen = true;
                else if (diffuseSeen)
                    sawDiffuseThenSpec = true;
            }
            if (sawDiffuseThenSpec && (!causticsOn || !lightContributesCaustics(l))) break;
        }

        if (l.type == kLightDome) {
            const bool primary = t == 2;
            if (primary && (!settings.envVisibleCamera || !l.visibleCamera)) break;
            const Vec3 dirW = -v.wo;
            const Vec3 Le = domeRadiance(scene, l, dirW, /*nearestTexel=*/t > 2);
            if (!isBlack(Le)) {
                float w = 1.0f;
                const Vert& prev = eye[t - 2];
                if (t > 2 && !prev.delta && prev.type == VType::Surface) {
                    const float lightPdf =
                        lightPdfDirection(scene, v.lightIndex, prev.p, dirW, prev.p, dirW) *
                        lightSelectionPdfIndex(scene, prev.p, v.lightIndex);
                    w = powerHeuristic(1.0f, v.pdfFwd, 1.0f, lightPdf);
                }
                const SampledWavelengths& wE = eyeWavePath[t - 1];
                SampledSpectrum c = eyeBeta[t - 1] * upsampleLightRadiance(Le, wE, filmCs, &l) * w;
                if (t > 2) c = clampSpectrumIndirect(c, settings.clampDirect);
                addFilm(c, wE);
#if SOLSTICE_HAVE_OPENPGL
                if (guiding && guiding->active())
                    guiding->recordBackground(eye[t - 2].p, dirW,
                                              spectrumToRgb(upsampleLightRadiance(Le, wE, filmCs, &l),
                                                            wE, filmCs),
                                              w);
#endif
            }
            break;
        }

        const Vec3 lightN = l.type == kLightSphere ? v.ng : areaLightNormal(l);
        const Vec3 wi = -v.wo;
        const Vec3 Le = areaLightEmission(scene, l, wi, lightN);
        if (isBlack(Le)) break;

        bool specularToLight = false;
        for (int i = 1; i <= t - 2; ++i) {
            if (eye[i].type == VType::Surface && eye[i].nearSpec) {
                specularToLight = true;
                break;
            }
        }
        if (causticsOn && specularToLight && t >= 4) {
            int j = t - 2;
            while (j >= 1 && eye[j].type == VType::Surface && eye[j].nearSpec) --j;
            if (j >= 1 && !eye[j].nearSpec && !eye[1].nearSpec &&
                (doSplats || photonEngine))
                break;
            // SDS: MNEE owns when active; with Photon engine drop s=0 to avoid stacking.
            if (j >= 1 && !eye[j].nearSpec && eye[1].nearSpec) break;
        }

        MisOverride ov;
        ov.splatStrategy = doSplats;
        ov.eyeLastRev = pdfLightOrigin(scene, l, v.lightIndex, eye[t - 2].p);
        const Vec3 emitToPrev = normalize(eye[t - 2].p - v.p);
        ov.eyePrevRev =
            toAreaPdf(pdfLightDirSa(l, lightN, emitToPrev), v.p, eye[t - 2].p,
                      eye[t - 2].type == VType::Surface ? eye[t - 2].ns : eye[t - 2].ng);
        const SampledWavelengths& wE = eyeWavePath[t - 1];
        SampledSpectrum c =
            eyeBeta[t - 1] * upsampleLightRadiance(Le, wE, filmCs, &l) * misWeight(eye, t, light, 0, ov);
        if (t > 2) c = clampSpectrumIndirect(c, settings.clampDirect);
        if (specularToLight) {
            c = clampSpectrumIndirect(c, causticFireflyCap(settings));
        }
        addFilm(c, wE);
#if SOLSTICE_HAVE_OPENPGL
        if (guiding && guiding->active())
            guiding->recordLightHit(v.p, v.wo,
                                    spectrumToRgb(upsampleLightRadiance(Le, wE, filmCs, &l), wE,
                                                  filmCs),
                                    misWeight(eye, t, light, 0, ov));
#endif
        break;
    }

    // s = 1: light resampling, including the through-glass MNEE upgrade.
    for (int t = 2; t <= nEye; ++t) {
        const Vert& E = eye[t - 1];
        if (E.type != VType::Surface || !E.connectable) continue;
        const SampledWavelengths& wE = eyeWavePath[t - 1];

        float selectPdf = 0.0f;
        const int li = sampleLightIndex(scene, E.p, rng.nextFloat(), selectPdf);
        if (li < 0 || selectPdf <= 0.0f) continue;
        const LightData& l = scene.lights[li];

        if (l.type == kLightDome || l.type == kLightDistant) {
            LightSample ls;
            if (!sampleLight(scene, li, E.p, rng.nextFloat(), rng.nextFloat(), ls) || ls.pdf <= 0.0f ||
                isBlack(ls.radiance))
                continue;
            const SampledSpectrum f = vertBsdfFSpectral(E, E.wo, ls.wi, wE, filmCs);
            if (spectrumNearBlack(f)) continue;
            const float lightPdf = ls.pdf * selectPdf;
            const float cosAbs = fabsf(dot(E.ns, ls.wi));
            const SampledSpectrum unshadowed =
                f * upsampleLightRadiance(ls.radiance, wE, filmCs, &l) * (cosAbs / lightPdf);
            float visibility = 1.0f;
            if (scene.lights[li].shadowEnable) {
                const Vec3 o = offsetRayOrigin(E.p, E.ng, ls.wi);
                visibility = shadowVisibility(scene, tracer, o, ls.wi, 1.0e8f);
            }
            if (visibility <= 1e-5f) continue;
            const float bsdfPdf = ls.delta ? 0.0f : bsdfPdfSa(E, E.wo, ls.wi);
            const float w = ls.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, bsdfPdf);
            SampledSpectrum local = unshadowed * (w * visibility);
            SampledSpectrum c = eyeBeta[t - 1] * local;
            if (t >= 2) c = clampSpectrumIndirect(c, settings.clampDirect);
            if (E.nearSpec) c = clampSpectrumIndirect(c, causticFireflyCap(settings));
            addFilm(c, wE);
#if SOLSTICE_HAVE_OPENPGL
            if (guiding && guiding->active() && E.guideSeg && !E.nearSpec)
                guiding->addScatteredAt(E.guideSeg, spectrumToRgb(local, wE, filmCs));
#endif
            continue;
        }
        if (!lightIsFinite(l)) continue;

        Vert Ls{};
        Ls.type = VType::Light;
        Ls.lightIndex = li;
        float pdfPosArea = 0.0f;
        Vec3 lightN;
        if (l.type == kLightPoint) {
            Ls.p = lightOrigin(l);
            Ls.ng = Ls.ns = Vec3(0.0f, 1.0f, 0.0f);
            Ls.delta = true;
            pdfPosArea = 1.0f;
            lightN = Ls.ns;
        } else if (l.type == kLightSphere) {
            const Vec3 center = lightOrigin(l);
            const float radius = srMax(1e-5f, sphereLightRadius(l));
            const Vec3 sampledDir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
            Ls.p = center + sampledDir * radius;
            Ls.ng = Ls.ns = sampledDir;
            pdfPosArea = 1.0f / (4.0f * kPi * radius * radius);
            lightN = sampledDir;
        } else {
            if (l.type == kLightRect) {
                const Vec3 pLocal((rng.nextFloat() - 0.5f) * l.width,
                                  (rng.nextFloat() - 0.5f) * l.height, 0.0f);
                Ls.p = transformPoint(l.xform, pLocal);
                pdfPosArea = 1.0f / srMax(1e-12f, rectLightArea(l));
            } else {
                const Vec2 d = sampleConcentricDisk(rng.nextFloat(), rng.nextFloat());
                Ls.p = transformPoint(l.xform, Vec3(d.x * l.radius, d.y * l.radius, 0.0f));
                pdfPosArea = 1.0f / srMax(1e-12f, diskLightArea(l));
            }
            Ls.ng = Ls.ns = areaLightNormal(l);
            lightN = Ls.ns;
        }
        Ls.pdfFwd = selectPdf * pdfPosArea;

        Vec3 toLight = Ls.p - E.p;
        const float dist2 = lengthSquared(toLight);
        if (dist2 < 1e-10f) continue;
        const float dist = sqrtf(dist2);
        const Vec3 wi = toLight / dist;

        bool clearPath = true;
        bool glassPath = false;
        int blockerInstance = -1;
        if (l.shadowEnable) {
            const Vec3 o = offsetRayOrigin(E.p, E.ng, wi);
            RayHit shadowHit;
            if (tracer.intersect(o, wi, dist * (1.0f - 1e-3f), shadowHit)) {
                SurfaceInteraction blockerSi;
                clearPath = false;
                if (buildSurfaceInteraction(scene, shadowHit, o, wi, blockerSi)) {
                    if (blockerSi.lightIndex == li) {
                        clearPath = true;
                    } else if (blockerSi.lightIndex < 0 && causticsOn &&
                               lightContributesCaustics(l)) {
                        Material blocker =
                            materialForCausticTransport(scene, blockerSi.materialIndex);
                        blocker = evaluateTexturedMaterial(
                            scene, blocker, blockerSi.uv, blockerSi.ns, blockerSi.pObject,
                            blockerSi.nObject, blockerSi.uvFilterWidth, blockerSi.pRef,
                            blockerSi.nRef, blockerSi.hasPref);
                        glassPath = mnee::isCausticCaster(blocker);
                        blockerInstance = blockerSi.instanceIndex;
                    }
                }
            }
        }

        bool eyeThroughSpec = false;
        for (int i = 1; i <= t - 2; ++i) {
            if (eye[i].type == VType::Surface && eye[i].nearSpec) {
                eyeThroughSpec = true;
                break;
            }
        }

        if (!clearPath) {
            if (!causticsUseMnee(settings, &scene) || photonEngine) continue;
            if (!(glassPath && eyeThroughSpec)) continue;
            const Vec3 LeMnee =
                l.type == kLightPoint ? l.emittedRadiance() : lightRadiance(l);
            if (isBlack(LeMnee)) continue;
            const mnee::MneeResult mr = mnee::manifoldConnect(
                scene, tracer, E.p, E.ns, E.wo, E.mat, li, Ls.p, lightN, LeMnee,
                pdfPosArea, selectPdf, blockerInstance, dispersion);
            if (!mr.solved || isBlack(mr.contribution)) continue;
            SampledSpectrum local = upsampleLightRadiance(mr.contribution, wE, filmCs);
            SampledSpectrum c = eyeBeta[t - 1] * local;
            if (t >= 2) c = clampSpectrumIndirect(c, settings.clampDirect);
            c = clampSpectrumIndirect(c, causticFireflyCap(settings));  // was opt-in; safety floor when 0
            addFilm(c, wE);
#if SOLSTICE_HAVE_OPENPGL
            if (guiding && guiding->active() && E.guideSeg && !E.nearSpec)
                guiding->addScatteredAt(E.guideSeg, spectrumToRgb(local, wE, filmCs));
#endif
            continue;
        }

        Vec3 Le(0.0f);
        if (l.type == kLightPoint) {
            Le = l.emittedRadiance() / dist2;
        } else if (l.type == kLightSphere) {
            if (dot(lightN, -wi) <= 1e-6f) continue;
            Le = lightRadiance(l);
        } else {
            Le = areaLightEmission(scene, l, wi, lightN);
        }
        if (isBlack(Le)) continue;
        const SampledSpectrum f = vertBsdfFSpectral(E, E.wo, wi, wE, filmCs);
        if (spectrumNearBlack(f)) continue;

        SampledSpectrum local;
        if (l.type == kLightPoint) {
            local = f * upsampleLightRadiance(Le, wE, filmCs, &l) *
                    (fabsf(dot(E.ns, wi)) / srMax(1e-12f, selectPdf));
        } else {
            const float G = geometryTerm(E, Ls);
            if (G <= 0.0f) continue;
            local = f * upsampleLightRadiance(Le, wE, filmCs, &l) * (G / srMax(1e-12f, Ls.pdfFwd));
        }

        MisOverride ov;
        ov.splatStrategy = doSplats;
        ov.lightOriginDelta = l.type == kLightPoint;
        ov.lightLastRev = toAreaPdf(bsdfPdfSa(E, E.wo, wi), E.p, Ls.p, Ls.ns);
        ov.eyeLastRev = toAreaPdf(pdfLightDirSa(l, lightN, -wi), Ls.p, E.p, E.ns);
        if (t >= 3)
            ov.eyePrevRev =
                toAreaPdf(bsdfPdfSa(E, wi, normalize(eye[t - 2].p - E.p)), E.p,
                          eye[t - 2].p,
                          eye[t - 2].type == VType::Surface ? eye[t - 2].ns : eye[t - 2].ng);
        Vert oneLight[1] = {Ls};
        const float w = misWeight(eye, t, oneLight, 1, ov);
        local *= w;
        SampledSpectrum c = eyeBeta[t - 1] * local;
        if (t >= 2) c = clampSpectrumIndirect(c, settings.clampDirect);
        if (E.nearSpec) c = clampSpectrumIndirect(c, causticFireflyCap(settings));
        addFilm(c, wE);
#if SOLSTICE_HAVE_OPENPGL
        if (guiding && guiding->active() && E.guideSeg && !E.nearSpec)
            guiding->addScatteredAt(E.guideSeg, spectrumToRgb(local, wE, filmCs));
#endif
    }

    // s >= 2: connect eye and light surface vertices.
    for (int t = 2; t <= nEye; ++t) {
        const Vert& E = eye[t - 1];
        if (E.type != VType::Surface || !E.connectable) continue;
        for (int s = 2; s <= nLight; ++s) {
            if (s + t > maxVerts + 1) break;
            const Vert& Lv = light[s - 1];
            if (Lv.type != VType::Surface || !Lv.connectable) continue;

            bool lightPrefixCaustic = false;
            for (int i = 1; i < s - 1; ++i)
                if (light[i].nearSpec && materialContributesCaustics(light[i].mat))
                    lightPrefixCaustic = true;
            if (!causticsOn) {
                bool diffuseSeen = false;
                bool chainAfterDiffuse = false;
                for (int i = 1; i < t; ++i) {
                    if (!eye[i].nearSpec)
                        diffuseSeen = true;
                    else if (diffuseSeen && materialContributesCaustics(eye[i].mat))
                        chainAfterDiffuse = true;
                }
                if (chainAfterDiffuse) continue;
            } else if (photonEngine && lightPrefixCaustic) {
                continue;
            } else if (lightPrefixCaustic && light[0].lightIndex >= 0 &&
                       !lightContributesCaustics(scene.lights[light[0].lightIndex])) {
                continue;
            }

            Vec3 d = Lv.p - E.p;
            const float dist2 = lengthSquared(d);
            if (dist2 < 1e-10f) continue;
            d *= 1.0f / sqrtf(dist2);
            const SampledWavelengths wConn =
                bdptConnectWavelengths(eyeWavePath[t - 1], lightWavePath[s - 1]);
            const SampledSpectrum fE = vertBsdfFSpectral(E, E.wo, d, wConn, filmCs);
            const SampledSpectrum fL = vertBsdfFSpectral(Lv, Lv.wo, -d, wConn, filmCs);
            if (spectrumNearBlack(fE) || spectrumNearBlack(fL)) continue;
            const float G = geometryTerm(E, Lv);
            if (G <= 0.0f ||
                !connectionVisible(scene, tracer, E.p, E.ng, Lv.p, -1))
                continue;

            MisOverride ov;
            ov.splatStrategy = doSplats;
            ov.s0Sampled = !(causticsOn && doSplats && lightPrefixCaustic);
            ov.lightOriginDelta = lightOriginDelta;
            ov.lightLastRev = toAreaPdf(bsdfPdfSa(E, E.wo, d), E.p, Lv.p, Lv.ns);
            ov.eyeLastRev = toAreaPdf(bsdfPdfSa(Lv, Lv.wo, -d), Lv.p, E.p, E.ns);
            if (t >= 3)
                ov.eyePrevRev =
                    toAreaPdf(bsdfPdfSa(E, d, normalize(eye[t - 2].p - E.p)), E.p,
                              eye[t - 2].p,
                              eye[t - 2].type == VType::Surface ? eye[t - 2].ns
                                                                : eye[t - 2].ng);
            if (s >= 2)
                ov.lightPrevRev =
                    toAreaPdf(bsdfPdfSa(Lv, -d, normalize(light[s - 2].p - Lv.p)), Lv.p,
                              light[s - 2].p,
                              light[s - 2].type == VType::Surface ? light[s - 2].ns
                                                                  : light[s - 2].ng);

            SampledSpectrum c =
                eyeBeta[t - 1] * fE * fL *
                lightBeta[s - 1] * (G * misWeight(eye, t, light, s, ov));
            c = clampSpectrumIndirect(c, settings.clampDirect);
            if (lightPrefixCaustic || Lv.nearSpec || E.nearSpec)
                c = clampSpectrumIndirect(c, causticFireflyCap(settings));
            addFilm(c, wConn);
        }
    }

    if (!spectrumIsFinite(radiance)) {
        radiance = SampledSpectrum::zero(waves.n);
        filmRgb = Vec3(0.0f);
    }
    for (int i = 0; i < radiance.n; ++i)
        radiance.values[i] = srMax(0.0f, radiance.values[i]);
    if (outSpectrum) *outSpectrum = radiance;
    return isFinite(filmRgb) ? filmRgb : Vec3(0.0f);
}

template <typename Tracer>
class SpectralBdptIntegrator final : public Integrator<Tracer> {
public:
    const char* name() const override { return "BDPT"; }

    Vec3 Li(IntegratorSampleContext<Tracer>& ctx) const override {
        const int sampleCount = kMaxSpectrumSamples;
        SampledWavelengths waves =
            SampledWavelengths::sampleVisible(sampleCount, ctx.rng->nextFloat());
        const int heroPick =
            std::clamp(int(ctx.rng->nextFloat() * float(waves.n)), 0, waves.n - 1);
        waves.promoteHero(heroPick);
        const int heroIdx = 0;
        SampledSpectrum radiance = SampledSpectrum::zero(waves.n);
#if SOLSTICE_HAVE_OPENPGL
        const Vec3 rgb = traceRadianceBdptSpectral(
            *ctx.scene, *ctx.tracer, ctx.origin, ctx.direction, *ctx.rng, waves, heroIdx,
            ctx.guiding, ctx.splatFb, ctx.dispersion, ctx.photons, &radiance);
#else
        const Vec3 rgb = traceRadianceBdptSpectral(
            *ctx.scene, *ctx.tracer, ctx.origin, ctx.direction, *ctx.rng, waves, heroIdx,
            ctx.splatFb, ctx.dispersion, ctx.photons, &radiance);
#endif
        return rgb;
    }
};

}  // namespace sol
