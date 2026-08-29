// Caustic-only photon map (VCM-style density estimation on diffuse after a
// specular chain). Used when RenderSettingsData::causticsEngine == Photon.
// CPU / Embree only — included from embree_device.cpp and the integrators.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/rng.h"
#include "render/integrator.h"
#include "render/lights.h"
#include "render/shading.h"
#include "render/spectral_common.h"
#include "scene/types.h"

namespace sol {

struct CausticPhoton {
    Vec3 p{0.0f};
    Vec3 wi{0.0f};   // incident direction at the deposit (toward the surface)
    Vec3 power{0.0f};
    Vec3 n{0.0f};
    float lambdaNm = 0.0f;   // hero λ used for Snell / Abbe (0 = RGB fallback)
    float lambdaPdf = 0.0f;  // visible-wavelength PDF at lambdaNm
};

// Flat hash-grid photon map. Built once per progressive pass from lights that
// Contribute to Caustics, tracing only through materials that Contribute to
// Caustics, depositing on the first diffuse hit after a specular/near-spec chain.
class CausticPhotonMap {
public:
    void clear() {
        photons_.clear();
        cells_.clear();
        cellStarts_.clear();
        invCellSize_ = 1.0f;
        origin_ = Vec3(0.0f);
    }

    int size() const { return int(photons_.size()); }
    bool empty() const { return photons_.empty(); }

    float gatherRadius(const RenderSettingsData& settings) const {
        const float base = srMax(1e-4f, settings.photonRadius);
        // Progressive photon mapping radius shrink (Jensen/Hachisuka style).
        const float s = float(srMax(0, settings.progressiveSample));
        return base * powf(1.0f / (1.0f + s * 0.25f), 1.0f / 6.0f);
    }

    template <typename Tracer>
    void build(const SceneView& scene, const Tracer& tracer, int photonCount, uint32_t seed) {
        clear();
        if (photonCount <= 0 || scene.lightCount <= 0) return;

        photons_.reserve(size_t(photonCount));
        const int maxDepth = srMax(2, scene.settings.maxDepth);

        for (int i = 0; i < photonCount; ++i) {
            Rng rng(hashCombine(uint32_t(i), seed), hashUint(uint32_t(i) * 747796405u ^ seed));
            emitOne(scene, tracer, rng, maxDepth);
        }
        if (photons_.empty()) return;
        // Flux → per-photon power (density estimation divides by gather area).
        const float invN = 1.0f / float(photonCount);
        for (CausticPhoton& ph : photons_) ph.power = ph.power * invN;
        buildGrid_(gatherRadius(scene.settings));
    }

    // Estimate reflected radiance at a diffuse-ish shading point (W/m²/sr).
    Vec3 gather(Vec3 p, Vec3 n, Vec3 wo, const Material& mat, float radius) const {
        Vec3 sum(0.0f);
        forEachPhotonInBall_(p, n, wo, mat, radius, [&](const CausticPhoton&, const Vec3& rgb) {
            sum += rgb;
        });
        return sum;
    }

    // Same kernel, reconstructed at the eye path's wavelengths. A stored hero λ
    // becomes a monochromatic illuminant (spectral-locus colour).
    SampledSpectrum gatherSpectral(Vec3 p, Vec3 n, Vec3 wo, const Material& mat, float radius,
                                   const SampledWavelengths& waves, const RGBColorSpace& cs) const {
        SampledSpectrum sum = SampledSpectrum::zero(waves.n);
        forEachPhotonInBall_(p, n, wo, mat, radius, [&](const CausticPhoton& ph, const Vec3& rgb) {
            sum += photonPowerSpectrum(ph, rgb, waves, cs);
        });
        return sum;
    }

    static SampledSpectrum photonPowerSpectrum(const CausticPhoton& ph, Vec3 rgbContrib,
                                              const SampledWavelengths& waves,
                                              const RGBColorSpace& cs) {
        if (isBlack(rgbContrib)) return SampledSpectrum::zero(waves.n);
        if (ph.lambdaNm >= kSpectrumLambdaMin && ph.lambdaNm <= kSpectrumLambdaMax &&
            ph.lambdaPdf > 1e-12f) {
            SampledWavelengths wPh;
            wPh.n = 1;
            wPh.lambda[0] = ph.lambdaNm;
            wPh.pdf[0] = ph.lambdaPdf;
            const Vec3 rgbSpike = spectrumToRgb(SampledSpectrum::constant(1, 1.0f), wPh, cs);
            const float spikeLum = srMax(luminance(rgbSpike), 1e-8f);
            return upsampleEmission(rgbSpike * (luminance(rgbContrib) / spikeLum), waves, cs);
        }
        return upsampleLightRadiance(rgbContrib, waves, cs);
    }

private:
    template <typename Fn>
    void forEachPhotonInBall_(Vec3 p, Vec3 n, Vec3 wo, const Material& mat, float radius, Fn&& fn) const {
        if (photons_.empty() || radius <= 1e-8f) return;
        const LobeWeights lw = computeLobes(mat, Frame(n).toLocal(wo));
        if (lw.delta && lw.diffuse < 1e-4f) return;

        const float r2 = radius * radius;
        if (r2 <= 1e-20f) return;
        const float norm = 3.0f / (kPi * r2);

        const Vec3 cellF = (p - origin_) * invCellSize_;
        const int cx = int(floorf(cellF.x));
        const int cy = int(floorf(cellF.y));
        const int cz = int(floorf(cellF.z));
        const int nHood = srMax(1, int(ceilf(radius * invCellSize_)) + 1);
        for (int dz = -nHood; dz <= nHood; ++dz)
            for (int dy = -nHood; dy <= nHood; ++dy)
                for (int dx = -nHood; dx <= nHood; ++dx) {
                    const uint32_t h = hashCell_(cx + dx, cy + dy, cz + dz);
                    const uint32_t start = cellStarts_[h];
                    const uint32_t end = cellStarts_[h + 1];
                    for (uint32_t i = start; i < end; ++i) {
                        const CausticPhoton& ph = photons_[cells_[i]];
                        const Vec3 d = ph.p - p;
                        const float dist2 = dot(d, d);
                        if (dist2 > r2) continue;
                        if (dot(ph.n, n) <= 0.1f) continue;
                        const float cosI = fabsf(dot(n, ph.wi));
                        if (cosI <= 1e-6f) continue;
                        const float u = dist2 / r2;
                        const float kern = (1.0f - u);
                        const float kern2 = kern * kern;
                        const Frame fr(n);
                        const BsdfEval be = bsdfEvalLocal(mat, fr.toLocal(wo), fr.toLocal(ph.wi));
                        if (be.pdf <= 0.0f || isBlack(be.f)) continue;
                        fn(ph, be.f * ph.power * kern2 * norm);
                    }
                }
    }

    std::vector<CausticPhoton> photons_;
    std::vector<uint32_t> cells_;
    std::vector<uint32_t> cellStarts_;
    float invCellSize_ = 1.0f;
    float cellSize_ = 1.0f;
    Vec3 origin_{0.0f};

    static uint32_t hashCell_(int x, int y, int z) {
        uint32_t h = uint32_t(x) * 73856093u ^ uint32_t(y) * 19349663u ^ uint32_t(z) * 83492791u;
        h ^= h >> 16;
        h *= 0x7feb352du;
        h ^= h >> 15;
        return h & 65535u;  // 65536 slots — fewer collisions → less structured noise
    }

    void buildGrid_(float gatherRadiusHint) {
        Bounds3 b;
        for (const CausticPhoton& ph : photons_) b.extend(ph.p);
        const Vec3 ext = b.extent();
        const float maxExt = srMax(ext.x, srMax(ext.y, ext.z));
        // Cell size tied to the gather radius so a 3×3×3 (or expanded) walk
        // covers the kernel. Cap by scene extent so tiny radii don't explode slots.
        float cell = srMax(1e-4f, gatherRadiusHint);
        if (maxExt > 1e-6f) cell = srMin(cell, maxExt / 8.0f);
        cellSize_ = cell;
        invCellSize_ = 1.0f / cell;
        origin_ = b.lo - Vec3(cell);

        constexpr int kSlots = 65536;
        std::vector<uint32_t> counts(kSlots, 0);
        for (const CausticPhoton& ph : photons_) {
            const Vec3 c = (ph.p - origin_) * invCellSize_;
            counts[hashCell_(int(floorf(c.x)), int(floorf(c.y)), int(floorf(c.z)))]++;
        }
        cellStarts_.assign(kSlots + 1, 0);
        for (int i = 0; i < kSlots; ++i) cellStarts_[i + 1] = cellStarts_[i] + counts[i];
        cells_.assign(photons_.size(), 0);
        std::vector<uint32_t> cursor = cellStarts_;
        for (uint32_t i = 0; i < photons_.size(); ++i) {
            const Vec3 c = (photons_[i].p - origin_) * invCellSize_;
            const uint32_t h = hashCell_(int(floorf(c.x)), int(floorf(c.y)), int(floorf(c.z)));
            cells_[cursor[h]++] = i;
        }
    }

    template <typename Tracer>
    void emitOne(const SceneView& scene, const Tracer& tracer, Rng& rng, int maxDepth) {
        float selectPdf = 0.0f;
        const int li = sampleLightIndex(scene, rng.nextFloat(), selectPdf);
        if (li < 0 || selectPdf <= 0.0f) return;
        const LightData& l = scene.lights[li];
        if (!lightContributesCaustics(l)) return;
        if (l.type == kLightDome || l.type == kLightDistant) return;

        const RGBColorSpace& filmCs = pathColorSpace(scene.settings);
        SampledWavelengths waves = SampledWavelengths::sampleVisible(kMaxSpectrumSamples, rng.nextFloat());
        const int heroPick = std::clamp(int(rng.nextFloat() * float(waves.n)), 0, std::max(0, waves.n - 1));
        waves.promoteHero(heroPick);
        const int heroIdx = 0;

        Vec3 p, n;
        Vec3 dir;
        float pdfDir = 0.0f;
        SampledSpectrum betaS = lightEmissionSpectrum(l, waves, filmCs);

        if (l.type == kLightPoint) {
            p = lightOrigin(l);
            n = Vec3(0.0f, 1.0f, 0.0f);
            dir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
            pdfDir = kInv4Pi;
            betaS *= 1.0f / srMax(1e-12f, selectPdf * pdfDir);
        } else {
            float area = 0.0f;
            if (l.type == kLightRect) {
                const Vec3 pLocal((rng.nextFloat() - 0.5f) * l.width, (rng.nextFloat() - 0.5f) * l.height, 0.0f);
                p = transformPoint(l.xform, pLocal);
                n = areaLightNormal(l);
                area = rectLightArea(l);
            } else if (l.type == kLightDisk) {
                const Vec2 d = sampleConcentricDisk(rng.nextFloat(), rng.nextFloat());
                p = transformPoint(l.xform, Vec3(d.x * l.radius, d.y * l.radius, 0.0f));
                n = areaLightNormal(l);
                area = diskLightArea(l);
            } else {  // sphere
                const Vec3 center = lightOrigin(l);
                const float radius = srMax(1e-5f, sphereLightRadius(l));
                n = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
                p = center + n * radius;
                area = 4.0f * kPi * radius * radius;
            }
            if (area <= 1e-12f) return;
            Vec3 nEmit = n;
            if (l.twoSided && rng.nextFloat() < 0.5f) nEmit = -nEmit;
            const Frame frame(nEmit);
            const Vec3 local = sampleCosineHemisphere(rng.nextFloat(), rng.nextFloat());
            dir = normalize(frame.toWorld(local));
            pdfDir = fabsf(dot(nEmit, dir)) * kInvPi * (l.twoSided ? 0.5f : 1.0f);
            if (pdfDir <= 0.0f) return;
            betaS *= fabsf(dot(n, dir)) / srMax(1e-12f, selectPdf * (1.0f / area) * pdfDir);
        }

        bool causticChain = false;
        Vec3 o = offsetRayOrigin(p, n, dir);
        Vec3 d = dir;

        for (int depth = 0; depth < maxDepth; ++depth) {
            RayHit hit;
            if (!tracer.intersect(o, d, kFloatMax, hit)) break;
            SurfaceInteraction si;
            if (!buildSurfaceInteraction(scene, hit, o, d, si)) break;
            if (si.lightIndex >= 0) break;

            Material mat = materialForCausticTransport(scene, si.materialIndex);
            mat = evaluateTexturedMaterial(scene, mat, si.uv, si.ns, si.pObject, si.nObject,
                                           si.uvFilterWidth, si.pRef, si.nRef, si.hasPref);
            // Abbe is often authored only on the camera port — still bend Snell.
            if (mat.dispersionAbbe <= 1e-3f) {
                Material cam = materialForRay(scene, si.materialIndex, RayShadeKind::Camera);
                cam = evaluateTexturedMaterial(scene, cam, si.uv, si.ns, si.pObject, si.nObject,
                                               si.uvFilterWidth, si.pRef, si.nRef, si.hasPref);
                if (cam.dispersionAbbe > 1e-3f) {
                    mat.dispersionAbbe = cam.dispersionAbbe;
                    if (mat.ior < 1.01f) mat.ior = cam.ior;
                }
            }

            if (mat.opacity <= 1e-6f || (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity)) {
                o = offsetRayOrigin(si.p, si.ng, d);
                continue;
            }

            const float baseIor = mat.ior;
            const LobeWeights lw = computeLobes(mat, Frame(si.ns).toLocal(-d));
            const bool causticLink =
                isPhotonCausticCasterLobe(lw) && materialContributesCaustics(mat);
            const bool diffuseHit = lw.diffuse > 1e-4f || (!lw.delta && !causticLink);

            if (causticLink) {
                causticChain = true;
                const Frame frame(si.ns);
                const Vec3 woLocal = frame.toLocal(-d);
                terminateSecondaryIfSpectralEta(mat, waves);
                const BsdfSampleSpectral ss =
                    bsdfSampleSpectral(mat, woLocal, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                       rng.nextFloat(), waves, baseIor, heroIdx, filmCs);
                if (!ss.valid || ss.pdf <= 0.0f) break;
                const Vec3 wiWorld = normalize(frame.toWorld(ss.wi));
                if (!shadingNormalConsistent(si.ng, si.ns, -d, wiWorld)) break;
                betaS *= ss.weight;
                if (spectrumMaxComponent(betaS) < 1e-20f) break;
                BsdfSample rgbBs;
                rgbBs.wi = ss.wi;
                rgbBs.pdf = ss.pdf;
                rgbBs.specular = ss.specular;
                rgbBs.transmitted = ss.transmitted;
                if (shouldTerminateSecondaryWavelengths(rgbBs, lw, mat) && !waves.secondaryTerminated())
                    waves.terminateSecondary();
                if (depth >= 2) {
                    const float q = clampf(spectrumMaxComponent(betaS), 0.05f, 1.0f);
                    if (rng.nextFloat() > q) break;
                    betaS *= 1.0f / q;
                }
                o = offsetRayOrigin(si.p, si.ng, wiWorld);
                d = wiWorld;
                continue;
            }

            if (diffuseHit && causticChain) {
                CausticPhoton ph;
                ph.p = si.p;
                ph.wi = -d;
                ph.n = si.ns;
                ph.lambdaNm = waves.lambda[0];
                ph.lambdaPdf = waves.pdf[0];
                ph.power = spectrumToRgb(betaS, waves, filmCs);
                if (isFinite(ph.power) && !isBlack(ph.power) && ph.lambdaPdf > 0.0f)
                    photons_.push_back(ph);
                break;
            }
            break;
        }
    }
};

// True when any material can cast refractive caustics that are too rough for
// delta MNEE (α > kDeltaAlpha). Used by MNEE+Photon (Auto) caustics routing.
SR_INL bool sceneHasRoughCausticCaster(const SceneView& scene) {
    for (int i = 0; i < scene.materialCount; ++i) {
        const Material& m = scene.materials[i];
        if (!materialContributesCaustics(m)) continue;
        if (m.transmission <= 0.25f) continue;
        const LobeWeights lw = computeLobes(m);
        // Rough refractive: photon map, not MNEE.
        if (isPhotonCausticCasterLobe(lw) && !lw.delta) return true;
    }
    return false;
}

// True when the active caustics engine should use the photon map.
// MNEE+Photon (Auto): Photon when the scene has rough refractive casters; else MNEE.
SR_INL bool causticsUsePhotonMap(const RenderSettingsData& s, const SceneView* scene = nullptr) {
    if (s.caustics == 0) return false;
    if (s.causticsEngine == kCausticsEnginePbrt) return false;
    if (s.causticsEngine == kCausticsEngineAimedLt || s.causticsEngine == kCausticsEngineAimedLtMnee)
        return false;
    if (s.causticsEngine == kCausticsEnginePhoton) return true;
    if (s.causticsEngine == kCausticsEngineAuto && scene && sceneHasRoughCausticCaster(*scene))
        return true;
    return false;
}

// True when MNEE should run (explicit MNEE, or MNEE+Photon on delta-only glass).
// pbrt (book) caustics never use MNEE.
SR_INL bool causticsUseMnee(const RenderSettingsData& s, const SceneView* scene = nullptr) {
    if (s.caustics == 0) return false;
    if (s.causticsEngine == kCausticsEnginePbrt) return false;
    if (s.causticsEngine == kCausticsEngineAimedLt) return false;
    if (s.causticsEngine == kCausticsEnginePhoton) return false;
    if (s.causticsEngine == kCausticsEngineMnee) return true;
    if (s.causticsEngine == kCausticsEngineAimedLtMnee) return true;
    if (s.causticsEngine == kCausticsEngineAuto) {
        // MNEE+Photon + rough glass → photons; + delta-only → MNEE.
        return !causticsUsePhotonMap(s, scene);
    }
    return false;
}

}  // namespace sol
