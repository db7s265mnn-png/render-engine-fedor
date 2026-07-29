// Caustic-only photon map (VCM-style density estimation on diffuse after a
// specular chain). Used when RenderSettingsData::causticsEngine == Photon.
// CPU / Embree only — included from embree_device.cpp and the integrators.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "core/rng.h"
#include "render/integrator.h"
#include "render/lights.h"
#include "render/shading.h"
#include "scene/types.h"

namespace sol {

struct CausticPhoton {
    Vec3 p{0.0f};
    Vec3 wi{0.0f};   // incident direction at the deposit (toward the surface)
    Vec3 power{0.0f};
    Vec3 n{0.0f};
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
        buildGrid_();
    }

    // Estimate reflected radiance at a diffuse-ish shading point (W/m²/sr).
    Vec3 gather(Vec3 p, Vec3 n, Vec3 wo, const Material& mat, float radius) const {
        if (photons_.empty() || radius <= 1e-8f) return Vec3(0.0f);
        const LobeWeights lw = computeLobes(mat);
        if (lw.diffuse < 1e-4f && !lw.delta) {
            // Still allow rough dielectrics with a diffuse leftover; pure delta skips.
        }
        if (lw.delta && lw.diffuse < 1e-4f) return Vec3(0.0f);

        const float r2 = radius * radius;
        const float area = kPi * r2;
        if (area <= 1e-20f) return Vec3(0.0f);

        Vec3 sum(0.0f);
        const Vec3 cellF = (p - origin_) * invCellSize_;
        const int cx = int(floorf(cellF.x));
        const int cy = int(floorf(cellF.y));
        const int cz = int(floorf(cellF.z));
        // Visit the 3×3×3 neighbourhood — cell size ≈ radius.
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const uint32_t h = hashCell_(cx + dx, cy + dy, cz + dz);
                    const uint32_t start = cellStarts_[h];
                    const uint32_t end = cellStarts_[h + 1];
                    for (uint32_t i = start; i < end; ++i) {
                        const CausticPhoton& ph = photons_[cells_[i]];
                        const Vec3 d = ph.p - p;
                        if (dot(d, d) > r2) continue;
                        if (dot(ph.n, n) <= 0.1f) continue;  // opposite-facing deposit
                        const float cosI = fabsf(dot(n, ph.wi));
                        if (cosI <= 1e-6f) continue;
                        // Lambertian-style BRDF * flux (caustic photons land on diffuse).
                        const Frame fr(n);
                        const BsdfEval be = bsdfEvalLocal(mat, fr.toLocal(wo), fr.toLocal(ph.wi));
                        if (be.pdf <= 0.0f || isBlack(be.f)) continue;
                        sum += be.f * ph.power;
                    }
                }
        return sum * (1.0f / area);
    }

private:
    std::vector<CausticPhoton> photons_;
    std::vector<uint32_t> cells_;
    std::vector<uint32_t> cellStarts_;
    float invCellSize_ = 1.0f;
    Vec3 origin_{0.0f};

    static uint32_t hashCell_(int x, int y, int z) {
        uint32_t h = uint32_t(x) * 73856093u ^ uint32_t(y) * 19349663u ^ uint32_t(z) * 83492791u;
        h ^= h >> 16;
        h *= 0x7feb352du;
        h ^= h >> 15;
        return h & 4095u;  // 4096 slots
    }

    void buildGrid_() {
        Bounds3 b;
        for (const CausticPhoton& ph : photons_) b.extend(ph.p);
        const Vec3 ext = b.extent();
        const float maxExt = srMax(ext.x, srMax(ext.y, ext.z));
        float cell = srMax(1e-3f, maxExt / 64.0f);
        // Prefer cell ≈ mean photon radius hint from settings is unknown here —
        // use a fraction of the scene extent; gather still uses its own radius.
        invCellSize_ = 1.0f / cell;
        origin_ = b.lo - Vec3(cell);

        constexpr int kSlots = 4096;
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

        Vec3 p, n, beta;
        Vec3 dir;
        float pdfDir = 0.0f;

        if (l.type == kLightPoint) {
            p = lightOrigin(l);
            n = Vec3(0.0f, 1.0f, 0.0f);
            dir = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
            pdfDir = kInv4Pi;
            beta = l.emittedRadiance() / srMax(1e-12f, selectPdf * pdfDir);
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
            beta = lightRadiance(l) * fabsf(dot(n, dir)) / srMax(1e-12f, selectPdf * (1.0f / area) * pdfDir);
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

            Material mat = si.materialIndex >= 0 && si.materialIndex < scene.materialCount
                               ? scene.materials[si.materialIndex]
                               : defaultMaterial();
            mat = evaluateTexturedMaterial(scene, mat, si.uv, si.ns, si.pObject, si.nObject, si.uvFilterWidth);

            if (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity) {
                o = offsetRayOrigin(si.p, si.ng, d);
                continue;
            }

            const LobeWeights lw = computeLobes(mat);
            const bool nearSpec = (lw.delta || isNearSpecularLobe(lw)) && materialContributesCaustics(mat);
            const bool diffuseHit = lw.diffuse > 1e-4f || (!lw.delta && !nearSpec);

            if (nearSpec) {
                causticChain = true;
                const Frame frame(si.ns);
                const Vec3 woLocal = frame.toLocal(-d);
                const BsdfSample bs =
                    bsdfSampleLocal(mat, woLocal, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
                if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;
                const Vec3 wiWorld = normalize(frame.toWorld(bs.wi));
                if (!shadingNormalConsistent(si.ng, si.ns, -d, wiWorld)) break;
                beta = beta * bs.weight;
                if (!isFinite(beta) || isBlack(beta)) break;
                // Russian roulette after a few specular bounces.
                if (depth >= 2) {
                    const float q = saturatef(1.0f - average(beta));
                    if (rng.nextFloat() < q) break;
                    beta = beta / srMax(1e-4f, 1.0f - q);
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
                ph.power = beta;
                if (isFinite(ph.power) && !isBlack(ph.power)) photons_.push_back(ph);
                break;
            }
            break;
        }
    }
};

// True when the active caustics engine is the photon map.
SR_INL bool causticsUsePhotonMap(const RenderSettingsData& s) {
    return s.caustics != 0 && s.causticsEngine == kCausticsEnginePhoton;
}

// True when MNEE should run (PT auto/mnee, and not photon engine).
SR_INL bool causticsUseMnee(const RenderSettingsData& s) {
    return s.caustics != 0 && s.causticsEngine != kCausticsEnginePhoton &&
           (s.causticsEngine == kCausticsEngineAuto || s.causticsEngine == kCausticsEngineMnee);
}

}  // namespace sol
