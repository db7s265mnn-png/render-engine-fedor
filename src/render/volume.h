// Homogeneous / null-scattering volume helpers (PBRT 4ed Ch.11).
// Media are authored on geometry (InstanceData::mediumIndex).
// Heterogeneous VDB fog free-flight + ratio-tracking shadow Tr: see volume_vdb.h.
#pragma once

#include "core/math.h"
#include "core/rng.h"
#include "scene/types.h"

namespace sol {

// Henyey–Greenstein phase function value (per steradian).
SR_INL SR_HD float henyeyGreenstein(float cosTheta, float g) {
    const float g2 = g * g;
    const float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    return kInv4Pi * (1.0f - g2) / srMax(1e-6f, denom * sqrtf(srMax(1e-6f, denom)));
}

SR_INL SR_HD float henyeyGreensteinPdf(float cosTheta, float g) {
    return henyeyGreenstein(cosTheta, g);
}

// Sample HG; returns wi given wo (both unit).
SR_INL SR_HD Vec3 sampleHenyeyGreenstein(Vec3 wo, float g, float u1, float u2, float& pdf) {
    float cosTheta = 0.0f;
    if (fabsf(g) < 1e-3f) {
        cosTheta = 1.0f - 2.0f * u1;
    } else {
        const float g2 = g * g;
        const float sqrTerm = (1.0f - g2) / (1.0f - g + 2.0f * g * u1);
        cosTheta = (1.0f + g2 - sqrTerm * sqrTerm) / (2.0f * g);
        cosTheta = clampf(cosTheta, -1.0f, 1.0f);
    }
    const float sinTheta = sqrtf(srMax(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = kTwoPi * u2;
    const Frame frame(wo);
    const Vec3 wi = frame.toWorld(Vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta));
    pdf = henyeyGreensteinPdf(cosTheta, g);
    return wi;
}

// Disney / Hyperion similarity (opt-in, biased): lerp g → 0 between volume
// scatters 5 and 20, keeping σs(1−g) so the mean free path grows. Low-order
// bounces stay anisotropic. scatterIndex is 0-based (0 = first scatter).
SR_INL SR_HD MediumData mediumWithVolumeSimilarity(const MediumData& m, int scatterIndex) {
    MediumData out = m;
    const float g0 = clampf(m.g, -0.999f, 0.999f);
    float t = 0.0f;
    if (scatterIndex >= 20) t = 1.0f;
    else if (scatterIndex > 5) t = float(scatterIndex - 5) / 15.0f;
    const float gStar = g0 * (1.0f - t);
    const float denom = srMax(1e-3f, 1.0f - gStar);
    const float scale = (1.0f - g0) / denom;
    out.g = gStar;
    out.sigmaS = m.sigmaS * srMax(0.0f, scale);
    return out;
}

SR_INL SR_HD Vec3 mediumSigmaA(const MediumData& m) {
    return m.sigmaA * srMax(0.0f, m.density);
}

SR_INL SR_HD Vec3 mediumSigmaS(const MediumData& m) {
    return m.sigmaS * srMax(0.0f, m.density);
}

SR_INL SR_HD Vec3 mediumSigmaT(const MediumData& m) {
    return mediumSigmaA(m) + mediumSigmaS(m);
}

// Majorant for null-scattering (max RGB channel of σt).
SR_INL SR_HD float mediumMajorant(const MediumData& m) {
    const Vec3 st = mediumSigmaT(m);
    return srMax(st.x, srMax(st.y, st.z));
}

SR_INL SR_HD bool mediumIsActive(const SceneView& scene, int mediumIndex) {
    if (mediumIndex < 0 || mediumIndex >= scene.mediumCount || !scene.media) return false;
    const MediumData& m = scene.media[mediumIndex];
    // type 3 = SDF surface (not a participating medium walk)
    if (m.type == 0 || m.type == 3) return false;
    if (m.type == 2) return true;  // fog VDB — majorant from grid at sample time
    return mediumMajorant(m) > 1e-8f;
}

// Beer–Lambert transmittance over distance t (homogeneous).
SR_INL SR_HD Vec3 mediumTr(const MediumData& m, float t) {
    const Vec3 st = mediumSigmaT(m);
    return Vec3(expf(-st.x * t), expf(-st.y * t), expf(-st.z * t));
}

// Delta-tracking / null-collision free-flight sample for a homogeneous medium.
// Returns true if a real scatter event occurs before tMax; then `t` is the
// distance and `throughput` is multiplied by σs/σt (ratio tracking).
// On a miss (reach surface), multiplies throughput by the analytic Tr and
// returns false with t = tMax.
struct MediumSample {
    float t = 0.0f;
    bool scattered = false;
    bool absorbed = false;
};

SR_INL SR_HD int rgbHeroChannel(Vec3 v) {
    int h = 0;
    if (v.y > v.x) h = 1;
    if (v.z > (h == 0 ? v.x : v.y)) h = 2;
    return h;
}

SR_INL SR_HD float rgbChannel(Vec3 v, int c) { return c == 0 ? v.x : (c == 1 ? v.y : v.z); }

// pbrt-v4 SimpleVolPath: discrete absorb / scatter / null on the hero channel,
// with a relative spectral weight so the other RGB channels stay unbiased.
SR_INL SR_HD int sampleHeroCollision(Vec3 sigmaA, Vec3 sigmaS, float majorant, float u, int hero,
                                     Vec3& weight) {
    const Vec3 sigmaT = sigmaA + sigmaS;
    const Vec3 sigmaN = Vec3(majorant) - sigmaT;
    const float pA = rgbChannel(sigmaA, hero) / srMax(majorant, 1e-12f);
    const float pS = rgbChannel(sigmaS, hero) / srMax(majorant, 1e-12f);
    if (u < pA) {
        weight = Vec3(0.0f);
        return 0;  // absorb
    }
    if (u < pA + pS) {
        weight = sigmaS / srMax(rgbChannel(sigmaS, hero), 1e-12f);
        return 1;  // scatter
    }
    weight = sigmaN / srMax(rgbChannel(sigmaN, hero), 1e-12f);
    return 2;  // null
}

SR_INL SR_HD MediumSample sampleMediumHomogeneous(const MediumData& m, float tMax, Rng& rng,
                                                  Vec3& throughput) {
    MediumSample out;
    const float majorant = mediumMajorant(m);
    if (majorant <= 1e-8f || tMax <= 0.0f) {
        out.t = tMax;
        return out;
    }
    const Vec3 sigmaA = mediumSigmaA(m);
    const Vec3 sigmaS = mediumSigmaS(m);
    const Vec3 sigmaT = sigmaA + sigmaS;
    const int hero = rgbHeroChannel(sigmaT);
    const float stHero = rgbChannel(sigmaT, hero);
    float t = 0.0f;
    bool anyEvent = false;
    constexpr int kNullCollisionMaxIters = 1 << 20;
    for (int iter = 0; iter < kNullCollisionMaxIters; ++iter) {
        const float u = srMax(1e-6f, 1.0f - rng.nextFloat());
        t += -logf(u) / majorant;
        if (t >= tMax) {
            // No remaining majorant events: relative Beer vs the hero channel.
            if (!anyEvent) {
                throughput = throughput * Vec3(expf(-(sigmaT.x - stHero) * tMax),
                                               expf(-(sigmaT.y - stHero) * tMax),
                                               expf(-(sigmaT.z - stHero) * tMax));
            }
            out.t = tMax;
            return out;
        }
        anyEvent = true;
        Vec3 w(1.0f);
        const int mode = sampleHeroCollision(sigmaA, sigmaS, majorant, rng.nextFloat(), hero, w);
        if (mode == 0) {
            throughput = Vec3(0.0f);
            out.t = t;
            out.absorbed = true;
            return out;
        }
        if (mode == 1) {
            throughput = throughput * w;
            out.t = t;
            out.scattered = true;
            return out;
        }
        throughput = throughput * w;
    }
    out.t = tMax;
    return out;
}

// Shadow-ray transmittance through a homogeneous medium of length dist.
SR_INL SR_HD Vec3 mediumShadowTr(const MediumData& m, float dist) {
    if (dist <= 0.0f) return Vec3(1.0f);
    return mediumTr(m, dist);
}

// Volume-path Russian roulette survival probability.
// Conservative fog (σs/σt ≈ 1) keeps luminance from dropping. A 0.95 cap would
// divide throughput by 0.95 on every bounce — unbiased in expectation, but the
// survivor weight explodes at high maxDepth and lights up NEE fireflies.
// Floor 0.05 still kills near-dead paths.
SR_INL SR_HD float volumeRussianRouletteQ(Vec3 throughput) {
    return clampf(luminance(throughput), 0.05f, 1.0f);
}

// pbrt-v4 VolPath: one RR on path throughput only. Always take volume NEE.
SR_INL SR_HD float volumeNeeRouletteP(int depth) {
    (void)depth;
    return 1.0f;
}

// Resolve medium for an instance (type 2 VDB falls back to homogeneous coeffs
// until a density grid is wired — same σa/σs/density still apply as a base).
SR_INL SR_HD const MediumData* getMedium(const SceneView& scene, int mediumIndex) {
    if (!mediumIsActive(scene, mediumIndex)) return nullptr;
    return &scene.media[mediumIndex];
}

}  // namespace sol
