// Light sampling shared by the Embree and OptiX integrators.
//
// Conventions (matching Houdini/USD): rect and disk lights live in the XY plane
// of their transform and emit along -Z, distant lights travel along -Z, sphere
// lights are centred on the transform origin, dome lights use an
// equirectangular map with +Y up.
#pragma once

#include "core/math.h"
#include "scene/types.h"

namespace sol {

struct LightSample {
    Vec3 wi{0.0f, 1.0f, 0.0f};       // direction from the shading point to the light
    Vec3 radiance{0.0f, 0.0f, 0.0f}; // incident radiance
    float distance = kFloatMax;
    float pdf = 0.0f;                // solid angle pdf of this light (light choice excluded)
    bool delta = false;
};

SR_INL SR_HD bool lightContributesCaustics(const LightData& l) { return l.contributeCaustics != 0; }

SR_INL SR_HD bool materialContributesCaustics(const Material& m) { return m.contributeCaustics != 0; }

// ---------------------------------------------------------------------------
// Environment map helpers
// ---------------------------------------------------------------------------
SR_INL SR_HD Vec3 envTexel(const EnvMapView& env, int x, int y) {
    x = x < 0 ? 0 : (x >= env.width ? env.width - 1 : x);
    y = y < 0 ? 0 : (y >= env.height ? env.height - 1 : y);
    const float* p = env.pixels + (size_t(y) * size_t(env.width) + size_t(x)) * 4;
    return Vec3(p[0], p[1], p[2]);
}

SR_INL SR_HD void envDirectionToTexel(const EnvMapView& env, Vec3 dirLocal, int& x, int& y) {
    const Vec2 uv = directionToEquirect(normalize(dirLocal));
    x = int(uv.x * float(env.width));
    y = int(uv.y * float(env.height));
    x = x < 0 ? 0 : (x >= env.width ? env.width - 1 : x);
    y = y < 0 ? 0 : (y >= env.height ? env.height - 1 : y);
}

SR_INL SR_HD Vec3 envLookup(const EnvMapView& env, Vec3 dirLocal) {
    if (!env.valid()) return Vec3(0.0f);
    const Vec2 uv = directionToEquirect(normalize(dirLocal));
    float u = uv.x - floorf(uv.x);
    const float v = clampf(uv.y, 0.0f, 1.0f);
    const float fx = u * float(env.width) - 0.5f;
    const float fy = v * float(env.height) - 0.5f;
    const int x0 = int(floorf(fx));
    const int y0 = int(floorf(fy));
    const float tx = fx - float(x0);
    const float ty = fy - float(y0);
    const int xa = ((x0 % env.width) + env.width) % env.width;
    const int xb = ((x0 + 1) % env.width + env.width) % env.width;
    const Vec3 c00 = envTexel(env, xa, y0);
    const Vec3 c10 = envTexel(env, xb, y0);
    const Vec3 c01 = envTexel(env, xa, y0 + 1);
    const Vec3 c11 = envTexel(env, xb, y0 + 1);
    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

// Discrete texel that owns `envPdf` — NEE must use this, not bilinear. Mixing a
// dark sampled texel with a neighbouring sun texel is the classic HDRI firefly.
SR_INL SR_HD Vec3 envLookupNearest(const EnvMapView& env, Vec3 dirLocal) {
    if (!env.valid()) return Vec3(0.0f);
    int x = 0, y = 0;
    envDirectionToTexel(env, dirLocal, x, y);
    return envTexel(env, x, y);
}

SR_INL SR_HD int cdfFindInterval(const float* cdf, int size, float u) {
    int first = 0;
    int len = size;
    while (len > 0) {
        const int half = len >> 1;
        const int middle = first + half;
        if (cdf[middle] <= u) {
            first = middle + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    int r = first - 1;
    if (r < 0) r = 0;
    if (r > size - 2) r = size - 2;
    return r;
}

// pdf with respect to solid angle for a direction in dome-local space.
SR_INL SR_HD float envPdf(const EnvMapView& env, Vec3 dirLocal) {
    if (!env.sampled()) return kInv4Pi;
    const Vec2 uv = directionToEquirect(normalize(dirLocal));
    const float sinTheta = sinf(clampf(uv.y, 0.0f, 1.0f) * kPi);
    if (sinTheta <= 0.0f) return 0.0f;
    int x = 0, y = 0;
    envDirectionToTexel(env, dirLocal, x, y);
    const float funcValue = env.func[size_t(y) * size_t(env.width) + size_t(x)];
    const float pdfUv = funcValue / env.integral;
    return pdfUv / (kTwoPi * kPi * sinTheta);
}

// Importance sample the environment; returns a direction in dome-local space.
SR_INL SR_HD Vec3 envSample(const EnvMapView& env, float u1, float u2, float& pdf) {
    if (!env.sampled()) {
        pdf = kInv4Pi;
        return sampleUniformSphere(u1, u2);
    }
    const int y = cdfFindInterval(env.margCdf, env.height + 1, u2);
    const float dyDen = env.margCdf[y + 1] - env.margCdf[y];
    const float dy = dyDen > 0.0f ? (u2 - env.margCdf[y]) / dyDen : 0.0f;

    const float* cdf = env.condCdf + size_t(y) * size_t(env.width + 1);
    const int x = cdfFindInterval(cdf, env.width + 1, u1);
    const float dxDen = cdf[x + 1] - cdf[x];
    const float dx = dxDen > 0.0f ? (u1 - cdf[x]) / dxDen : 0.0f;

    const float u = (float(x) + dx) / float(env.width);
    const float v = (float(y) + dy) / float(env.height);
    const float theta = v * kPi;
    const float sinTheta = sinf(theta);
    if (sinTheta <= 0.0f) {
        pdf = 0.0f;
        return Vec3(0.0f, 1.0f, 0.0f);
    }
    const float funcValue = env.func[size_t(y) * size_t(env.width) + size_t(x)];
    pdf = (funcValue / env.integral) / (kTwoPi * kPi * sinTheta);
    return equirectToDirection(u, v);
}

// ---------------------------------------------------------------------------
// Light geometry helpers
// ---------------------------------------------------------------------------
SR_INL SR_HD Vec3 lightAxisX(const LightData& l) { return transformVector(l.xform, Vec3(1.0f, 0.0f, 0.0f)); }
SR_INL SR_HD Vec3 lightAxisY(const LightData& l) { return transformVector(l.xform, Vec3(0.0f, 1.0f, 0.0f)); }
SR_INL SR_HD Vec3 lightAxisZ(const LightData& l) { return transformVector(l.xform, Vec3(0.0f, 0.0f, 1.0f)); }
SR_INL SR_HD Vec3 lightOrigin(const LightData& l) { return transformPoint(l.xform, Vec3(0.0f, 0.0f, 0.0f)); }

SR_INL SR_HD float rectLightArea(const LightData& l) {
    return length(cross(lightAxisX(l) * l.width, lightAxisY(l) * l.height));
}

SR_INL SR_HD float diskLightArea(const LightData& l) {
    return kPi * length(cross(lightAxisX(l) * l.radius, lightAxisY(l) * l.radius));
}

SR_INL SR_HD float sphereLightRadius(const LightData& l) {
    const float sx = length(lightAxisX(l));
    const float sy = length(lightAxisY(l));
    const float sz = length(lightAxisZ(l));
    return l.radius * (sx + sy + sz) * (1.0f / 3.0f);
}

// Radiance emitted by a light, before any geometric term.
SR_INL SR_HD Vec3 lightRadiance(const LightData& l) {
    Vec3 e = l.emittedRadiance();
    if (!l.normalize) return e;
    switch (l.type) {
        case kLightRect: {
            const float area = rectLightArea(l);
            return area > 0.0f ? e / area : e;
        }
        case kLightDisk: {
            const float area = diskLightArea(l);
            return area > 0.0f ? e / area : e;
        }
        case kLightSphere: {
            const float r = sphereLightRadius(l);
            const float area = 4.0f * kPi * r * r;
            return area > 0.0f ? e / area : e;
        }
        case kLightDistant: {
            const float halfAngle = radians(srMax(0.0f, l.angle)) * 0.5f;
            const float solidAngle = kTwoPi * (1.0f - cosf(halfAngle));
            return solidAngle > 1e-9f ? e / solidAngle : e;
        }
        default: return e;
    }
}

// Radiance leaving an area light towards -wi (wi points from the surface to the
// light). Returns black when the light does not emit in that direction.
SR_INL SR_HD Vec3 areaLightEmission(const SceneView& scene, const LightData& l, Vec3 wi, Vec3 lightNormal) {
    if (l.type == kLightSphere) return lightRadiance(l);
    const float cosTheta = dot(lightNormal, -wi);
    if (cosTheta <= 0.0f && !l.twoSided) return Vec3(0.0f);
    (void)scene;
    return lightRadiance(l);
}

// Emitted normal of a rect/disk light in world space (-Z of the transform).
SR_INL SR_HD Vec3 areaLightNormal(const LightData& l) {
    return normalize(-lightAxisZ(l));
}

// Radiance of the dome light for a world space direction.
SR_INL SR_HD Vec3 domeRadiance(const SceneView& scene, const LightData& l, Vec3 dirWorld) {
    Vec3 tint = l.emittedRadiance();
    if (l.envIndex >= 0 && l.envIndex < scene.envMapCount) {
        const EnvMapView& env = scene.envMaps[l.envIndex];
        if (env.valid()) {
            const Vec3 dirLocal = normalize(transformVector(l.xformInv, dirWorld));
            return tint * envLookup(env, dirLocal);
        }
    }
    return tint;
}

// Radiance seen by a ray that left the scene.
SR_INL SR_HD Vec3 environmentRadiance(const SceneView& scene, Vec3 dirWorld) {
    if (scene.domeLightIndex < 0) return Vec3(0.0f);
    return domeRadiance(scene, scene.lights[scene.domeLightIndex], dirWorld);
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------
SR_INL SR_HD bool sampleLight(const SceneView& scene, int lightIndex, Vec3 refP, float u1, float u2,
                              LightSample& out) {
    if (lightIndex < 0 || lightIndex >= scene.lightCount) return false;
    const LightData& l = scene.lights[lightIndex];

    switch (l.type) {
        case kLightDistant: {
            const Vec3 axis = normalize(lightAxisZ(l));
            const float halfAngle = radians(srMax(0.0f, l.angle)) * 0.5f;
            out.distance = kFloatMax;
            if (halfAngle < 1e-4f) {
                out.wi = axis;
                out.pdf = 1.0f;
                out.delta = true;
                out.radiance = l.emittedRadiance();
                return true;
            }
            const float cosThetaMax = cosf(halfAngle);
            const Frame frame(axis);
            const Vec3 local = sampleUniformCone(u1, u2, cosThetaMax);
            out.wi = normalize(frame.toWorld(local));
            out.pdf = 1.0f / (kTwoPi * (1.0f - cosThetaMax));
            out.delta = false;
            out.radiance = lightRadiance(l);
            return true;
        }
        case kLightPoint: {
            const Vec3 p = lightOrigin(l);
            const Vec3 d = p - refP;
            const float dist2 = lengthSquared(d);
            if (dist2 <= 1e-12f) return false;
            const float dist = sqrtf(dist2);
            out.wi = d / dist;
            out.distance = dist;
            out.pdf = 1.0f;
            out.delta = true;
            out.radiance = l.emittedRadiance() / dist2;
            return true;
        }
        case kLightRect:
        case kLightDisk: {
            Vec3 pLocal;
            if (l.type == kLightRect) {
                pLocal = Vec3((u1 - 0.5f) * l.width, (u2 - 0.5f) * l.height, 0.0f);
            } else {
                const Vec2 d = sampleConcentricDisk(u1, u2);
                pLocal = Vec3(d.x * l.radius, d.y * l.radius, 0.0f);
            }
            const Vec3 p = transformPoint(l.xform, pLocal);
            const Vec3 toLight = p - refP;
            const float dist2 = lengthSquared(toLight);
            if (dist2 <= 1e-12f) return false;
            const float dist = sqrtf(dist2);
            const Vec3 wi = toLight / dist;
            Vec3 n = areaLightNormal(l);
            float cosLight = dot(n, -wi);
            if (cosLight <= 0.0f) {
                if (!l.twoSided) return false;
                cosLight = -cosLight;
            }
            if (cosLight <= 1e-6f) return false;
            const float area = l.type == kLightRect ? rectLightArea(l) : diskLightArea(l);
            if (area <= 0.0f) return false;
            out.wi = wi;
            out.distance = dist;
            out.pdf = dist2 / (cosLight * area);
            out.delta = false;
            out.radiance = lightRadiance(l);
            return true;
        }
        case kLightSphere: {
            const Vec3 center = lightOrigin(l);
            const float radius = srMax(1e-5f, sphereLightRadius(l));
            const Vec3 toCenter = center - refP;
            const float dist2 = lengthSquared(toCenter);
            const float dist = sqrtf(dist2);
            const Vec3 radiance = lightRadiance(l);
            if (dist <= radius * 1.0001f) {
                // Inside the light: sample the full sphere surface.
                const Vec3 dir = sampleUniformSphere(u1, u2);
                const Vec3 p = center + dir * radius;
                const Vec3 toLight = p - refP;
                const float d2 = lengthSquared(toLight);
                if (d2 <= 1e-12f) return false;
                const float d = sqrtf(d2);
                const Vec3 wi = toLight / d;
                const float cosLight = absDot(dir, -wi);
                if (cosLight <= 1e-6f) return false;
                const float area = 4.0f * kPi * radius * radius;
                out.wi = wi;
                out.distance = d;
                out.pdf = d2 / (cosLight * area);
                out.delta = false;
                out.radiance = radiance;
                return true;
            }
            const float sinThetaMax2 = (radius * radius) / dist2;
            const float cosThetaMax = sqrtf(srMax(0.0f, 1.0f - sinThetaMax2));
            const Frame frame(toCenter / dist);
            const Vec3 wi = normalize(frame.toWorld(sampleUniformCone(u1, u2, cosThetaMax)));
            // Distance to the sphere along wi.
            const float b = dot(wi, toCenter);
            const float disc = b * b - (dist2 - radius * radius);
            const float t = disc > 0.0f ? b - sqrtf(disc) : b;
            out.wi = wi;
            out.distance = srMax(1e-4f, t);
            out.pdf = 1.0f / (kTwoPi * (1.0f - cosThetaMax));
            out.delta = false;
            out.radiance = radiance;
            return true;
        }
        case kLightDome: {
            float pdf = 0.0f;
            Vec3 dirLocal;
            if (l.envIndex >= 0 && l.envIndex < scene.envMapCount && scene.envMaps[l.envIndex].sampled()) {
                dirLocal = envSample(scene.envMaps[l.envIndex], u1, u2, pdf);
            } else {
                dirLocal = sampleUniformSphere(u1, u2);
                pdf = kInv4Pi;
            }
            if (pdf <= 0.0f) return false;
            const Vec3 dirWorld = normalize(transformVector(l.xform, dirLocal));
            out.wi = dirWorld;
            out.distance = kFloatMax;
            out.pdf = pdf;
            out.delta = false;
            // Camera rays keep bilinear domeRadiance; NEE uses the discrete PDF texel.
            Vec3 tint = l.emittedRadiance();
            if (l.envIndex >= 0 && l.envIndex < scene.envMapCount &&
                scene.envMaps[l.envIndex].valid()) {
                out.radiance = tint * envLookupNearest(scene.envMaps[l.envIndex], dirLocal);
            } else {
                out.radiance = tint;
            }
            return true;
        }
        default: return false;
    }
}

// Solid angle pdf of sampling `wi` on the given light, used for MIS when a BSDF
// ray hits the light. `hitP`/`hitN` describe the hit on the light geometry.
SR_INL SR_HD float lightPdfDirection(const SceneView& scene, int lightIndex, Vec3 refP, Vec3 wi, Vec3 hitP,
                                     Vec3 hitN) {
    if (lightIndex < 0 || lightIndex >= scene.lightCount) return 0.0f;
    const LightData& l = scene.lights[lightIndex];
    switch (l.type) {
        case kLightRect:
        case kLightDisk: {
            const float area = l.type == kLightRect ? rectLightArea(l) : diskLightArea(l);
            if (area <= 0.0f) return 0.0f;
            const float dist2 = lengthSquared(hitP - refP);
            float cosLight = absDot(hitN, wi);
            if (cosLight <= 1e-6f) return 0.0f;
            return dist2 / (cosLight * area);
        }
        case kLightSphere: {
            const Vec3 center = lightOrigin(l);
            const float radius = srMax(1e-5f, sphereLightRadius(l));
            const float dist2 = lengthSquared(center - refP);
            if (dist2 <= radius * radius * 1.0002f) {
                const float area = 4.0f * kPi * radius * radius;
                const float cosLight = absDot(hitN, wi);
                if (cosLight <= 1e-6f) return 0.0f;
                return lengthSquared(hitP - refP) / (cosLight * area);
            }
            const float sinThetaMax2 = (radius * radius) / dist2;
            const float cosThetaMax = sqrtf(srMax(0.0f, 1.0f - sinThetaMax2));
            const float denom = kTwoPi * (1.0f - cosThetaMax);
            return denom > 0.0f ? 1.0f / denom : 0.0f;
        }
        case kLightDistant: {
            const float halfAngle = radians(srMax(0.0f, l.angle)) * 0.5f;
            if (halfAngle < 1e-4f) return 0.0f;
            const float cosThetaMax = cosf(halfAngle);
            if (dot(normalize(lightAxisZ(l)), wi) < cosThetaMax) return 0.0f;
            return 1.0f / (kTwoPi * (1.0f - cosThetaMax));
        }
        case kLightDome: {
            if (l.envIndex >= 0 && l.envIndex < scene.envMapCount && scene.envMaps[l.envIndex].sampled()) {
                const Vec3 dirLocal = normalize(transformVector(l.xformInv, wi));
                return envPdf(scene.envMaps[l.envIndex], dirLocal);
            }
            return kInv4Pi;
        }
        default: return 0.0f;
    }
}

// Approximate emitted flux used to pick lights (brighter / larger → more samples).
SR_INL SR_HD float lightFluxWeight(const SceneView& scene, int lightIndex) {
    if (lightIndex < 0 || lightIndex >= scene.lightCount) return 0.0f;
    const LightData& l = scene.lights[lightIndex];
    const float intens = srMax(1e-8f, average(vmax(Vec3(0.0f), l.emittedRadiance())));
    switch (l.type) {
        case kLightDome:
            if (l.envIndex >= 0 && l.envIndex < scene.envMapCount && scene.envMaps[l.envIndex].sampled())
                return intens * srMax(1e-4f, scene.envMaps[l.envIndex].integral);
            return intens * 4.0f;
        case kLightRect:
            return intens * srMax(1e-6f, rectLightArea(l));
        case kLightDisk:
            return intens * srMax(1e-6f, diskLightArea(l));
        case kLightSphere: {
            const float r = sphereLightRadius(l);
            return intens * srMax(1e-6f, 4.0f * kPi * r * r);
        }
        case kLightDistant: {
            const float halfAngle = radians(srMax(0.0f, l.angle)) * 0.5f;
            if (halfAngle < 1e-4f) return intens;
            // normalize=1: intensity is irradiance (Karma / Physical Sky). Multiplying
            // by the disc solid angle underweights the sun vs a dome by ~1/ω (~15000×).
            if (l.normalize) return intens;
            return intens * srMax(1e-6f, kTwoPi * (1.0f - cosf(halfAngle)));
        }
        case kLightPoint:
            return intens;
        default:
            return intens;
    }
}

SR_INL SR_HD float lightSelectionPdf(const SceneView& scene) {
    return scene.lightCount > 0 ? 1.0f / float(scene.lightCount) : 0.0f;
}

// Flux-weighted probability of selecting a specific light.
SR_INL SR_HD float lightSelectionPdfIndex(const SceneView& scene, int lightIndex) {
    if (scene.lightCount <= 0 || lightIndex < 0 || lightIndex >= scene.lightCount) return 0.0f;
    float total = 0.0f;
    float chosen = 0.0f;
    for (int i = 0; i < scene.lightCount; ++i) {
        const float w = lightFluxWeight(scene, i);
        total += w;
        if (i == lightIndex) chosen = w;
    }
    if (total <= 1e-20f) return lightSelectionPdf(scene);
    return chosen / total;
}

// Sample a light index with probability ∝ flux. `pdf` is the selection pdf.
SR_INL SR_HD int sampleLightIndex(const SceneView& scene, float u, float& pdf) {
    if (scene.lightCount <= 0) {
        pdf = 0.0f;
        return -1;
    }
    float total = 0.0f;
    for (int i = 0; i < scene.lightCount; ++i) total += lightFluxWeight(scene, i);
    if (total <= 1e-20f) {
        int idx = int(u * float(scene.lightCount));
        if (idx >= scene.lightCount) idx = scene.lightCount - 1;
        pdf = lightSelectionPdf(scene);
        return idx;
    }
    float r = clampf(u, 0.0f, 0.999999f) * total;
    for (int i = 0; i < scene.lightCount; ++i) {
        const float w = lightFluxWeight(scene, i);
        if (r < w) {
            pdf = w / total;
            return i;
        }
        r -= w;
    }
    pdf = lightFluxWeight(scene, scene.lightCount - 1) / total;
    return scene.lightCount - 1;
}

// ---------------------------------------------------------------------------
// BVH-accelerated light sampling (position-aware importance)
// ---------------------------------------------------------------------------

// Importance of a BVH node from refP. Uses distance to the AABB *center*
// softened by the node's extent — not min-distance-to-box (that is zero under
// the entire footprint and jumps at AABB faces → axis-aligned "bucket" noise
// in caustic shadows on floors).
SR_INL SR_HD float bvhNodeImportance(const LightBvhNode& node, Vec3 refP) {
    if (node.power <= 0.f) return 0.f;
    const Vec3 center = (node.bMin + node.bMax) * 0.5f;
    const Vec3 ext = node.bMax - node.bMin;
    const float r2 = 0.25f * (ext.x * ext.x + ext.y * ext.y + ext.z * ext.z) + 1e-4f;
    const float d2 = lengthSquared(refP - center);
    return node.power / (d2 + r2);
}

// Returns true if the BVH subtree rooted at `nodeIdx` contains `lightIdx`.
// Iterative DFS; stack depth ≤ tree height (≤ log2(N)+1 for a balanced tree).
SR_INL SR_HD bool bvhContainsLight(const LightBvhNode* nodes, int nodeCount,
                                   int nodeIdx, int lightIdx) {
    int stack[64];
    int top = 0;
    if (nodeIdx >= 0 && nodeIdx < nodeCount) stack[top++] = nodeIdx;
    while (top > 0) {
        const int idx = stack[--top];
        const LightBvhNode& n = nodes[idx];
        if (n.isLeaf) {
            if (n.childOrLight == lightIdx) return true;
        } else {
            // Safe: tree height h ≤ log2(N), so max stack depth ≤ h + 1 ≤ 64.
            const int lc = n.childOrLight;
            const int rc = n.rightChild;
            if (lc >= 0 && lc < nodeCount && top < 63) stack[top++] = lc;
            if (rc >= 0 && rc < nodeCount && top < 63) stack[top++] = rc;
        }
    }
    return false;
}

// Position-aware light selection via the prebuilt BVH (center+extent importance).
// Infinite lights (dome/distant) are weighted by flux and kept outside the BVH.
// Falls back to the flux-only overload when the BVH is unavailable (or skipped
// for scenes with few finite lights).
SR_INL SR_HD int sampleLightIndex(const SceneView& scene, Vec3 refP, float u, float& pdf) {
    if (scene.lightCount <= 0) { pdf = 0.f; return -1; }

    if (!scene.lightBvh || scene.lightBvhNodeCount == 0)
        return sampleLightIndex(scene, u, pdf);

    const float wInf = scene.infiniteLightPower;
    const float wFin = bvhNodeImportance(scene.lightBvh[0], refP);
    const float wTotal = wInf + wFin;
    if (wTotal <= 1e-30f) return sampleLightIndex(scene, u, pdf);

    float r = clampf(u, 0.f, 0.999999f) * wTotal;

    // -- Infinite-light branch --------------------------------------------------
    if (r < wInf && scene.infiniteLightCount > 0) {
        float cumW = 0.f;
        for (int i = 0; i < scene.infiniteLightCount; ++i) {
            const int   idx = scene.infiniteLightIndices[i];
            const float w   = lightFluxWeight(scene, idx);
            cumW += w;
            if (r < cumW || i == scene.infiniteLightCount - 1) {
                pdf = (wInf > 0.f ? w / wInf : 1.f) * (wInf / wTotal);
                if (pdf <= 0.f) { pdf = 0.f; return -1; }
                return idx;
            }
        }
    }

    // -- Finite-light branch: traverse BVH -------------------------------------
    if (wFin <= 0.f) {
        pdf = 0.f; return -1;
    }
    float uFin = clampf((r - wInf) / wFin, 0.f, 0.999999f);
    float travPdf = wFin / wTotal;

    int nodeIdx = 0;
    for (int depth = 0; depth < 64 && !scene.lightBvh[nodeIdx].isLeaf; ++depth) {
        const LightBvhNode& nd = scene.lightBvh[nodeIdx];
        const int lc = nd.childOrLight;
        const int rc = nd.rightChild;
        if (lc < 0 || lc >= scene.lightBvhNodeCount ||
            rc < 0 || rc >= scene.lightBvhNodeCount) break;

        const float wL    = bvhNodeImportance(scene.lightBvh[lc], refP);
        const float wR    = bvhNodeImportance(scene.lightBvh[rc], refP);
        const float wSum  = wL + wR;
        const float pLeft = wSum > 0.f ? wL / wSum : 0.5f;

        if (uFin < pLeft) {
            travPdf *= pLeft;
            uFin = pLeft > 0.f ? uFin / pLeft : 0.5f;
            nodeIdx = lc;
        } else {
            travPdf *= (1.f - pLeft);
            const float rem = 1.f - pLeft;
            uFin = rem > 0.f ? (uFin - pLeft) / rem : 0.5f;
            nodeIdx = rc;
        }
        uFin = clampf(uFin, 0.f, 0.999999f);
    }

    pdf = travPdf;
    if (pdf <= 0.f) { pdf = 0.f; return -1; }
    return scene.lightBvh[nodeIdx].childOrLight;
}

// Position-aware selection PDF for a specific light index.
// Must be called with the same `refP` used during sampling for MIS correctness.
SR_INL SR_HD float lightSelectionPdfIndex(const SceneView& scene, Vec3 refP, int lightIndex) {
    if (!scene.lightBvh || scene.lightBvhNodeCount == 0 ||
        lightIndex < 0 || lightIndex >= scene.lightCount)
        return lightSelectionPdfIndex(scene, lightIndex);

    const float wInf   = scene.infiniteLightPower;
    const float wFin   = bvhNodeImportance(scene.lightBvh[0], refP);
    const float wTotal = wInf + wFin;
    if (wTotal <= 1e-30f) return lightSelectionPdfIndex(scene, lightIndex);

    const LightData& l = scene.lights[lightIndex];
    if (l.type == kLightDome || l.type == kLightDistant) {
        return lightFluxWeight(scene, lightIndex) / wTotal;
    }

    // Walk BVH path to the leaf containing lightIndex.
    float travPdf = wFin / wTotal;
    int nodeIdx = 0;
    for (int depth = 0; depth < 64 && !scene.lightBvh[nodeIdx].isLeaf; ++depth) {
        const LightBvhNode& nd = scene.lightBvh[nodeIdx];
        const int lc = nd.childOrLight;
        const int rc = nd.rightChild;
        if (lc < 0 || lc >= scene.lightBvhNodeCount ||
            rc < 0 || rc >= scene.lightBvhNodeCount) break;

        const float wL    = bvhNodeImportance(scene.lightBvh[lc], refP);
        const float wR    = bvhNodeImportance(scene.lightBvh[rc], refP);
        const float wSum  = wL + wR;
        const float pLeft = wSum > 0.f ? wL / wSum : 0.5f;

        if (bvhContainsLight(scene.lightBvh, scene.lightBvhNodeCount, lc, lightIndex)) {
            travPdf *= pLeft;
            nodeIdx = lc;
        } else {
            travPdf *= (1.f - pLeft);
            nodeIdx = rc;
        }
    }
    return travPdf;
}

// Solar disc for Physical Sky distant lights. The baked sky map has no disc
// (avoids double lighting / HDRI fireflies); camera and glossy rays see this.
SR_INL SR_HD Vec3 cameraSunDiscRadiance(const SceneView& scene, Vec3 origin, Vec3 dirWorld, float bsdfPdf,
                                        bool specularBounce, bool primary, bool skipNonCausticLights) {
    Vec3 sum(0.0f);
    const Vec3 wi = normalize(dirWorld);
    for (int i = 0; i < scene.lightCount; ++i) {
        const LightData& l = scene.lights[i];
        if (l.type != kLightDistant || l.cameraSunDisc == 0) continue;
        if (primary && l.visibleCamera == 0) continue;
        if (skipNonCausticLights && !lightContributesCaustics(l)) continue;
        const float halfAngle = radians(srMax(0.0f, l.angle)) * 0.5f;
        if (halfAngle < 1e-4f) continue;
        const Vec3 axis = normalize(lightAxisZ(l));
        const float cosThetaMax = cosf(halfAngle);
        if (dot(axis, wi) < cosThetaMax) continue;
        const Vec3 Le = lightRadiance(l);
        float weight = 1.0f;
        if (!specularBounce) {
            const float lp = lightPdfDirection(scene, i, origin, wi, origin, wi) *
                             lightSelectionPdfIndex(scene, origin, i);
            weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
        }
        sum += Le * weight;
    }
    return sum;
}

}  // namespace sol
