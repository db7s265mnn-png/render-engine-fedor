// Path tracer with Manifold Next-Event Estimation for refractive caustics
// (Hanika et al. 2015, formulated as a 2D Newton solve on the initial direction
// of a specular refraction chain — robust for glass with several interfaces).
// CPU / Embree only — included from embree_device.cpp.
//
// At every connectable vertex the light sample is first checked with a plain
// shadow segment. If the first blocker is a delta-transmissive "caustic caster",
// the solver walks the refraction chain from the shading point and Newton-steps
// the launch direction until the chain exits toward the light point. The
// converged Jacobian |dx⊥/dω| provides the generalized geometry term, so the
// estimator matches plain NEE when no glass is present.
//
// To avoid double counting, diffuse→(all-transmissive delta chain)→finite-light
// BSDF hits are suppressed — MNEE is the (much lower variance) estimator for
// that family. Reflective caustics stay with BSDF sampling; environment light
// stays with regular NEE/MIS.
#pragma once

#include "core/rng.h"
#include "render/integrator.h"
#include "render/lights.h"
#include "render/shading.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPENPGL
#include "render/cpu/path_guiding.h"
#endif

namespace sol {
namespace mnee {

constexpr int kMaxChain = 6;      // max specular interfaces (glass sphere = 2)
constexpr int kNewtonIters = 24;
constexpr int kBacktrackSteps = 5;
// Multi-branch seeding (SMS-inspired, Zeltner et al. 2020, made deterministic):
// the straight-line seed plus a fixed ring of seeds inside the caster cone.
// Both the NEE estimator and the BSDF-suppression check run the SAME seed set,
// so every branch is counted exactly once regardless of how many seeds converge.
constexpr int kSeedRing = 4;                    // ring seeds beside the straight one
constexpr float kSeedRingRadius = 0.75f;        // fraction of the cone angle


SR_INL bool isCausticCaster(const Material& m) {
    const LobeWeights lw = computeLobes(m);
    return lw.delta && lw.transmission > 0.25f && lw.diffuse < 1e-3f;
}

// Refract direction d (travel direction) through a surface with normal n.
// Returns false on total internal reflection.
SR_INL bool refractTravel(Vec3 d, Vec3 n, float ior, Vec3& out, float& etaRel, float& fresnel) {
    float cosI = -dot(d, n);
    Vec3 nn = n;
    float eta = 1.0f / ior;  // entering (air → glass)
    if (cosI < 0.0f) {       // leaving the medium
        cosI = -cosI;
        nn = -n;
        eta = ior;
    }
    const float sin2T = eta * eta * (1.0f - cosI * cosI);
    if (sin2T >= 1.0f) return false;
    const float cosT = sqrtf(srMax(0.0f, 1.0f - sin2T));
    out = normalize(d * eta + nn * (eta * cosI - cosT));
    etaRel = eta;
    fresnel = fresnelDielectric(cosI, ior);
    return true;
}

struct ChainState {
    Vec3 exitP{0.0f};    // last surface point of the chain
    Vec3 exitDir{0.0f};  // travel direction after the last refraction
    Vec3 exitN{0.0f};
    Vec3 throughput{1.0f};
    int interfaces = 0;
    bool valid = false;
};

// Trace the refraction chain from p along dir. Stops when the current segment
// carries no more caustic casters before `maxDist` along its line.
template <typename Tracer>
SR_INL ChainState traceChain(const SceneView& scene, const Tracer& tracer, Vec3 p, Vec3 n, Vec3 dir,
                             int targetLight) {
    ChainState st;
    Vec3 o = offsetRayOrigin(p, n, dir);
    Vec3 d = dir;
    Vec3 T(1.0f);
    Vec3 lastP = p;
    Vec3 lastN = n;
    for (int k = 0; k <= kMaxChain; ++k) {
        RayHit hit;
        if (!tracer.intersect(o, d, kFloatMax, hit)) {
            st.exitP = lastP;
            st.exitDir = d;
            st.exitN = lastN;
            st.throughput = T;
            st.interfaces = k;
            st.valid = k > 0;
            return st;
        }
        SurfaceInteraction si;
        if (!buildSurfaceInteraction(scene, hit, o, d, si)) return st;
        if (si.lightIndex >= 0) {
            // Reached light geometry — chain complete (only the target light counts).
            if (si.lightIndex == targetLight || targetLight < 0) {
                st.exitP = lastP;
                st.exitDir = d;
                st.exitN = lastN;
                st.throughput = T;
                st.interfaces = k;
                st.valid = k > 0;
            }
            return st;
        }
        Material mat = si.materialIndex >= 0 && si.materialIndex < scene.materialCount
                           ? scene.materials[si.materialIndex]
                           : defaultMaterial();
        mat = evaluateTexturedMaterial(scene, mat, si.uv, si.ns, si.pObject, si.nObject, si.uvFilterWidth);
        if (!isCausticCaster(mat)) return st;  // opaque blocker → fail
        if (k == kMaxChain) return st;

        Vec3 dNext;
        float etaRel = 1.0f, fr = 0.0f;
        const LobeWeights lw = computeLobes(mat);
        if (!refractTravel(d, si.ns, lw.eta, dNext, etaRel, fr)) return st;  // TIR → fail
        // Radiance scaling 1/eta² per interface cancels over enter/exit pairs.
        T = T * lw.transmissionTint * ((1.0f - fr) / srMax(1e-4f, etaRel * etaRel));
        if (isBlack(T)) return st;
        lastP = si.p;
        lastN = si.ns;
        o = offsetRayOrigin(si.p, si.ng, dNext);
        d = dNext;
    }
    return st;
}

// Miss of the chain exit ray measured on the fixed plane through y with normal
// `planeN` (the seed direction p→y). Stays smooth for strongly diverging lens
// exits, which is what Newton needs to walk back onto the manifold.
template <typename Tracer>
SR_INL bool chainError(const SceneView& scene, const Tracer& tracer, Vec3 p, Vec3 n, Vec3 dir, Vec3 y,
                       int targetLight, Vec3 planeN, Vec3 b1, Vec3 b2, float& e1, float& e2,
                       ChainState* outChain) {
    const ChainState st = traceChain(scene, tracer, p, n, dir, targetLight);
    if (!st.valid) return false;
    const float denom = dot(st.exitDir, planeN);
    if (denom <= 1e-5f) return false;  // exit ray parallel to / away from the plane
    const float t = dot(y - st.exitP, planeN) / denom;
    if (t <= 1e-6f) return false;  // plane behind the exit point
    const Vec3 hit = st.exitP + st.exitDir * t;
    const Vec3 e = hit - y;
    e1 = dot(e, b1);
    e2 = dot(e, b2);
    if (outChain) *outChain = st;
    return true;
}

struct MneeResult {
    Vec3 contribution{0.0f};
    bool solved = false;
};

struct ManifoldSolution {
    Vec3 omega{0.0f};
    ChainState chain;
    float detJ = 0.0f;
    Vec3 planeN{0.0f};
    bool solved = false;
};

// Newton solve for the launch direction connecting p → (specular chain) → y,
// starting from `omegaInit` (SMS-style random seeds discover distinct branches).
template <typename Tracer>
SR_INL ManifoldSolution solveManifold(const SceneView& scene, const Tracer& tracer, Vec3 p, Vec3 n,
                                      int lightIndex, Vec3 y, Vec3 omegaInit) {
    ManifoldSolution sol;
    Vec3 dir = y - p;
    const float distPy = length(dir);
    if (distPy < 1e-5f) return sol;
    dir = dir / distPy;

    const Frame errFrame(dir);
    const Vec3 planeN = dir;
    const Vec3 b1 = errFrame.t;
    const Vec3 b2 = errFrame.b;

    Vec3 omega = normalize(omegaInit);
    const float tol = srMax(1e-5f, 1e-4f * distPy);
    float e1 = 0.0f, e2 = 0.0f;
    ChainState chain;
    float j11 = 0, j12 = 0, j21 = 0, j22 = 0;

    if (!chainError(scene, tracer, p, n, omega, y, lightIndex, planeN, b1, b2, e1, e2, &chain)) return sol;
    float err = sqrtf(e1 * e1 + e2 * e2);
    bool converged = err < tol;

    for (int iter = 0; iter < kNewtonIters && !converged; ++iter) {
        const Frame dirFrame(omega);
        const float h = 1e-3f;
        float p1 = 0, p2 = 0, q1 = 0, q2 = 0;
        const Vec3 omU = normalize(omega + dirFrame.t * h);
        const Vec3 omV = normalize(omega + dirFrame.b * h);
        if (!chainError(scene, tracer, p, n, omU, y, lightIndex, planeN, b1, b2, p1, p2, nullptr))
            return sol;
        if (!chainError(scene, tracer, p, n, omV, y, lightIndex, planeN, b1, b2, q1, q2, nullptr))
            return sol;
        j11 = (p1 - e1) / h;
        j21 = (p2 - e2) / h;
        j12 = (q1 - e1) / h;
        j22 = (q2 - e2) / h;
        const float det = j11 * j22 - j12 * j21;
        if (fabsf(det) < 1e-12f) return sol;
        float du = (j22 * e1 - j12 * e2) / det;
        float dv = (-j21 * e1 + j11 * e2) / det;
        const float stepLen = sqrtf(du * du + dv * dv);
        if (stepLen > 0.5f) {
            du *= 0.5f / stepLen;
            dv *= 0.5f / stepLen;
        }
        bool accepted = false;
        float scale = 1.0f;
        for (int k = 0; k < kBacktrackSteps; ++k, scale *= 0.5f) {
            const Vec3 cand = normalize(omega - (dirFrame.t * du + dirFrame.b * dv) * scale);
            if (dot(cand, dir) < -0.1f) continue;
            float c1 = 0.0f, c2 = 0.0f;
            ChainState candChain;
            if (!chainError(scene, tracer, p, n, cand, y, lightIndex, planeN, b1, b2, c1, c2, &candChain))
                continue;
            const float cErr = sqrtf(c1 * c1 + c2 * c2);
            if (cErr < err) {
                omega = cand;
                e1 = c1;
                e2 = c2;
                err = cErr;
                chain = candChain;
                accepted = true;
                break;
            }
        }
        if (!accepted) return sol;  // stuck — no manifold in reach
        converged = err < tol;
    }
    if (!converged) return sol;

    // Final Jacobian at the solution (generalized geometry term). Central
    // differences with a small step: near caustic folds the mapping curvature is
    // large and forward differences overestimate |det J| several-fold, dimming
    // the caustic.
    {
        const Frame dirFrame(omega);
        const float h = 2e-4f;
        float a1 = 0, a2 = 0, c1 = 0, c2 = 0;
        const Vec3 omUp = normalize(omega + dirFrame.t * h);
        const Vec3 omUm = normalize(omega - dirFrame.t * h);
        const Vec3 omVp = normalize(omega + dirFrame.b * h);
        const Vec3 omVm = normalize(omega - dirFrame.b * h);
        if (!chainError(scene, tracer, p, n, omUp, y, lightIndex, planeN, b1, b2, a1, a2, nullptr))
            return sol;
        if (!chainError(scene, tracer, p, n, omUm, y, lightIndex, planeN, b1, b2, c1, c2, nullptr))
            return sol;
        j11 = (a1 - c1) / (2.0f * h);
        j21 = (a2 - c2) / (2.0f * h);
        if (!chainError(scene, tracer, p, n, omVp, y, lightIndex, planeN, b1, b2, a1, a2, nullptr))
            return sol;
        if (!chainError(scene, tracer, p, n, omVm, y, lightIndex, planeN, b1, b2, c1, c2, nullptr))
            return sol;
        j12 = (a1 - c1) / (2.0f * h);
        j22 = (a2 - c2) / (2.0f * h);
    }
    const float detJ = fabsf(j11 * j22 - j12 * j21);
    if (!(detJ > 1e-10f) || !srIsFinite(detJ)) return sol;

    sol.omega = omega;
    sol.chain = chain;
    sol.detJ = detJ;
    sol.planeN = planeN;
    sol.solved = true;
    return sol;
}

// Contribution of a converged manifold solution (BSDF at p included, light
// radiance included, pdfs divided out). Zero when occluded / non-emitting.
template <typename Tracer>
SR_INL Vec3 solutionContribution(const SceneView& scene, const Tracer& tracer, const ManifoldSolution& sol,
                                 Vec3 p, Vec3 n, Vec3 wo, const Material& shadeMat, int lightIndex, Vec3 y,
                                 Vec3 yN, Vec3 Le, float pdfArea, float selectPdf) {
    const ChainState& chain = sol.chain;

    // Verify the final segment reaches y unoccluded.
    {
        const Vec3 toY = y - chain.exitP;
        const float d = length(toY);
        if (d < 1e-5f) return Vec3(0.0f);
        const Vec3 wd = toY / d;
        const Vec3 o = offsetRayOrigin(chain.exitP, chain.exitN, wd);
        RayHit hit;
        if (tracer.intersect(o, wd, d * (1.0f - 1e-3f), hit)) {
            SurfaceInteraction si;
            if (!buildSurfaceInteraction(scene, hit, o, wd, si) || si.lightIndex != lightIndex)
                return Vec3(0.0f);
        }
    }

    const Frame frame(n);
    const BsdfEval be = bsdfEvalLocal(shadeMat, frame.toLocal(wo), frame.toLocal(sol.omega));
    if (be.pdf <= 0.0f || isBlack(be.f)) return Vec3(0.0f);
    const float cosP = fabsf(dot(n, sol.omega));

    // Plane → light-surface area conversion (the error plane has normal planeN).
    // Point lights integrate radiant intensity directly (no cosine).
    float planeToLight = 1.0f;
    const LightData& l = scene.lights[lightIndex];
    if (l.type != kLightPoint) {
        const float cEmit = dot(yN, -chain.exitDir);
        const float emitOk = (l.type != kLightSphere && l.twoSided) ? fabsf(cEmit) : srMax(0.0f, cEmit);
        if (emitOk <= 1e-6f) return Vec3(0.0f);  // light does not emit toward the chain exit
        planeToLight = fabsf(dot(yN, sol.planeN));
        if (planeToLight <= 1e-6f) return Vec3(0.0f);
    }

    // dω/dA_y = |dot(yN, planeN)| / |det J|  (straight line: |det J| = dist² → 1/r²).
    const float geom = planeToLight / sol.detJ;
    const Vec3 c = be.f * cosP * chain.throughput * Le * geom / srMax(1e-12f, pdfArea * selectPdf);
    if (!isFinite(c)) return Vec3(0.0f);
    return vmax(Vec3(0.0f), c);
}

// Seed cone that covers the caustic caster instance as seen from p. Falls back
// to a narrow cone around the straight line when bounds are unavailable.
SR_INL void seedCone(const SceneView& scene, Vec3 p, Vec3 straightDir, int casterInstance, Vec3& axis,
                     float& cosThetaMax) {
    axis = straightDir;
    cosThetaMax = 0.9962f;  // ~5° fallback
    if (casterInstance < 0 || casterInstance >= scene.instanceCount) return;
    const InstanceData& inst = scene.instances[casterInstance];
    if (inst.meshIndex < 0 || inst.meshIndex >= scene.meshCount) return;
    const MeshView& mesh = scene.meshes[inst.meshIndex];
    const Vec3 lo = mesh.boundsLo;
    const Vec3 hi = mesh.boundsHi;
    if (!(hi.x >= lo.x && hi.y >= lo.y && hi.z >= lo.z)) return;
    Bounds3 world;
    for (int i = 0; i < 8; ++i) {
        const Vec3 corner(i & 1 ? hi.x : lo.x, i & 2 ? hi.y : lo.y, i & 4 ? hi.z : lo.z);
        world.extend(transformPoint(inst.xform, corner));
    }
    const Vec3 center = world.center();
    const float radius = 0.5f * length(world.extent());
    const Vec3 toC = center - p;
    const float dist = length(toC);
    if (dist <= radius * 1.05f || dist < 1e-5f) {
        // Shading point effectively inside the caster bounds — sample widely.
        cosThetaMax = 0.0f;  // hemisphere around the straight direction
        return;
    }
    axis = toC / dist;
    const float sinT = clampf(radius / dist, 0.0f, 0.9999f);
    cosThetaMax = sqrtf(srMax(0.0f, 1.0f - sinT * sinT));
}

// Deterministic multi-branch seed set: the straight line to y plus a ring of
// directions inside the caster cone (catches rim chains and secondary images).
struct SeedSet {
    Vec3 dirs[1 + kSeedRing];
    int count = 0;
};

SR_INL SeedSet buildSeedSet(const SceneView& scene, Vec3 p, Vec3 straightDir, int casterInstance) {
    SeedSet s;
    s.dirs[s.count++] = straightDir;
    Vec3 axis;
    float cosThetaMax = 1.0f;
    seedCone(scene, p, straightDir, casterInstance, axis, cosThetaMax);
    const float thetaMax = acosf(clampf(cosThetaMax, -1.0f, 1.0f));
    if (thetaMax < 1e-3f) return s;  // cone degenerate — straight seed only
    const float theta = thetaMax * kSeedRingRadius;
    const float st = sinf(theta);
    const float ct = cosf(theta);
    const Frame frame(axis);
    for (int i = 0; i < kSeedRing; ++i) {
        const float phi = (float(i) + 0.5f) * (kTwoPi / float(kSeedRing));
        const Vec3 local(st * cosf(phi), st * sinf(phi), ct);
        s.dirs[s.count++] = normalize(frame.toWorld(local));
    }
    return s;
}

SR_INL bool sameBranch(Vec3 a, Vec3 b) { return dot(a, b) > 0.99996f; }  // ≈0.5°

// Multi-branch MNEE: run the deterministic seed set, deduplicate the converged
// branches and sum each unique contribution once. The BSDF-suppression check in
// the path loop replays the same seed set, so a BSDF-found chain is suppressed
// exactly when this estimator counts its branch — totals stay unbiased no matter
// how many branches the solver finds.
template <typename Tracer>
SR_INL MneeResult manifoldConnect(const SceneView& scene, const Tracer& tracer, Vec3 p, Vec3 n, Vec3 wo,
                                  const Material& shadeMat, int lightIndex, Vec3 y, Vec3 yN, Vec3 Le,
                                  float pdfArea, float selectPdf, int casterInstance) {
    MneeResult res;
    Vec3 dir = y - p;
    const float distPy = length(dir);
    if (distPy < 1e-5f) return res;
    dir = dir / distPy;

    const SeedSet seeds = buildSeedSet(scene, p, dir, casterInstance);
    Vec3 found[1 + kSeedRing];
    int foundCount = 0;
    Vec3 total(0.0f);
    for (int i = 0; i < seeds.count; ++i) {
        const ManifoldSolution sol = solveManifold(scene, tracer, p, n, lightIndex, y, seeds.dirs[i]);
        if (!sol.solved) continue;
        bool duplicate = false;
        for (int k = 0; k < foundCount; ++k)
            if (sameBranch(sol.omega, found[k])) {
                duplicate = true;
                break;
            }
        if (duplicate) continue;
        found[foundCount++] = sol.omega;
        res.solved = true;
        total += solutionContribution(scene, tracer, sol, p, n, wo, shadeMat, lightIndex, y, yN, Le, pdfArea,
                                      selectPdf);
    }
    res.contribution = total;
    return res;
}

// True when the seed set converges to the branch a BSDF path actually took —
// i.e. the MNEE estimator above covers this exact chain.
template <typename Tracer>
SR_INL bool branchCovered(const SceneView& scene, const Tracer& tracer, Vec3 p, Vec3 n, int lightIndex,
                          Vec3 y, int casterInstance, Vec3 pathDir) {
    Vec3 dir = y - p;
    const float distPy = length(dir);
    if (distPy < 1e-5f) return false;
    dir = dir / distPy;
    const SeedSet seeds = buildSeedSet(scene, p, dir, casterInstance);
    for (int i = 0; i < seeds.count; ++i) {
        const ManifoldSolution sol = solveManifold(scene, tracer, p, n, lightIndex, y, seeds.dirs[i]);
        if (sol.solved && sameBranch(sol.omega, pathDir)) return true;
    }
    return false;
}

// Sample a point on a finite light in area measure.
SR_INL bool sampleFiniteLightPoint(const SceneView& scene, int lightIndex, Rng& rng, Vec3& y, Vec3& yN,
                                   Vec3& Le, float& pdfArea) {
    const LightData& l = scene.lights[lightIndex];
    switch (l.type) {
        case kLightRect: {
            const float area = rectLightArea(l);
            if (area <= 1e-12f) return false;
            y = transformPoint(l.xform, Vec3((rng.nextFloat() - 0.5f) * l.width,
                                             (rng.nextFloat() - 0.5f) * l.height, 0.0f));
            yN = areaLightNormal(l);
            Le = lightRadiance(l);
            pdfArea = 1.0f / area;
            return true;
        }
        case kLightDisk: {
            const float area = diskLightArea(l);
            if (area <= 1e-12f) return false;
            const Vec2 d = sampleConcentricDisk(rng.nextFloat(), rng.nextFloat());
            y = transformPoint(l.xform, Vec3(d.x * l.radius, d.y * l.radius, 0.0f));
            yN = areaLightNormal(l);
            Le = lightRadiance(l);
            pdfArea = 1.0f / area;
            return true;
        }
        case kLightSphere: {
            const Vec3 center = lightOrigin(l);
            const float radius = srMax(1e-5f, sphereLightRadius(l));
            yN = sampleUniformSphere(rng.nextFloat(), rng.nextFloat());
            y = center + yN * radius;
            Le = lightRadiance(l);
            pdfArea = 1.0f / (4.0f * kPi * radius * radius);
            return true;
        }
        case kLightPoint: {
            y = lightOrigin(l);
            yN = Vec3(0.0f, 1.0f, 0.0f);
            Le = l.emittedRadiance();  // radiant intensity (W/sr)
            pdfArea = 1.0f;
            return true;
        }
        default:
            return false;
    }
}

}  // namespace mnee

// ---------------------------------------------------------------------------
// Path tracer with MNEE refractive caustics (dispatched by the Embree backend
// when Integrator = Path Tracer and render-settings caustics are enabled).
// ---------------------------------------------------------------------------
template <typename Tracer, typename Guiding>
SR_INL Vec3 traceRadiancePtMnee(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                                Rng& rng, Guiding* guiding) {
    Vec3 radiance(0.0f);
    Vec3 throughput(1.0f);
    float bsdfPdf = 0.0f;
    bool specularBounce = true;
    // True while the path suffix after the last non-specular vertex is a pure
    // delta-transmissive chain — the family MNEE covers for finite lights.
    bool sawNonSpecular = false;
    bool mneeFamily = false;
    int familyChainLen = 0;
    Vec3 anchorP(0.0f);  // last non-specular surface vertex (MNEE launch point)
    Vec3 anchorN(0.0f, 1.0f, 0.0f);
    Vec3 anchorDir(0.0f, 1.0f, 0.0f);  // direction the path left the anchor with
    int depth = 0;
    int passThrough = 0;
    const RenderSettingsData& settings = scene.settings;
    const int maxDepth = srMax(1, settings.maxDepth);

    while (depth <= maxDepth) {
        RayHit hit;
        const bool didHit = tracer.intersect(origin, direction, kFloatMax, hit);

        if (!didHit) {
            if (scene.domeLightIndex >= 0) {
                const LightData& dome = scene.lights[scene.domeLightIndex];
                const bool primary = depth == 0 && passThrough == 0;
                if (!(primary && (!settings.envVisibleCamera || !dome.visibleCamera))) {
                    Vec3 envL = domeRadiance(scene, dome, direction);
                    if (!isBlack(envL)) {
                        float weight = 1.0f;
                        if (!specularBounce) {
                            const float lp = lightPdfDirection(scene, scene.domeLightIndex, origin, direction,
                                                               origin, direction) *
                                             lightSelectionPdfIndex(scene, scene.domeLightIndex);
                            weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                        }
                        Vec3 contrib = throughput * envL * weight;
                        if (depth > 0 && !specularBounce)
                            contrib = clampContribution(contrib, settings.clampIndirect);
                        radiance += contrib;
#if !defined(__CUDACC__)
                        if (guiding && guiding->active())
                            guiding->recordBackground(origin, direction, envL, weight);
#endif
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
            // Suppress a BSDF-found caustic chain exactly when the multi-seed MNEE
            // estimator covers its branch: the straight anchor→light segment must
            // hit a caustic caster (the NEE trigger) AND replaying the same seed
            // set must converge onto the direction this path actually took. Chains
            // the seeds miss stay with unbiased BSDF sampling — totals preserved.
            const LightData& light = scene.lights[si.lightIndex];
            const bool finiteLight = light.type == kLightRect || light.type == kLightDisk ||
                                     light.type == kLightSphere || light.type == kLightPoint;
            if (mneeFamily && finiteLight && settings.caustics != 0) {
                Vec3 seg = si.p - anchorP;
                const float segLen = length(seg);
                if (segLen > 1e-5f) {
                    seg = seg / segLen;
                    const Vec3 o = offsetRayOrigin(anchorP, anchorN, seg);
                    RayHit sh;
                    if (tracer.intersect(o, seg, segLen * (1.0f - 1e-3f), sh)) {
                        SurfaceInteraction ssi;
                        if (buildSurfaceInteraction(scene, sh, o, seg, ssi) && ssi.lightIndex < 0) {
                            Material smat = ssi.materialIndex >= 0 && ssi.materialIndex < scene.materialCount
                                                ? scene.materials[ssi.materialIndex]
                                                : defaultMaterial();
                            smat = evaluateTexturedMaterial(scene, smat, ssi.uv, ssi.ns, ssi.pObject,
                                                            ssi.nObject, ssi.uvFilterWidth);
                            if (mnee::isCausticCaster(smat) &&
                                mnee::branchCovered(scene, tracer, anchorP, anchorN, si.lightIndex, si.p,
                                                    ssi.instanceIndex, anchorDir))
                                break;  // MNEE-covered — suppress the noisy BSDF copy
                        }
                    }
                }
            }
            if (mneeFamily && finiteLight && settings.caustics == 0) break;

            const Vec3 lightN = light.type == kLightSphere ? si.ng : areaLightNormal(light);
            Vec3 emitted = areaLightEmission(scene, light, direction, lightN);
            if (!isBlack(emitted)) {
                float weight = 1.0f;
                if (!specularBounce) {
                    const float lp = lightPdfDirection(scene, si.lightIndex, origin, direction, si.p, lightN) *
                                     lightSelectionPdfIndex(scene, si.lightIndex);
                    weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                }
                Vec3 contrib = throughput * emitted * weight;
                if (depth > 0 && !specularBounce)
                    contrib = clampContribution(contrib, settings.clampIndirect);
                radiance += contrib;
#if !defined(__CUDACC__)
                if (guiding && guiding->active())
                    guiding->recordLightHit(si.p, -direction, emitted, weight);
#endif
            }
            break;
        }

        Material baseMat = si.materialIndex >= 0 && si.materialIndex < scene.materialCount
                               ? scene.materials[si.materialIndex]
                               : defaultMaterial();
        Material mat =
            evaluateTexturedMaterial(scene, baseMat, si.uv, si.ns, si.pObject, si.nObject, si.uvFilterWidth);

        if (mat.transmission <= 0.0f && mat.doubleSided && dot(si.ns, -direction) < 0.0f) {
            si.ns = -si.ns;
            si.ng = -si.ng;
        }

        if (mat.emissionStrength > 0.0f && !isBlack(mat.emissionColor)) {
            const bool frontFacing = dot(si.ns, -direction) > 0.0f;
            if (frontFacing || mat.doubleSided)
                radiance += throughput * mat.emissionColor * mat.emissionStrength;
        }

        if (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity) {
            origin = offsetRayOrigin(si.p, si.ng, direction);
            ++passThrough;
            if (passThrough > 32) break;
            continue;
        }

        if (depth >= maxDepth) break;

        const Vec3 wo = -direction;
        const Frame frame(si.ns);
        const LobeWeights lw = computeLobes(mat);

#if !defined(__CUDACC__)
        bool guideReady = false;
        if (guiding && guiding->active() && !lw.delta) {
            guiding->beginSegment(si.p, wo);
            guideReady = guiding->prepare(si.p, si.ns, rng);
        }
#else
        const bool guideReady = false;
        (void)guiding;
#endif

        // --- NEE with lazy MNEE upgrade -------------------------------------
        const bool connectable = lw.diffuse > 1e-4f || !lw.delta;
        if (connectable) {
            const int nLightSamples = srMax(1, settings.lightSamples);
            Vec3 neeSum(0.0f);
            for (int ls = 0; ls < nLightSamples; ++ls) {
                float selectPdf = 0.0f;
                const int li = sampleLightIndex(scene, rng.nextFloat(), selectPdf);
                if (li < 0 || selectPdf <= 0.0f) continue;
                const LightData& l = scene.lights[li];

                if (l.type == kLightDome || l.type == kLightDistant) {
                    LightSample lsam;
                    if (!sampleLight(scene, li, si.p, rng.nextFloat(), rng.nextFloat(), lsam)) continue;
                    if (lsam.pdf <= 0.0f || isBlack(lsam.radiance)) continue;
                    float visibility = 1.0f;
                    if (l.shadowEnable) {
                        const Vec3 o = offsetRayOrigin(si.p, si.ng, lsam.wi);
                        visibility = shadowVisibility(scene, tracer, o, lsam.wi, 1.0e8f);
                    }
                    if (visibility <= 1e-5f) continue;
                    const Vec3 wiL = frame.toLocal(lsam.wi);
                    const BsdfEval be = bsdfEvalLocal(mat, frame.toLocal(wo), wiL);
                    if (be.pdf <= 0.0f || isBlack(be.f)) continue;
                    float scatterPdf = be.pdf;
#if !defined(__CUDACC__)
                    if (guideReady) {
                        const float pg = guiding->guideProbability();
                        scatterPdf = pg * guiding->pdf(lsam.wi) + (1.0f - pg) * be.pdf;
                    }
#endif
                    const float lightPdf = lsam.pdf * selectPdf;
                    const float w = lsam.delta ? 1.0f : powerHeuristic(1.0f, lightPdf, 1.0f, scatterPdf);
                    neeSum += lsam.radiance * be.f * (fabsf(wiL.z) * w * visibility / lightPdf);
                    continue;
                }

                // Finite light: sample an area point, one shadow segment first.
                Vec3 y, yN, Le;
                float pdfArea = 0.0f;
                if (!mnee::sampleFiniteLightPoint(scene, li, rng, y, yN, Le, pdfArea)) continue;
                if (pdfArea <= 0.0f || isBlack(Le)) continue;

                Vec3 toY = y - si.p;
                const float dist2 = lengthSquared(toY);
                if (dist2 < 1e-10f) continue;
                const float dist = sqrtf(dist2);
                const Vec3 wi = toY / dist;

                // Peek along the shadow segment: clear, glass, or blocked?
                bool clearPath = true;
                bool glassPath = false;
                int blockerInstance = -1;
                if (l.shadowEnable) {
                    const Vec3 o = offsetRayOrigin(si.p, si.ng, wi);
                    RayHit sh;
                    if (tracer.intersect(o, wi, dist * (1.0f - 1e-3f), sh)) {
                        SurfaceInteraction bsi;
                        clearPath = false;
                        if (buildSurfaceInteraction(scene, sh, o, wi, bsi)) {
                            if (bsi.lightIndex == li) {
                                clearPath = true;
                            } else if (bsi.lightIndex < 0 && settings.caustics != 0) {
                                Material bmat = bsi.materialIndex >= 0 && bsi.materialIndex < scene.materialCount
                                                    ? scene.materials[bsi.materialIndex]
                                                    : defaultMaterial();
                                bmat = evaluateTexturedMaterial(scene, bmat, bsi.uv, bsi.ns, bsi.pObject,
                                                                bsi.nObject, bsi.uvFilterWidth);
                                glassPath = mnee::isCausticCaster(bmat);
                                blockerInstance = bsi.instanceIndex;
                            }
                        }
                    }
                }

                if (clearPath) {
                    Vec3 lightN = yN;
                    float cosL = dot(lightN, -wi);
                    if (l.type == kLightSphere) {
                        // Back-facing sphere samples are self-occluded — reject.
                        if (cosL <= 1e-6f) continue;
                    } else if (l.type != kLightPoint) {
                        if (cosL <= 0.0f && !l.twoSided) continue;
                        cosL = fabsf(cosL);
                        if (cosL <= 1e-6f) continue;
                    }
                    const Vec3 wiLocal = frame.toLocal(wi);
                    const BsdfEval be = bsdfEvalLocal(mat, frame.toLocal(wo), wiLocal);
                    if (be.pdf <= 0.0f || isBlack(be.f)) continue;
                    float scatterPdf = be.pdf;
#if !defined(__CUDACC__)
                    if (guideReady) {
                        const float pg = guiding->guideProbability();
                        scatterPdf = pg * guiding->pdf(wi) + (1.0f - pg) * be.pdf;
                    }
#endif
                    if (l.type == kLightPoint) {
                        neeSum += Le * be.f * (fabsf(wiLocal.z) / (dist2 * selectPdf));
                    } else {
                        const float pdfSa = pdfArea * dist2 / cosL;  // area → solid angle
                        const float lightPdf = pdfSa * selectPdf;
                        const float w = powerHeuristic(1.0f, lightPdf, 1.0f, scatterPdf);
                        neeSum += Le * be.f * (fabsf(wiLocal.z) * w / lightPdf);
                    }
                } else if (glassPath) {
                    // Multi-seed MNEE: manifold connections through the refraction
                    // chain (matching BSDF path copies are suppressed at light hits).
                    const mnee::MneeResult mr =
                        mnee::manifoldConnect(scene, tracer, si.p, si.ns, wo, mat, li, y, yN, Le, pdfArea,
                                              selectPdf, blockerInstance);
                    if (mr.solved && !isBlack(mr.contribution)) {
                        Vec3 c = mr.contribution;
                        c = clampContribution(c, settings.clampIndirect > 0.0f
                                                     ? settings.clampIndirect * 4.0f
                                                     : 0.0f);  // caustics keep more energy
                        neeSum += c;
                    }
                }
            }
            const Vec3 nee = neeSum * (1.0f / float(srMax(1, settings.lightSamples)));
            Vec3 contrib = throughput * nee;
            if (depth > 0 && !specularBounce) contrib = clampContribution(contrib, settings.clampIndirect);
            radiance += contrib;
#if !defined(__CUDACC__)
            if (guiding && guiding->active()) guiding->addScattered(nee);
#endif
        }

        // --- BSDF continuation (guided mixture) ------------------------------
        const Vec3 woLocal = frame.toLocal(wo);
        BsdfSample bs;
        bool gotSample = false;
#if !defined(__CUDACC__)
        if (guideReady) {
            const float pg = guiding->guideProbability();
            if (rng.nextFloat() < pg) {
                Vec3 wiWorld;
                float gPdf = 0.0f;
                if (guiding->sample(rng.nextFloat(), rng.nextFloat(), wiWorld, gPdf) && gPdf > 0.0f) {
                    const Vec3 wiLocal = frame.toLocal(wiWorld);
                    const BsdfEval be = bsdfEvalLocal(mat, woLocal, wiLocal);
                    if (be.pdf > 0.0f && !isBlack(be.f)) {
                        const float mixPdf = pg * gPdf + (1.0f - pg) * be.pdf;
                        if (mixPdf > 0.0f) {
                            bs.wi = wiLocal;
                            bs.pdf = mixPdf;
                            bs.weight = be.f * (fabsf(wiLocal.z) / mixPdf);
                            bs.specular = false;
                            bs.transmitted = wiLocal.z < 0.0f;
                            gotSample = true;
                        }
                    }
                }
            }
        }
#endif
        if (!gotSample) {
            bs = bsdfSampleLocal(mat, woLocal, rng.nextFloat(), rng.nextFloat(), rng.nextFloat(),
                                 rng.nextFloat());
#if !defined(__CUDACC__)
            if (bs.pdf > 0.0f && guideReady && !bs.specular) {
                const float pg = guiding->guideProbability();
                const float gPdf = guiding->pdf(normalize(frame.toWorld(bs.wi)));
                const float mixPdf = pg * gPdf + (1.0f - pg) * bs.pdf;
                if (mixPdf > 0.0f) {
                    bs.weight *= bs.pdf / mixPdf;
                    bs.pdf = mixPdf;
                }
            }
#endif
        }
        if (bs.pdf <= 0.0f || isBlack(bs.weight)) break;

        // Track the MNEE-covered family: non-specular vertex → pure delta
        // transmissive chain afterwards.
        const bool deltaTransmit = bs.specular && bs.transmitted && mnee::isCausticCaster(mat);
        const bool deltaReflectInsideChain = bs.specular && !bs.transmitted && mnee::isCausticCaster(mat);
        if (!bs.specular) {
            sawNonSpecular = true;
            mneeFamily = false;
            familyChainLen = 0;
            anchorP = si.p;
            anchorN = si.ns;
            anchorDir = normalize(frame.toWorld(bs.wi));
        } else if (sawNonSpecular && settings.caustics != 0) {
            if (deltaTransmit) {
                ++familyChainLen;
                // The solver walks pure refraction chains up to kMaxChain interfaces;
                // longer chains and any reflection (incl. TIR) stay with BSDF sampling.
                mneeFamily = familyChainLen <= mnee::kMaxChain;
            } else {
                mneeFamily = false;
                familyChainLen = 0;
            }
        }
        // Caustics disabled: suppress all diffuse→specular→light transport.
        if (settings.caustics == 0 && bs.specular && sawNonSpecular) mneeFamily = true;

        Vec3 weight = bs.weight;
        if (settings.clampIndirect > 0.0f) {
            const float m = maxComponent(weight);
            if (m > settings.clampIndirect) weight *= settings.clampIndirect / m;
        }

        const Vec3 wiWorld = normalize(frame.toWorld(bs.wi));
#if !defined(__CUDACC__)
        if (guiding && guiding->active())
            guiding->recordBounce(si.ns, wiWorld, bs.pdf, weight, bs.specular, mat.roughness, lw.eta, 1.0f);
#endif

        throughput *= weight;
        if (!isFinite(throughput) || isBlack(throughput)) break;

        origin = offsetRayOrigin(si.p, si.ng, wiWorld);
        direction = wiWorld;
        bsdfPdf = bs.pdf;
        specularBounce = bs.specular;
        ++depth;

        if (depth >= srMax(1, settings.rrStartDepth)) {
            const float q = clampf(maxComponent(throughput), 0.05f, 1.0f);
            if (rng.nextFloat() > q) break;
#if !defined(__CUDACC__)
            if (guiding && guiding->active()) guiding->setRussianRoulette(q);
#endif
            throughput /= q;
        }
    }

    if (!isFinite(radiance)) return Vec3(0.0f);
    return radiance;
}

template <typename Tracer>
SR_INL Vec3 traceRadiancePtMnee(const SceneView& scene, const Tracer& tracer, Vec3 origin, Vec3 direction,
                                Rng& rng) {
    return traceRadiancePtMnee<Tracer, NullGuiding>(scene, tracer, origin, direction, rng, nullptr);
}

}  // namespace sol
