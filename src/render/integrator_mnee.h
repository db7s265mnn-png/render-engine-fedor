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
// To avoid double counting, diffuse→(delta glass chain)→finite-light BSDF hits
// share energy with MNEE via power-heuristic MIS (same seed set). When the photon
// map owns caustics, those BSDF caustic hits are suppressed entirely.
// Reflective caustics stay with BSDF sampling; environment light stays with
// regular NEE/MIS (no dome MNEE).
#pragma once

#include "core/rng.h"
#include "render/integrator.h"
#include "render/lights.h"
#include "render/photon_map.h"
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


SR_INL bool isCausticCaster(const Material& m) { return isDeltaCausticCaster(m); }

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
    fresnel = fresnelDielectric(cosI, 1.0f / srMax(1e-6f, eta));
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
                             int targetLight, DispersionContext* dispersion) {
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
        Material mat = materialForCausticTransport(scene, si.materialIndex);
        mat = evaluateTexturedMaterial(scene, mat, si.uv, si.ns, si.pObject, si.nObject, si.uvFilterWidth, si.pRef, si.nRef, si.hasPref);
        applyDispersion(mat, dispersion);
        if (!isCausticCaster(mat)) return st;  // opaque blocker → fail
        if (k == kMaxChain) return st;

        Vec3 dNext;
        float etaRel = 1.0f, fr = 0.0f;
        const LobeWeights lw = computeLobes(mat, Frame(si.ns).toLocal(-d));
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
                       ChainState* outChain, DispersionContext* dispersion) {
    const ChainState st = traceChain(scene, tracer, p, n, dir, targetLight, dispersion);
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
                                      int lightIndex, Vec3 y, Vec3 omegaInit, DispersionContext* dispersion) {
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

    if (!chainError(scene, tracer, p, n, omega, y, lightIndex, planeN, b1, b2, e1, e2, &chain, dispersion)) return sol;
    float err = sqrtf(e1 * e1 + e2 * e2);
    bool converged = err < tol;

    for (int iter = 0; iter < kNewtonIters && !converged; ++iter) {
        const Frame dirFrame(omega);
        const float h = 1e-3f;
        float p1 = 0, p2 = 0, q1 = 0, q2 = 0;
        const Vec3 omU = normalize(omega + dirFrame.t * h);
        const Vec3 omV = normalize(omega + dirFrame.b * h);
        if (!chainError(scene, tracer, p, n, omU, y, lightIndex, planeN, b1, b2, p1, p2, nullptr, dispersion))
            return sol;
        if (!chainError(scene, tracer, p, n, omV, y, lightIndex, planeN, b1, b2, q1, q2, nullptr, dispersion))
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
            if (!chainError(scene, tracer, p, n, cand, y, lightIndex, planeN, b1, b2, c1, c2, &candChain, dispersion))
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
        if (!chainError(scene, tracer, p, n, omUp, y, lightIndex, planeN, b1, b2, a1, a2, nullptr, dispersion))
            return sol;
        if (!chainError(scene, tracer, p, n, omUm, y, lightIndex, planeN, b1, b2, c1, c2, nullptr, dispersion))
            return sol;
        j11 = (a1 - c1) / (2.0f * h);
        j21 = (a2 - c2) / (2.0f * h);
        if (!chainError(scene, tracer, p, n, omVp, y, lightIndex, planeN, b1, b2, a1, a2, nullptr, dispersion))
            return sol;
        if (!chainError(scene, tracer, p, n, omVm, y, lightIndex, planeN, b1, b2, c1, c2, nullptr, dispersion))
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

// BSDF density at the anchor mapped to light-surface area through a manifold
// solution (dω → dA_y via the chain Jacobian). Used for MIS between MNEE and
// the BSDF-sampled copy of the same caustic family.
SR_INL float bsdfPdfOnLightArea(const Material& anchorMat, Vec3 anchorN, Vec3 anchorWo, Vec3 omega, Vec3 yN,
                                const ManifoldSolution& sol, bool pointLight) {
    const Frame frame(anchorN);
    const BsdfEval e = bsdfEvalLocal(anchorMat, frame.toLocal(anchorWo), frame.toLocal(omega));
    if (!(e.pdf > 0.0f) || !srIsFinite(e.pdf)) return 0.0f;
    const float planeToLight = pointLight ? 1.0f : fabsf(dot(yN, sol.planeN));
    return e.pdf * planeToLight / srMax(1e-12f, sol.detJ);
}

// Multi-branch MNEE with MIS against the BSDF-sampled copies (power heuristic in
// light-area measure, chain Jacobian from the solver). Where the solver cannot
// find a branch the BSDF copy keeps full weight in the path loop below, so no
// energy is lost on geometry too hard for Newton walks (complex glass).
template <typename Tracer>
SR_INL MneeResult manifoldConnect(const SceneView& scene, const Tracer& tracer, Vec3 p, Vec3 n, Vec3 wo,
                                  const Material& shadeMat, int lightIndex, Vec3 y, Vec3 yN, Vec3 Le,
                                  float pdfArea, float selectPdf, int casterInstance, DispersionContext* dispersion) {
    MneeResult res;
    Vec3 dir = y - p;
    const float distPy = length(dir);
    if (distPy < 1e-5f) return res;
    dir = dir / distPy;

    const bool pointLight = scene.lights[lightIndex].type == kLightPoint;
    const float pM = pdfArea * selectPdf;
    const SeedSet seeds = buildSeedSet(scene, p, dir, casterInstance);
    Vec3 found[1 + kSeedRing];
    int foundCount = 0;
    Vec3 total(0.0f);
    for (int i = 0; i < seeds.count; ++i) {
        const ManifoldSolution sol = solveManifold(scene, tracer, p, n, lightIndex, y, seeds.dirs[i], dispersion);
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
        Vec3 c = solutionContribution(scene, tracer, sol, p, n, wo, shadeMat, lightIndex, y, yN, Le, pdfArea,
                                      selectPdf);
        if (isBlack(c)) continue;
        // Power-heuristic MIS vs the BSDF copy (delta lights are BSDF-unreachable).
        if (!pointLight) {
            const float pB = bsdfPdfOnLightArea(shadeMat, n, wo, sol.omega, yN, sol, pointLight);
            const float w = (pM * pM) / srMax(1e-20f, pM * pM + pB * pB);
            c = c * w;
        }
        total += c;
    }
    res.contribution = total;
    return res;
}

struct BranchMatch {
    bool matched = false;
    ManifoldSolution sol;
};

// Replays the seed set for the anchor→light-point pair and returns the solution
// matching the branch a BSDF path actually took (if the solver can find it).
template <typename Tracer>
SR_INL BranchMatch matchBranch(const SceneView& scene, const Tracer& tracer, Vec3 p, Vec3 n, int lightIndex,
                               Vec3 y, int casterInstance, Vec3 pathDir, DispersionContext* dispersion) {
    BranchMatch out;
    Vec3 dir = y - p;
    const float distPy = length(dir);
    if (distPy < 1e-5f) return out;
    dir = dir / distPy;
    const SeedSet seeds = buildSeedSet(scene, p, dir, casterInstance);
    for (int i = 0; i < seeds.count; ++i) {
        const ManifoldSolution sol = solveManifold(scene, tracer, p, n, lightIndex, y, seeds.dirs[i], dispersion);
        if (sol.solved && sameBranch(sol.omega, pathDir)) {
            out.matched = true;
            out.sol = sol;
            return out;
        }
    }
    return out;
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
                                Rng& rng, Guiding* guiding, DispersionContext* dispersion = nullptr,
                                const CausticPhotonMap* photons = nullptr) {
    Vec3 radiance(0.0f);
    Vec3 throughput(1.0f);
    float bsdfPdf = 0.0f;
    bool specularBounce = true;
    // True while the path suffix after the last non-specular vertex is a pure
    // delta-transmissive chain — the family MNEE covers for finite lights.
    bool sawNonSpecular = false;
    bool mneeFamily = false;
    bool photonFamily = false;  // diffuse → photon-caster → … (owned by photon map)
    int familyChainLen = 0;
    Vec3 anchorP(0.0f);  // last non-specular surface vertex (MNEE launch point)
    Vec3 anchorN(0.0f, 1.0f, 0.0f);
    Vec3 anchorDir(0.0f, 1.0f, 0.0f);  // direction the path left the anchor with
    Vec3 anchorWo(0.0f, 1.0f, 0.0f);   // outgoing (toward camera) at the anchor
    Material anchorMat{};
    int depth = 0;
    int passThrough = 0;
    RayShadeKind rayKind = RayShadeKind::Camera;
    const RenderSettingsData& settings = scene.settings;
    const int maxDepth = srMax(1, settings.maxDepth);
    const bool photonEngine = photons != nullptr;
    const bool photonCaustics = photonEngine && !photons->empty();
    const float photonRadius = photonCaustics ? photons->gatherRadius(settings) : 0.0f;

    while (depth <= maxDepth) {
        RayHit hit;
        const bool didHit = tracer.intersect(origin, direction, kFloatMax, hit);

        if (!didHit) {
            if (scene.domeLightIndex >= 0) {
                const LightData& dome = scene.lights[scene.domeLightIndex];
                // Per-light caustics off: skip env after a diffuse→specular suffix.
                if (!(mneeFamily && !lightContributesCaustics(dome))) {
                const bool primary = depth == 0 && passThrough == 0;
                if (!(primary && (!settings.envVisibleCamera || !dome.visibleCamera))) {
                    Vec3 envL = domeRadiance(scene, dome, direction, /*nearestTexel=*/depth > 0);
                    if (!isBlack(envL)) {
                        float weight = 1.0f;
                        if (!specularBounce) {
                            const float lp = lightPdfDirection(scene, scene.domeLightIndex, origin, direction,
                                                               origin, direction) *
                                             lightSelectionPdfIndex(scene, origin, scene.domeLightIndex);
                            weight = powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                        }
                        Vec3 contrib = throughput * envL * weight;
                        if (depth > 0) contrib = clampContribution(contrib, settings.clampDirect);
                        radiance += contrib;
#if !defined(__CUDACC__)
                        if (guiding && guiding->active())
                            guiding->recordBackground(origin, direction, envL, weight);
#endif
                    }
                }
                }
            }
            {
                const bool primarySun = depth == 0 && passThrough == 0;
                if (!(primarySun && !settings.envVisibleCamera)) {
                    const Vec3 sunL =
                        cameraSunDiscRadiance(scene, origin, direction, bsdfPdf, specularBounce,
                                              primarySun, mneeFamily);
                    if (!isBlack(sunL)) {
                        Vec3 contrib = throughput * sunL;
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
            continue;
        }

        if (si.lightIndex >= 0) {
            // Photon map owns the caustic family (light→spec→diffuse and SDS). Do not
            // also accept eye→spec→light BSDF hits or apply MNEE MIS against a disabled
            // technique — that double-counted and biased the power heuristic.
            const LightData& light = scene.lights[si.lightIndex];
            const bool finiteLight = light.type == kLightRect || light.type == kLightDisk ||
                                     light.type == kLightSphere || light.type == kLightPoint;
            // Photon map owns the caustic family even when this pass deposited
            // zero photons (e.g. all casters have Contribute to Caustics off).
            if (photonEngine && photonFamily && finiteLight) break;
            if (photonEngine && sawNonSpecular && specularBounce && finiteLight &&
                lightContributesCaustics(light))
                break;
            if (photonEngine && mneeFamily && finiteLight) break;

            // MIS with the MNEE estimator: when replaying the seed set converges to
            // the branch this BSDF path took, the two estimators sample the same
            // family and the contribution gets the power-heuristic weight (in
            // light-area measure via the chain Jacobian). If the solver cannot find
            // the branch (complex glass), full weight stays here — no energy loss.
            float misScale = 1.0f;
            if (!photonEngine && mneeFamily && finiteLight && settings.caustics != 0 &&
                lightContributesCaustics(light)) {
                Vec3 seg = si.p - anchorP;
                const float segLen = length(seg);
                if (segLen > 1e-5f) {
                    seg = seg / segLen;
                    const Vec3 o = offsetRayOrigin(anchorP, anchorN, seg);
                    RayHit sh;
                    if (tracer.intersect(o, seg, segLen * (1.0f - 1e-3f), sh)) {
                        SurfaceInteraction ssi;
                        if (buildSurfaceInteraction(scene, sh, o, seg, ssi) && ssi.lightIndex < 0) {
                            Material smat = materialForCausticTransport(scene, ssi.materialIndex);
                            smat = evaluateTexturedMaterial(scene, smat, ssi.uv, ssi.ns, ssi.pObject,
                                                            ssi.nObject, ssi.uvFilterWidth,
                                                            ssi.pRef, ssi.nRef, ssi.hasPref);
                            if (mnee::isCausticCaster(smat)) {
                                const mnee::BranchMatch bm =
                                    mnee::matchBranch(scene, tracer, anchorP, anchorN, si.lightIndex, si.p,
                                                      ssi.instanceIndex, anchorDir, dispersion);
                                if (bm.matched) {
                                    const Vec3 lightNHit =
                                        light.type == kLightSphere ? si.ng : areaLightNormal(light);
                                    float area = 1.0f;
                                    if (light.type == kLightRect) area = rectLightArea(light);
                                    else if (light.type == kLightDisk) area = diskLightArea(light);
                                    else if (light.type == kLightSphere) {
                                        const float r = sphereLightRadius(light);
                                        area = 4.0f * kPi * r * r;
                                    }
                                    const float pM = lightSelectionPdfIndex(scene, anchorP, si.lightIndex) /
                                                     srMax(1e-12f, area);
                                    const float pB = mnee::bsdfPdfOnLightArea(
                                        anchorMat, anchorN, anchorWo, anchorDir, lightNHit, bm.sol, false);
                                    misScale = (pB * pB) / srMax(1e-20f, pB * pB + pM * pM);
                                }
                            }
                        }
                    }
                }
            }
            if (mneeFamily && finiteLight &&
                (settings.caustics == 0 || !lightContributesCaustics(light)))
                break;

            const Vec3 lightN = light.type == kLightSphere ? si.ng : areaLightNormal(light);
            Vec3 emitted = areaLightEmission(scene, light, direction, lightN);
            if (!isBlack(emitted)) {
                float weight = misScale;
                if (!specularBounce) {
                    const float lp = lightPdfDirection(scene, si.lightIndex, origin, direction, si.p, lightN) *
                                     lightSelectionPdfIndex(scene, origin, si.lightIndex);
                    weight *= powerHeuristic(1.0f, bsdfPdf, 1.0f, lp);
                }
                Vec3 contrib = throughput * emitted * weight;
                if (depth > 0 && !specularBounce)
                    contrib = clampContribution(contrib, settings.clampDirect);
                radiance += contrib;
#if !defined(__CUDACC__)
                if (guiding && guiding->active())
                    guiding->recordLightHit(si.p, -direction, emitted, weight);
#endif
            }
            break;
        }

        // Arnold ray_switch by incoming ray type (camera rays → camera port only).
        Material baseMat = materialForRay(scene, si.materialIndex, rayKind);
        Material mat =
            evaluateTexturedMaterial(scene, baseMat, si.uv, si.ns, si.pObject, si.nObject, si.uvFilterWidth, si.pRef, si.nRef, si.hasPref);
        applyDispersion(mat, dispersion);
        // Glass classification for MNEE / caustic family uses caustic-transport ports.
        Material matCau = materialForCausticTransport(scene, si.materialIndex);
        matCau = evaluateTexturedMaterial(scene, matCau, si.uv, si.ns, si.pObject, si.nObject,
                                          si.uvFilterWidth, si.pRef, si.nRef, si.hasPref);
        // No applyDispersion(matCau): camera-port Abbe must not mark shadow/LT paths.

        if (mat.transmission <= 0.0f && mat.doubleSided && dot(si.ns, -direction) < 0.0f) {
            si.ns = -si.ns;
            si.ng = -si.ng;
        }

        if (mat.emissionStrength > 0.0f && !isBlack(mat.emissionColor)) {
            const bool frontFacing = dot(si.ns, -direction) > 0.0f;
            if (frontFacing || mat.doubleSided)
                radiance += throughput * mat.emissionColor * mat.emissionStrength;
        }

        if (mat.opacity <= 1e-6f || (mat.opacity < 0.999f && rng.nextFloat() > mat.opacity)) {
            origin = offsetRayOrigin(si.p, si.ng, direction);
            ++passThrough;
            continue;
        }

        if (depth >= maxDepth) break;

        const Vec3 wo = -direction;
        const Frame frame(si.ns);
        const Vec3 woLocal = frame.toLocal(wo);
        const LobeWeights lw = computeLobes(mat, woLocal);

        // Arnold / Autodesk Standard Surface base mix — same lottery as PathIntegrator.
        // PathMnee previously skipped SSS entirely whenever caustics were on.
        const float sssWeight = saturatef(mat.subsurface);
        if (materialSupportsSss(mat) && rng.nextFloat() < sssWeight) {
            Material specMat = sssSpecularEntryMaterial(mat);
            const Vec3 woLocalEntry = frame.toLocal(wo);
            const LobeWeights specLw = computeLobes(specMat, woLocalEntry);
            const float pSpec = sssEntrySpecularProb(specMat, woLocalEntry);

            if (pSpec > 0.0f) {
                const Vec3 nee =
                    nextEventEstimation(scene, tracer, si, specMat, frame, wo, rng, guiding, -1,
                                        depth > 0 ? 1 : 0);
                Vec3 contrib = throughput * nee;
                if (depth > 0) contrib = clampContribution(contrib, settings.clampDirect);
                radiance += contrib;
#if !defined(__CUDACC__)
                if (guiding && guiding->active()) guiding->addScattered(nee);
#endif
            }

            if (pSpec > 0.0f && rng.nextFloat() < pSpec) {
                throughput /= pSpec;
                const float uSpec = specLw.diffuse + specLw.specular * rng.nextFloat();
                const BsdfSample specBs =
                    bsdfSampleLocal(specMat, woLocalEntry, uSpec, rng.nextFloat(), rng.nextFloat(),
                                    rng.nextFloat());
                if (specBs.pdf > 0.0f && !isBlack(specBs.weight)) {
                    const Vec3 wiWorld = normalize(frame.toWorld(specBs.wi));
#if !defined(__CUDACC__)
                    if (guiding && guiding->active())
                        guiding->recordBounce(si.ns, wiWorld, specBs.pdf, specBs.weight, true,
                                              mat.roughness, computeLobes(specMat).eta, 1.0f);
#endif
                    throughput *= specBs.weight;
                    origin = offsetRayOrigin(si.p, si.ng, wiWorld);
                    direction = wiWorld;
                    bsdfPdf = specBs.pdf;
                    specularBounce = specBs.specular;
                    rayKind = nextRayShadeKind(specBs, specLw);
                    ++depth;
                    continue;
                }
                break;
            }
            if (pSpec > 0.0f && pSpec < 0.999f) throughput /= (1.0f - pSpec);

            const SssWalkResult walk = sampleSssRandomWalk(scene, tracer, si, wo, mat, rng);
            if (!walk.escaped || isBlack(walk.pathWeight) || !isFinite(walk.pathWeight)) break;
            Material lambert = sssExitLambertMaterial();
            SurfaceInteraction ssSi = si;
            ssSi.p = walk.exitP;
            ssSi.ns = walk.exitN;
            ssSi.ng = walk.exitN;
            const Frame ssFrame(walk.exitN);
            const Vec3 nee =
                nextEventEstimation(scene, tracer, ssSi, lambert, ssFrame, walk.exitWo, rng, guiding, -1,
                                    depth > 0 ? 1 : 0);
            Vec3 contrib = throughput * walk.pathWeight * nee;
            if (depth > 0) contrib = clampContribution(contrib, settings.clampDirect);
            radiance += contrib;
#if !defined(__CUDACC__)
            if (guiding && guiding->active()) guiding->addScattered(walk.pathWeight * nee);
#endif
            const BsdfSample ssBs =
                bsdfSampleLocal(lambert, ssFrame.toLocal(walk.exitWo), rng.nextFloat(), rng.nextFloat(),
                                rng.nextFloat(), rng.nextFloat());
            if (ssBs.pdf > 0.0f && !isBlack(ssBs.weight)) {
                const Vec3 wiWorld = normalize(ssFrame.toWorld(ssBs.wi));
#if !defined(__CUDACC__)
                if (guiding && guiding->active())
                    guiding->recordBounce(walk.exitN, wiWorld, ssBs.pdf, walk.pathWeight * ssBs.weight,
                                          false, 1.0f, 1.0f, 1.0f);
#endif
                throughput *= walk.pathWeight * ssBs.weight;
                origin = offsetRayOrigin(walk.exitP, walk.exitN, wiWorld);
                direction = wiWorld;
                bsdfPdf = ssBs.pdf;
                specularBounce = false;
                rayKind = RayShadeKind::DiffuseReflection;
                sawNonSpecular = true;
                mneeFamily = false;
                photonFamily = false;
                ++depth;
                continue;
            }
            break;
        }

#if !defined(__CUDACC__)
        bool guideReady = false;
        bool trainGuide = false;
        if (guiding && guiding->active()) {
            // Always open a segment so recordBounce cannot overwrite a prior
            // diffuse vertex when we hit delta glass (was corrupting OpenPGL chains).
            guiding->beginSegment(si.p, wo);
            // Do not guide-sample specular / near-specular / caustic casters.
            // Still train at diffuse receivers (including caustic energy below).
            trainGuide = !lw.delta && !isNearSpecularLobe(lw) && lw.diffuse > 1e-4f;
            if (trainGuide) guideReady = guiding->prepare(si.p, si.ns, rng);
        }
#else
        const bool guideReady = false;
        const bool trainGuide = false;
        (void)guiding;
#endif

        // --- NEE with lazy MNEE upgrade -------------------------------------
        const bool connectable = lw.diffuse > 1e-4f || !lw.delta;
        if (connectable) {
            if (photonCaustics) {
                Vec3 g = photons->gather(si.p, si.ns, wo, mat, photonRadius);
                if (!isBlack(g) && isFinite(g)) {
                    Vec3 contrib = throughput * g;
                    if (depth > 0 && !specularBounce)
                        contrib = clampContribution(contrib, settings.clampDirect);
                    radiance += contrib;
#if !defined(__CUDACC__)
                    // Photons are the caustic estimator — teach the guide at this
                    // diffuse receiver so later eye samples favour bright regions.
                    if (trainGuide) guiding->addScattered(g);
#endif
                }
            }
            (void)settings.lightSamples;
            const int nLightSamples = 1;  // pbrt-v4: one light sample per vertex, MIS with BSDF
            Vec3 neeSum(0.0f);
            Vec3 neeSumGuide(0.0f);  // clear-path + MNEE at diffuse receivers
            for (int ls = 0; ls < nLightSamples; ++ls) {
                float selectPdf = 0.0f;
                const int li = sampleLightIndex(scene, si.p, rng.nextFloat(), selectPdf);
                if (li < 0 || selectPdf <= 0.0f) continue;
                const LightData& l = scene.lights[li];

                if (l.type == kLightDome || l.type == kLightDistant) {
                    LightSample lsam;
                    if (!sampleLight(scene, li, si.p, rng.nextFloat(), rng.nextFloat(), lsam)) continue;
                    if (lsam.pdf <= 0.0f || isBlack(lsam.radiance)) continue;
                    float visibility = 1.0f;
                    if (l.shadowEnable) {
                        const Vec3 o = offsetRayOrigin(si.p, si.ng, lsam.wi);
                        visibility = shadowVisibility(scene, tracer, o, lsam.wi, 1.0e8f,
                                                      depth > 0 ? 1 : 0);
                    }
                    if (visibility <= 1e-5f) continue;
                    if (!shadingNormalConsistent(si.ng, si.ns, wo, lsam.wi)) continue;
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
                    const Vec3 c = lsam.radiance * be.f * (fabsf(wiL.z) * w * visibility / lightPdf);
                    neeSum += c;
                    neeSumGuide += c;
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
                            } else if (bsi.lightIndex < 0 && settings.caustics != 0 &&
                                       lightContributesCaustics(l)) {
                                Material bmat = materialForCausticTransport(scene, bsi.materialIndex);
                                bmat = evaluateTexturedMaterial(scene, bmat, bsi.uv, bsi.ns, bsi.pObject,
                                                                bsi.nObject, bsi.uvFilterWidth,
                                                                bsi.pRef, bsi.nRef, bsi.hasPref);
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
                    if (!shadingNormalConsistent(si.ng, si.ns, wo, wi)) continue;
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
                        const Vec3 c = Le * be.f * (fabsf(wiLocal.z) / (dist2 * selectPdf));
                        neeSum += c;
                        neeSumGuide += c;
                    } else {
                        const float pdfSa = pdfArea * dist2 / cosL;  // area → solid angle
                        const float lightPdf = pdfSa * selectPdf;
                        const float w = powerHeuristic(1.0f, lightPdf, 1.0f, scatterPdf);
                        const Vec3 c = Le * be.f * (fabsf(wiLocal.z) * w / lightPdf);
                        neeSum += c;
                        neeSumGuide += c;
                    }
                } else if (glassPath && !photonEngine) {
                    // Multi-seed MNEE: manifold connections through the refraction
                    // chain (matching BSDF path copies are MIS'd at light hits).
                    // Skipped when the photon engine owns caustics.
                    const mnee::MneeResult mr =
                        mnee::manifoldConnect(scene, tracer, si.p, si.ns, wo, mat, li, y, yN, Le, pdfArea,
                                              selectPdf, blockerInstance, dispersion);
                    if (mr.solved && !isBlack(mr.contribution)) {
                        // Clamp once on the final pixel contribution below — not here
                        // (and not at clamp*4). Double application forced ~1e6 thresholds.
                        neeSum += mr.contribution;
                        // Train the diffuse receiver (floor under glass). Do not
                        // guide-sample on the glass itself — only learn incident
                        // caustic radiance here when Indirect Guides is on.
                        if (trainGuide) neeSumGuide += mr.contribution;
                    }
                }
            }
            const float invLs = 1.0f;
            const Vec3 nee = neeSum * invLs;
            Vec3 contrib = throughput * nee;
            if (depth > 0 && !specularBounce) contrib = clampContribution(contrib, settings.clampDirect);
            radiance += contrib;
#if !defined(__CUDACC__)
            if (guiding && guiding->active()) guiding->addScattered(neeSumGuide * invLs);
#endif
        }

        // --- BSDF continuation (guided mixture) ------------------------------
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

        // Track caustic families. MNEE covers delta transmission only; the photon
        // engine also owns rough refractive casters (isPhotonCausticCasterLobe).
        const bool deltaTransmit = bs.specular && bs.transmitted && mnee::isCausticCaster(matCau);
        const bool photonCasterBounce =
            isPhotonCausticCasterLobe(lw) &&
            (materialContributesCaustics(matCau) ||
             (bs.transmitted && matCau.transmission > 0.25f));  // non-contrib glass still suppresses
        if (!bs.specular && !isNearSpecularLobe(lw) && !isPhotonCausticCasterLobe(lw)) {
            sawNonSpecular = true;
            mneeFamily = false;
            photonFamily = false;
            familyChainLen = 0;
            anchorP = si.p;
            anchorN = si.ns;
            anchorDir = normalize(frame.toWorld(bs.wi));
            anchorWo = wo;
            anchorMat = mat;
        } else if (sawNonSpecular && settings.caustics != 0) {
            if (photonEngine && photonCasterBounce) {
                photonFamily = true;
            }
            if (deltaTransmit) {
                ++familyChainLen;
                mneeFamily = familyChainLen <= mnee::kMaxChain;
            } else {
                // Transmissive glass with Contribute to Caustics OFF is not an MNEE
                // caster, but floor→glass→light must still be suppressed — otherwise
                // disabling the flag leaks caustic-family energy via BSDF.
                const bool nonContribGlass =
                    bs.transmitted && matCau.transmission > 0.25f && !materialContributesCaustics(matCau);
                if (nonContribGlass) {
                    mneeFamily = true;
                    familyChainLen = 0;
                    if (photonEngine) photonFamily = true;
                } else if (!photonCasterBounce) {
                    mneeFamily = false;
                    familyChainLen = 0;
                }
            }
        }
        // Caustics disabled: suppress all diffuse→specular→light transport.
        if (settings.caustics == 0 && bs.specular && sawNonSpecular) mneeFamily = true;

        // Indirect Clamp is pixel-radiance only (see clampContribution) — do not
        // crush BSDF weights mid-path (that forced users toward clamp ≈ 1e6).
        const Vec3 weight = bs.weight;

        const Vec3 wiWorld = normalize(frame.toWorld(bs.wi));
        if (!shadingNormalConsistent(si.ng, si.ns, wo, wiWorld)) break;
#if !defined(__CUDACC__)
        // Record every bounce: delta/near-spec flagged so OpenPGL propagates
        // caustic radiance back to the diffuse receiver. Guide sampling stays
        // diffuse-only (see guideReady / guideable above).
        if (guiding && guiding->active()) {
            const bool deltaSeg = bs.specular || isNearSpecularLobe(lw);
            guiding->recordBounce(si.ns, wiWorld, bs.pdf, weight, deltaSeg, mat.roughness, lw.eta, 1.0f);
        }
#endif

        throughput *= weight;
        if (bs.transmitted) throughput = applyFakeDispersionThroughput(throughput, mat, dispersion);
        if (!isFinite(throughput) || isBlack(throughput)) break;

        origin = offsetRayOrigin(si.p, si.ng, wiWorld);
        direction = wiWorld;
        bsdfPdf = bs.pdf;
        specularBounce = bs.specular;
        rayKind = nextRayShadeKind(bs, lw);
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
                                Rng& rng, DispersionContext* dispersion = nullptr, const CausticPhotonMap* photons = nullptr) {
    return traceRadiancePtMnee<Tracer, NullGuiding>(scene, tracer, origin, direction, rng, nullptr,
                                                    dispersion, photons);
}

}  // namespace sol
