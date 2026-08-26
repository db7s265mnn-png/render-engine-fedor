// Eye-path MNEE for OptiX shade_surface (Hanika et al. 2015, same 2D Newton as
// CPU integrator_mnee.h). Shade kernels otherwise never call optixTrace; MNEE
// is the exception — each probe save/restores GpuHit because CH writes hits[pixel].
// Distant lights use an angular residual (CPU MNEE is finite-light only).
#pragma once

#include "render/lights.h"
#include "render/optix/optix_bsdf.cuh"
#include "render/optix/optix_geom.cuh"
#include "render/optix/optix_spectral.cuh"
#include "render/optix/optix_trace.cuh"
#include "render/shading_bsdf.h"

namespace sol {

constexpr int kGpuMneeMaxChain = 6;
constexpr int kGpuMneeNewtonIters = 12;
constexpr int kGpuMneeBacktrack = 4;
constexpr int kGpuMneeSeedRing = 4;
constexpr float kGpuMneeSeedRingRadius = 0.75f;

__device__ inline GpuHit gpuTraceHit(int pixel, Vec3 origin, Vec3 direction, float tMax) {
    LaunchParams& params = launchParamsMutable();
    const GpuHit saved = params.hits[pixel];
    traceClosest(pixel, origin, direction, tMax);
    const GpuHit hit = params.hits[pixel];
    params.hits[pixel] = saved;
    return hit;
}

__device__ inline bool gpuTraceSurf(int pixel, const SceneView& scene, Vec3 origin, Vec3 direction,
                                    float tMax, Surf& si) {
    const GpuHit hit = gpuTraceHit(pixel, origin, direction, tMax);
    if (!hit.didHit) return false;
    return buildSurf(scene, hit, origin, direction, si);
}

__device__ inline bool gpuRefractTravel(Vec3 d, Vec3 n, float ior, Vec3& out, float& etaRel,
                                        float& fresnel) {
    float cosI = -dot(d, n);
    Vec3 nn = n;
    float eta = 1.0f / ior;
    if (cosI < 0.0f) {
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

struct GpuChainState {
    Vec3 exitP{0.0f};
    Vec3 exitDir{0.0f};
    Vec3 exitN{0.0f};
    Vec3 throughput{1.0f};
    int interfaces = 0;
    int valid = 0;
};

__device__ inline Material gpuMneeCasterMat(const SceneView& scene, const Surf& si, const GpuPath& path,
                                            Vec3& ns) {
    ns = si.ns;
    Material mat = gpuMaterialForCausticSlot(scene, si.materialIndex);
    mat = optixpt::evaluateMaps(scene, mat, si.uv, ns);
    if (path.nLambda > 0)
        mat.ior = specDielectricIor(mat.ior, mat.dispersionAbbe, path.lambda[0]);
    return mat;
}

__device__ inline GpuChainState gpuTraceChain(int pixel, const SceneView& scene, const GpuPath& path,
                                              Vec3 p, Vec3 n, Vec3 dir, int targetLight) {
    GpuChainState st;
    Vec3 o = offsetRay(p, n, dir);
    Vec3 d = dir;
    Vec3 T(1.0f);
    Vec3 lastP = p;
    Vec3 lastN = n;
    for (int k = 0; k <= kGpuMneeMaxChain; ++k) {
        Surf si;
        if (!gpuTraceSurf(pixel, scene, o, d, kFloatMax, si)) {
            st.exitP = lastP;
            st.exitDir = d;
            st.exitN = lastN;
            st.throughput = T;
            st.interfaces = k;
            st.valid = k > 0 ? 1 : 0;
            return st;
        }
        if (si.lightIndex >= 0) {
            if (si.lightIndex == targetLight || targetLight < 0) {
                st.exitP = lastP;
                st.exitDir = d;
                st.exitN = lastN;
                st.throughput = T;
                st.interfaces = k;
                st.valid = k > 0 ? 1 : 0;
            }
            return st;
        }
        Vec3 ns = si.ns;
        const Material mat = gpuMneeCasterMat(scene, si, path, ns);
        if (!isDeltaCausticCaster(mat)) return st;
        if (k == kGpuMneeMaxChain) return st;
        Vec3 dNext;
        float etaRel = 1.0f, fr = 0.0f;
        const LobeWeights lw = computeLobes(mat, Frame(ns).toLocal(-d));
        if (!gpuRefractTravel(d, ns, lw.eta, dNext, etaRel, fr)) return st;
        T = T * lw.transmissionTint * ((1.0f - fr) / srMax(1e-4f, etaRel * etaRel));
        if (isBlack(T)) return st;
        lastP = si.p;
        lastN = ns;
        o = offsetRay(si.p, si.ng, dNext);
        d = dNext;
    }
    return st;
}

__device__ inline bool gpuChainPlaneError(int pixel, const SceneView& scene, const GpuPath& path, Vec3 p,
                                          Vec3 n, Vec3 dir, Vec3 y, int targetLight, Vec3 planeN, Vec3 b1,
                                          Vec3 b2, float& e1, float& e2, GpuChainState* outChain) {
    const GpuChainState st = gpuTraceChain(pixel, scene, path, p, n, dir, targetLight);
    if (!st.valid) return false;
    const float denom = dot(st.exitDir, planeN);
    if (denom <= 1e-5f) return false;
    const float t = dot(y - st.exitP, planeN) / denom;
    if (t <= 1e-6f) return false;
    const Vec3 hit = st.exitP + st.exitDir * t;
    const Vec3 e = hit - y;
    e1 = dot(e, b1);
    e2 = dot(e, b2);
    if (outChain) *outChain = st;
    return true;
}

__device__ inline bool gpuChainAngularError(int pixel, const SceneView& scene, const GpuPath& path, Vec3 p,
                                            Vec3 n, Vec3 dir, Vec3 lightDir, Vec3 b1, Vec3 b2, float& e1,
                                            float& e2, GpuChainState* outChain) {
    const GpuChainState st = gpuTraceChain(pixel, scene, path, p, n, dir, -1);
    if (!st.valid) return false;
    if (dot(st.exitDir, lightDir) < 0.2f) return false;
    e1 = dot(st.exitDir, b1);
    e2 = dot(st.exitDir, b2);
    if (outChain) *outChain = st;
    return true;
}

struct GpuManifoldSolution {
    Vec3 omega{0.0f};
    GpuChainState chain;
    float detJ = 0.0f;
    Vec3 planeN{0.0f};
    int solved = 0;
    int distant = 0;
};

__device__ inline GpuManifoldSolution gpuSolvePlane(int pixel, const SceneView& scene, const GpuPath& path,
                                                    Vec3 p, Vec3 n, int lightIndex, Vec3 y, Vec3 omegaInit) {
    GpuManifoldSolution sol;
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
    GpuChainState chain;
    float j11 = 0, j12 = 0, j21 = 0, j22 = 0;
    if (!gpuChainPlaneError(pixel, scene, path, p, n, omega, y, lightIndex, planeN, b1, b2, e1, e2, &chain))
        return sol;
    float err = sqrtf(e1 * e1 + e2 * e2);
    bool converged = err < tol;
    for (int iter = 0; iter < kGpuMneeNewtonIters && !converged; ++iter) {
        const Frame dirFrame(omega);
        const float h = 1e-3f;
        float p1 = 0, p2 = 0, q1 = 0, q2 = 0;
        if (!gpuChainPlaneError(pixel, scene, path, p, n, normalize(omega + dirFrame.t * h), y, lightIndex,
                                planeN, b1, b2, p1, p2, nullptr))
            return sol;
        if (!gpuChainPlaneError(pixel, scene, path, p, n, normalize(omega + dirFrame.b * h), y, lightIndex,
                                planeN, b1, b2, q1, q2, nullptr))
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
        for (int k = 0; k < kGpuMneeBacktrack; ++k, scale *= 0.5f) {
            const Vec3 cand = normalize(omega - (dirFrame.t * du + dirFrame.b * dv) * scale);
            if (dot(cand, dir) < -0.1f) continue;
            float c1 = 0.0f, c2 = 0.0f;
            GpuChainState candChain;
            if (!gpuChainPlaneError(pixel, scene, path, p, n, cand, y, lightIndex, planeN, b1, b2, c1, c2,
                                    &candChain))
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
        if (!accepted) return sol;
        converged = err < tol;
    }
    if (!converged) return sol;
    {
        const Frame dirFrame(omega);
        const float h = 2e-4f;
        float a1 = 0, a2 = 0, c1 = 0, c2 = 0;
        if (!gpuChainPlaneError(pixel, scene, path, p, n, normalize(omega + dirFrame.t * h), y, lightIndex,
                                planeN, b1, b2, a1, a2, nullptr) ||
            !gpuChainPlaneError(pixel, scene, path, p, n, normalize(omega - dirFrame.t * h), y, lightIndex,
                                planeN, b1, b2, c1, c2, nullptr))
            return sol;
        j11 = (a1 - c1) / (2.0f * h);
        j21 = (a2 - c2) / (2.0f * h);
        if (!gpuChainPlaneError(pixel, scene, path, p, n, normalize(omega + dirFrame.b * h), y, lightIndex,
                                planeN, b1, b2, a1, a2, nullptr) ||
            !gpuChainPlaneError(pixel, scene, path, p, n, normalize(omega - dirFrame.b * h), y, lightIndex,
                                planeN, b1, b2, c1, c2, nullptr))
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
    sol.solved = 1;
    return sol;
}

__device__ inline GpuManifoldSolution gpuSolveAngular(int pixel, const SceneView& scene, const GpuPath& path,
                                                      Vec3 p, Vec3 n, Vec3 lightDir, Vec3 omegaInit) {
    GpuManifoldSolution sol;
    sol.distant = 1;
    const Vec3 L = normalize(lightDir);
    const Frame errFrame(L);
    const Vec3 b1 = errFrame.t;
    const Vec3 b2 = errFrame.b;
    Vec3 omega = normalize(omegaInit);
    const float tol = 2e-3f;
    float e1 = 0.0f, e2 = 0.0f;
    GpuChainState chain;
    float j11 = 0, j12 = 0, j21 = 0, j22 = 0;
    if (!gpuChainAngularError(pixel, scene, path, p, n, omega, L, b1, b2, e1, e2, &chain)) return sol;
    float err = sqrtf(e1 * e1 + e2 * e2);
    bool converged = err < tol;
    for (int iter = 0; iter < kGpuMneeNewtonIters && !converged; ++iter) {
        const Frame dirFrame(omega);
        const float h = 1e-3f;
        float p1 = 0, p2 = 0, q1 = 0, q2 = 0;
        if (!gpuChainAngularError(pixel, scene, path, p, n, normalize(omega + dirFrame.t * h), L, b1, b2, p1,
                                  p2, nullptr) ||
            !gpuChainAngularError(pixel, scene, path, p, n, normalize(omega + dirFrame.b * h), L, b1, b2, q1,
                                  q2, nullptr))
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
        for (int k = 0; k < kGpuMneeBacktrack; ++k, scale *= 0.5f) {
            const Vec3 cand = normalize(omega - (dirFrame.t * du + dirFrame.b * dv) * scale);
            if (dot(cand, L) < -0.1f) continue;
            float c1 = 0.0f, c2 = 0.0f;
            GpuChainState candChain;
            if (!gpuChainAngularError(pixel, scene, path, p, n, cand, L, b1, b2, c1, c2, &candChain))
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
        if (!accepted) return sol;
        converged = err < tol;
    }
    if (!converged) return sol;
    {
        const Frame dirFrame(omega);
        const float h = 2e-4f;
        float a1 = 0, a2 = 0, c1 = 0, c2 = 0;
        if (!gpuChainAngularError(pixel, scene, path, p, n, normalize(omega + dirFrame.t * h), L, b1, b2, a1,
                                  a2, nullptr) ||
            !gpuChainAngularError(pixel, scene, path, p, n, normalize(omega - dirFrame.t * h), L, b1, b2, c1,
                                  c2, nullptr))
            return sol;
        j11 = (a1 - c1) / (2.0f * h);
        j21 = (a2 - c2) / (2.0f * h);
        if (!gpuChainAngularError(pixel, scene, path, p, n, normalize(omega + dirFrame.b * h), L, b1, b2, a1,
                                  a2, nullptr) ||
            !gpuChainAngularError(pixel, scene, path, p, n, normalize(omega - dirFrame.b * h), L, b1, b2, c1,
                                  c2, nullptr))
            return sol;
        j12 = (a1 - c1) / (2.0f * h);
        j22 = (a2 - c2) / (2.0f * h);
    }
    const float detJ = fabsf(j11 * j22 - j12 * j21);
    if (!(detJ > 1e-10f) || !srIsFinite(detJ)) return sol;
    sol.omega = omega;
    sol.chain = chain;
    sol.detJ = detJ;
    sol.planeN = L;
    sol.solved = 1;
    return sol;
}

__device__ inline void gpuMneeSeedCone(const SceneView& scene, Vec3 p, Vec3 straightDir, int casterInstance,
                                       Vec3& axis, float& cosThetaMax) {
    axis = straightDir;
    cosThetaMax = 0.9962f;
    if (casterInstance < 0 || casterInstance >= scene.instanceCount) return;
    const InstanceData& inst = scene.instances[casterInstance];
    if (inst.meshIndex < 0 || inst.meshIndex >= scene.meshCount || !scene.meshes) return;
    const MeshView& mesh = scene.meshes[inst.meshIndex];
    Bounds3 local;
    local.lo = mesh.boundsLo;
    local.hi = mesh.boundsHi;
    if (!local.valid()) return;
    const Bounds3 world = transformBounds(inst.xform, local);
    const Vec3 center = world.center();
    const float radius = 0.5f * length(world.extent());
    const Vec3 toC = center - p;
    const float dist = length(toC);
    if (dist <= radius * 1.05f || dist < 1e-5f) {
        cosThetaMax = 0.0f;
        return;
    }
    axis = toC / dist;
    const float sinT = clampf(radius / dist, 0.0f, 0.9999f);
    cosThetaMax = sqrtf(srMax(0.0f, 1.0f - sinT * sinT));
}

__device__ inline int gpuBuildSeedDirs(const SceneView& scene, Vec3 p, Vec3 straightDir, int casterInstance,
                                       Vec3* dirs) {
    int count = 0;
    dirs[count++] = straightDir;
    Vec3 axis;
    float cosThetaMax = 1.0f;
    gpuMneeSeedCone(scene, p, straightDir, casterInstance, axis, cosThetaMax);
    const float thetaMax = acosf(clampf(cosThetaMax, -1.0f, 1.0f));
    if (thetaMax < 1e-3f) return count;
    const float theta = thetaMax * kGpuMneeSeedRingRadius;
    const float st = sinf(theta);
    const float ct = cosf(theta);
    const Frame frame(axis);
    for (int i = 0; i < kGpuMneeSeedRing; ++i) {
        const float phi = (float(i) + 0.5f) * (kTwoPi / float(kGpuMneeSeedRing));
        dirs[count++] = normalize(frame.toWorld(Vec3(st * cosf(phi), st * sinf(phi), ct)));
    }
    return count;
}

__device__ inline bool gpuSameBranch(Vec3 a, Vec3 b) { return dot(a, b) > 0.99996f; }

__device__ inline bool gpuVerifyAreaExit(int pixel, const SceneView& scene, const GpuChainState& chain,
                                         int lightIndex, Vec3 y) {
    const Vec3 toY = y - chain.exitP;
    const float d = length(toY);
    if (d < 1e-5f) return false;
    const Vec3 wd = toY / d;
    Surf si;
    if (!gpuTraceSurf(pixel, scene, offsetRay(chain.exitP, chain.exitN, wd), wd, d * (1.0f - 1e-3f), si))
        return true;
    return si.lightIndex == lightIndex;
}

__device__ inline bool gpuVerifyDistantExit(int pixel, const SceneView& scene, const GpuChainState& chain,
                                            Vec3 lightDir) {
    Surf si;
    if (!gpuTraceSurf(pixel, scene, offsetRay(chain.exitP, chain.exitN, lightDir), lightDir, 1.0e8f, si))
        return true;
    Vec3 ns = si.ns;
    GpuPath dummy{};
    dummy.nLambda = 0;
    const Material mat = gpuMneeCasterMat(scene, si, dummy, ns);
    return isDeltaCausticCaster(mat);
}

__device__ inline void gpuAddMneeContribution(GpuPath& path, const LightData& light, const Material& shadeMat,
                                              Vec3 wo, Vec3 n, const GpuManifoldSolution& sol, Vec3 LeRgb,
                                              Vec3 yN, float pdfArea, float selectPdf, float clampValue) {
    const GpuChainState& chain = sol.chain;
    const Frame frame(n);
    const Vec3 woLocal = frame.toLocal(wo);
    const Vec3 wiLocal = frame.toLocal(sol.omega);
    const BsdfEval be = bsdfEvalLocal(shadeMat, woLocal, wiLocal);
    if (be.pdf <= 0.0f || isBlack(be.f)) return;
    const float cosP = fabsf(dot(n, sol.omega));
    float planeToLight = 1.0f;
    if (!sol.distant && light.type != kLightPoint) {
        const float cEmit = dot(yN, -chain.exitDir);
        const float emitOk = (light.type != kLightSphere && light.twoSided) ? fabsf(cEmit) : srMax(0.0f, cEmit);
        if (emitOk <= 1e-6f) return;
        planeToLight = fabsf(dot(yN, sol.planeN));
        if (planeToLight <= 1e-6f) return;
    }
    const float geom = planeToLight / sol.detJ;
    const float scale = cosP * geom / srMax(1e-12f, pdfArea * selectPdf);
    if (!(scale > 0.0f) || !srIsFinite(scale)) return;

    float Le[kMaxSpectrumSamples];
    float fS[kMaxSpectrumSamples];
    float tS[kMaxSpectrumSamples];
    float neeS[kMaxSpectrumSamples];
    specAuthoredRadiance(light, LeRgb, path, Le);
    evalBsdfSpectralGpu(shadeMat, woLocal, wiLocal, path, shadeMat.ior, fS);
    specUpsampleLinear(chain.throughput, path.lambda, path.nLambda, tS);
    for (int i = 0; i < path.nLambda; ++i) neeS[i] = Le[i] * fS[i] * tS[i] * scale;
    float baked[kMaxSpectrumSamples];
    bakeNeeAtVertexS(path, neeS, clampValue, baked);
    addBakedRadianceS(path, baked);
}

enum GpuNeePeek : int { kGpuNeeClear = 0, kGpuNeeGlass = 1, kGpuNeeBlocked = 2 };

__device__ inline GpuNeePeek gpuPeekNee(int pixel, const SceneView& scene, const GpuPath& path, Vec3 origin,
                                        Vec3 dir, float tMax, int lightIndex, int& casterInstance) {
    casterInstance = -1;
    Surf si;
    if (!gpuTraceSurf(pixel, scene, origin, dir, tMax, si)) return kGpuNeeClear;
    if (si.lightIndex == lightIndex) return kGpuNeeClear;
    if (si.lightIndex >= 0) return kGpuNeeBlocked;
    Vec3 ns = si.ns;
    const Material mat = gpuMneeCasterMat(scene, si, path, ns);
    if (isDeltaCausticCaster(mat)) {
        casterInstance = si.instanceIndex;
        return kGpuNeeGlass;
    }
    return kGpuNeeBlocked;
}

// Returns true when the NEE sample was a glass SDS (MNEE ran or failed) — skip
// the regular shadow NEE, which would be blocked by opaque-caustic glass.
__device__ inline bool tryGpuMneeNee(int pixel, GpuPath& path, GpuShadow& shadow, const Surf& si,
                                     const Material& mat, const Frame& frame, Vec3 wo, int lightIndex,
                                     const LightSample& ls, float selectPdf) {
    (void)shadow;
    const LaunchParams& params = launchParams();
    const SceneView& scene = params.scene;
    if (scene.settings.caustics == 0) return false;
    if (scene.settings.causticsEngineGpu != kGpuCausticsMneeLt) return false;
    if (lightIndex < 0 || lightIndex >= scene.lightCount) return false;
    const LightData& light = scene.lights[lightIndex];
    if (!lightContributesCaustics(light)) return false;
    if (light.type == kLightDome) return false;

    const Vec3 shadowOrigin = offsetRay(si.p, si.ng, ls.wi);
    float tMax = 1.0e8f;
    if (ls.distance < 1.0e7f) tMax = ls.distance * (1.0f - 1e-3f);
    int casterInstance = -1;
    const GpuNeePeek peek = gpuPeekNee(pixel, scene, path, shadowOrigin, ls.wi, tMax, lightIndex,
                                       casterInstance);
    if (peek != kGpuNeeGlass) return false;

    const float clampValue =
        pathContributionClamp(scene.settings, path.depth, path.specularBounce != 0, path.causticSuffix != 0);

    Vec3 seeds[1 + kGpuMneeSeedRing];
    Vec3 found[1 + kGpuMneeSeedRing];
    int foundCount = 0;

    if (light.type == kLightDistant) {
        const int nSeeds = gpuBuildSeedDirs(scene, si.p, ls.wi, casterInstance, seeds);
        for (int i = 0; i < nSeeds; ++i) {
            const GpuManifoldSolution sol = gpuSolveAngular(pixel, scene, path, si.p, si.ns, ls.wi, seeds[i]);
            if (!sol.solved) continue;
            bool dup = false;
            for (int k = 0; k < foundCount; ++k)
                if (gpuSameBranch(sol.omega, found[k])) {
                    dup = true;
                    break;
                }
            if (dup) continue;
            found[foundCount++] = sol.omega;
            if (!gpuVerifyDistantExit(pixel, scene, sol.chain, ls.wi)) continue;
            gpuAddMneeContribution(path, light, mat, wo, si.ns, sol, ls.radiance, Vec3(0.0f, 1.0f, 0.0f),
                                   ls.pdf, selectPdf, clampValue);
        }
        return true;
    }

    const Vec3 y = si.p + ls.wi * ls.distance;
    Vec3 yN = areaLightNormal(light);
    if (light.type == kLightSphere) yN = normalize(y - lightOrigin(light));
    if (light.type == kLightPoint) yN = Vec3(0.0f, 1.0f, 0.0f);

    float pdfArea = 1.0f;
    Vec3 LeRgb = ls.radiance;
    if (light.type == kLightPoint) {
        pdfArea = 1.0f;
        LeRgb = light.emittedRadiance();
    } else if (light.type == kLightRect) {
        const float area = rectLightArea(light);
        if (area <= 1e-12f) return true;
        pdfArea = 1.0f / area;
        LeRgb = lightRadiance(light);
    } else if (light.type == kLightDisk) {
        const float area = diskLightArea(light);
        if (area <= 1e-12f) return true;
        pdfArea = 1.0f / area;
        LeRgb = lightRadiance(light);
    } else if (light.type == kLightSphere) {
        const float radius = srMax(1e-5f, sphereLightRadius(light));
        pdfArea = 1.0f / (4.0f * kPi * radius * radius);
        LeRgb = lightRadiance(light);
    }

    Vec3 dir = y - si.p;
    const float distPy = length(dir);
    if (distPy < 1e-5f) return true;
    dir = dir / distPy;
    const int nSeeds = gpuBuildSeedDirs(scene, si.p, dir, casterInstance, seeds);
    for (int i = 0; i < nSeeds; ++i) {
        const GpuManifoldSolution sol =
            gpuSolvePlane(pixel, scene, path, si.p, si.ns, lightIndex, y, seeds[i]);
        if (!sol.solved) continue;
        bool dup = false;
        for (int k = 0; k < foundCount; ++k)
            if (gpuSameBranch(sol.omega, found[k])) {
                dup = true;
                break;
            }
        if (dup) continue;
        found[foundCount++] = sol.omega;
        if (!gpuVerifyAreaExit(pixel, scene, sol.chain, lightIndex, y)) continue;
        gpuAddMneeContribution(path, light, mat, wo, si.ns, sol, LeRgb, yN, pdfArea, selectPdf, clampValue);
    }
    (void)frame;
    return true;
}

}  // namespace sol
