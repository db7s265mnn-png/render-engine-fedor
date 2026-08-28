// Eye-path MNEE Newton (Hanika et al. 2015). Only the dedicated MNEE OptiX
// pipeline includes this file (SOLSTICE_OPTIX_MNEE_KERNEL). shade_surface and
// path_tail must not: Newton probes call optixTrace, and inlining them into
// the interactive shade/tail TUs hangs cicc / optixModuleCreate.
#pragma once

#if defined(__CUDACC__) && !defined(SOLSTICE_OPTIX_MNEE_KERNEL)
#error "optix_mnee.cuh must not be included in shade / path_tail (optixTrace+BSDF hangs cicc)"
#endif

#include "render/camera_proj.h"
#include "render/lights.h"
#include "render/optix/optix_bsdf.cuh"
#include "render/optix/optix_geom.cuh"
#include "render/optix/optix_light_emit.cuh"
#include "render/optix/optix_spectral.cuh"
#include "render/optix/optix_trace.cuh"
#include "render/shading_bsdf.h"

namespace sol {

constexpr int kGpuMneeMaxChain = 6;
constexpr int kGpuMneeNewtonIters = 24;
constexpr int kGpuMneeBacktrack = 5;
constexpr int kGpuMneeSeedRing = 4;
constexpr float kGpuMneeSeedRingRadius = 0.75f;
// gpuTraceChain: not a scene light. Miss after glass is success; area lights block.
constexpr int kGpuMneeCameraTarget = -2;

// __noinline__: inlined optixTrace would put Newton's live Material / Jacobian
// on the continuation stack. Overflow is a CUDA illegal access; the next
// optixLaunch then reports OPTIX_ERROR_CUDA_ERROR 7900 / "query command list event".
__device__ __noinline__ GpuHit gpuTraceHit(int pixel, Vec3 origin, Vec3 direction, float tMax) {
    GpuHit miss{};
    if (pixel < 0) return miss;
    if (!isFinite(origin) || !isFinite(direction)) return miss;
    const float dir2 = lengthSquared(direction);
    if (!(dir2 > 1e-20f) || !srIsFinite(dir2)) return miss;
    if (!(tMax > 1e-8f) || !srIsFinite(tMax)) return miss;
    if (tMax > 1.0e8f) tMax = 1.0e8f;
    const LaunchParams& params = launchParams();
    if (!params.hits) return miss;
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
#pragma unroll 1
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
            // Camera MNEE: lights along the floor→camera chain are blockers.
            if (targetLight == kGpuMneeCameraTarget) return st;
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
    if (lengthSquared(omegaInit) < 1e-12f) return sol;
    Vec3 omega = normalize(omegaInit);
    if (!isFinite(omega)) return sol;
    const float tol = srMax(1e-5f, 1e-4f * distPy);
    float e1 = 0.0f, e2 = 0.0f;
    GpuChainState chain;
    float j11 = 0, j12 = 0, j21 = 0, j22 = 0;
    if (!gpuChainPlaneError(pixel, scene, path, p, n, omega, y, lightIndex, planeN, b1, b2, e1, e2, &chain))
        return sol;
    float err = sqrtf(e1 * e1 + e2 * e2);
    bool converged = err < tol;
#pragma unroll 1
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
#pragma unroll 1
        for (int k = 0; k < kGpuMneeBacktrack; ++k, scale *= 0.5f) {
            const Vec3 cand = normalize(omega - (dirFrame.t * du + dirFrame.b * dv) * scale);
            if (!isFinite(cand) || dot(cand, dir) < -0.1f) continue;
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
    if (lengthSquared(lightDir) < 1e-12f || lengthSquared(omegaInit) < 1e-12f) return sol;
    const Vec3 L = normalize(lightDir);
    const Frame errFrame(L);
    const Vec3 b1 = errFrame.t;
    const Vec3 b2 = errFrame.b;
    Vec3 omega = normalize(omegaInit);
    if (!isFinite(L) || !isFinite(omega)) return sol;
    const float tol = 2e-3f;
    float e1 = 0.0f, e2 = 0.0f;
    GpuChainState chain;
    float j11 = 0, j12 = 0, j21 = 0, j22 = 0;
    if (!gpuChainAngularError(pixel, scene, path, p, n, omega, L, b1, b2, e1, e2, &chain)) return sol;
    float err = sqrtf(e1 * e1 + e2 * e2);
    bool converged = err < tol;
#pragma unroll 1
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
#pragma unroll 1
        for (int k = 0; k < kGpuMneeBacktrack; ++k, scale *= 0.5f) {
            const Vec3 cand = normalize(omega - (dirFrame.t * du + dirFrame.b * dv) * scale);
            if (!isFinite(cand) || dot(cand, L) < -0.1f) continue;
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

__device__ inline void gpuAddMneeContribution(GpuPath& path, const float* throughputS, const LightData& light,
                                              const Material& shadeMat, Vec3 wo, Vec3 n,
                                              const GpuManifoldSolution& sol, Vec3 LeRgb, Vec3 yN,
                                              float pdfArea, float selectPdf, float clampValue) {
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
    const int nLambda = path.nLambda;
    for (int i = 0; i < nLambda; ++i) neeS[i] = Le[i] * fS[i] * tS[i] * scale;
    float baked[kMaxSpectrumSamples];
    for (int i = 0; i < nLambda; ++i) baked[i] = throughputS[i] * neeS[i];
    specClampIndirect(baked, nLambda, clampValue);
    addBakedRadianceS(path, baked);
}

__device__ inline bool gpuVerifyCameraExit(int pixel, const SceneView& scene, const GpuChainState& chain,
                                           Vec3 camPos) {
    Vec3 toC = camPos - chain.exitP;
    const float d = length(toC);
    if (d < 1e-5f) return false;
    toC = toC / d;
    if (dot(toC, chain.exitDir) < 0.2f) return false;
    Surf si;
    if (!gpuTraceSurf(pixel, scene, offsetRay(chain.exitP, chain.exitN, toC), toC, d * (1.0f - 1e-3f), si))
        return true;
    return false;
}

__device__ inline void gpuAddCameraMneeSplat(GpuPath& path, const GpuMneeJob& job, const Material& shadeMat,
                                             const GpuManifoldSolution& sol) {
    const LaunchParams& params = launchParams();
    if (!params.camProj.valid || params.splatInvLightPaths <= 0.0f) return;
    const GpuChainState& chain = sol.chain;
    const Frame frame(job.ns);
    const Vec3 woLocal = frame.toLocal(job.wo);
    const Vec3 wiLocal = frame.toLocal(sol.omega);
    const BsdfEval be = bsdfEvalLocal(shadeMat, woLocal, wiLocal);
    if (isBlack(be.f) || !isFinite(be.f)) return;

    float px = 0.0f, py = 0.0f, cosTheta = 0.0f, dist2 = 0.0f;
    if (!projectToPixel(params.camProj, chain.exitP, px, py, cosTheta, dist2) || dist2 < 1e-8f) return;
    const int ix = int(px);
    const int iy = int(py);
    if (ix < 0 || iy < 0 || ix >= params.width || iy >= params.height) return;
    const int dest = iy * params.width + ix;

    const float cosP = fabsf(dot(job.ns, sol.omega));
    const float pdfOmega = cameraPdfOmega(params.camProj, cosTheta);
    const float geom = cosP * pdfOmega / srMax(1e-12f, sol.detJ);
    if (!(geom > 0.0f) || !srIsFinite(geom)) return;

    float fS[kMaxSpectrumSamples];
    float tS[kMaxSpectrumSamples];
    float tmp[kMaxSpectrumSamples];
    evalBsdfSpectralGpu(shadeMat, woLocal, wiLocal, path, shadeMat.ior, fS);
    specUpsampleLinear(chain.throughput, path.lambda, path.nLambda, tS);
    specZero(tmp, path.nLambda);
    for (int i = 0; i < path.nLambda; ++i)
        tmp[i] = job.throughputS[i] * fS[i] * tS[i] * geom;
    specClampIndirect(tmp, path.nLambda, gpuLightTraceSplatClamp(params.scene.settings));
    if (!specIsFinite(tmp, path.nLambda) || specIsBlack(tmp, path.nLambda)) return;
    Vec3 rgb = specToRgb(params.spec, tmp, path.lambda, path.pdf, path.nLambda);
    rgb = rgb * params.splatInvLightPaths;
    if (!isFinite(rgb) || isBlack(rgb)) return;
    addSplatRadiance(dest, rgb);
}

__device__ inline void tryGpuCameraMneeSplat(int pixel, GpuPath& path, GpuMneeJob& job) {
    const LaunchParams& params = launchParams();
    const SceneView& scene = params.scene;
    if (!gpuSdsRefractionEnabled(scene.settings)) return;
    if (!path.lightPath) return;
    if (!params.camProj.valid) return;
    if (job.selectPdf <= 0.0f) return;

    Material mat = gpuMaterialAt(scene, job.materialIndex);
    mat = optixpt::evaluateMaps(scene, mat, job.uv, job.ns);

    Vec3 seeds[1 + kGpuMneeSeedRing];
    Vec3 found[1 + kGpuMneeSeedRing];
    int foundCount = 0;
    const int nSeeds = gpuBuildSeedDirs(scene, job.p, job.wi, job.casterInstance, seeds);
    for (int i = 0; i < nSeeds; ++i) {
        const GpuManifoldSolution sol =
            gpuSolvePlane(pixel, scene, path, job.p, job.ns, kGpuMneeCameraTarget, job.y, seeds[i]);
        if (!sol.solved) continue;
        bool dup = false;
        for (int k = 0; k < foundCount; ++k)
            if (gpuSameBranch(sol.omega, found[k])) {
                dup = true;
                break;
            }
        if (dup) continue;
        found[foundCount++] = sol.omega;
        if (!gpuVerifyCameraExit(pixel, scene, sol.chain, job.y)) continue;
        gpuAddCameraMneeSplat(path, job, mat, sol);
    }
}

// Lazy MNEE upgrade after intersect_shadow peeked a delta-glass blocker.
// Matches CPU integrator_mnee: finite (and distant) lights, not the dome.
// Hybrid SDS refraction: same Newton to the camera when an LT splat is blocked.
__device__ inline void tryGpuMneeJob(int pixel, GpuPath& path, GpuMneeJob& job) {
    const LaunchParams& params = launchParams();
    const SceneView& scene = params.scene;
    if (!job.pending) return;
    job.pending = 0;
    if (job.cameraSplat) {
        tryGpuCameraMneeSplat(pixel, path, job);
        return;
    }
    if (!gpuEyePathMneeEnabled(scene.settings)) return;
    if (path.lightPath) return;
    if (job.lightIndex < 0 || job.lightIndex >= scene.lightCount) return;
    const LightData& light = scene.lights[job.lightIndex];
    if (!lightContributesCaustics(light)) return;
    if (light.type == kLightDome) return;
    if (job.selectPdf <= 0.0f) return;
    if (!job.distant && light.type != kLightDistant && job.pdfArea <= 0.0f) return;

    Material mat = gpuMaterialAt(scene, job.materialIndex);
    mat = optixpt::evaluateMaps(scene, mat, job.uv, job.ns);

    const float clampValue =
        pathContributionClamp(scene.settings, job.clampDepth, job.clampSpec != 0, job.clampCaustic != 0);

    Vec3 seeds[1 + kGpuMneeSeedRing];
    Vec3 found[1 + kGpuMneeSeedRing];
    int foundCount = 0;

    if (job.distant || light.type == kLightDistant) {
        const int nSeeds = gpuBuildSeedDirs(scene, job.p, job.wi, job.casterInstance, seeds);
        for (int i = 0; i < nSeeds; ++i) {
            const GpuManifoldSolution sol =
                gpuSolveAngular(pixel, scene, path, job.p, job.ns, job.wi, seeds[i]);
            if (!sol.solved) continue;
            bool dup = false;
            for (int k = 0; k < foundCount; ++k)
                if (gpuSameBranch(sol.omega, found[k])) {
                    dup = true;
                    break;
                }
            if (dup) continue;
            found[foundCount++] = sol.omega;
            if (!gpuVerifyDistantExit(pixel, scene, sol.chain, job.wi)) continue;
            gpuAddMneeContribution(path, job.throughputS, light, mat, job.wo, job.ns, sol, job.LeRgb,
                                   Vec3(0.0f, 1.0f, 0.0f), job.pdfArea, job.selectPdf, clampValue);
        }
        return;
    }

    const int nSeeds = gpuBuildSeedDirs(scene, job.p, job.wi, job.casterInstance, seeds);
    for (int i = 0; i < nSeeds; ++i) {
        const GpuManifoldSolution sol =
            gpuSolvePlane(pixel, scene, path, job.p, job.ns, job.lightIndex, job.y, seeds[i]);
        if (!sol.solved) continue;
        bool dup = false;
        for (int k = 0; k < foundCount; ++k)
            if (gpuSameBranch(sol.omega, found[k])) {
                dup = true;
                break;
            }
        if (dup) continue;
        found[foundCount++] = sol.omega;
        if (!gpuVerifyAreaExit(pixel, scene, sol.chain, job.lightIndex, job.y)) continue;
        gpuAddMneeContribution(path, job.throughputS, light, mat, job.wo, job.ns, sol, job.LeRgb, job.yN,
                               job.pdfArea, job.selectPdf, clampValue);
    }
}

}  // namespace sol
