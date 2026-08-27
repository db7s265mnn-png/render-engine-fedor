// SDS-family camera splat for OptiX light paths. No optixTrace.
#pragma once

#include "render/camera_proj.h"
#include "render/optix/optix_light_emit.cuh"
#include "render/optix/optix_spectral.cuh"

namespace sol {

__device__ inline bool gpuLightVertexConnectable(const Material& mat, Vec3 woLocal) {
    const LobeWeights lw = computeLobes(mat, woLocal);
    return !(lw.delta && lw.diffuse < 1e-4f);
}

// SDS family only: light → near-spec chain → first connectable → camera, then stop.
// Unweighted vs camera PT: camera skips the SDS suffix while LT is running.
__device__ inline void tryEnqueueCausticSplat(int pixel, GpuPath& path, GpuShadow& shadow, const Surf& si,
                                              const Material& mat, const Frame& frame, Vec3 wo) {
    const LaunchParams& params = launchParams();
    (void)pixel;
    if (!path.lightPath || !path.specPrefix) return;
    if (params.splatInvLightPaths <= 0.0f || !params.camProj.valid) return;
    if (path.lightIndex >= 0 && path.lightIndex < params.scene.lightCount &&
        !lightContributesCaustics(params.scene.lights[path.lightIndex]))
        return;

    const Vec3 woLocal = frame.toLocal(wo);
    if (!gpuLightVertexConnectable(mat, woLocal)) return;

    float px = 0.0f, py = 0.0f, cosTheta = 0.0f, dist2 = 0.0f;
    if (!projectToPixel(params.camProj, si.p, px, py, cosTheta, dist2) || dist2 < 1e-8f) return;
    const int ix = int(px);
    const int iy = int(py);
    if (ix < 0 || iy < 0 || ix >= params.width || iy >= params.height) return;
    const int dest = iy * params.width + ix;

    const Vec3 toCam = normalize(params.camProj.camPos - si.p);
    const Vec3 wiLocal = frame.toLocal(toCam);
    const BsdfEval be = bsdfEvalLocal(mat, woLocal, wiLocal);
    if (isBlack(be.f) || !isFinite(be.f)) return;

    const float pdfOmega = cameraPdfOmega(params.camProj, cosTheta);
    const float cosV = fabsf(dot(si.ns, toCam));
    const float geom = cosV * pdfOmega / dist2;
    if (!(geom > 0.0f) || !srIsFinite(geom)) return;

    float fS[kMaxSpectrumSamples];
    evalBsdfSpectralGpu(mat, woLocal, wiLocal, path, mat.ior, fS);
    float tmp[kMaxSpectrumSamples];
    specZero(tmp, path.nLambda);
    for (int i = 0; i < path.nLambda; ++i) tmp[i] = path.throughputS[i] * fS[i] * geom;
    specClampIndirect(tmp, path.nLambda, gpuLightTraceSplatClamp(params.scene.settings));
    if (!specIsFinite(tmp, path.nLambda) || specIsBlack(tmp, path.nLambda)) return;
    Vec3 rgb = specToRgb(params.spec, tmp, path.lambda, path.pdf, path.nLambda);
    rgb = rgb * params.splatInvLightPaths;
    if (!isFinite(rgb) || isBlack(rgb)) return;

    const Vec3 shadowOrigin = offsetRay(si.p, si.ng, toCam);
    const float dist = sqrtf(srMax(1e-12f, dist2));
    enqueueShadow(shadow, shadowOrigin, toCam, dist * (1.0f - 1e-3f), rgb, path.mediumIndex, dest);
}

}  // namespace sol
