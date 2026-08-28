// SDS-family camera splat for OptiX light paths. No optixTrace.
// Connectable = lightTraceConnectable (not the caster). After a splat the path
// continues; specPrefix is cleared by shade so later diffuse hits do not re-splat.
// Hybrid (Aimed LT + SDS refraction): if the splat shadow is later blocked by
// delta glass, the dedicated MNEE pipeline connects this vertex to the camera.
#pragma once

#include "render/camera_proj.h"
#include "render/optix/optix_light_emit.cuh"
#include "render/optix/optix_spectral.cuh"
#include "render/shading_bsdf.h"

namespace sol {

__device__ inline bool gpuLightVertexConnectable(const Material& mat, Vec3 woLocal) {
    return lightTraceConnectable(mat, woLocal);
}

// SDS family: light → near-spec chain → connectable → camera splat, then the
// path continues (caster is not connectable, so glass does not splat-and-die).
// Camera PT skips the SDS suffix while LT is running so the two don't double-count.
__device__ inline void tryEnqueueCausticSplat(int pixel, GpuPath& path, GpuShadow& shadow, const Surf& si,
                                              const Material& mat, const Frame& frame, Vec3 wo) {
    const LaunchParams& params = launchParams();
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
    if (shadow.queue != kShadowTrace || shadow.splatPixel < 0) return;

    if (!params.mneeJobs || !gpuSdsRefractionEnabled(params.scene.settings)) return;
    GpuMneeJob& job = params.mneeJobs[pixel];
    job.p = si.p;
    job.ns = si.ns;
    job.ng = si.ng;
    job.wo = wo;
    job.uv = si.uv;
    job.y = params.camProj.camPos;
    job.yN = Vec3(0.0f, 1.0f, 0.0f);
    job.LeRgb = Vec3(0.0f);
    job.wi = toCam;
    job.distance = dist;
    job.pdfArea = 1.0f;
    job.selectPdf = 1.0f;
    job.materialIndex = si.materialIndex;
    job.lightIndex = path.lightIndex;
    job.casterInstance = -1;
    job.armed = 1;
    job.pending = 0;
    job.distant = 0;
    job.cameraSplat = 1;
    job.clampDepth = path.depth;
    job.clampSpec = 1;
    job.clampCaustic = 1;
    for (int i = 0; i < path.nLambda && i < kMaxSpectrumSamples; ++i) job.throughputS[i] = path.throughputS[i];
}

}  // namespace sol
