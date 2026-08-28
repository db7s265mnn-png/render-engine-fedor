// Cycles analogue: integrator_shade_shadow.
// Residual-ratio Tr. Camera NEE contribS is already throughput × Ld at the vertex.
#include "render/optix/optix_spawn.cuh"

namespace sol {

__device__ inline bool shadeShadowPixel(int pixel) {
    const LaunchParams& params = launchParams();
    GpuShadow& shadow = params.shadows[pixel];
    GpuPath& path = params.paths[pixel];
    if (shadow.occluded && shadow.mneeCaster >= 0 && params.mneeJobs &&
        params.mneeJobs[pixel].armed) {
        GpuMneeJob& job = params.mneeJobs[pixel];
        if (!path.lightPath && shadow.splatPixel < 0) {
            job.casterInstance = shadow.mneeCaster;
            job.pending = 1;
            shadow.queue = kShadowMnee;
            return false;
        }
    }
    if (!shadow.occluded) {
        if (shadow.splatPixel >= 0) {
            Vec3 contrib = shadow.contrib;
            if (shadow.volumeTr)
                contrib = contrib * gpuVolumeShadowTr(params, shadow.origin, shadow.direction,
                                                      shadow.tMax, shadow.mediumIndex, path.rng);
            addSplatRadiance(shadow.splatPixel, contrib);
        } else if (shadow.specContrib) {
            float s[kMaxSpectrumSamples];
            const int n = path.nLambda;
            for (int i = 0; i < n; ++i) s[i] = shadow.contribS[i];
            if (shadow.volumeTr) {
                const Vec3 tr = gpuVolumeShadowTr(params, shadow.origin, shadow.direction, shadow.tMax,
                                                  shadow.mediumIndex, path.rng);
                float trS[kMaxSpectrumSamples];
                specUpsampleLinear(tr, path.lambda, n, trS);
                specMul(s, trS, n);
            }
            addBakedRadianceS(path, s);
        }
    }
    shadow.queue = kShadowIdle;
    shadow.splatPixel = -1;
    shadow.specContrib = 0;
    shadow.mneeCaster = -1;
    shadow.eyeBounceNee = 0;
    if (params.mneeJobs) {
        params.mneeJobs[pixel].armed = 0;
        params.mneeJobs[pixel].pending = 0;
    }
    flushPathFilm(pixel);
    return maybeRegeneratePath(pixel, path);
}

#ifndef SOLSTICE_OPTIX_OPS_ONLY
extern "C" __global__ void __raygen__shade_shadow() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;
    GpuShadow& shadow = launchParams().shadows[pixel];
    if (!launchParams().compactLaunch && shadow.queue != kShadowShade) return;
    if (shadeShadowPixel(pixel)) enqueuePathContinuation(pixel);
}
#endif

}  // namespace sol
