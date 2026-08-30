// Dedicated MNEE pipeline: Newton probes call optixTrace here, never from
// shade_surface / path_tail (that combination hangs cicc / optixModuleCreate).
// Do not include optix_spawn.cuh — spawn/film regen stays in shade_shadow.
#define SOLSTICE_OPTIX_MNEE_KERNEL
#include "render/optix/optix_mnee.cuh"

namespace sol {

extern "C" __global__ void __raygen__mnee() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    if (!params.mneeJobs || !params.paths || !params.shadows) return;
    GpuMneeJob& job = params.mneeJobs[pixel];
    if (!job.pending) return;

    GpuPath& path = params.paths[pixel];
    GpuShadow& shadow = params.shadows[pixel];
    tryGpuMneeJob(pixel, path, job);
    job.armed = 0;
    job.pending = 0;
    job.casterInstance = -1;
    shadow.mneeCaster = -1;
    // shade_shadow skipped flush/regen while this job was pending. Hand the
    // slot back so the follow-up shade_shadow launch can close the path.
    shadow.queue = kShadowShade;
}

}  // namespace sol
