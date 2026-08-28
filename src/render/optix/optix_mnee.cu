// Dedicated MNEE pipeline: Newton probes call optixTrace here, never from
// shade_surface / path_tail (that combination hangs cicc / optixModuleCreate).
#define SOLSTICE_OPTIX_MNEE_KERNEL
#include "render/optix/optix_mnee.cuh"
#include "render/optix/optix_spawn.cuh"

namespace sol {

extern "C" __global__ void __raygen__mnee() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    LaunchParams& params = launchParamsMutable();
    if (!params.mneeJobs || !params.paths || !params.shadows) return;
    GpuMneeJob& job = params.mneeJobs[pixel];
    if (!job.pending) return;

    GpuPath& path = params.paths[pixel];
    GpuShadow& shadow = params.shadows[pixel];
    tryGpuMneeJob(pixel, path, job);
    job.armed = 0;
    job.pending = 0;
    job.casterInstance = -1;
    shadow.queue = kShadowIdle;
    shadow.mneeCaster = -1;
    flushPathFilm(pixel);
    maybeRegeneratePath(pixel, path);
}

}  // namespace sol
