// Cycles analogue: integrator_init_from_camera.
// Pinhole only. Thin-lens DoF and polynomial optics stay on Embree.
#include "render/blue_noise.h"
#include "render/optix/optix_geom.cuh"

namespace sol {

extern "C" __global__ void __raygen__init_from_camera() {
    int x = 0, y = 0;
    const int pixel = wavefrontPixel(x, y);
    if (pixel < 0) return;

    const LaunchParams& params = launchParams();
    GpuPath& path = params.paths[pixel];
    GpuHit& hit = params.hits[pixel];
    GpuShadow& shadow = params.shadows[pixel];

    path.rng = makePixelRng(x, y, params.sampleIndex, params.frameSeed);
    float jitterX = 0.5f, jitterY = 0.5f;
    if (params.pixelSampler == kPixelSamplerBlueNoise) {
        blueNoisePixelJitter(x, y, params.sampleIndex, jitterX, jitterY);
    } else {
        jitterX = path.rng.nextFloat();
        jitterY = path.rng.nextFloat();
    }

    pinholeRay(params.scene, float(x) + jitterX, float(y) + jitterY, path.origin, path.direction);
    path.throughput = Vec3(1.0f);
    path.bsdfPdf = 0.0f;
    path.depth = 0;
    path.hops = 0;
    path.queue = kQueueIntersectClosest;
    path.specularBounce = 1;

    hit = GpuHit{};
    shadow = GpuShadow{};

    params.accumBuffer[pixel].w += 1.0f;
}

}  // namespace sol
