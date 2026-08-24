// Wavefront path state for the OptiX backend.
//
// Cycles keeps one IntegratorState slot per path and a queue id for the next
// kernel (intersect_closest, shade_surface, …). Same idea here: unidirectional
// spectral PT (hero-λ, matching Embree SpectralPathIntegrator). Shade kernels
// never call optixTrace; intersect kernels never include the BSDF.
#pragma once

#include "core/math.h"
#include "core/rng.h"
#include "render/spectrum_constants.h"

namespace sol {

enum PathQueue : int {
    kQueueDead = 0,
    kQueueIntersectClosest = 1,
    kQueueShadeSurface = 2,
    kQueueShadeBackground = 3,
    kQueueShadeVolume = 4,
};

enum ShadowQueue : int {
    kShadowIdle = 0,
    kShadowTrace = 1,
    kShadowShade = 2,
};

struct GpuHit {
    float t = 0.0f;
    int instanceIndex = -1;
    unsigned int primIndex = 0;
    float u = 0.0f;
    float v = 0.0f;
    int didHit = 0;
};

struct GpuPath {
    Vec3 origin{0.0f};
    Vec3 direction{0.0f, 0.0f, 1.0f};
    Vec3 sampleRgb{0.0f, 0.0f, 0.0f};  // this-spp radiance for the variance oracle
    float lambda[kMaxSpectrumSamples]{};
    float pdf[kMaxSpectrumSamples]{};
    float throughputS[kMaxSpectrumSamples]{};
    float radianceS[kMaxSpectrumSamples]{};
    int nLambda = 0;
    int filmOpen = 0;  // 1 until spectrumToRgb is flushed into the accum buffer
    float bsdfPdf = 0.0f;
    int depth = 0;
    int hops = 0;
    int queue = kQueueDead;
    int specularBounce = 1;
    int mediumIndex = -1;
    int volumeScatters = 0;
    Rng rng;
};

struct GpuShadow {
    Vec3 origin{0.0f};
    Vec3 direction{0.0f, 0.0f, 1.0f};
    Vec3 contrib{0.0f};  // RGB NEE aggregate (no throughput); Tr applied in shade_shadow
    float throughputS[kMaxSpectrumSamples]{};  // path throughput at the NEE vertex
    int nLambda = 0;
    float clampValue = 0.0f;
    float tMax = 0.0f;
    int queue = kShadowIdle;
    int occluded = 0;
    int volumeTr = 0;     // 1 = multiply by GPU volume / homogeneous transmittance
    int mediumIndex = -1;  // current path medium for homogeneous Beer–Lambert on the shadow ray
};

}  // namespace sol
