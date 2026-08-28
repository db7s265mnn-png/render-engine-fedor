// Wavefront path state for the OptiX backend.
//
// Cycles keeps one IntegratorState slot per path and a queue id for the next
// kernel (intersect_closest, shade_surface, …). Same idea here: unidirectional
// spectral PT (hero-λ, matching Embree Path Tracer). Shade kernels
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

enum WorkSlot : int {
    kSlotIntersect = 0,
    kSlotVolume = 1,
    kSlotSurface = 2,
    kSlotBackground = 3,
    kSlotShadow = 4,
    kSlotIntersectNext = 5,
    kSlotCount = 6
};

enum ShadowQueue : int {
    kShadowIdle = 0,
    kShadowTrace = 1,
    kShadowShade = 2,
    kShadowMnee = 3,  // glass-blocked NEE; dedicated MNEE pipeline owns the slot
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
    int transmittedBounce = 0;  // 1 = last BSDF sample was refraction (not mirror / TIR)
    int mediumIndex = -1;
    int volumeScatters = 0;
    int localSample = 0;  // 0 .. batchSamples-1 when regenerating into the next spp
    int lightPath = 0;    // 1 = Iray-style caustic light trace (not camera PT)
    int specPrefix = 0;   // 1 = light prefix went through contributing near-spec / delta
    int lightIndex = -1;  // SampleLe emitter for this light path
    int sawNonSpecular = 0;  // camera PT: had a diffuse/volume bounce
    int causticSuffix = 0;   // camera PT: spec after diffuse (SDS); skip if LT is on
    // 1 = this camera path already refracted through a delta caustic caster.
    // Aimed LT + MNEE: those pixels are the glass, not the floor — keep BSDF.
    // Aimed LT + SDS: still skip camera SDS here (BDPT s=0 under glass).
    int throughGlass = 0;
    // MCMC / ERPT: stored SampleLe so mutations respawn without a new light pick.
    Vec3 mcmcOrigin{0.0f};
    Vec3 mcmcDir{0.0f, 0.0f, 1.0f};
    Vec3 mcmcN{0.0f, 1.0f, 0.0f};
    Vec3 mcmcBetaRgb{0.0f};
    int mcmcInfinite = 0;
    int mcmcRemain = 0;  // mutations left in this slot (not spp)
    // AoS: shade kernels touch many fields of one path. Iray's SoA win was in a
    // separate logic kernel that streamed one member across 1M slots — not here.
    // Packed flags would misalign the 4λ arrays; leave ints.
    Rng rng;
};

struct GpuShadow {
    Vec3 origin{0.0f};
    Vec3 direction{0.0f, 0.0f, 1.0f};
    Vec3 contrib{0.0f};  // RGB LT splat (already includes throughput)
    float tMax = 0.0f;
    int queue = kShadowIdle;
    int occluded = 0;
    int volumeTr = 0;     // 1 = multiply by GPU volume / homogeneous transmittance
    int mediumIndex = -1;  // current path medium for homogeneous Beer–Lambert on the shadow ray
    int splatPixel = -1;  // >=0: unoccluded contrib atomicAdds RGB to that film pixel (no .w)
    int mneeCaster = -1;  // instanceIndex of a delta glass blocker; -1 = none
    // 1 = camera NEE from depth>0: Iray Fresnel-continue through contributing glass.
    // 0 = primary NEE or LT camera splat (glass stays opaque; SDS on the floor).
    // Stored here because shade increments path.depth after enqueueing the shadow.
    int eyeBounceNee = 0;
    // Camera NEE baked at the vertex: throughput × illuminant(Le) × albedo(f) × geom.
    // shade_shadow must add this as-is — live path throughput has already stepped.
    float contribS[kMaxSpectrumSamples]{};
    int specContrib = 0;  // 1 = shade_shadow uses contribS (not RGB contrib)
};

// Eye-path MNEE job filled at the NEE vertex (before BSDF steps throughput).
// The Newton pipeline reads this after intersect_shadow peeks a glass blocker.
struct GpuMneeJob {
    Vec3 p{0.0f};
    Vec3 ns{0.0f, 0.0f, 1.0f};
    Vec3 ng{0.0f, 0.0f, 1.0f};
    Vec3 wo{0.0f, 0.0f, 1.0f};
    Vec2 uv{0.0f, 0.0f};
    Vec3 y{0.0f};
    Vec3 yN{0.0f, 1.0f, 0.0f};
    Vec3 LeRgb{0.0f};
    Vec3 wi{0.0f, 0.0f, 1.0f};
    float distance = 0.0f;
    float pdfArea = 0.0f;
    float selectPdf = 0.0f;
    float throughputS[kMaxSpectrumSamples]{};
    int materialIndex = -1;
    int lightIndex = -1;
    int casterInstance = -1;
    int armed = 0;
    int pending = 0;
    int distant = 0;
    int clampDepth = 0;
    int clampSpec = 0;
    int clampCaustic = 0;
};

}  // namespace sol
