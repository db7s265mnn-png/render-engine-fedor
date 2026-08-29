// CPU integrator interface + thin wrappers around existing transport kernels.
// OptiX keeps calling traceRadiance directly; Embree dispatches through these.
#pragma once

#include "core/rng.h"
#include "render/integrator.h"
#include "render/integrator_bdpt.h"
#include "render/integrator_mnee.h"
#include "render/photon_map.h"
#include "scene/types.h"

#if !defined(__CUDACC__) && SOLSTICE_HAVE_OPENPGL
#include "render/cpu/path_guiding.h"
#endif

namespace sol {

// Shared per-pixel arguments for CPU integrators.
template <typename Tracer>
struct IntegratorSampleContext {
    const SceneView* scene = nullptr;
    Tracer* tracer = nullptr;
    Vec3 origin{0.0f};
    Vec3 direction{0.0f};
    Rng* rng = nullptr;
    DispersionContext* dispersion = nullptr;
    Framebuffer* splatFb = nullptr;
    const CausticPhotonMap* photons = nullptr;
    struct BdptPassStats* bdptStats = nullptr;
#if !defined(__CUDACC__) && SOLSTICE_HAVE_OPENPGL
    PathGuiding::ThreadState* guiding = nullptr;
#endif
};

template <typename Tracer>
class Integrator {
public:
    virtual ~Integrator() = default;
    virtual Vec3 Li(IntegratorSampleContext<Tracer>& ctx) const = 0;
    virtual const char* name() const = 0;
};

template <typename Tracer>
class PathIntegrator final : public Integrator<Tracer> {
public:
    const char* name() const override { return "Path Tracer"; }
    Vec3 Li(IntegratorSampleContext<Tracer>& ctx) const override {
#if !defined(__CUDACC__) && SOLSTICE_HAVE_OPENPGL
        if (ctx.guiding)
            return traceRadiance(*ctx.scene, *ctx.tracer, ctx.origin, ctx.direction, *ctx.rng, ctx.guiding,
                                 ctx.dispersion);
#endif
        return traceRadiance(*ctx.scene, *ctx.tracer, ctx.origin, ctx.direction, *ctx.rng, ctx.dispersion);
    }
};

template <typename Tracer>
class DirectLightingIntegrator final : public Integrator<Tracer> {
public:
    const char* name() const override { return "Direct Lighting"; }
    Vec3 Li(IntegratorSampleContext<Tracer>& ctx) const override {
        // DL is maxDepth=1 inside traceRadiance when settings.integrator says so.
        return PathIntegrator<Tracer>{}.Li(ctx);
    }
};

template <typename Tracer>
class AmbientOcclusionIntegrator final : public Integrator<Tracer> {
public:
    const char* name() const override { return "Ambient Occlusion"; }
    Vec3 Li(IntegratorSampleContext<Tracer>& ctx) const override {
        return PathIntegrator<Tracer>{}.Li(ctx);
    }
};

template <typename Tracer>
class WireframeIntegrator final : public Integrator<Tracer> {
public:
    const char* name() const override { return "Wireframe"; }
    Vec3 Li(IntegratorSampleContext<Tracer>& ctx) const override {
        return PathIntegrator<Tracer>{}.Li(ctx);
    }
};

template <typename Tracer>
class BdptIntegrator final : public Integrator<Tracer> {
public:
    const char* name() const override { return "BDPT"; }
    Vec3 Li(IntegratorSampleContext<Tracer>& ctx) const override {
#if SOLSTICE_HAVE_OPENPGL
        return traceRadianceBdpt(*ctx.scene, *ctx.tracer, ctx.origin, ctx.direction, *ctx.rng, ctx.guiding,
                                 ctx.splatFb, ctx.dispersion, ctx.photons);
#else
        return traceRadianceBdpt(*ctx.scene, *ctx.tracer, ctx.origin, ctx.direction, *ctx.rng, ctx.splatFb,
                                 ctx.dispersion, ctx.photons);
#endif
    }
};

template <typename Tracer>
class PathMneeIntegrator final : public Integrator<Tracer> {
public:
    const char* name() const override { return "Path Tracer (MNEE/Photon)"; }
    Vec3 Li(IntegratorSampleContext<Tracer>& ctx) const override {
#if !defined(__CUDACC__) && SOLSTICE_HAVE_OPENPGL
        if (ctx.guiding)
            return traceRadiancePtMnee(*ctx.scene, *ctx.tracer, ctx.origin, ctx.direction, *ctx.rng, ctx.guiding,
                                       ctx.dispersion, ctx.photons);
        return traceRadiancePtMnee(*ctx.scene, *ctx.tracer, ctx.origin, ctx.direction, *ctx.rng, ctx.dispersion,
                                   ctx.photons);
#else
        return traceRadiancePtMnee(*ctx.scene, *ctx.tracer, ctx.origin, ctx.direction, *ctx.rng, ctx.dispersion,
                                   ctx.photons);
#endif
    }
};

}  // namespace sol
