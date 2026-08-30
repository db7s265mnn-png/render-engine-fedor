// CPU Aimed LT splats for Path Tracer (and PT+MNEE). BDPT already has t=1;
// this is the unidirectional twin: one aimed light path per camera sample,
// weight 1, family partition on the eye path (cpuAimedSkipCameraSds).
// Only runs when causticsUseAimedLt — pbrt / MNEE / Photon are unchanged.
#pragma once

#include "render/bdpt_scratch.h"
#include "render/camera_proj.h"
#include "render/framebuffer.h"
#include "render/integrator_bdpt.h"

namespace sol {
namespace bdpt {

template <typename Tracer>
SR_INL void splatAimedLightTrace(const SceneView& scene, const Tracer& tracer, Rng& rng,
                                 Framebuffer* splatFb, DispersionContext* dispersion,
                                 BdptScratch* scratch) {
    if (!splatFb || !causticsUseAimedLt(scene.settings)) return;
    if (scene.lightCount <= 0) return;
    const CameraProj camProj = buildCameraProj(scene);
    if (!camProj.valid) return;

    const RenderSettingsData& settings = scene.settings;
    const int maxVerts = bdptSessionVerts(settings.maxDepth);
    BdptScratch* buf = scratch;
    if (!buf) buf = &bdptThreadScratch();
    buf->ensure(maxVerts);
    Vert* light = buf->light.data();
    if (!light) return;

    splatFb->addSplatPath();

    Vec3 emitDir;
    float pdfDirSa = 0.0f;
    if (!startLightPath(scene, rng, light[0], emitDir, pdfDirSa)) return;
    WalkConfig cfg;
    cfg.eyePath = false;
    cfg.dispersion = dispersion;
    const bool inf = lightIsInfinite(scene.lights[light[0].lightIndex]);
    const Vec3 o = inf ? light[0].p : offsetRayOrigin(light[0].p, light[0].ng, emitDir);
    const int nLight = randomWalk(scene, tracer, rng, light, 1, o, emitDir, pdfDirSa, maxVerts, cfg);
    correctInfiniteLightSubpathPdfs(scene, light, nLight, emitDir);
    if (nLight < 2) return;

    for (int s = 2; s <= nLight; ++s) {
        const Vert& v = light[s - 1];
        if (v.type != VType::Surface || !v.connectable) continue;
        bool lightPrefixCaustic = false;
        for (int i = 1; i < s - 1; ++i)
            if (light[i].nearSpec && materialContributesCaustics(light[i].mat)) lightPrefixCaustic = true;
        if (!settings.caustics) {
            if (lightPrefixCaustic) continue;
        } else if (lightPrefixCaustic && light[0].lightIndex >= 0 &&
                   !lightContributesCaustics(scene.lights[light[0].lightIndex])) {
            continue;
        }
        float px = 0.0f, py = 0.0f, cosTheta = 0.0f, dist2 = 0.0f;
        if (!projectToPixel(camProj, v.p, px, py, cosTheta, dist2) || dist2 < 1e-8f) continue;
        const Vec3 toCam = normalize(camProj.camPos - v.p);
        const Vec3 f = bsdfF(v, v.wo, toCam);
        if (isBlack(f)) continue;
        if (!connectionVisible(scene, tracer, v.p, v.ng, camProj.camPos, -1)) continue;

        const float pdfOmega = cameraPdfOmega(camProj, cosTheta);
        const float cosV = fabsf(dot(v.ns, toCam));
        Vec3 c = v.beta * f * (cosV * pdfOmega / dist2);
        if (s >= 2) c = clampContribution(c, lightTraceSplatClamp(settings));
        if (!isFinite(c)) continue;
        if (dispersion && dispersion->heroChannel >= 0 && dispersion->used &&
            (dispersion->mode == kDispersionHero || dispersion->mode == kDispersionOptimized ||
             dispersion->mode == kDispersionSpectral3)) {
            const int ch = dispersion->heroChannel;
            const float hero = (ch == 0 ? c.x : (ch == 1 ? c.y : c.z)) * 3.0f;
            c = Vec3(0.0f);
            if (ch == 0) c.x = hero;
            else if (ch == 1) c.y = hero;
            else c.z = hero;
        }
        splatFb->addSplat(int(px), int(py), c);
    }
}

}  // namespace bdpt
}  // namespace sol
