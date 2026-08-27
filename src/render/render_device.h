// Interface implemented by the Embree (CPU) and OptiX (GPU) backends.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>

#include "render/framebuffer.h"
#include "scene/scene.h"

namespace sol {

// Nav preview (tumble): same integrator / VDB / lights. Only the film shrinks.
// Cycles resolution_divider + RenderMan Progressive Pixels: start coarse so the
// first picture is a screenful of fat pixels, then refine 32→16→8→4 from the
// last frame's wall time. Viewport nearest-neighbor upscale is the splat.
constexpr int kNavPreviewDividerMin = 4;
constexpr int kNavPreviewDividerMax = 32;
constexpr int kNavPreviewDividerStart = 16;
constexpr double kNavPreviewTargetMs = 80.0;

inline int clampNavPreviewDivider(int d) {
    if (d < kNavPreviewDividerMin) return kNavPreviewDividerMin;
    if (d > kNavPreviewDividerMax) return kNavPreviewDividerMax;
    int p = kNavPreviewDividerMin;
    while (p * 2 <= d && p * 2 <= kNavPreviewDividerMax) p *= 2;
    return p;
}

inline int adaptNavPreviewDivider(int divider, double frameMs,
                                  double targetMs = kNavPreviewTargetMs) {
    int d = clampNavPreviewDivider(divider);
    if (!(frameMs > 0.0) || !(targetMs > 0.0)) return d;
    if (frameMs > targetMs * 1.8 && d < kNavPreviewDividerMax) return d * 2;
    if (frameMs < targetMs * 0.4 && d > kNavPreviewDividerMin) return d / 2;
    return d;
}

struct RenderProgress {
    int samplesDone = 0;
    int samplesTarget = 0;
    int noiseSkipCount = 0;   // Variance oracle: pixels that stopped taking samples
    int noisePixelCount = 0;  // width*height; 0 if the film is empty
    double elapsedSeconds = 0.0;
    double samplesPerSecond = 0.0;
    double backendGpuMs = 0.0;  // CUDA time of last sample; 0 on Embree
    bool running = false;
    std::string backendName;
    std::string message;
};

// Optional mid-sample preview hook (e.g. after each bootstrap phase).
using RenderMidProgressFn = std::function<void()>;

// XPU Mixture scheduling plus optional sub-rect launches.
// Standalone devices leave remaining/target at 0 and clip empty.
struct RenderSampleOptions {
    int xpuRemainingSamples = 0;        // spp left in the session (legacy cap)
    int xpuTargetSamples = 0;           // absolute spp target for Mixture GPU stop
    int xpuSchedule = 0;                // XpuSchedule: 0 Overlap, 1 Mixture
    int clipX0 = 0, clipY0 = 0, clipX1 = 0, clipY1 = 0;  // exclusive x1/y1; empty = full frame
    bool skipFramebufferStore = false;  // OptiX: keep this sample internal (XPU add)
    bool resetAccum = false;            // OptiX: memset accum before this sample
    bool deferHostCopy = false;         // OptiX: skip D2H; caller downloads the batch
    bool skipPhotonRebuild = false;     // Embree: keep the photon map from an earlier tile
    bool skipGuidingCommit = false;     // Embree: delay OpenPGL commit until finishSample()
    // Same integrator / scene; session shrank the film. Forces 1 spp present.
    bool navPreview = false;
};

// Pixel grid is the framebuffer. Camera NDC must use this, not the authored
// Film res — otherwise a smaller FB is a top-left crop, not a 1/4 downscale.
inline void bindFilmToFramebuffer(SceneView& scene, int width, int height) {
    scene.settings.resolutionX = width > 0 ? width : 1;
    scene.settings.resolutionY = height > 0 ? height : 1;
}

class RenderDevice {
public:
    virtual ~RenderDevice() = default;

    virtual std::string name() const = 0;
    virtual bool isAvailable() const = 0;
    // CUDA event time of the last renderSample(); 0 on CPU backends.
    virtual double lastGpuSampleMs() const { return 0.0; }
    // Samples folded into the film by the last renderSample() (XPU may do many).
    virtual int lastCompletedSamples() const { return 1; }

    // Uploads/builds acceleration structures. Returns false and fills `error`
    // when the scene cannot be prepared.
    virtual bool buildScene(const ScenePtr& scene, std::string& error) = 0;

    // Renders `sampleIndex` (one sample per pixel) into the framebuffer.
    // `cancel` is polled frequently so the UI stays responsive.
    // `midProgress` may be invoked after a *complete* bootstrap phase. The session
    // leaves it empty so the viewport only ever sees a hole-free sample.
    virtual void renderSample(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                              const RenderMidProgressFn& midProgress,
                              const RenderSampleOptions* options = nullptr) = 0;

    // OptiX internal accum after skipFramebufferStore (XPU merge). Count is width*height.
    virtual bool copyInternalAccum(Vec4* /*dst*/, size_t /*count*/) const { return false; }
    // D2H of the device accum (XPU batch). Default falls back to copyInternalAccum.
    virtual bool downloadInternalAccum(Vec4* dst, size_t count) { return copyInternalAccum(dst, count); }
    // D2H of a pixel rect. dst is a full-frame buffer with pitch dstPitchPixels.
    virtual bool downloadInternalAccumRect(Vec4* /*dst*/, int /*dstPitchPixels*/, int /*x0*/, int /*y0*/,
                                           int /*x1*/, int /*y1*/) {
        return false;
    }
    virtual bool downloadInternalLumSq(float* /*dst*/, size_t /*count*/) { return false; }

    // Embree: commit delayed OpenPGL training after a batch of clipped tiles.
    virtual void finishSample() {}
    // XPU: stop the persistent GPU worker at the end of a session / on failure.
    virtual void finishRender() {}

    // Picks up in-place edits of the scene that do not touch geometry, such as
    // a camera move or a change of film settings, without rebuilding the
    // acceleration structures.
    virtual void refreshSceneData() {}

    virtual void release() {}
};

using RenderDevicePtr = std::shared_ptr<RenderDevice>;

// Factories. The OptiX one returns nullptr when the build has no CUDA support
// or no capable device is present.
RenderDevicePtr createEmbreeDevice(int threadCount = 0);
RenderDevicePtr createOptixDevice();
RenderDevicePtr createXpuDevice(int threadCount = 0);
bool optixBackendCompiledIn();

// Live CUDA + optixInit probe (cached). HUD must not start this — only
// createOptixDevice() on the render thread. initialize() overwrites the result.
bool optixRuntimeAvailable(std::string* error = nullptr);
bool optixRuntimeProbePending();

}  // namespace sol
