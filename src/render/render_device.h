// Interface implemented by the Embree (CPU) and OptiX (GPU) backends.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>

#include "render/framebuffer.h"
#include "scene/scene.h"

#include <vector>

namespace sol {

struct RenderProgress {
    int samplesDone = 0;
    int samplesTarget = 0;
    double elapsedSeconds = 0.0;
    double samplesPerSecond = 0.0;
    double backendGpuMs = 0.0;  // CUDA time of last sample; 0 on Embree
    bool running = false;
    std::string backendName;
    std::string message;
};

// Optional mid-sample preview hook (e.g. after each bootstrap phase).
using RenderMidProgressFn = std::function<void()>;

// XPU work split: checkerboard tiles. GPU-only / CPU-only leave xpuTileRole < 0.
struct RenderSampleOptions {
    int xpuTileRole = -1;   // -1 = all pixels, 0 = CPU tiles, 1 = GPU tiles
    int xpuTileSize = 32;
    int xpuGpuParity = 0;   // GPU owns (tx+ty)%2 == parity
    bool skipFramebufferStore = false;  // OptiX: keep accum internal (XPU merge)
};

class RenderDevice {
public:
    virtual ~RenderDevice() = default;

    virtual std::string name() const = 0;
    virtual bool isAvailable() const = 0;
    // CUDA event time of the last renderSample(); 0 on CPU backends.
    virtual double lastGpuSampleMs() const { return 0.0; }

    // Uploads/builds acceleration structures. Returns false and fills `error`
    // when the scene cannot be prepared.
    virtual bool buildScene(const ScenePtr& scene, std::string& error) = 0;

    // Renders `sampleIndex` (one sample per pixel) into the framebuffer.
    // `cancel` is polled frequently so the UI stays responsive.
    // `midProgress` may be invoked during long passes (bootstrap) for smoother IPR.
    virtual void renderSample(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                              const RenderMidProgressFn& midProgress,
                              const RenderSampleOptions* options = nullptr) = 0;

    // OptiX internal accum after skipFramebufferStore (XPU merge). Count is width*height.
    virtual bool copyInternalAccum(Vec4* /*dst*/, size_t /*count*/) const { return false; }

    // Picks up in-place edits of the scene that do not touch geometry, such as
    // a camera move or a change of film settings, without rebuilding the
    // acceleration structures.
    virtual void refreshSceneData() {}

    virtual void release() {}

    // PT Spectral: optional fixed-bin spectral accumulation (Embree only).
    virtual bool copySpectralBins(int& /*width*/, int& /*height*/, int& /*bins*/,
                                  std::vector<float>& /*accum*/) const {
        return false;
    }
};

using RenderDevicePtr = std::shared_ptr<RenderDevice>;

// Factories. The OptiX one returns nullptr when the build has no CUDA support
// or no capable device is present.
RenderDevicePtr createEmbreeDevice(int threadCount = 0);
RenderDevicePtr createOptixDevice();
RenderDevicePtr createXpuDevice(int threadCount = 0);
bool optixBackendCompiledIn();

// Align CPU Path Tracer with the OptiX wavefront estimator (1 NEE, box filter,
// no MNEE/SSS/OpenPGL/motion blur / polynomial optics).
void applyXpuDeviceMatch(Scene& scene);

// Live CUDA + optixInit probe (cached, never on the Qt UI thread).
// createOptixDevice() overwrites this with the real initialize() result.
bool optixRuntimeAvailable(std::string* error = nullptr);
bool optixRuntimeProbePending();

}  // namespace sol
