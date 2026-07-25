// Interface implemented by the Embree (CPU) and OptiX (GPU) backends.
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "render/framebuffer.h"
#include "scene/scene.h"

namespace sol {

struct RenderProgress {
    int samplesDone = 0;
    int samplesTarget = 0;
    double elapsedSeconds = 0.0;
    double samplesPerSecond = 0.0;
    bool running = false;
    std::string backendName;
    std::string message;
};

class RenderDevice {
public:
    virtual ~RenderDevice() = default;

    virtual std::string name() const = 0;
    virtual bool isAvailable() const = 0;

    // Uploads/builds acceleration structures. Returns false and fills `error`
    // when the scene cannot be prepared.
    virtual bool buildScene(const ScenePtr& scene, std::string& error) = 0;

    // Renders `sampleIndex` (one sample per pixel) into the framebuffer.
    // `cancel` is polled frequently so the UI stays responsive.
    virtual void renderSample(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel) = 0;

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
bool optixBackendCompiledIn();

}  // namespace sol
