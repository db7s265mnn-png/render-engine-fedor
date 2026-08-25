// Owns the render thread and drives a RenderDevice progressively.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "render/render_device.h"

namespace sol {

class RenderSession {
public:
    RenderSession();
    ~RenderSession();

    RenderSession(const RenderSession&) = delete;
    RenderSession& operator=(const RenderSession&) = delete;

    // Replaces the scene. A render in flight is cancelled; call start() again.
    void setScene(ScenePtr scene);
    ScenePtr scene() const;

    void start();
    void stop();
    void restart();
    // Restarts only the accumulation, e.g. after a camera move.
    void invalidate();
    // Picks up in-place scene edits (camera, film settings) and resets the
    // accumulation without rebuilding acceleration structures.
    void updateSceneData();
    // Interactive camera / film tweak while a render is running: abort the current
    // sample and restart accumulation on the *render* thread (no UI-thread join).
    // Call after mutating scene->camera (etc.). If idle, falls back to update+start.
    void pushInteractiveRestart();
    // Mouse-drag orbit/pan/dolly: same scene, 1 spp, adaptive film (64→32→16→8→4).
    // Display-only fat-pixel upscale; release holds until full-res 1 spp.
    void setInteractivePreview(bool on);
    bool interactivePreview() const { return interactivePreview_.load(std::memory_order_relaxed); }
    // Bump so the worker can pick up the latest camera after the current sample.
    // Does not cancel an in-flight preview frame.
    void noteCameraMoved();
    // Stop the render thread and free *all* previous-render state: device BVH,
    // cooked scene, accumulation, and display hold. Call before a heavy
    // tessellation so peak RAM is not previous_render + new_tess.
    void discardPreviousRender();
    // Free device / cooked scene / accum buffers but keep `displayHold_` so the
    // viewport can keep showing the last beauty while re-dicing (Stop→Start).
    void releaseDeviceKeepDisplay();

    bool isRendering() const { return rendering_.load(std::memory_order_relaxed); }
    RenderProgress progress() const;

    Framebuffer& framebuffer() { return framebuffer_; }
    const Framebuffer& framebuffer() const { return framebuffer_; }
    Image displayImage() const;
    Image linearImage() const;

    // Called from the render thread whenever new samples are available.
    void setUpdateCallback(std::function<void()> callback);
    // Called once a render finishes or is cancelled.
    void setFinishedCallback(std::function<void()> callback);

    std::string backendName() const;
    void waitForCompletion();

private:
    void threadMain();
    bool prepareDevice(std::string& error);

    ScenePtr scene_;
    mutable std::mutex sceneMutex_;

    RenderDevicePtr device_;
    int deviceBackend_ = -1;
    int deviceThreads_ = -1;

    Framebuffer framebuffer_;
    // Last resolved display image — shown while the accumulation buffer is empty
    // so IPR restarts do not flash black tile holes.
    mutable Image displayHold_;
    mutable std::mutex displayHoldMutex_;
    std::thread thread_;
    std::atomic<bool> cancel_{false};
    std::atomic<bool> hardStop_{false};
    std::atomic<bool> softRestart_{false};
    std::atomic<bool> rendering_{false};
    std::atomic<bool> sceneDirty_{true};
    std::atomic<bool> interactivePreview_{false};
    std::atomic<bool> completeFramesOnly_{false};
    std::atomic<int> navDivider_{kNavPreviewDividerStart};
    std::atomic<uint64_t> cameraEpoch_{0};

    mutable std::mutex progressMutex_;
    RenderProgress progress_;

    std::function<void()> updateCallback_;
    std::function<void()> finishedCallback_;
    mutable std::mutex callbackMutex_;
};

}  // namespace sol
