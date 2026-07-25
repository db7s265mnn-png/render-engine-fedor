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
    std::thread thread_;
    std::atomic<bool> cancel_{false};
    std::atomic<bool> rendering_{false};
    std::atomic<bool> sceneDirty_{true};

    mutable std::mutex progressMutex_;
    RenderProgress progress_;

    std::function<void()> updateCallback_;
    std::function<void()> finishedCallback_;
    mutable std::mutex callbackMutex_;
};

}  // namespace sol
