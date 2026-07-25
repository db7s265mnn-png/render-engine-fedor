#include "render/render_session.h"

#include <algorithm>
#include <chrono>

#include "core/log.h"

namespace sol {

RenderSession::RenderSession() = default;

RenderSession::~RenderSession() { stop(); }

void RenderSession::setScene(ScenePtr scene) {
    stop();
    {
        std::lock_guard<std::mutex> lock(sceneMutex_);
        scene_ = std::move(scene);
    }
    sceneDirty_.store(true, std::memory_order_relaxed);
}

ScenePtr RenderSession::scene() const {
    std::lock_guard<std::mutex> lock(sceneMutex_);
    return scene_;
}

void RenderSession::setUpdateCallback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    updateCallback_ = std::move(callback);
}

void RenderSession::setFinishedCallback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    finishedCallback_ = std::move(callback);
}

std::string RenderSession::backendName() const { return device_ ? device_->name() : std::string("none"); }

RenderProgress RenderSession::progress() const {
    std::lock_guard<std::mutex> lock(progressMutex_);
    return progress_;
}

Image RenderSession::displayImage() const {
    ScenePtr scene = this->scene();
    RenderSettingsData settings = scene ? scene->settings : RenderSettingsData();
    return framebuffer_.resolveDisplay(settings);
}

Image RenderSession::linearImage() const { return framebuffer_.resolveLinear(); }

void RenderSession::stop() {
    cancel_.store(true, std::memory_order_relaxed);
    if (thread_.joinable()) thread_.join();
    cancel_.store(false, std::memory_order_relaxed);
    rendering_.store(false, std::memory_order_relaxed);
}

void RenderSession::start() {
    stop();
    ScenePtr scene = this->scene();
    if (!scene) {
        logWarning("Render start requested without a scene");
        return;
    }
    rendering_.store(true, std::memory_order_relaxed);
    thread_ = std::thread([this] { threadMain(); });
}

void RenderSession::restart() {
    stop();
    sceneDirty_.store(true, std::memory_order_relaxed);
    start();
}

void RenderSession::invalidate() {
    stop();
    framebuffer_.clear();
    start();
}

void RenderSession::waitForCompletion() {
    if (thread_.joinable()) thread_.join();
    rendering_.store(false, std::memory_order_relaxed);
}

bool RenderSession::prepareDevice(std::string& error) {
    ScenePtr scene = this->scene();
    if (!scene) {
        error = "no scene";
        return false;
    }
    const int backend = scene->settings.backend;
    const int threads = scene->settings.threads;
    if (!device_ || backend != deviceBackend_ || threads != deviceThreads_) {
        device_.reset();
        if (backend == kBackendGpuOptix) {
            device_ = createOptixDevice();
            if (!device_) {
                logWarning("OptiX backend is unavailable, falling back to Embree");
                device_ = createEmbreeDevice(threads);
            }
        } else {
            device_ = createEmbreeDevice(threads);
        }
        deviceBackend_ = backend;
        deviceThreads_ = threads;
        sceneDirty_.store(true, std::memory_order_relaxed);
    }
    if (!device_) {
        error = "no render device available";
        return false;
    }
    if (sceneDirty_.exchange(false, std::memory_order_relaxed)) {
        if (!device_->buildScene(scene, error)) return false;
    }
    return true;
}

void RenderSession::threadMain() {
    ScenePtr scene = this->scene();
    if (!scene) {
        rendering_.store(false, std::memory_order_relaxed);
        return;
    }

    std::string error;
    if (!prepareDevice(error)) {
        logError("Render failed: " + error);
        {
            std::lock_guard<std::mutex> lock(progressMutex_);
            progress_.running = false;
            progress_.message = error;
        }
        rendering_.store(false, std::memory_order_relaxed);
        std::function<void()> finished;
        {
            std::lock_guard<std::mutex> lock(callbackMutex_);
            finished = finishedCallback_;
        }
        if (finished) finished();
        return;
    }

    const RenderSettingsData& settings = scene->settings;
    const int width = std::max(1, settings.resolutionX);
    const int height = std::max(1, settings.resolutionY);
    if (framebuffer_.width() != width || framebuffer_.height() != height) framebuffer_.resize(width, height);

    const int startSample = framebuffer_.sampleCount();
    const int targetSamples = std::max(1, settings.samplesPerPixel);

    {
        std::lock_guard<std::mutex> lock(progressMutex_);
        progress_.running = true;
        progress_.samplesDone = startSample;
        progress_.samplesTarget = targetSamples;
        progress_.backendName = device_->name();
        progress_.message.clear();
        progress_.elapsedSeconds = 0.0;
        progress_.samplesPerSecond = 0.0;
    }

    const auto startTime = std::chrono::steady_clock::now();
    auto lastNotify = startTime;

    for (int sample = startSample; sample < targetSamples; ++sample) {
        if (cancel_.load(std::memory_order_relaxed)) break;
        device_->renderSample(framebuffer_, sample, cancel_);
        if (cancel_.load(std::memory_order_relaxed)) break;

        framebuffer_.setSampleCount(sample + 1);
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - startTime).count();
        {
            std::lock_guard<std::mutex> lock(progressMutex_);
            progress_.samplesDone = sample + 1;
            progress_.elapsedSeconds = elapsed;
            progress_.samplesPerSecond = elapsed > 0.0 ? double(sample + 1 - startSample) / elapsed : 0.0;
        }

        // Throttle UI notifications: early samples update immediately, later
        // ones at roughly 10 Hz.
        const double sinceNotify = std::chrono::duration<double>(now - lastNotify).count();
        if (sample < 4 || sinceNotify > 0.1 || sample + 1 == targetSamples) {
            lastNotify = now;
            std::function<void()> update;
            {
                std::lock_guard<std::mutex> lock(callbackMutex_);
                update = updateCallback_;
            }
            if (update) update();
        }
    }

    {
        std::lock_guard<std::mutex> lock(progressMutex_);
        progress_.running = false;
    }
    rendering_.store(false, std::memory_order_relaxed);

    std::function<void()> finished;
    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        finished = finishedCallback_;
    }
    if (finished) finished();
}

}  // namespace sol
