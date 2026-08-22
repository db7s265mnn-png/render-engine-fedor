#include "render/render_session.h"

#include <algorithm>
#include <chrono>
#include <new>
#include <string>

#include "core/log.h"
#include "io/ocio_util.h"

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

    // Keep OCIO config in sync with Film settings for Nuke-style views.
    ocioEnsureConfig(settings.ocioUseEnv != 0, settings.ocioConfigPath);

    // After a clear / restart / re-dice, keep the last finished preview until the
    // first bootstrap pixels land — avoids a harsh black flash.
    if (!framebuffer_.hasAccumulatedData()) {
        std::lock_guard<std::mutex> lock(displayHoldMutex_);
        if (!displayHold_.empty()) {
            // FB may be released (size 0) during tess, or already resized to match.
            if (framebuffer_.width() <= 0 || framebuffer_.height() <= 0 ||
                (displayHold_.width() == framebuffer_.width() &&
                 displayHold_.height() == framebuffer_.height())) {
                return displayHold_;
            }
        }
        // Soft charcoal placeholder (less jarring than pure black).
        return Image(std::max(1, framebuffer_.width()), std::max(1, framebuffer_.height()),
                     Vec4(0.07f, 0.07f, 0.08f, 1.0f));
    }

    Image image = framebuffer_.resolveDisplay(settings);
    {
        std::lock_guard<std::mutex> lock(displayHoldMutex_);
        displayHold_ = image;
    }
    return image;
}

Image RenderSession::linearImage() const { return framebuffer_.resolveLinear(); }

void RenderSession::stop() {
    hardStop_.store(true, std::memory_order_relaxed);
    cancel_.store(true, std::memory_order_relaxed);
    if (thread_.joinable()) thread_.join();
    cancel_.store(false, std::memory_order_relaxed);
    hardStop_.store(false, std::memory_order_relaxed);
    softRestart_.store(false, std::memory_order_relaxed);
    rendering_.store(false, std::memory_order_relaxed);
}

void RenderSession::start() {
    stop();
    ScenePtr scene = this->scene();
    if (!scene) {
        logWarning("Render start requested without a scene");
        return;
    }
    hardStop_.store(false, std::memory_order_relaxed);
    softRestart_.store(false, std::memory_order_relaxed);
    cancel_.store(false, std::memory_order_relaxed);
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

void RenderSession::updateSceneData() {
    stop();
    if (device_) device_->refreshSceneData();
    framebuffer_.clear();
}

void RenderSession::pushInteractiveRestart() {
    // Already running: ask the worker to abort the current sample and restart
    // accumulation after refreshSceneData — UI thread must not join().
    if (rendering_.load(std::memory_order_relaxed) && thread_.joinable()) {
        softRestart_.store(true, std::memory_order_relaxed);
        cancel_.store(true, std::memory_order_relaxed);
        return;
    }
    // Idle: full path (stop is cheap when no thread).
    updateSceneData();
    start();
}

void RenderSession::discardPreviousRender() {
    stop();
    {
        std::lock_guard<std::mutex> lock(sceneMutex_);
        scene_.reset();
    }
    if (device_) {
        device_->release();
        device_.reset();
    }
    deviceBackend_ = -1;
    deviceThreads_ = -1;
    framebuffer_.release();
    {
        std::lock_guard<std::mutex> lock(displayHoldMutex_);
        displayHold_ = Image();
    }
    {
        std::lock_guard<std::mutex> lock(progressMutex_);
        progress_ = RenderProgress();
    }
    sceneDirty_.store(true, std::memory_order_relaxed);
}

void RenderSession::releaseDeviceKeepDisplay() {
    stop();
    {
        std::lock_guard<std::mutex> lock(sceneMutex_);
        scene_.reset();
    }
    if (device_) {
        device_->release();
        device_.reset();
    }
    deviceBackend_ = -1;
    deviceThreads_ = -1;
    // Free accum RAM; keep displayHold_ so Stop→Start re-dice does not flash black.
    framebuffer_.release();
    {
        std::lock_guard<std::mutex> lock(progressMutex_);
        progress_ = RenderProgress();
    }
    sceneDirty_.store(true, std::memory_order_relaxed);
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
            if (!optixBackendCompiledIn()) {
                error = "GPU (OptiX) is selected, but this build has no OptiX/CUDA backend. "
                        "Rebuild with BUILD_WINDOWS.bat.";
                return false;
            }
            if (scene->settings.integrator != kIntegratorPathTracer) {
                error = "GPU (OptiX) only supports Path Tracer. Switch Integrator to Path Tracer, "
                        "or set Render Backend to CPU (Embree).";
                return false;
            }
            device_ = createOptixDevice();
            if (!device_) {
                std::string err;
                optixRuntimeAvailable(&err);
                error = "GPU (OptiX) cannot start";
                if (!err.empty()) error += ": " + err;
                else error += ": NVIDIA GPU / driver / CUDA runtime unavailable";
                return false;
            }
        } else {
            device_ = createEmbreeDevice(threads);
        }
        if (!device_) {
            error = "no render device available";
            return false;
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
        if (!device_->buildScene(scene, error)) {
            if (backend == kBackendGpuOptix) {
                error = std::string("GPU (OptiX) failed to build the scene") +
                        (error.empty() ? std::string() : ": " + error);
            }
            return false;
        }
    }
    return true;
}

void RenderSession::threadMain() {
    ScenePtr scene = this->scene();
    if (!scene) {
        rendering_.store(false, std::memory_order_relaxed);
        return;
    }

    auto fail = [&](const std::string& error) {
        logError(error);
        {
            std::lock_guard<std::mutex> lock(progressMutex_);
            progress_.running = false;
            progress_.message = error;
            if (scene && scene->settings.backend == kBackendGpuOptix)
                progress_.backendName = "GPU / OptiX";
            else if (device_)
                progress_.backendName = device_->name();
        }
        rendering_.store(false, std::memory_order_relaxed);
        std::function<void()> finished;
        {
            std::lock_guard<std::mutex> lock(callbackMutex_);
            finished = finishedCallback_;
        }
        if (finished) finished();
    };

    std::string error;
    try {
        if (!prepareDevice(error)) {
            fail(error.empty() ? std::string("Render failed: no device") : error);
            return;
        }
    } catch (const std::bad_alloc&) {
        fail("out of memory while building the scene (lower Subdiv Iterations)");
        return;
    } catch (const std::exception& ex) {
        fail(ex.what());
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
        progress_.backendGpuMs = 0.0;
    }

    const auto startTime = std::chrono::steady_clock::now();
    auto lastNotify = startTime;
    auto sampleStartTime = startTime;

    auto notifyUi = [&](bool force) {
        const auto now = std::chrono::steady_clock::now();
        const double sinceNotify = std::chrono::duration<double>(now - lastNotify).count();
        // Bootstrap phases update ~20 Hz; early samples immediately; later ~10 Hz.
        if (!force && sinceNotify < 0.05) return;
        lastNotify = now;
        std::function<void()> update;
        {
            std::lock_guard<std::mutex> lock(callbackMutex_);
            update = updateCallback_;
        }
        if (update) update();
    };

    int sample = startSample;
    while (sample < targetSamples) {
        if (hardStop_.load(std::memory_order_relaxed)) break;

        if (softRestart_.exchange(false, std::memory_order_relaxed)) {
            cancel_.store(false, std::memory_order_relaxed);
            if (device_) device_->refreshSceneData();
            framebuffer_.clear();
            sample = 0;
            sampleStartTime = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(progressMutex_);
                progress_.samplesDone = 0;
                progress_.samplesTarget = targetSamples;
                progress_.elapsedSeconds = 0.0;
                progress_.samplesPerSecond = 0.0;
                progress_.backendGpuMs = 0.0;
            }
            notifyUi(true);
            continue;
        }

        if (cancel_.load(std::memory_order_relaxed)) break;

        RenderMidProgressFn midProgress;
        if (sample == 0) {
            midProgress = [&] {
                if (cancel_.load(std::memory_order_relaxed)) return;
                notifyUi(false);
            };
        }

        try {
            device_->renderSample(framebuffer_, sample, cancel_, midProgress);
        } catch (const std::exception& ex) {
            const std::string prefix = (scene->settings.backend == kBackendGpuOptix)
                                           ? "GPU (OptiX) render failed: "
                                           : "Render failed: ";
            fail(prefix + ex.what());
            return;
        }

        if (hardStop_.load(std::memory_order_relaxed)) break;
        if (softRestart_.load(std::memory_order_relaxed)) continue;  // handled at loop top
        if (cancel_.load(std::memory_order_relaxed)) break;

        framebuffer_.setSampleCount(sample + 1);
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - sampleStartTime).count();
        {
            std::lock_guard<std::mutex> lock(progressMutex_);
            progress_.samplesDone = sample + 1;
            progress_.elapsedSeconds = elapsed;
            progress_.samplesPerSecond = elapsed > 0.0 ? double(sample + 1) / elapsed : 0.0;
            progress_.backendGpuMs = device_ ? device_->lastGpuSampleMs() : 0.0;
        }

        // Throttle UI notifications: early samples update immediately, later
        // ones at roughly 10 Hz.
        const double sinceNotify = std::chrono::duration<double>(now - lastNotify).count();
        if (sample < 4 || sinceNotify > 0.1 || sample + 1 == targetSamples) {
            notifyUi(true);
        }
        ++sample;
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
