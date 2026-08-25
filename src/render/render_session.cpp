#include "render/render_session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <new>
#include <string>
#include <thread>

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

    // After a clear / restart / re-dice, keep the last finished preview until a
    // complete sample lands — never resolve scanline / bootstrap holes.
    if (!framebuffer_.hasAccumulatedData() || !framebuffer_.isPresentable()) {
        std::lock_guard<std::mutex> lock(displayHoldMutex_);
        if (!displayHold_.empty()) {
            // FB may be released (size 0) during tess, or already resized to match.
            const int fw = framebuffer_.width();
            const int fh = framebuffer_.height();
            if (fw <= 0 || fh <= 0 ||
                (displayHold_.width() == fw && displayHold_.height() == fh)) {
                return displayHold_;
            }
            // Nav preview 1/4 ↔ full: keep the last beauty instead of a charcoal flash.
            const int hw = displayHold_.width();
            const int hh = displayHold_.height();
            if (hw > 0 && hh > 0) {
                const double fa = double(fw) / double(fh);
                const double ha = double(hw) / double(hh);
                if (std::abs(fa - ha) < 0.05) return displayHold_;
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

void RenderSession::setInteractivePreview(bool on) {
    const bool was = interactivePreview_.exchange(on, std::memory_order_relaxed);
    if (was == on) return;
    // Coarsest first picture on press (RenderMan PP). Refine after each frame.
    if (on) navDivider_.store(kNavPreviewDividerStart, std::memory_order_relaxed);
    completeFramesOnly_.store(true, std::memory_order_relaxed);
    if (rendering_.load(std::memory_order_relaxed) && thread_.joinable()) {
        softRestart_.store(true, std::memory_order_relaxed);
        cancel_.store(true, std::memory_order_relaxed);
    }
}

void RenderSession::noteCameraMoved() {
    cameraEpoch_.fetch_add(1, std::memory_order_relaxed);
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
        if (backend == kBackendGpuOptix || backend == kBackendXpu) {
            const char* label = backend == kBackendXpu ? "XPU (Embree+OptiX)" : "GPU (OptiX)";
            if (!optixBackendCompiledIn()) {
                error = std::string(label) + " is selected, but this build has no OptiX/CUDA backend. "
                        "Rebuild with BUILD_WINDOWS.bat.";
                return false;
            }
            if (scene->settings.integrator != kIntegratorPathTracer) {
                error = std::string(label) + " only supports Path Tracer. Switch Integrator to Path Tracer, "
                        "or set Render Device to CPU (Embree).";
                return false;
            }
            if (backend == kBackendXpu) {
                device_ = createXpuDevice(threads);
                if (!device_) {
                    std::string err;
                    optixRuntimeAvailable(&err);
                    error = "XPU (Embree+OptiX) cannot start";
                    if (!err.empty()) error += ": " + err;
                    else error += ": NVIDIA GPU / driver / CUDA runtime unavailable";
                    return false;
                }
            } else {
                device_ = createOptixDevice();
                if (!device_) {
                    std::string err;
                    optixRuntimeAvailable(&err);
                    error = "GPU (OptiX) cannot start";
                    if (!err.empty()) error += ": " + err;
                    else error += ": NVIDIA GPU / driver / CUDA runtime unavailable";
                    return false;
                }
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
            } else if (backend == kBackendXpu) {
                error = std::string("XPU (Embree+OptiX) failed to build the scene") +
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
        if (device_) {
            try {
                device_->finishRender();
            } catch (...) {
            }
        }
        logError(error);
        {
            std::lock_guard<std::mutex> lock(progressMutex_);
            progress_.running = false;
            progress_.message = error;
            if (scene && scene->settings.backend == kBackendXpu)
                progress_.backendName = "XPU / Embree+OptiX";
            else if (scene && scene->settings.backend == kBackendGpuOptix)
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
    const int fullWidth = std::max(1, settings.resolutionX);
    const int fullHeight = std::max(1, settings.resolutionY);
    const int targetSamples = std::max(1, settings.samplesPerPixel);

    int navDivUsed = 1;
    auto applyFilmSize = [&](int& sampleIndex) -> bool {
        const bool preview = interactivePreview_.load(std::memory_order_relaxed);
        navDivUsed = preview ? clampNavPreviewDivider(navDivider_.load(std::memory_order_relaxed)) : 1;
        const int width = std::max(1, fullWidth / navDivUsed);
        const int height = std::max(1, fullHeight / navDivUsed);
        if (framebuffer_.width() != width || framebuffer_.height() != height) {
            framebuffer_.resize(width, height);
            sampleIndex = 0;
        }
        return preview;
    };

    int startSample = framebuffer_.sampleCount();
    applyFilmSize(startSample);

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
        progress_.noiseSkipCount = 0;
        progress_.noisePixelCount = framebuffer_.width() * framebuffer_.height();
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
            // Keep the last complete blit. Do not resolve an empty / half-filled FB.
            continue;
        }

        if (cancel_.load(std::memory_order_relaxed)) break;

        const bool preview = applyFilmSize(sample);
        if (preview && device_) device_->refreshSceneData();
        const uint64_t epochLaunched = cameraEpoch_.load(std::memory_order_relaxed);

        // Never blit a partial sample. Embree bootstrap 2x2 / Progressive scanlines
        // would otherwise paint holes; OptiX abort-before-D2H would too.
        const RenderMidProgressFn midProgress;

        RenderSampleOptions opt;
        const bool xpu = scene->settings.backend == kBackendXpu;
        const bool gpu = scene->settings.backend == kBackendGpuOptix;
        if (xpu) {
            opt.xpuRemainingSamples = targetSamples - sample;
            opt.xpuTargetSamples = targetSamples;
            opt.xpuSchedule = scene->settings.xpuSchedule;
        } else if (gpu) {
            opt.xpuRemainingSamples = targetSamples - sample;
        }
        if (preview) {
            opt.navPreview = true;
            opt.xpuRemainingSamples = 1;
        }

        const auto pass0 = std::chrono::steady_clock::now();
        try {
            device_->renderSample(framebuffer_, sample, cancel_, midProgress, &opt);
        } catch (const std::exception& ex) {
            const int backend = scene->settings.backend;
            const std::string prefix = backend == kBackendXpu
                                           ? "XPU (Embree+OptiX) render failed: "
                                           : (backend == kBackendGpuOptix ? "GPU (OptiX) render failed: "
                                                                          : "Render failed: ");
            fail(prefix + ex.what());
            return;
        }

        if (hardStop_.load(std::memory_order_relaxed)) break;
        if (softRestart_.load(std::memory_order_relaxed)) continue;  // handled at loop top
        if (cancel_.load(std::memory_order_relaxed)) break;

        const int sampleStep = device_ ? device_->lastCompletedSamples() : 1;
        if (sampleStep <= 0) continue;
        const double passMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pass0)
                .count();
        framebuffer_.setSampleCount(sample + sampleStep);
        const float noiseT = (preview || scene->settings.samplingDebug != 0)
                                 ? 0.0f
                                 : scene->settings.noiseThreshold;
        if (noiseT > 0.0f) {
            framebuffer_.refreshNoiseOracle(noiseT, sample + sampleStep, targetSamples);
        }
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - sampleStartTime).count();
        {
            std::lock_guard<std::mutex> lock(progressMutex_);
            progress_.samplesDone = sample + sampleStep;
            progress_.elapsedSeconds = elapsed;
            progress_.samplesPerSecond = elapsed > 0.0 ? double(sample + sampleStep) / elapsed : 0.0;
            progress_.backendGpuMs = device_ ? device_->lastGpuSampleMs() : 0.0;
            progress_.noiseSkipCount = framebuffer_.noiseOracleSkipCount();
            progress_.noisePixelCount = framebuffer_.width() * framebuffer_.height();
        }

        // Only present a finished sample — never scanline / bootstrap holes.
        framebuffer_.setPresentable(true);
        notifyUi(true);
        if (!preview) completeFramesOnly_.store(false, std::memory_order_relaxed);

        if (preview) {
            sample = 0;
            const int nextDiv = adaptNavPreviewDivider(navDivUsed, passMs);
            navDivider_.store(nextDiv, std::memory_order_relaxed);
            // Same camera, cheaper than target: refine 64→32→16→8→4 without waiting.
            const bool refine = nextDiv < navDivUsed;
            while (!refine && !hardStop_.load(std::memory_order_relaxed) &&
                   !softRestart_.load(std::memory_order_relaxed) &&
                   !cancel_.load(std::memory_order_relaxed)) {
                if (!interactivePreview_.load(std::memory_order_relaxed)) break;
                if (cameraEpoch_.load(std::memory_order_relaxed) != epochLaunched) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
            }
            if (hardStop_.load(std::memory_order_relaxed)) break;
            if (softRestart_.load(std::memory_order_relaxed) || cancel_.load(std::memory_order_relaxed) ||
                !interactivePreview_.load(std::memory_order_relaxed)) {
                continue;
            }
            if (device_) device_->refreshSceneData();
            framebuffer_.clear();
            sample = 0;
            continue;
        }

        sample += sampleStep;
        if (framebuffer_.noiseOracleDone()) {
            std::lock_guard<std::mutex> lock(progressMutex_);
            progress_.samplesDone = targetSamples;
            break;
        }
    }

    if (device_) {
        try {
            device_->finishRender();
        } catch (...) {
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
