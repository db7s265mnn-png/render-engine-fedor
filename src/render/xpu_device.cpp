// Hybrid CPU+GPU device. Two schedules:
//   Overlap (default): GPU fills even spp until Embree finishes one odd spp, one D2H add.
//   Mixture (Karma): independent full-frame films, persistent GPU worker, host blend.
#include "render/render_device.h"

#include "core/log.h"
#include "render/xpu_split.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sol {
namespace {

void blendAdd(Vec4* dst, const Vec4* a, const Vec4* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i].x = a[i].x + b[i].x;
        dst[i].y = a[i].y + b[i].y;
        dst[i].z = a[i].z + b[i].z;
        dst[i].w = a[i].w + b[i].w;
    }
}

class XpuDevice final : public RenderDevice {
public:
    XpuDevice(RenderDevicePtr cpu, RenderDevicePtr gpu) : cpu_(std::move(cpu)), gpu_(std::move(gpu)) {}
    ~XpuDevice() override {
        try {
            stopGpuWorker();
        } catch (...) {
        }
    }

    std::string name() const override { return "XPU / Embree+OptiX"; }
    bool isAvailable() const override {
        return cpu_ && cpu_->isAvailable() && gpu_ && gpu_->isAvailable();
    }
    double lastGpuSampleMs() const override { return gpu_ ? gpu_->lastGpuSampleMs() : 0.0; }
    int lastCompletedSamples() const override { return lastCompletedSamples_; }

    bool buildScene(const ScenePtr& scene, std::string& error) override {
        stopGpuWorker();
        if (!cpu_ || !gpu_) {
            error = "XPU devices missing";
            return false;
        }
        if (!gpu_->buildScene(scene, error)) return false;
        std::string cpuError;
        if (!cpu_->buildScene(scene, cpuError)) {
            error = cpuError.empty() ? std::string("Embree failed to build the XPU scene") : cpuError;
            return false;
        }
        resetFrame();
        return true;
    }

    void renderSample(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                      const RenderMidProgressFn& midProgress, const RenderSampleOptions* options) override {
        lastCompletedSamples_ = 0;
        if (!cpu_ || !gpu_) return;
        if (fb.width() <= 0 || fb.height() <= 0) return;

        const RenderSampleOptions opt = options ? *options : RenderSampleOptions{};
        if (opt.xpuSchedule == kXpuScheduleMixture) {
            renderMixture(fb, sampleIndex, cancel, midProgress, opt);
            return;
        }
        stopGpuWorker();
        renderOverlap(fb, sampleIndex, cancel, midProgress, opt);
    }

    void finishRender() override { stopGpuWorker(); }

    void refreshSceneData() override {
        stopGpuWorker();
        if (gpu_) gpu_->refreshSceneData();
        if (cpu_) cpu_->refreshSceneData();
        resetFrame();
    }

    void release() override {
        stopGpuWorker();
        if (gpu_) gpu_->release();
        if (cpu_) cpu_->release();
        gpuHost_.clear();
        gpuHost_.shrink_to_fit();
        cpuFb_.clear();
    }

private:
    void resetFrame() {
        nextGpuSample_ = 0;
        nextCpuSample_ = 1;
        cpuSpp_.store(0, std::memory_order_relaxed);
        gpuSpp_.store(0, std::memory_order_relaxed);
        reportedSpp_ = 0;
        batchIndex_ = 0;
        loggedMode_ = false;
    }

    void stopGpuWorker() {
        gpuStop_.store(true, std::memory_order_relaxed);
        snapshotWanted_.store(false, std::memory_order_relaxed);
        snapshotCv_.notify_all();
        if (gpuThread_.joinable()) gpuThread_.join();
        gpuAlive_.store(false, std::memory_order_release);
        gpuStop_.store(false, std::memory_order_relaxed);
        if (gpuEx_) {
            std::exception_ptr ex = gpuEx_;
            gpuEx_ = nullptr;
            std::rethrow_exception(ex);
        }
    }

    void ensureGpuWorker(Framebuffer& dimFb, const std::atomic<bool>& cancel, int target) {
        fbPtr_ = &dimFb;
        cancelPtr_ = &cancel;
        targetSpp_ = target;
        if (gpuThread_.joinable() && gpuAlive_.load(std::memory_order_acquire)) return;
        if (gpuThread_.joinable()) {
            gpuThread_.join();
            gpuAlive_.store(false, std::memory_order_release);
            if (gpuEx_) {
                std::exception_ptr ex = gpuEx_;
                gpuEx_ = nullptr;
                std::rethrow_exception(ex);
            }
        }
        gpuStop_.store(false, std::memory_order_relaxed);
        gpuEx_ = nullptr;
        gpuAlive_.store(true, std::memory_order_release);
        try {
            gpuThread_ = std::thread([this] { gpuLoop(); });
        } catch (...) {
            gpuAlive_.store(false, std::memory_order_release);
            throw;
        }
    }

    void gpuLoop() {
        try {
            while (!gpuStop_.load(std::memory_order_relaxed) && fbPtr_ && cancelPtr_ &&
                   !cancelPtr_->load(std::memory_order_relaxed)) {
                const int cpu = cpuSpp_.load(std::memory_order_relaxed);
                const int gpu = gpuSpp_.load(std::memory_order_relaxed);
                if (cpu + gpu >= targetSpp_) break;
                RenderSampleOptions gpuOpt;
                gpuOpt.skipFramebufferStore = true;
                gpuOpt.deferHostCopy = true;
                gpuOpt.resetAccum = (gpu == 0);
                gpu_->renderSample(*fbPtr_, gpu, *cancelPtr_, RenderMidProgressFn{}, &gpuOpt);
                gpuSpp_.fetch_add(1, std::memory_order_relaxed);
                if (snapshotWanted_.load(std::memory_order_relaxed)) takeGpuSnapshot();
            }
        } catch (...) {
            gpuEx_ = std::current_exception();
            gpuStop_.store(true, std::memory_order_relaxed);
        }
        try {
            if (fbPtr_ && gpuSpp_.load(std::memory_order_relaxed) > 0) takeGpuSnapshot();
        } catch (...) {
            if (!gpuEx_) gpuEx_ = std::current_exception();
        }
        snapshotWanted_.store(false, std::memory_order_relaxed);
        gpuAlive_.store(false, std::memory_order_release);
        snapshotCv_.notify_all();
    }

    void takeGpuSnapshot() {
        if (!fbPtr_ || !gpu_) {
            snapshotWanted_.store(false, std::memory_order_relaxed);
            snapshotCv_.notify_all();
            return;
        }
        const size_t n = size_t(fbPtr_->width()) * size_t(fbPtr_->height());
        std::lock_guard<std::mutex> lock(gpuHostMutex_);
        gpuHost_.resize(n);
        gpu_->downloadInternalAccum(gpuHost_.data(), n);
        snapshotWanted_.store(false, std::memory_order_relaxed);
        snapshotCv_.notify_all();
    }

    void waitGpuSnapshot() {
        if (!gpuAlive_.load(std::memory_order_acquire)) {
            if (fbPtr_ && gpuSpp_.load(std::memory_order_relaxed) > 0) takeGpuSnapshot();
            return;
        }
        snapshotWanted_.store(true, std::memory_order_relaxed);
        {
            std::unique_lock<std::mutex> lock(gpuHostMutex_);
            snapshotCv_.wait(lock, [&] {
                return !snapshotWanted_.load(std::memory_order_relaxed) ||
                       !gpuAlive_.load(std::memory_order_acquire) ||
                       gpuStop_.load(std::memory_order_relaxed);
            });
        }
        if (snapshotWanted_.load(std::memory_order_relaxed) && fbPtr_ &&
            gpuSpp_.load(std::memory_order_relaxed) > 0) {
            takeGpuSnapshot();
        }
    }

    // GPU launches even spp into device accum until Embree's odd spp returns,
    // then one D2H add. No 1:1 barrier — a faster GPU does more even spp.
    void renderOverlap(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                       const RenderMidProgressFn& midProgress, const RenderSampleOptions& opt) {
        const int width = fb.width();
        const int height = fb.height();
        if (sampleIndex <= 0) {
            nextGpuSample_ = 0;
            nextCpuSample_ = 1;
        }

        int remaining = opt.xpuRemainingSamples;
        if (remaining <= 0) remaining = 2;
        const bool runCpu = remaining >= 2;
        const int maxGpu = runCpu ? remaining - 1 : remaining;
        if (maxGpu <= 0 && !runCpu) return;

        if (!loggedMode_) {
            logInfo("XPU Overlap: GPU fills even spp until Embree finishes one odd spp, then one D2H add; "
                    "Embree keeps MNEE / SSS / OpenPGL / N lights / filters");
            loggedMode_ = true;
        }

        std::atomic<bool> cpuDone{false};
        std::atomic<int> gpuCount{0};
        std::exception_ptr gpuEx;

        std::thread gpuThread([&] {
            try {
                RenderSampleOptions gpuOpt;
                gpuOpt.skipFramebufferStore = true;
                gpuOpt.deferHostCopy = true;
                for (;;) {
                    if (cancel.load(std::memory_order_relaxed)) break;
                    const int done = gpuCount.load(std::memory_order_relaxed);
                    if (done >= maxGpu) break;
                    if (done > 0 && cpuDone.load(std::memory_order_relaxed)) break;
                    gpuOpt.resetAccum = (done == 0);
                    gpu_->renderSample(fb, nextGpuSample_, cancel, RenderMidProgressFn{}, &gpuOpt);
                    nextGpuSample_ += 2;
                    gpuCount.fetch_add(1, std::memory_order_relaxed);
                }
            } catch (...) {
                gpuEx = std::current_exception();
                cpuDone.store(true, std::memory_order_relaxed);
            }
        });

        const auto cpuT0 = std::chrono::steady_clock::now();
        try {
            if (runCpu && !cancel.load(std::memory_order_relaxed)) {
                cpu_->renderSample(fb, nextCpuSample_, cancel, midProgress, nullptr);
                nextCpuSample_ += 2;
            }
        } catch (...) {
            cpuDone.store(true, std::memory_order_relaxed);
            gpuThread.join();
            throw;
        }
        const double cpuMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuT0).count();
        cpuDone.store(true, std::memory_order_relaxed);
        gpuThread.join();
        if (gpuEx) std::rethrow_exception(gpuEx);

        const int g = gpuCount.load(std::memory_order_relaxed);
        const int c = runCpu ? 1 : 0;
        lastCompletedSamples_ = std::max(1, g + c);

        if (g > 0) {
            const size_t pixelCount = size_t(width) * size_t(height);
            gpuHost_.resize(pixelCount);
            if (gpu_->downloadInternalAccum(gpuHost_.data(), pixelCount)) {
                Vec4* dst = fb.data();
                for (size_t i = 0; i < pixelCount; ++i) {
                    dst[i].x += gpuHost_[i].x;
                    dst[i].y += gpuHost_[i].y;
                    dst[i].z += gpuHost_[i].z;
                    dst[i].w += gpuHost_[i].w;
                }
                fb.markHasData();
                gpuLumSq_.resize(pixelCount);
                if (gpu_->downloadInternalLumSq(gpuLumSq_.data(), pixelCount)) {
                    fb.addLumSq(gpuLumSq_.data(), pixelCount);
                }
            }
        }

        if (batchIndex_ < 2 || batchIndex_ % 8 == 0) {
            logInfo("XPU Overlap: GPU " + std::to_string(g) + " spp (" +
                    std::to_string(int(gpu_ ? gpu_->lastGpuSampleMs() : 0.0)) + " ms/spp) + Embree " +
                    std::to_string(c) + " spp (" + std::to_string(int(cpuMs)) + " ms)");
        }
        ++batchIndex_;
    }

    void presentMixture(Framebuffer& fb) {
        const size_t n = size_t(fb.width()) * size_t(fb.height());
        if (cpuFb_.width() != fb.width() || cpuFb_.height() != fb.height()) return;
        std::lock_guard<std::mutex> lock(gpuHostMutex_);
        if (gpuHost_.size() != n) gpuHost_.assign(n, Vec4{});
        blendAdd(fb.data(), cpuFb_.data(), gpuHost_.data(), n);
        fb.markHasData();
        gpuLumSq_.resize(n);
        if (gpu_->downloadInternalLumSq(gpuLumSq_.data(), n)) {
            fb.copyLumSq(cpuFb_.lumSq().data(), n);
            fb.addLumSq(gpuLumSq_.data(), n);
        } else if (cpuFb_.lumSq().size() == n) {
            fb.copyLumSq(cpuFb_.lumSq().data(), n);
        }
    }

    void renderMixture(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                       const RenderMidProgressFn& midProgress, const RenderSampleOptions& opt) {
        if (sampleIndex <= 0) {
            stopGpuWorker();
            resetFrame();
            cpuFb_.resize(fb.width(), fb.height());
            cpuFb_.clear();
            gpuHost_.assign(size_t(fb.width()) * size_t(fb.height()), Vec4{});
        }
        if (cpuFb_.width() != fb.width() || cpuFb_.height() != fb.height()) {
            cpuFb_.resize(fb.width(), fb.height());
            cpuFb_.clear();
        }

        int target = opt.xpuTargetSamples;
        if (target <= 0) target = sampleIndex + std::max(1, opt.xpuRemainingSamples);
        if (target <= 0) target = 1;

        if (!loggedMode_) {
            logInfo("XPU Mixture: independent CPU/GPU films, automatic spp share, host blend (Karma); "
                    "Embree keeps MNEE / SSS / OpenPGL / N lights / filters");
            loggedMode_ = true;
        }

        ensureGpuWorker(fb, cancel, target);

        RenderMidProgressFn wrappedMid = midProgress;
        if (midProgress) {
            wrappedMid = [&] {
                snapshotWanted_.store(true, std::memory_order_relaxed);
                presentMixture(fb);
                midProgress();
            };
        }

        const auto cpuT0 = std::chrono::steady_clock::now();
        try {
            if (!cancel.load(std::memory_order_relaxed) &&
                cpuSpp_.load(std::memory_order_relaxed) + gpuSpp_.load(std::memory_order_relaxed) < target) {
                cpuFb_.copySkipMask(fb.skipMask());
                const int cpuIndex = xpuCpuSampleIndex(cpuSpp_.load(std::memory_order_relaxed));
                cpu_->renderSample(cpuFb_, cpuIndex, cancel, wrappedMid, nullptr);
                cpuSpp_.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (...) {
            try {
                stopGpuWorker();
            } catch (...) {
            }
            throw;
        }
        const double cpuMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpuT0).count();

        waitGpuSnapshot();
        if (gpuEx_) {
            std::exception_ptr ex = gpuEx_;
            gpuEx_ = nullptr;
            try {
                stopGpuWorker();
            } catch (...) {
            }
            std::rethrow_exception(ex);
        }
        presentMixture(fb);

        const int total = cpuSpp_.load(std::memory_order_relaxed) + gpuSpp_.load(std::memory_order_relaxed);
        lastCompletedSamples_ = std::max(1, total - reportedSpp_);
        reportedSpp_ = total;

        if (batchIndex_ < 2 || batchIndex_ % 8 == 0) {
            logInfo("XPU Mixture: GPU " + std::to_string(gpuSpp_.load(std::memory_order_relaxed)) + " spp (" +
                    std::to_string(int(gpu_ ? gpu_->lastGpuSampleMs() : 0.0)) + " ms/spp) + Embree " +
                    std::to_string(cpuSpp_.load(std::memory_order_relaxed)) + " spp (" +
                    std::to_string(int(cpuMs)) + " ms)");
        }
        ++batchIndex_;
    }

    RenderDevicePtr cpu_;
    RenderDevicePtr gpu_;
    Framebuffer cpuFb_;
    std::vector<Vec4> gpuHost_;
    std::vector<float> gpuLumSq_;
    std::mutex gpuHostMutex_;
    std::condition_variable snapshotCv_;
    std::atomic<bool> snapshotWanted_{false};
    std::atomic<bool> gpuStop_{false};
    std::atomic<bool> gpuAlive_{false};
    std::atomic<int> cpuSpp_{0};
    std::atomic<int> gpuSpp_{0};
    int targetSpp_ = 0;
    int reportedSpp_ = 0;
    int lastCompletedSamples_ = 1;
    int batchIndex_ = 0;
    int nextGpuSample_ = 0;
    int nextCpuSample_ = 1;
    bool loggedMode_ = false;
    Framebuffer* fbPtr_ = nullptr;
    const std::atomic<bool>* cancelPtr_ = nullptr;
    std::thread gpuThread_;
    std::exception_ptr gpuEx_;
};

}  // namespace

RenderDevicePtr createXpuDevice(int threadCount) {
    if (!optixBackendCompiledIn()) {
        logError("XPU (Embree+OptiX) needs an OptiX/CUDA build");
        return nullptr;
    }
    const int cpuThreads = xpuEmbreeThreadCount(threadCount);
    RenderDevicePtr cpu = createEmbreeDevice(cpuThreads);
    if (!cpu) return nullptr;
    RenderDevicePtr gpu = createOptixDevice();
    if (!gpu) {
        logError("XPU (Embree+OptiX) cannot start: OptiX GPU is unavailable");
        return nullptr;
    }
    logInfo("XPU: Overlap (default) or Mixture; Embree threads=" + std::to_string(cpuThreads));
    return std::make_shared<XpuDevice>(std::move(cpu), std::move(gpu));
}

}  // namespace sol
