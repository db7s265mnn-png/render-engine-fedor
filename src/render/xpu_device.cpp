// Hybrid CPU+GPU device: Embree and OptiX run together. GPU keeps launching
// even spp into device accum until the CPU odd spp finishes, then one D2H add.
#include "render/render_device.h"

#include "core/log.h"
#include "render/xpu_split.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sol {
namespace {

class XpuDevice final : public RenderDevice {
public:
    XpuDevice(RenderDevicePtr cpu, RenderDevicePtr gpu) : cpu_(std::move(cpu)), gpu_(std::move(gpu)) {}

    std::string name() const override { return "XPU / Embree+OptiX"; }
    bool isAvailable() const override {
        return cpu_ && cpu_->isAvailable() && gpu_ && gpu_->isAvailable();
    }
    double lastGpuSampleMs() const override { return gpu_ ? gpu_->lastGpuSampleMs() : 0.0; }
    int lastCompletedSamples() const override { return lastCompletedSamples_; }

    bool buildScene(const ScenePtr& scene, std::string& error) override {
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
        nextGpuSample_ = 0;
        nextCpuSample_ = 1;
        batchIndex_ = 0;
        return true;
    }

    void renderSample(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                      const RenderMidProgressFn& midProgress, const RenderSampleOptions* options) override {
        lastCompletedSamples_ = 0;
        if (!cpu_ || !gpu_) return;
        const int width = fb.width();
        const int height = fb.height();
        if (width <= 0 || height <= 0) return;

        if (sampleIndex <= 0) {
            nextGpuSample_ = 0;
            nextCpuSample_ = 1;
        }

        const RenderSampleOptions opt = options ? *options : RenderSampleOptions{};
        int remaining = opt.xpuRemainingSamples;
        if (remaining <= 0) remaining = 2;
        const bool runCpu = remaining >= 2;
        const int maxGpu = runCpu ? remaining - 1 : remaining;
        if (maxGpu <= 0 && !runCpu) return;

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
        lastCompletedSamples_ = g + c;
        if (lastCompletedSamples_ <= 0) lastCompletedSamples_ = 1;

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
            }
        }

        if (batchIndex_ < 2 || batchIndex_ % 8 == 0) {
            logInfo("XPU: GPU " + std::to_string(g) + " spp (" +
                    std::to_string(int(gpu_ ? gpu_->lastGpuSampleMs() : 0.0)) + " ms/spp) + Embree " +
                    std::to_string(c) + " spp (" + std::to_string(int(cpuMs)) + " ms)");
        }
        ++batchIndex_;
    }

    void refreshSceneData() override {
        if (gpu_) gpu_->refreshSceneData();
        if (cpu_) cpu_->refreshSceneData();
    }

    void release() override {
        if (gpu_) gpu_->release();
        if (cpu_) cpu_->release();
        gpuHost_.clear();
        gpuHost_.shrink_to_fit();
    }

private:
    RenderDevicePtr cpu_;
    RenderDevicePtr gpu_;
    std::vector<Vec4> gpuHost_;
    int nextGpuSample_ = 0;
    int nextCpuSample_ = 1;
    int lastCompletedSamples_ = 1;
    int batchIndex_ = 0;
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
    logInfo("XPU: GPU fills even spp until Embree odd spp finishes (no 1:1 barrier); Embree threads=" +
            std::to_string(cpuThreads));
    return std::make_shared<XpuDevice>(std::move(cpu), std::move(gpu));
}

}  // namespace sol
