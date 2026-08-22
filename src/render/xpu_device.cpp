// Hybrid CPU+GPU device: Embree and OptiX alternate samples per pixel.
#include "render/render_device.h"

#include "core/log.h"

#include <exception>
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
        return true;
    }

    void renderSample(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                      const RenderMidProgressFn& midProgress, const RenderSampleOptions* options) override {
        if (!cpu_ || !gpu_) return;
        const int width = fb.width();
        const int height = fb.height();
        if (width <= 0 || height <= 0) return;

        const RenderSampleOptions opt = options ? *options : RenderSampleOptions{};
        const int cpuSample = opt.xpuPartnerSample;

        RenderSampleOptions gpuOpt;
        gpuOpt.skipFramebufferStore = true;
        gpuOpt.resetAccum = true;

        std::exception_ptr gpuEx;
        std::thread gpuThread([&] {
            try {
                gpu_->renderSample(fb, sampleIndex, cancel, RenderMidProgressFn{}, &gpuOpt);
            } catch (...) {
                gpuEx = std::current_exception();
            }
        });
        if (cpuSample >= 0 && cpu_)
            cpu_->renderSample(fb, cpuSample, cancel, midProgress, nullptr);
        gpuThread.join();
        if (gpuEx) std::rethrow_exception(gpuEx);
        if (cancel.load(std::memory_order_relaxed)) return;

        const size_t pixelCount = size_t(width) * size_t(height);
        gpuHost_.resize(pixelCount);
        if (!gpu_->copyInternalAccum(gpuHost_.data(), pixelCount)) return;

        Vec4* dst = fb.data();
        for (size_t i = 0; i < pixelCount; ++i) {
            dst[i].x += gpuHost_[i].x;
            dst[i].y += gpuHost_[i].y;
            dst[i].z += gpuHost_[i].z;
            dst[i].w += gpuHost_[i].w;
        }
        fb.markHasData();
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
};

}  // namespace

RenderDevicePtr createXpuDevice(int threadCount) {
    if (!optixBackendCompiledIn()) {
        logError("XPU (Embree+OptiX) needs an OptiX/CUDA build");
        return nullptr;
    }
    RenderDevicePtr cpu = createEmbreeDevice(threadCount);
    if (!cpu) return nullptr;
    RenderDevicePtr gpu = createOptixDevice();
    if (!gpu) {
        logError("XPU (Embree+OptiX) cannot start: OptiX GPU is unavailable");
        return nullptr;
    }
    logInfo("XPU: even spp on OptiX, odd spp on Embree (full frame, no feature strip)");
    return std::make_shared<XpuDevice>(std::move(cpu), std::move(gpu));
}

}  // namespace sol
