// Hybrid CPU+GPU device: Embree and OptiX render checkerboard buckets together.
#include "render/render_device.h"

#include "core/log.h"
#include "render/xpu_split.h"
#include "scene/scene.h"

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
        scene_ = scene;
        if (!gpu_->buildScene(scene, error)) return false;
        std::string cpuError;
        if (!cpu_->buildScene(scene, cpuError)) {
            error = cpuError.empty() ? std::string("Embree failed to build the XPU scene") : cpuError;
            return false;
        }
        return true;
    }

    void renderSample(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                      const RenderMidProgressFn& midProgress, const RenderSampleOptions* /*options*/) override {
        if (!cpu_ || !gpu_) return;
        const int width = fb.width();
        const int height = fb.height();
        if (width <= 0 || height <= 0) return;

        const int splitTile = xpuTileSizeOrDefault(scene_ ? scene_->settings.tileSize : 32);

        RenderSampleOptions cpuOpt;
        cpuOpt.xpuTileRole = 0;
        cpuOpt.xpuTileSize = splitTile;
        cpuOpt.xpuGpuParity = 0;

        RenderSampleOptions gpuOpt;
        gpuOpt.xpuTileRole = 1;
        gpuOpt.xpuTileSize = splitTile;
        gpuOpt.xpuGpuParity = 0;
        gpuOpt.skipFramebufferStore = true;

        std::exception_ptr gpuEx;
        std::thread gpuThread([&] {
            try {
                gpu_->renderSample(fb, sampleIndex, cancel, RenderMidProgressFn{}, &gpuOpt);
            } catch (...) {
                gpuEx = std::current_exception();
            }
        });
        cpu_->renderSample(fb, sampleIndex, cancel, midProgress, &cpuOpt);
        gpuThread.join();
        if (gpuEx) std::rethrow_exception(gpuEx);
        if (cancel.load(std::memory_order_relaxed)) return;

        const size_t pixelCount = size_t(width) * size_t(height);
        gpuHost_.resize(pixelCount);
        if (!gpu_->copyInternalAccum(gpuHost_.data(), pixelCount)) return;

        Vec4* dst = fb.data();
        const int parity = gpuOpt.xpuGpuParity;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (!xpuGpuOwnsPixel(x, y, splitTile, parity)) continue;
                dst[size_t(y) * size_t(width) + size_t(x)] = gpuHost_[size_t(y) * size_t(width) + size_t(x)];
            }
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
        scene_.reset();
        gpuHost_.clear();
        gpuHost_.shrink_to_fit();
    }

private:
    RenderDevicePtr cpu_;
    RenderDevicePtr gpu_;
    ScenePtr scene_;
    std::vector<Vec4> gpuHost_;
};

}  // namespace

void applyXpuDeviceMatch(Scene& scene) {
    RenderSettingsData& s = scene.settings;
    s.lightSamples = 1;
    s.pathGuiding = 0;
    s.motionBlur = 0;
    s.pixelFilter = kPixelFilterBox;
    s.filterRadius = 0.5f;
    s.samplingEngine = kSamplingEngineBuckets;
    scene.camera.opticalModel = 0;
    for (Material& mat : scene.materials) mat.subsurface = 0.0f;
}

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
    logInfo("XPU: Embree and OptiX share checkerboard buckets (GPU estimator on both)");
    return std::make_shared<XpuDevice>(std::move(cpu), std::move(gpu));
}

}  // namespace sol
