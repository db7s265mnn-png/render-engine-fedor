// GPU Iray-style wavefront path tracing on NVIDIA OptiX (4λ spectral PT).
//
// Thin wavefront kernels (init / intersect / shade) plus a separate tail
// pipeline so the interactive stack is not poisoned by the megakernel.
// Closest-hit writes GpuHit / occlusion; no payload. Host never reads live
// counts — full W×H pool, GPU regen into the next spp, one D2H at UI rate.
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPTIX

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/log.h"
#include "render/cie_tables.h"
#include "render/camera_proj.h"
#include "render/color_space.h"
#include "render/illuminant_spd.h"
#include "render/optix/launch_params.h"
#include "render/render_device.h"
#include "render/rgb_spectrum_tables.h"
#include "scene/volume_grid.h"

// Emitted by the build from the wavefront OptiX modules.
extern "C" const unsigned char solsticeOptixInitIr[];
extern "C" const unsigned long long solsticeOptixInitIrSize;
extern "C" const unsigned char solsticeOptixInitFromLightIr[];
extern "C" const unsigned long long solsticeOptixInitFromLightIrSize;
extern "C" const unsigned char solsticeOptixIntersectClosestIr[];
extern "C" const unsigned long long solsticeOptixIntersectClosestIrSize;
extern "C" const unsigned char solsticeOptixIntersectShadowIr[];
extern "C" const unsigned long long solsticeOptixIntersectShadowIrSize;
extern "C" const unsigned char solsticeOptixShadeSurfaceIr[];
extern "C" const unsigned long long solsticeOptixShadeSurfaceIrSize;
extern "C" const unsigned char solsticeOptixShadeBackgroundIr[];
extern "C" const unsigned long long solsticeOptixShadeBackgroundIrSize;
extern "C" const unsigned char solsticeOptixShadeShadowIr[];
extern "C" const unsigned long long solsticeOptixShadeShadowIrSize;
extern "C" const unsigned char solsticeOptixShadeVolumeIr[];
extern "C" const unsigned long long solsticeOptixShadeVolumeIrSize;
extern "C" const unsigned char solsticeOptixPathTailIr[];
extern "C" const unsigned long long solsticeOptixPathTailIrSize;
extern "C" const unsigned char solsticeOptixHitIr[];
extern "C" const unsigned long long solsticeOptixHitIrSize;

namespace sol {
namespace {

#define CUDA_CHECK(call)                                                                              \
    do {                                                                                              \
        const cudaError_t status = (call);                                                            \
        if (status != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status) + " (" + \
                                     #call + ")");                                                   \
    } while (0)

std::mutex gOptixLogMutex;
std::string gLastOptixLog;

void contextLog(unsigned int level, const char* tag, const char* message, void*) {
    const std::string text = std::string("[OptiX ") + (tag ? tag : "") + "] " + (message ? message : "");
    {
        std::lock_guard<std::mutex> lock(gOptixLogMutex);
        gLastOptixLog = text;
    }
    if (level <= 2) {
        logError(text);
    } else if (level == 3) {
        logWarning(text);
    } else {
        logDebug(text);
    }
}

std::string optixFailMessage(OptixResult result, const char* call) {
    std::ostringstream oss;
    oss << "OptiX error " << int(result);
    if (result == 7900) oss << " OPTIX_ERROR_CUDA_ERROR";
#ifdef OPTIX_ERROR_INVALID_VALUE
    else if (result == OPTIX_ERROR_INVALID_VALUE) oss << " OPTIX_ERROR_INVALID_VALUE";
#endif
    oss << " in " << call;
    const cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        oss << "; CUDA " << int(cudaErr) << " " << cudaGetErrorString(cudaErr);
    }
    std::string log;
    {
        std::lock_guard<std::mutex> lock(gOptixLogMutex);
        log = gLastOptixLog;
    }
    if (!log.empty()) oss << "; " << log;
    return oss.str();
}

#define OPTIX_CHECK(call)                                                                    \
    do {                                                                                     \
        const OptixResult result = (call);                                                   \
        if (result != OPTIX_SUCCESS)                                                         \
            throw std::runtime_error(optixFailMessage(result, #call));                       \
    } while (0)

// Small owning wrapper around a device allocation.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { free(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { *this = std::move(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            pointer_ = other.pointer_;
            size_ = other.size_;
            other.pointer_ = 0;
            other.size_ = 0;
        }
        return *this;
    }

    void alloc(size_t bytes) {
        free();
        if (bytes == 0) return;
        void* raw = nullptr;
        CUDA_CHECK(cudaMalloc(&raw, bytes));
        pointer_ = reinterpret_cast<CUdeviceptr>(raw);
        size_ = bytes;
    }

    template <typename T>
    void upload(const T* host, size_t count) {
        alloc(sizeof(T) * count);
        if (count == 0) return;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(pointer_), host, sizeof(T) * count, cudaMemcpyHostToDevice));
    }

    template <typename T>
    void upload(const std::vector<T>& host) {
        upload(host.data(), host.size());
    }

    template <typename T>
    void download(T* host, size_t count) const {
        if (count == 0 || !pointer_) return;
        CUDA_CHECK(cudaMemcpy(host, reinterpret_cast<const void*>(pointer_), sizeof(T) * count,
                              cudaMemcpyDeviceToHost));
    }

    void clear() {
        if (pointer_ && size_) CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(pointer_), 0, size_));
    }

    void free() {
        if (pointer_) {
            cudaFree(reinterpret_cast<void*>(pointer_));
            pointer_ = 0;
            size_ = 0;
        }
    }

    CUdeviceptr device() const { return pointer_; }
    template <typename T>
    T* as() const {
        return reinterpret_cast<T*>(pointer_);
    }
    size_t size() const { return size_; }
    bool valid() const { return pointer_ != 0; }

private:
    CUdeviceptr pointer_ = 0;
    size_t size_ = 0;
};

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyRecord {
    int unused = 0;
};

using RayGenRecord = SbtRecord<EmptyRecord>;
using MissRecord = SbtRecord<EmptyRecord>;
using HitGroupRecord = SbtRecord<EmptyRecord>;

static_assert(sizeof(RayGenRecord) % OPTIX_SBT_RECORD_ALIGNMENT == 0,
              "raygen SBT records must pack without gaps");

enum RaygenId : int {
    kRgInit = 0,
    kRgInitFromLight,
    kRgIntersectClosest,
    kRgIntersectShadow,
    kRgShadeSurface,
    kRgShadeBackground,
    kRgShadeShadow,
    kRgShadeVolume,
    kRgPathTail,
    kRgCount
};

enum ModuleId : int {
    kModInit = 0,
    kModInitFromLight,
    kModIntersectClosest,
    kModIntersectShadow,
    kModShadeSurface,
    kModShadeBackground,
    kModShadeShadow,
    kModShadeVolume,
    kModPathTail,
    kModHit,
    kModCount
};

class OptixPathTracer final : public RenderDevice {
public:
    OptixPathTracer() = default;

    ~OptixPathTracer() override { shutdown(); }

    std::string name() const override { return deviceName_.empty() ? "GPU / OptiX" : "GPU / OptiX (" + deviceName_ + ")"; }

    bool isAvailable() const override { return initialized_; }

    double lastGpuSampleMs() const override { return lastGpuSampleMs_; }
    int lastCompletedSamples() const override { return lastCompletedSamples_; }

    bool initialize(std::string& error) {
        try {
            int deviceCount = 0;
            const cudaError_t status = cudaGetDeviceCount(&deviceCount);
            if (status != cudaSuccess || deviceCount == 0) {
                error = "no CUDA capable device found";
                return false;
            }
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaFree(nullptr));  // create the primary context

            cudaDeviceProp properties{};
            CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
            deviceName_ = properties.name;

            OPTIX_CHECK(optixInit());

            OptixDeviceContextOptions options{};
            options.logCallbackFunction = &contextLog;
            options.logCallbackLevel = 2;
#if OPTIX_VERSION >= 70200
            options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif
            OPTIX_CHECK(optixDeviceContextCreate(nullptr, &options, &context_));

            CUDA_CHECK(cudaStreamCreate(&stream_));
            CUDA_CHECK(cudaEventCreate(&gpuStartEvent_));
            CUDA_CHECK(cudaEventCreate(&gpuStopEvent_));

            uploadSpectralTables();
            buildPipeline();
            warmupPrograms();
            initialized_ = true;
            logInfo("OptiX backend initialised on " + deviceName_ +
                    " (" + std::to_string(properties.multiProcessorCount) +
                    " SMs, Iray wavefront + tail pipeline, LaunchParams " +
                    std::to_string(sizeof(LaunchParams)) + " bytes"
#if SOLSTICE_IEEE_FP32
                    ", fast-math ftz, precise div/sqrt, OptiX DEFAULT"
#endif
                    ")");
            logInfo("OptiX submits CUDA/Compute work. Windows Task Manager defaults to the 3D graph "
                    "(~0% for path tracing) — switch a GPU graph to CUDA or Compute_0, or watch the HUD ms.");
            return true;
        } catch (const std::exception& e) {
            error = e.what();
            shutdown();
            return false;
        }
    }

    bool buildScene(const ScenePtr& scene, std::string& error) override {
        try {
            releaseScene();
            scene_ = scene;
            if (!scene_) {
                error = "no scene";
                return false;
            }

            const SceneView hostView = scene_->view();

            // Per mesh geometry buffers and acceleration structures.
            std::vector<MeshView> meshViews;
            meshViews.reserve(scene_->meshes.size());
            gasHandles_.assign(scene_->meshes.size(), 0);

            for (size_t i = 0; i < scene_->meshes.size(); ++i) {
                const MeshPtr& mesh = scene_->meshes[i];
                MeshView view;
                if (!mesh || mesh->indices.empty()) {
                    meshViews.push_back(view);
                    continue;
                }
                DeviceBuffer positions, normals, uvs, indices, restPositions, restNormals;
                positions.upload(mesh->positions);
                indices.upload(mesh->indices);
                if (mesh->normals.size() == mesh->positions.size()) normals.upload(mesh->normals);
                if (mesh->uvs.size() == mesh->positions.size()) uvs.upload(mesh->uvs);
                if (mesh->restPositions.size() == mesh->positions.size() && !mesh->restPositions.empty())
                    restPositions.upload(mesh->restPositions);
                if (mesh->restNormals.size() == mesh->positions.size() && !mesh->restNormals.empty())
                    restNormals.upload(mesh->restNormals);

                view.positions = positions.as<const Vec3>();
                view.normals = normals.as<const Vec3>();
                view.uvs = uvs.as<const Vec2>();
                view.indices = indices.as<const uint32_t>();
                view.restPositions = restPositions.as<const Vec3>();
                view.restNormals = restNormals.as<const Vec3>();
                view.triangleCount = uint32_t(mesh->indices.size() / 3);
                view.vertexCount = uint32_t(mesh->positions.size());
                meshViews.push_back(view);

                gasHandles_[i] = buildTriangleGas(positions, indices, view.vertexCount, view.triangleCount);

                geometryBuffers_.push_back(std::move(positions));
                geometryBuffers_.push_back(std::move(indices));
                geometryBuffers_.push_back(std::move(normals));
                geometryBuffers_.push_back(std::move(uvs));
                geometryBuffers_.push_back(std::move(restPositions));
                geometryBuffers_.push_back(std::move(restNormals));
            }

            // Top level instance acceleration structure.
            std::vector<OptixInstance> instances;
            instances.reserve(scene_->instances.size());
            for (size_t i = 0; i < scene_->instances.size(); ++i) {
                const InstanceData& source = scene_->instances[i];
                if (source.meshIndex < 0 || size_t(source.meshIndex) >= gasHandles_.size()) continue;
                const OptixTraversableHandle gas = gasHandles_[size_t(source.meshIndex)];
                if (!gas) continue;
                OptixInstance instance{};
                // Our matrices are row major with column vectors, matching the
                // 3x4 affine layout OptiX expects.
                std::memcpy(instance.transform, source.xform.m, sizeof(float) * 12);
                instance.instanceId = unsigned(i);
                instance.sbtOffset = 0;
                instance.visibilityMask = static_cast<unsigned char>(source.visibilityMask & 0xFF);
                instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                instance.traversableHandle = gas;
                instances.push_back(instance);
            }
            iasHandle_ = buildInstanceAccel(instances);

            // Environment maps: pixels plus the sampling tables.
            std::vector<EnvMapView> envViews;
            envViews.reserve(scene_->envMaps.size());
            for (const std::shared_ptr<EnvironmentMap>& env : scene_->envMaps) {
                EnvMapView view;
                if (env && !env->image.empty()) {
                    const EnvMapView hostEnv = env->view();
                    DeviceBuffer pixels, func, condCdf, condIntegral, margCdf, margFunc;
                    pixels.upload(hostEnv.pixels, size_t(hostEnv.width) * size_t(hostEnv.height) * 4);
                    view.pixels = pixels.as<const float>();
                    view.width = hostEnv.width;
                    view.height = hostEnv.height;
                    if (hostEnv.sampled()) {
                        const size_t texels = size_t(hostEnv.width) * size_t(hostEnv.height);
                        func.upload(hostEnv.func, texels);
                        condCdf.upload(hostEnv.condCdf, size_t(hostEnv.width + 1) * size_t(hostEnv.height));
                        condIntegral.upload(hostEnv.condIntegral, size_t(hostEnv.height));
                        margCdf.upload(hostEnv.margCdf, size_t(hostEnv.height + 1));
                        margFunc.upload(hostEnv.margFunc, size_t(hostEnv.height));
                        view.func = func.as<const float>();
                        view.condCdf = condCdf.as<const float>();
                        view.condIntegral = condIntegral.as<const float>();
                        view.margCdf = margCdf.as<const float>();
                        view.margFunc = margFunc.as<const float>();
                        view.integral = hostEnv.integral;
                    }
                    geometryBuffers_.push_back(std::move(pixels));
                    geometryBuffers_.push_back(std::move(func));
                    geometryBuffers_.push_back(std::move(condCdf));
                    geometryBuffers_.push_back(std::move(condIntegral));
                    geometryBuffers_.push_back(std::move(margCdf));
                    geometryBuffers_.push_back(std::move(margFunc));
                }
                envViews.push_back(view);
            }

            // Upload textures (including UDIM atlases) — host pointers are not valid on device.
            std::vector<TextureView> textureViews;
            textureViews.reserve(scene_->textures.size());
            for (const std::shared_ptr<Image>& image : scene_->textures) {
                TextureView view;
                if (image && !image->empty()) {
                    DeviceBuffer pixels;
                    pixels.upload(image->data(), image->sizeInFloats());
                    view.pixels = pixels.as<const float>();
                    view.width = image->width();
                    view.height = image->height();
                    view.mipCount = image->mipCount() > 0 ? image->mipCount() : 1;
                    if (image->isUdimAtlas()) {
                        view.udimGridU = image->udimGridU();
                        view.udimGridV = image->udimGridV();
                    }
                    geometryBuffers_.push_back(std::move(pixels));
                }
                textureViews.push_back(view);
            }
            textureViewBuffer_.upload(textureViews);

            proceduralBuffer_.upload(scene_->procedurals);
            mediaBuffer_.upload(scene_->media);

            std::vector<GpuVolumeGrid> volumeViews(scene_->volumes.size());
            volumeDensityBuffers_.clear();
            int uploadedVolumes = 0;
            for (size_t i = 0; i < scene_->volumes.size(); ++i) {
                const VolumeGridPtr& grid = scene_->volumes[i];
                if (!grid || !grid->valid()) continue;
                VolumeGpuExport exp;
                if (!grid->exportGpuTracking(exp) || exp.occupancy.empty()) continue;
                DeviceBuffer occ;
                occ.upload(exp.occupancy);
                GpuVolumeGrid view;
                view.density = occ.as<const float>();
                view.nx = exp.nx;
                view.ny = exp.ny;
                view.nz = exp.nz;
                view.kind = exp.kind;
                view.nearest = exp.nearest;
                view.bmin = exp.bmin;
                view.bmax = exp.bmax;
                view.majorant = exp.majorant;
                view.voxelSize = exp.voxelSize;
                view.majOrigin = exp.majOrigin;
                view.majCell = exp.majCell;
                view.majNx = exp.majNx;
                view.majNy = exp.majNy;
                view.majNz = exp.majNz;
                view.brNx = exp.brNx;
                view.brNy = exp.brNy;
                view.brNz = exp.brNz;
                view.brickSize = exp.brickSize > 0 ? exp.brickSize : 4;
                volumeDensityBuffers_.push_back(std::move(occ));
                if (!exp.majMin.empty() && !exp.majMax.empty()) {
                    DeviceBuffer mn, mx;
                    mn.upload(exp.majMin);
                    mx.upload(exp.majMax);
                    view.majMin = mn.as<const float>();
                    view.majMax = mx.as<const float>();
                    volumeDensityBuffers_.push_back(std::move(mn));
                    volumeDensityBuffers_.push_back(std::move(mx));
                }
                if (!exp.bricks.empty()) {
                    DeviceBuffer br;
                    br.upload(exp.bricks);
                    view.bricks = br.as<const unsigned char>();
                    volumeDensityBuffers_.push_back(std::move(br));
                }
                volumeViews[i] = view;
                ++uploadedVolumes;
                logInfo("OptiX: volume[" + std::to_string(i) + "] " + std::to_string(exp.nx) + "x" +
                        std::to_string(exp.ny) + "x" + std::to_string(exp.nz) +
                        (exp.kind == 1 ? " fog" : " sdf") + " maj " + std::to_string(exp.majNx) + "x" +
                        std::to_string(exp.majNy) + "x" + std::to_string(exp.majNz));
            }
            volumeViewBuffer_.upload(volumeViews);

            meshViewBuffer_.upload(meshViews);
            instanceBuffer_.upload(scene_->instances);
            materialBuffer_.upload(scene_->materials);
            lightBuffer_.upload(scene_->lights);
            envViewBuffer_.upload(envViews);

            deviceScene_ = hostView;
            deviceScene_.meshes = meshViewBuffer_.as<const MeshView>();
            deviceScene_.instances = instanceBuffer_.as<const InstanceData>();
            deviceScene_.materials = materialBuffer_.as<const Material>();
            deviceScene_.lights = lightBuffer_.as<const LightData>();
            deviceScene_.envMaps = envViewBuffer_.as<const EnvMapView>();
            deviceScene_.textures = textureViewBuffer_.as<const TextureView>();
            deviceScene_.textureCount = int(textureViews.size());
            deviceScene_.procedurals = proceduralBuffer_.as<const ProceduralNode>();
            deviceScene_.proceduralCount = int(scene_->procedurals.size());
            deviceScene_.media = mediaBuffer_.as<const MediumData>();
            deviceScene_.mediumCount = int(scene_->media.size());
            deviceScene_.volumes = nullptr;
            deviceScene_.volumeCount = int(scene_->volumes.size());
            if (hostView.lightBvh && hostView.lightBvhNodeCount > 0) {
                lightBvhBuffer_.upload(hostView.lightBvh, size_t(hostView.lightBvhNodeCount));
                deviceScene_.lightBvh = lightBvhBuffer_.as<const LightBvhNode>();
                deviceScene_.lightBvhNodeCount = hostView.lightBvhNodeCount;
            } else {
                deviceScene_.lightBvh = nullptr;
                deviceScene_.lightBvhNodeCount = 0;
            }
            if (hostView.infiniteLightIndices && hostView.infiniteLightCount > 0) {
                infiniteLightIndexBuffer_.upload(hostView.infiniteLightIndices,
                                                 size_t(hostView.infiniteLightCount));
                deviceScene_.infiniteLightIndices = infiniteLightIndexBuffer_.as<const int>();
                deviceScene_.infiniteLightCount = hostView.infiniteLightCount;
            } else {
                deviceScene_.infiniteLightIndices = nullptr;
                deviceScene_.infiniteLightCount = 0;
            }
            deviceScene_.motionXforms = nullptr;
            deviceScene_.cameraMotionXforms = nullptr;
            gpuVolumeCount_ = int(volumeViews.size());

            logInfo("OptiX: uploaded " + std::to_string(scene_->instances.size()) + " instances, " +
                    std::to_string(scene_->totalTriangles()) + " triangles, " +
                    std::to_string(textureViews.size()) + " textures, " +
                    std::to_string(uploadedVolumes) + " GPU volumes (VDB tracking)");
            if (!iasHandle_ && uploadedVolumes == 0) {
                logWarning("OptiX: no triangle IAS — GPU will only shade the environment.");
            }
            if (!scene_->volumes.empty() && uploadedVolumes == 0 && !warnedVolumes_) {
                logWarning("OptiX: VDB grids present but GPU volume upload failed — volume PT skipped.");
                warnedVolumes_ = true;
            }
            if (!scene_->procedurals.empty() && !warnedProcedurals_) {
                logWarning("GPU (OptiX) uses basic surface shaders (image maps + Lambert/GGX/glass). "
                           "MaterialX procedurals stay on Embree.");
                warnedProcedurals_ = true;
            }
            if (hostView.camera.opticalModel != 0 && !warnedOptics_) {
                logWarning("GPU (OptiX) uses thin-lens DoF. Polynomial / Lentil optics stay on Embree.");
                warnedOptics_ = true;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            return true;
        } catch (const std::exception& e) {
            error = e.what();
            return false;
        }
    }

    void renderSample(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                      const RenderMidProgressFn& midProgress, const RenderSampleOptions* options) override {
        lastCompletedSamples_ = 0;
        if (!initialized_ || !scene_ || !stream_) return;
        if (cancel.load(std::memory_order_relaxed)) return;
        try {
            CUDA_CHECK(cudaSetDevice(0));
            if (scene_->settings.integrator != kIntegratorPathTracer && !warnedNonPath_) {
                logWarning("GPU (OptiX) is Path Tracer only. Other integrators are rejected by the session.");
                warnedNonPath_ = true;
            }
            const int width = fb.width();
            const int height = fb.height();
            if (width <= 0 || height <= 0) return;

            const size_t pixelCount = size_t(width) * size_t(height);
            const RenderSampleOptions opt = options ? *options : RenderSampleOptions{};
            if (accumBuffer_.size() != pixelCount * sizeof(Vec4)) {
                destroyGraph();
                accumBuffer_.alloc(pixelCount * sizeof(Vec4));
                CUDA_CHECK(cudaMemsetAsync(accumBuffer_.as<void>(), 0, accumBuffer_.size(), stream_));
                lumSqBuffer_.alloc(pixelCount * sizeof(float));
                CUDA_CHECK(cudaMemsetAsync(lumSqBuffer_.as<void>(), 0, lumSqBuffer_.size(), stream_));
            } else if (lumSqBuffer_.size() != pixelCount * sizeof(float)) {
                lumSqBuffer_.alloc(pixelCount * sizeof(float));
                CUDA_CHECK(cudaMemsetAsync(lumSqBuffer_.as<void>(), 0, lumSqBuffer_.size(), stream_));
            }
            if (pathBuffer_.size() != pixelCount * sizeof(GpuPath) ||
                hitBuffer_.size() != pixelCount * sizeof(GpuHit) ||
                shadowBuffer_.size() != pixelCount * sizeof(GpuShadow)) {
                destroyGraph();
                pathBuffer_.alloc(pixelCount * sizeof(GpuPath));
                hitBuffer_.alloc(pixelCount * sizeof(GpuHit));
                shadowBuffer_.alloc(pixelCount * sizeof(GpuShadow));
                CUDA_CHECK(cudaMemsetAsync(pathBuffer_.as<void>(), 0, pathBuffer_.size(), stream_));
                CUDA_CHECK(cudaMemsetAsync(hitBuffer_.as<void>(), 0, hitBuffer_.size(), stream_));
                CUDA_CHECK(cudaMemsetAsync(shadowBuffer_.as<void>(), 0, shadowBuffer_.size(), stream_));
            }
            accumWidth_ = width;
            accumHeight_ = height;
            int offX = 0, offY = 0;
            int launchW = width, launchH = height;
            const bool clipped = opt.clipX1 > opt.clipX0 && opt.clipY1 > opt.clipY0;
            if (clipped) {
                offX = std::max(0, opt.clipX0);
                offY = std::max(0, opt.clipY0);
                const int x1 = std::min(width, opt.clipX1);
                const int y1 = std::min(height, opt.clipY1);
                launchW = std::max(0, x1 - offX);
                launchH = std::max(0, y1 - offY);
            }
            if (launchW <= 0 || launchH <= 0) return;

            if (clipped) {
                CUDA_CHECK(cudaMemset2DAsync(
                    accumBuffer_.as<Vec4>() + size_t(offY) * size_t(width) + size_t(offX),
                    size_t(width) * sizeof(Vec4), 0, size_t(launchW) * sizeof(Vec4), unsigned(launchH),
                    stream_));
            } else if (sampleIndex == 0 || opt.resetAccum) {
                CUDA_CHECK(cudaMemsetAsync(accumBuffer_.as<void>(), 0, accumBuffer_.size(), stream_));
                if (lumSqBuffer_.size() == pixelCount * sizeof(float)) {
                    CUDA_CHECK(cudaMemsetAsync(lumSqBuffer_.as<void>(), 0, lumSqBuffer_.size(), stream_));
                }
            }

            LaunchParams launchParams{};
            launchParams.scene = deviceScene_;
            bindFilmToFramebuffer(launchParams.scene, width, height);
            launchParams.accumBuffer = accumBuffer_.as<Vec4>();
            launchParams.lumSq = lumSqBuffer_.as<float>();
            launchParams.skipMask = nullptr;
            if (scene_->settings.noiseThreshold > 0.0f && fb.skipMask().size() == pixelCount) {
                skipMaskBuffer_.upload(fb.skipMask());
                launchParams.skipMask = skipMaskBuffer_.as<unsigned char>();
            }
            launchParams.paths = pathBuffer_.as<GpuPath>();
            launchParams.hits = hitBuffer_.as<GpuHit>();
            launchParams.shadows = shadowBuffer_.as<GpuShadow>();
            launchParams.qIntersect = nullptr;
            launchParams.qIntersectNext = nullptr;
            launchParams.qVolume = nullptr;
            launchParams.qSurface = nullptr;
            launchParams.qBackground = nullptr;
            launchParams.qShadow = nullptr;
            launchParams.workCounts = nullptr;
            launchParams.workItems = nullptr;
            launchParams.workCount = 0;
            launchParams.workSlot = -1;
            launchParams.compactLaunch = 0;
            int remaining = opt.xpuRemainingSamples > 0 ? opt.xpuRemainingSamples : 1;
            int batch = 4;
            if (pixelCount > size_t(1920 * 1080)) batch = 2;
            if (pixelCount > size_t(3840 * 2160)) batch = 1;
            // First present and tumble preview: 1 spp, D2H immediately (not a 4-spp hitch).
            if (sampleIndex == 0 || opt.navPreview) batch = 1;
            if (batch > remaining) batch = remaining;
            if (batch < 1) batch = 1;
            launchParams.batchSamples = batch;
            lastCompletedSamples_ = batch;
            launchParams.width = width;
            launchParams.height = height;
            launchParams.pixelOffsetX = offX;
            launchParams.pixelOffsetY = offY;
            launchParams.sampleIndex = sampleIndex;
            launchParams.frameSeed = unsigned(scene_->settings.seed) * 9781u + unsigned(sampleIndex) * 6271u;
            fillSpectralLaunch(launchParams);
            launchParams.camProj = buildCameraProj(launchParams.scene);
            if (launchParams.scene.camera.opticalModel != 0) launchParams.camProj.valid = false;
            const bool gpuCaustics = launchParams.scene.settings.caustics != 0 &&
                                     launchParams.scene.lightCount > 0 && launchParams.camProj.valid;
            const int lightSlots = srMax(1, launchW * launchH);
            launchParams.splatInvLightPaths = gpuCaustics ? 1.0f / float(lightSlots) : 0.0f;
            launchParams.traversable = static_cast<unsigned long long>(iasHandle_);
            launchParams.volumes = volumeViewBuffer_.as<const GpuVolumeGrid>();
            launchParams.volumeCount = gpuVolumeCount_;

            if (!launchParamsBuffer_.valid()) launchParamsBuffer_.alloc(sizeof(LaunchParams));

            const int maxDepth = scene_->settings.maxDepth > 0 ? scene_->settings.maxDepth : 1;

            const auto wall0 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaEventRecord(gpuStartEvent_, stream_));
            int launches = 0;
            launchIrayWavefront(launchParams, unsigned(launchW), unsigned(launchH), maxDepth, cancel,
                                launches);
            CUDA_CHECK(cudaEventRecord(gpuStopEvent_, stream_));
            CUDA_CHECK(cudaEventSynchronize(gpuStopEvent_));
            float gpuMs = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&gpuMs, gpuStartEvent_, gpuStopEvent_));
            lastGpuSampleMs_ = double(gpuMs);

            // Aborted wavefront is missing bounces / tail. Do not D2H or markHasData
            // — a partial copy is charcoal holes in the viewport.
            if (cancel.load(std::memory_order_relaxed)) {
                lastCompletedSamples_ = 0;
                return;
            }

            const bool lastBatch = remaining <= batch;
            const bool needOracle = scene_->settings.noiseThreshold > 0.0f;
            const auto copyNow = std::chrono::steady_clock::now();
            const double copyAge = std::chrono::duration<double>(copyNow - lastHostCopy_).count();
            const bool wantCopy =
                !opt.deferHostCopy &&
                (needOracle || sampleIndex < 4 || lastBatch || !hostCopyEver_ || copyAge >= 0.08);
            if (wantCopy) {
                if (opt.skipFramebufferStore) {
                    hostAccum_.resize(pixelCount);
                    hostLumSq_.resize(pixelCount);
                    CUDA_CHECK(cudaMemcpyAsync(hostAccum_.data(), accumBuffer_.as<Vec4>(),
                                               pixelCount * sizeof(Vec4), cudaMemcpyDeviceToHost, stream_));
                    if (lumSqBuffer_.size() == pixelCount * sizeof(float)) {
                        CUDA_CHECK(cudaMemcpyAsync(hostLumSq_.data(), lumSqBuffer_.as<float>(),
                                                   pixelCount * sizeof(float), cudaMemcpyDeviceToHost,
                                                   stream_));
                    }
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(fb.data(), accumBuffer_.as<Vec4>(), pixelCount * sizeof(Vec4),
                                               cudaMemcpyDeviceToHost, stream_));
                    hostLumSq_.resize(pixelCount);
                    if (lumSqBuffer_.size() == pixelCount * sizeof(float)) {
                        CUDA_CHECK(cudaMemcpyAsync(hostLumSq_.data(), lumSqBuffer_.as<float>(),
                                                   pixelCount * sizeof(float), cudaMemcpyDeviceToHost,
                                                   stream_));
                    }
                }
                lastHostCopy_ = copyNow;
                hostCopyEver_ = true;
                CUDA_CHECK(cudaStreamSynchronize(stream_));
            }
            const double wallMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();

            if (wantCopy && !opt.navPreview && (sampleIndex < 2 || sampleIndex % 32 == 0)) {
                std::ostringstream msg;
                msg.setf(std::ios::fixed);
                msg.precision(2);
                msg << "OptiX GPU " << lastGpuSampleMs_ << " ms  wall " << wallMs << " ms  " << deviceName_
                    << "  " << width << "x" << height << "  batch=" << batch
                    << "  wavefront=" << launches << " launches";
                logInfo(msg.str());
            }

            if (wantCopy && !opt.skipFramebufferStore) {
                fb.markHasData();
                if (!hostLumSq_.empty()) fb.copyLumSq(hostLumSq_.data(), hostLumSq_.size());
            }
            if (midProgress) midProgress();
        } catch (const std::exception& e) {
            lastGpuSampleMs_ = 0.0;
            logError(std::string("OptiX render failed: ") + e.what());
            throw;
        }
    }

    void refreshSceneData() override {
        if (!scene_) return;
        // Camera and film settings live in the launch parameters, so nothing
        // has to be re-uploaded or rebuilt.
        deviceScene_.camera = scene_->camera;
        deviceScene_.settings = scene_->settings;
    }

    void release() override { releaseScene(); }

    bool copyInternalAccum(Vec4* dst, size_t count) const override {
        if (!dst || hostAccum_.size() != count) return false;
        std::memcpy(dst, hostAccum_.data(), count * sizeof(Vec4));
        return true;
    }

    bool downloadInternalAccum(Vec4* dst, size_t count) override {
        if (!dst || !stream_ || accumBuffer_.size() != count * sizeof(Vec4)) return false;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpyAsync(dst, accumBuffer_.as<Vec4>(), count * sizeof(Vec4),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return true;
    }

    bool downloadInternalLumSq(float* dst, size_t count) override {
        if (!dst || !stream_ || lumSqBuffer_.size() != count * sizeof(float)) return false;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpyAsync(dst, lumSqBuffer_.as<float>(), count * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return true;
    }

    bool downloadInternalAccumRect(Vec4* dst, int dstPitchPixels, int x0, int y0, int x1, int y1) override {
        if (!dst || !stream_ || accumWidth_ <= 0 || dstPitchPixels <= 0) return false;
        x0 = std::max(0, x0);
        y0 = std::max(0, y0);
        x1 = std::min(accumWidth_, x1);
        y1 = std::min(accumHeight_, y1);
        if (x1 <= x0 || y1 <= y0) return false;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy2DAsync(
            dst + size_t(y0) * size_t(dstPitchPixels) + size_t(x0), size_t(dstPitchPixels) * sizeof(Vec4),
            accumBuffer_.as<Vec4>() + size_t(y0) * size_t(accumWidth_) + size_t(x0),
            size_t(accumWidth_) * sizeof(Vec4), size_t(x1 - x0) * sizeof(Vec4), unsigned(y1 - y0),
            cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return true;
    }

private:
    void launchKernel(int raygenIndex, unsigned width, unsigned height) {
        static const char* kNames[kRgCount] = {
            "init_from_camera", "init_from_light", "intersect_closest", "intersect_shadow",
            "shade_surface",    "shade_background", "shade_shadow",     "shade_volume",
            "path_tail",
        };
        const OptixPipeline pipe = (raygenIndex == kRgPathTail) ? pipelineTail_ : pipeline_;
        const OptixResult result =
            optixLaunch(pipe, stream_, launchParamsBuffer_.device(), sizeof(LaunchParams),
                        &sbts_[raygenIndex], width, height, 1);
        if (result != OPTIX_SUCCESS) {
            throw std::runtime_error(optixFailMessage(result, "optixLaunch") + " [" +
                                     kNames[raygenIndex] + " " + std::to_string(width) + "x" +
                                     std::to_string(height) + "]");
        }
    }

    void uploadLaunch(const LaunchParams& lp) {
        CUDA_CHECK(cudaMemcpyAsync(launchParamsBuffer_.as<void>(), &lp, sizeof(LaunchParams),
                                   cudaMemcpyHostToDevice, stream_));
    }

    void launchIrayWavefront(LaunchParams& lp, unsigned launchW, unsigned launchH, int maxDepth,
                             const std::atomic<bool>& cancel, int& launches) {
        launches = 0;
        lp.compactLaunch = 0;
        lp.workItems = nullptr;
        lp.workCount = 0;
        lp.workSlot = -1;
        uploadLaunch(lp);
        auto bounceAndTail = [&]() {
            constexpr int kWavefrontBounces = 3;
            const int wave = std::max(1, std::min(maxDepth, kWavefrontBounces));
            for (int iter = 0; iter < wave; ++iter) {
                if (cancel.load(std::memory_order_relaxed)) return;
                launchKernel(kRgIntersectClosest, launchW, launchH);
                launchKernel(kRgShadeVolume, launchW, launchH);
                launchKernel(kRgShadeBackground, launchW, launchH);
                launchKernel(kRgShadeSurface, launchW, launchH);
                launchKernel(kRgIntersectShadow, launchW, launchH);
                launchKernel(kRgShadeShadow, launchW, launchH);
                launches += 6;
            }
            if (cancel.load(std::memory_order_relaxed)) return;
            launchKernel(kRgPathTail, launchW, launchH);
            ++launches;
        };

        launchKernel(kRgInit, launchW, launchH);
        ++launches;
        bounceAndTail();
        if (cancel.load(std::memory_order_relaxed)) return;
        if (lp.splatInvLightPaths > 0.0f) {
            launchKernel(kRgInitFromLight, launchW, launchH);
            ++launches;
            bounceAndTail();
        }
    }

    void destroyGraph() {
        if (graphExec_) {
            cudaGraphExecDestroy(graphExec_);
            graphExec_ = nullptr;
        }
        graphW_ = 0;
        graphH_ = 0;
        graphIters_ = 0;
    }

    void uploadSpectralTables() {
        using namespace rgb_spec;
        jakobAlbedoScale_.upload(albedoScaleTable(), size_t(kJakobTableRes));
        jakobAlbedoCoeffs_.upload(albedoCoeffsTable(), size_t(kJakobCoeffCount));
        jakobIllumScale_.upload(illuminantScaleTable(), size_t(kJakobTableRes));
        jakobIllumCoeffs_.upload(illuminantCoeffsTable(), size_t(kJakobCoeffCount));
        jakobAcesAlbedoScale_.upload(acesAlbedoScaleTable(), size_t(kJakobTableRes));
        jakobAcesAlbedoCoeffs_.upload(acesAlbedoCoeffsTable(), size_t(kJakobCoeffCount));
        jakobAcesIllumScale_.upload(acesIlluminantScaleTable(), size_t(kJakobTableRes));
        jakobAcesIllumCoeffs_.upload(acesIlluminantCoeffsTable(), size_t(kJakobCoeffCount));
        cieX_.upload(cie_tab::kCieX, size_t(cie_tab::kCieTabSamples));
        cieY_.upload(cie_tab::kCieY, size_t(cie_tab::kCieTabSamples));
        cieZ_.upload(cie_tab::kCieZ, size_t(cie_tab::kCieTabSamples));
        illumD65_.upload(illum_tab::kIllumD65, size_t(illum_tab::kIllumTabSamples));
        illumD60_.upload(illum_tab::kIllumD60, size_t(illum_tab::kIllumTabSamples));
    }

    void fillSpectralLaunch(LaunchParams& lp) const {
        const RenderSettingsData& st = scene_->settings;
        const RGBColorSpace& cs = (st.workingSpace == kWorkingSpaceAcesCg)
                                      ? colorSpaceAcesCg()
                                      : colorSpaceSrgb();
        const bool aces = cs.whiteIlluminant == kWhiteIlluminantD60;
        GpuSpectralTables spec;
        spec.albedoScale = aces ? jakobAcesAlbedoScale_.as<float>() : jakobAlbedoScale_.as<float>();
        spec.albedoCoeffs = aces ? jakobAcesAlbedoCoeffs_.as<float>() : jakobAlbedoCoeffs_.as<float>();
        spec.illuminantScale = aces ? jakobAcesIllumScale_.as<float>() : jakobIllumScale_.as<float>();
        spec.illuminantCoeffs = aces ? jakobAcesIllumCoeffs_.as<float>() : jakobIllumCoeffs_.as<float>();
        spec.cieX = cieX_.as<float>();
        spec.cieY = cieY_.as<float>();
        spec.cieZ = cieZ_.as<float>();
        spec.illuminantSpd =
            (cs.whiteIlluminant == kWhiteIlluminantD60) ? illumD60_.as<float>() : illumD65_.as<float>();
        for (int i = 0; i < 9; ++i) spec.rgbFromXyz[i] = cs.rgbFromXyz[i];
        spec.samples = kMaxSpectrumSamples;
        lp.spec = spec;
    }

    void warmupPrograms() {
        if (!launchParamsBuffer_.valid()) launchParamsBuffer_.alloc(sizeof(LaunchParams));
        LaunchParams dummy{};
        dummy.width = 1;
        dummy.height = 1;
        CUDA_CHECK(cudaMemcpyAsync(launchParamsBuffer_.as<void>(), &dummy, sizeof(LaunchParams),
                                   cudaMemcpyHostToDevice, stream_));
        for (int i = 0; i < kRgCount; ++i) launchKernel(i, 1, 1);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    OptixTraversableHandle buildTriangleGas(const DeviceBuffer& positions, const DeviceBuffer& indices,
                                            uint32_t vertexCount, uint32_t triangleCount) {
        OptixAccelBuildOptions accelOptions{};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        CUdeviceptr vertexBuffer = positions.device();
        unsigned int triangleFlags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};

        OptixBuildInput buildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(Vec3);
        buildInput.triangleArray.numVertices = vertexCount;
        buildInput.triangleArray.vertexBuffers = &vertexBuffer;
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
        buildInput.triangleArray.numIndexTriplets = triangleCount;
        buildInput.triangleArray.indexBuffer = indices.device();
        buildInput.triangleArray.flags = triangleFlags;
        buildInput.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes bufferSizes{};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &accelOptions, &buildInput, 1, &bufferSizes));

        DeviceBuffer temp;
        temp.alloc(bufferSizes.tempSizeInBytes);
        DeviceBuffer output;
        output.alloc(bufferSizes.outputSizeInBytes);

        DeviceBuffer compactedSize;
        compactedSize.alloc(sizeof(uint64_t));
        OptixAccelEmitDesc emitDesc{};
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = compactedSize.device();

        OptixTraversableHandle handle = 0;
        OPTIX_CHECK(optixAccelBuild(context_, nullptr, &accelOptions, &buildInput, 1, temp.device(),
                                    bufferSizes.tempSizeInBytes, output.device(), bufferSizes.outputSizeInBytes,
                                    &handle, &emitDesc, 1));
        CUDA_CHECK(cudaDeviceSynchronize());

        uint64_t compacted = 0;
        compactedSize.download(&compacted, 1);
        if (compacted > 0 && compacted < bufferSizes.outputSizeInBytes) {
            DeviceBuffer compactedBuffer;
            compactedBuffer.alloc(compacted);
            OPTIX_CHECK(optixAccelCompact(context_, nullptr, handle, compactedBuffer.device(), compacted, &handle));
            CUDA_CHECK(cudaDeviceSynchronize());
            accelBuffers_.push_back(std::move(compactedBuffer));
        } else {
            accelBuffers_.push_back(std::move(output));
        }
        return handle;
    }

    OptixTraversableHandle buildInstanceAccel(const std::vector<OptixInstance>& instances) {
        if (instances.empty()) return 0;
        instanceDescBuffer_.upload(instances);

        OptixBuildInput buildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances = instanceDescBuffer_.device();
        buildInput.instanceArray.numInstances = unsigned(instances.size());

        OptixAccelBuildOptions accelOptions{};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes bufferSizes{};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &accelOptions, &buildInput, 1, &bufferSizes));

        DeviceBuffer temp;
        temp.alloc(bufferSizes.tempSizeInBytes);
        DeviceBuffer output;
        output.alloc(bufferSizes.outputSizeInBytes);

        OptixTraversableHandle handle = 0;
        OPTIX_CHECK(optixAccelBuild(context_, nullptr, &accelOptions, &buildInput, 1, temp.device(),
                                    bufferSizes.tempSizeInBytes, output.device(), bufferSizes.outputSizeInBytes,
                                    &handle, nullptr, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        accelBuffers_.push_back(std::move(output));
        return handle;
    }

    void buildPipeline() {
        OptixModuleCompileOptions moduleOptions{};
        moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        // IEEE ladder: OptiX DEFAULT (step 1) + nvcc --fmad=true (step 2).
        // Still no --use_fast_math, so isfinite() stays.
        moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

        OptixPipelineCompileOptions pipelineOptions{};
        pipelineOptions.usesMotionBlur = 0;
        // IAS of triangle GAS — one instance level. Depth 2 (IAS + GAS).
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipelineOptions.numPayloadValues = 0;
        pipelineOptions.numAttributeValues = 2;
        pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.pipelineLaunchParamsVariableName = "solsticeLaunchParams";
        pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
#if OPTIX_VERSION >= 90000
        pipelineOptions.pipelineLaunchParamsSizeInBytes = sizeof(LaunchParams);
#endif

        char log[4096];
        size_t logSize = sizeof(log);
        auto loadModule = [&](const unsigned char* ir, unsigned long long irSize, OptixModule& out) {
            logSize = sizeof(log);
#if OPTIX_VERSION >= 70700
            OPTIX_CHECK(optixModuleCreate(context_, &moduleOptions, &pipelineOptions,
                                          reinterpret_cast<const char*>(ir), size_t(irSize), log,
                                          &logSize, &out));
#else
            OPTIX_CHECK(optixModuleCreateFromPTX(context_, &moduleOptions, &pipelineOptions,
                                                 reinterpret_cast<const char*>(ir), size_t(irSize), log,
                                                 &logSize, &out));
#endif
        };

        const unsigned char* ir[kModCount] = {
            solsticeOptixInitIr,
            solsticeOptixInitFromLightIr,
            solsticeOptixIntersectClosestIr,
            solsticeOptixIntersectShadowIr,
            solsticeOptixShadeSurfaceIr,
            solsticeOptixShadeBackgroundIr,
            solsticeOptixShadeShadowIr,
            solsticeOptixShadeVolumeIr,
            solsticeOptixPathTailIr,
            solsticeOptixHitIr,
        };
        const unsigned long long irSize[kModCount] = {
            solsticeOptixInitIrSize,
            solsticeOptixInitFromLightIrSize,
            solsticeOptixIntersectClosestIrSize,
            solsticeOptixIntersectShadowIrSize,
            solsticeOptixShadeSurfaceIrSize,
            solsticeOptixShadeBackgroundIrSize,
            solsticeOptixShadeShadowIrSize,
            solsticeOptixShadeVolumeIrSize,
            solsticeOptixPathTailIrSize,
            solsticeOptixHitIrSize,
        };
        for (int i = 0; i < kModCount; ++i) loadModule(ir[i], irSize[i], modules_[i]);

        OptixProgramGroupOptions groupOptions{};
        const char* raygenNames[kRgCount] = {
            "__raygen__init_from_camera",     "__raygen__init_from_light",   "__raygen__intersect_closest",
            "__raygen__intersect_shadow",     "__raygen__shade_surface",     "__raygen__shade_background",
            "__raygen__shade_shadow",         "__raygen__shade_volume",      "__raygen__path_tail",
        };
        for (int i = 0; i < kRgCount; ++i) {
            OptixProgramGroupDesc raygenDesc{};
            raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygenDesc.raygen.module = modules_[i];
            raygenDesc.raygen.entryFunctionName = raygenNames[i];
            logSize = sizeof(log);
            OPTIX_CHECK(optixProgramGroupCreate(context_, &raygenDesc, 1, &groupOptions, log, &logSize,
                                                &raygenGroups_[i]));
        }

        OptixProgramGroupDesc missDesc[2]{};
        missDesc[0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDesc[0].miss.module = modules_[kModHit];
        missDesc[0].miss.entryFunctionName = "__miss__radiance";
        missDesc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDesc[1].miss.module = modules_[kModHit];
        missDesc[1].miss.entryFunctionName = "__miss__shadow";
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(context_, missDesc, 2, &groupOptions, log, &logSize, missGroups_));

        OptixProgramGroupDesc hitDesc[2]{};
        hitDesc[0].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitDesc[0].hitgroup.moduleCH = modules_[kModHit];
        hitDesc[0].hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        hitDesc[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitDesc[1].hitgroup.moduleCH = modules_[kModHit];
        hitDesc[1].hitgroup.entryFunctionNameCH = "__closesthit__shadow";
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(context_, hitDesc, 2, &groupOptions, log, &logSize, hitGroups_));

        OptixProgramGroup groupsWf[kRgPathTail + 4];
        for (int i = 0; i < kRgPathTail; ++i) groupsWf[i] = raygenGroups_[i];
        groupsWf[kRgPathTail + 0] = missGroups_[0];
        groupsWf[kRgPathTail + 1] = missGroups_[1];
        groupsWf[kRgPathTail + 2] = hitGroups_[0];
        groupsWf[kRgPathTail + 3] = hitGroups_[1];
        OptixProgramGroup groupsTail[5] = {raygenGroups_[kRgPathTail], missGroups_[0], missGroups_[1],
                                           hitGroups_[0], hitGroups_[1]};

        OptixPipelineLinkOptions linkOptions{};
        linkOptions.maxTraceDepth = 1;
        logSize = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(context_, &pipelineOptions, &linkOptions, groupsWf,
                                        unsigned(sizeof(groupsWf) / sizeof(groupsWf[0])), log, &logSize,
                                        &pipeline_));
        logSize = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(context_, &pipelineOptions, &linkOptions, groupsTail,
                                        unsigned(sizeof(groupsTail) / sizeof(groupsTail[0])), log, &logSize,
                                        &pipelineTail_));

        auto setStack = [&](OptixPipeline pipe, OptixProgramGroup* groups, int n, unsigned floor) {
            OptixStackSizes stackSizes{};
            for (int i = 0; i < n; ++i) {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(groups[i], &stackSizes, pipe));
            }
            unsigned int directCallableFromTraversal = 0;
            unsigned int directCallableFromState = 0;
            unsigned int continuationStack = 0;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, linkOptions.maxTraceDepth, 0, 0,
                                                   &directCallableFromTraversal, &directCallableFromState,
                                                   &continuationStack));
            if (continuationStack < floor) continuationStack = floor;
            constexpr unsigned int kTraversableGraphDepth = 2;
            OPTIX_CHECK(optixPipelineSetStackSize(pipe, directCallableFromTraversal, directCallableFromState,
                                                  continuationStack, kTraversableGraphDepth));
        };
        // IAS→GAS is two traversables. Depth 1 is OPTIX_ERROR_INVALID_VALUE (7001)
        // with ALLOW_SINGLE_LEVEL_INSTANCING.
        setStack(pipeline_, groupsWf, int(sizeof(groupsWf) / sizeof(groupsWf[0])), 1024u);
        setStack(pipelineTail_, groupsTail, int(sizeof(groupsTail) / sizeof(groupsTail[0])), 8192u);

        RayGenRecord raygenRecords[kRgCount]{};
        for (int i = 0; i < kRgCount; ++i) {
            OPTIX_CHECK(optixSbtRecordPackHeader(raygenGroups_[i], &raygenRecords[i]));
        }
        raygenRecordBuffer_.upload(raygenRecords, kRgCount);

        MissRecord missRecords[2]{};
        OPTIX_CHECK(optixSbtRecordPackHeader(missGroups_[0], &missRecords[0]));
        OPTIX_CHECK(optixSbtRecordPackHeader(missGroups_[1], &missRecords[1]));
        missRecordBuffer_.upload(missRecords, 2);

        HitGroupRecord hitRecords[2]{};
        OPTIX_CHECK(optixSbtRecordPackHeader(hitGroups_[0], &hitRecords[0]));
        OPTIX_CHECK(optixSbtRecordPackHeader(hitGroups_[1], &hitRecords[1]));
        hitRecordBuffer_.upload(hitRecords, 2);

        sbt_ = {};
        sbt_.raygenRecord = raygenRecordBuffer_.device();
        sbt_.missRecordBase = missRecordBuffer_.device();
        sbt_.missRecordStrideInBytes = sizeof(MissRecord);
        sbt_.missRecordCount = 2;
        sbt_.hitgroupRecordBase = hitRecordBuffer_.device();
        sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        sbt_.hitgroupRecordCount = 2;
        for (int i = 0; i < kRgCount; ++i) {
            sbts_[i] = sbt_;
            sbts_[i].raygenRecord =
                raygenRecordBuffer_.device() + sizeof(RayGenRecord) * static_cast<size_t>(i);
        }
    }

    void releaseScene() {
        destroyGraph();
        geometryBuffers_.clear();
        accelBuffers_.clear();
        gasHandles_.clear();
        instanceDescBuffer_.free();
        meshViewBuffer_.free();
        instanceBuffer_.free();
        materialBuffer_.free();
        lightBuffer_.free();
        envViewBuffer_.free();
        textureViewBuffer_.free();
        proceduralBuffer_.free();
        mediaBuffer_.free();
        volumeViewBuffer_.free();
        volumeDensityBuffers_.clear();
        gpuVolumeCount_ = 0;
        iasHandle_ = 0;
        destroyGraph();
        deviceScene_ = SceneView();
        scene_.reset();
        hostAccum_.clear();
        hostAccum_.shrink_to_fit();
        hostLumSq_.clear();
        hostLumSq_.shrink_to_fit();
    }

    void shutdown() {
        releaseScene();
        destroyGraph();
        if (gpuStartEvent_) {
            cudaEventDestroy(gpuStartEvent_);
            gpuStartEvent_ = nullptr;
        }
        if (gpuStopEvent_) {
            cudaEventDestroy(gpuStopEvent_);
            gpuStopEvent_ = nullptr;
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        accumBuffer_.free();
        lumSqBuffer_.free();
        skipMaskBuffer_.free();
        pathBuffer_.free();
        hitBuffer_.free();
        shadowBuffer_.free();
        qIntersect_.free();
        qIntersectNext_.free();
        qVolume_.free();
        qSurface_.free();
        qBackground_.free();
        qShadow_.free();
        workCounts_.free();
        launchParamsBuffer_.free();
        raygenRecordBuffer_.free();
        missRecordBuffer_.free();
        hitRecordBuffer_.free();
        if (pipeline_) optixPipelineDestroy(pipeline_);
        if (pipelineTail_) optixPipelineDestroy(pipelineTail_);
        for (OptixProgramGroup& group : raygenGroups_) {
            if (group) optixProgramGroupDestroy(group);
            group = nullptr;
        }
        for (OptixProgramGroup& group : missGroups_) {
            if (group) optixProgramGroupDestroy(group);
            group = nullptr;
        }
        for (OptixProgramGroup& group : hitGroups_) {
            if (group) optixProgramGroupDestroy(group);
            group = nullptr;
        }
        for (OptixModule& module : modules_) {
            if (module) optixModuleDestroy(module);
            module = nullptr;
        }
        if (context_) optixDeviceContextDestroy(context_);
        pipeline_ = nullptr;
        pipelineTail_ = nullptr;
        context_ = nullptr;
        initialized_ = false;
    }

    bool initialized_ = false;
    bool warnedNonPath_ = false;
    bool warnedProcedurals_ = false;
    bool warnedOptics_ = false;
    bool warnedVolumes_ = false;
    double lastGpuSampleMs_ = 0.0;
    int lastCompletedSamples_ = 1;
    int accumWidth_ = 0;
    int accumHeight_ = 0;
    bool hostCopyEver_ = false;
    std::chrono::steady_clock::time_point lastHostCopy_{};
    std::string deviceName_;

    OptixDeviceContext context_ = nullptr;
    OptixModule modules_[kModCount] = {};
    OptixProgramGroup raygenGroups_[kRgCount] = {};
    OptixProgramGroup missGroups_[2] = {nullptr, nullptr};
    OptixProgramGroup hitGroups_[2] = {nullptr, nullptr};
    OptixPipeline pipeline_ = nullptr;
    OptixPipeline pipelineTail_ = nullptr;
    OptixShaderBindingTable sbt_{};
    OptixShaderBindingTable sbts_[kRgCount]{};

    cudaStream_t stream_ = nullptr;
    cudaEvent_t gpuStartEvent_ = nullptr;
    cudaEvent_t gpuStopEvent_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
    int graphW_ = 0;
    int graphH_ = 0;
    int graphIters_ = 0;

    DeviceBuffer raygenRecordBuffer_, missRecordBuffer_, hitRecordBuffer_;
    DeviceBuffer launchParamsBuffer_, accumBuffer_, lumSqBuffer_, skipMaskBuffer_;
    DeviceBuffer pathBuffer_, hitBuffer_, shadowBuffer_;
    DeviceBuffer qIntersect_, qIntersectNext_, qVolume_, qSurface_, qBackground_, qShadow_, workCounts_;
    DeviceBuffer jakobAlbedoScale_, jakobAlbedoCoeffs_, jakobIllumScale_, jakobIllumCoeffs_;
    DeviceBuffer jakobAcesAlbedoScale_, jakobAcesAlbedoCoeffs_, jakobAcesIllumScale_, jakobAcesIllumCoeffs_;
    DeviceBuffer cieX_, cieY_, cieZ_, illumD65_, illumD60_;
    DeviceBuffer meshViewBuffer_, instanceBuffer_, materialBuffer_, lightBuffer_, envViewBuffer_;
    DeviceBuffer textureViewBuffer_;
    DeviceBuffer proceduralBuffer_;
    DeviceBuffer mediaBuffer_;
    DeviceBuffer volumeViewBuffer_;
    DeviceBuffer lightBvhBuffer_;
    DeviceBuffer infiniteLightIndexBuffer_;
    std::vector<DeviceBuffer> volumeDensityBuffers_;
    int gpuVolumeCount_ = 0;
    DeviceBuffer instanceDescBuffer_;
    std::vector<DeviceBuffer> geometryBuffers_;
    std::vector<DeviceBuffer> accelBuffers_;
    std::vector<OptixTraversableHandle> gasHandles_;
    OptixTraversableHandle iasHandle_ = 0;

    ScenePtr scene_;
    SceneView deviceScene_;
    std::vector<Vec4> hostAccum_;
    std::vector<float> hostLumSq_;
};

enum class OptixRuntimeState { Unknown, Ok, Fail };

std::mutex gOptixRuntimeMutex;
std::condition_variable gOptixProbeCv;
bool gOptixProbeRunning = false;
OptixRuntimeState gOptixRuntimeState = OptixRuntimeState::Unknown;
std::string gOptixRuntimeError;

void setOptixRuntime(bool ok, std::string error) {
    gOptixRuntimeState = ok ? OptixRuntimeState::Ok : OptixRuntimeState::Fail;
    gOptixRuntimeError = std::move(error);
}

// CUDA + optixInit. Must not run on the Qt UI thread: with an Intel display GPU
// plus an NVIDIA card, cudaGetDeviceCount there often returns 0 / no-device and
// then the cached failure silently keeps OptiX off for the rest of the session.
bool probeOptixRuntimeUnlocked(std::string& error) {
    int deviceCount = 0;
    const cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
        error = std::string("CUDA: ") + cudaGetErrorString(status);
        return false;
    }
    if (deviceCount <= 0) {
        error = "no CUDA GPU";
        return false;
    }
    const OptixResult init = optixInit();
    if (init != OPTIX_SUCCESS) {
#ifdef OPTIX_ERROR_LIBRARY_NOT_FOUND
        if (init == OPTIX_ERROR_LIBRARY_NOT_FOUND) {
            error = "nvoptix.dll missing (update NVIDIA driver)";
            return false;
        }
#endif
        error = "optixInit failed (" + std::to_string(int(init)) + ")";
        return false;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess && properties.name[0]) {
        logInfo(std::string("OptiX runtime probe: CUDA sees ") + properties.name);
    }
    return true;
}

void kickOptixProbeLocked() {
    if (gOptixRuntimeState != OptixRuntimeState::Unknown || gOptixProbeRunning) return;
    gOptixProbeRunning = true;
    std::thread([] {
        std::string err;
        bool ok = false;
        try {
            ok = probeOptixRuntimeUnlocked(err);
        } catch (const std::exception& e) {
            err = e.what();
        } catch (...) {
            err = "CUDA probe crashed";
        }
        {
            std::lock_guard<std::mutex> lock(gOptixRuntimeMutex);
            setOptixRuntime(ok, err);
            gOptixProbeRunning = false;
        }
        gOptixProbeCv.notify_all();
        if (ok) logInfo("OptiX runtime probe: available");
        else logWarning("OptiX runtime probe: not available (" + err + ")");
    }).detach();
}

void waitForOptixProbe() {
    std::unique_lock<std::mutex> lock(gOptixRuntimeMutex);
    kickOptixProbeLocked();
    gOptixProbeCv.wait(lock, [] { return !gOptixProbeRunning; });
}

}  // namespace

bool optixBackendCompiledIn() { return true; }

bool optixRuntimeProbePending() {
    std::lock_guard<std::mutex> lock(gOptixRuntimeMutex);
    kickOptixProbeLocked();
    return gOptixRuntimeState == OptixRuntimeState::Unknown || gOptixProbeRunning;
}

bool optixRuntimeAvailable(std::string* error) {
    std::lock_guard<std::mutex> lock(gOptixRuntimeMutex);
    kickOptixProbeLocked();
    if (error) *error = gOptixRuntimeError;
    return gOptixRuntimeState == OptixRuntimeState::Ok;
}

RenderDevicePtr createOptixDevice() {
    waitForOptixProbe();
    auto device = std::make_shared<OptixPathTracer>();
    std::string error;
    if (!device->initialize(error)) {
        {
            std::lock_guard<std::mutex> lock(gOptixRuntimeMutex);
            setOptixRuntime(false, error);
        }
        logWarning("OptiX backend unavailable: " + error);
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(gOptixRuntimeMutex);
        setOptixRuntime(true, {});
    }
    return device;
}

}  // namespace sol

#endif  // SOLSTICE_HAVE_OPTIX
