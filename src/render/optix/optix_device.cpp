// GPU wavefront path tracing on NVIDIA OptiX (Cycles-style small modules).
//
// Geometry acceleration structures are built per mesh and instanced through a
// top level IAS, mirroring the Embree backend so both produce the same image.
// Each sample launches init → (intersect_closest → shade → intersect_shadow)*
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPTIX

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <chrono>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/log.h"
#include "render/optix/launch_params.h"
#include "render/render_device.h"
#include "scene/volume_grid.h"

// Emitted by the build from the wavefront OptiX modules.
extern "C" const unsigned char solsticeOptixInitIr[];
extern "C" const unsigned long long solsticeOptixInitIrSize;
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

#define OPTIX_CHECK(call)                                                                    \
    do {                                                                                     \
        const OptixResult result = (call);                                                   \
        if (result != OPTIX_SUCCESS)                                                         \
            throw std::runtime_error(std::string("OptiX error ") + std::to_string(int(result)) + \
                                     " in " + #call);                                        \
    } while (0)

void contextLog(unsigned int level, const char* tag, const char* message, void*) {
    const std::string text = std::string("[OptiX ") + (tag ? tag : "") + "] " + (message ? message : "");
    if (level <= 2) {
        logError(text);
    } else if (level == 3) {
        logWarning(text);
    } else {
        logDebug(text);
    }
}

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
    kRgIntersectClosest,
    kRgIntersectShadow,
    kRgShadeSurface,
    kRgShadeBackground,
    kRgShadeShadow,
    kRgShadeVolume,
    kRgCount
};

enum ModuleId : int {
    kModInit = 0,
    kModIntersectClosest,
    kModIntersectShadow,
    kModShadeSurface,
    kModShadeBackground,
    kModShadeShadow,
    kModShadeVolume,
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
            options.logCallbackLevel = 3;
            OPTIX_CHECK(optixDeviceContextCreate(nullptr, &options, &context_));

            CUDA_CHECK(cudaStreamCreate(&stream_));
            CUDA_CHECK(cudaEventCreate(&gpuStartEvent_));
            CUDA_CHECK(cudaEventCreate(&gpuStopEvent_));

            buildPipeline();
            warmupPrograms();
            initialized_ = true;
            logInfo("OptiX backend initialised on " + deviceName_ +
                    " (" + std::to_string(properties.multiProcessorCount) +
                    " SMs, wavefront PT: surfaces + thin-lens + GPU volumes)");
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
                std::vector<float> dense;
                int nx = 0, ny = 0, nz = 0;
                if (!grid->exportDense(160, dense, nx, ny, nz) || dense.empty()) continue;
                DeviceBuffer buf;
                buf.upload(dense);
                GpuVolumeGrid view;
                view.density = buf.as<const float>();
                view.nx = nx;
                view.ny = ny;
                view.nz = nz;
                view.kind = grid->kind() == VolumeGridKind::Sdf ? 0 : 1;
                view.bmin = grid->worldBounds().lo;
                view.bmax = grid->worldBounds().hi;
                view.majorant = srMax(grid->majorant(), 1e-4f);
                volumeViews[i] = view;
                volumeDensityBuffers_.push_back(std::move(buf));
                ++uploadedVolumes;
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
            deviceScene_.lightBvh = nullptr;
            deviceScene_.lightBvhNodeCount = 0;
            deviceScene_.infiniteLightIndices = nullptr;
            deviceScene_.infiniteLightCount = 0;
            deviceScene_.motionXforms = nullptr;
            deviceScene_.cameraMotionXforms = nullptr;
            gpuVolumeCount_ = int(volumeViews.size());

            logInfo("OptiX: uploaded " + std::to_string(scene_->instances.size()) + " instances, " +
                    std::to_string(scene_->totalTriangles()) + " triangles, " +
                    std::to_string(textureViews.size()) + " textures, " +
                    std::to_string(uploadedVolumes) + " GPU volume bricks");
            if (!iasHandle_ && uploadedVolumes == 0) {
                logWarning("OptiX: no triangle IAS — GPU will only shade the environment.");
            }
            if (!scene_->volumes.empty() && uploadedVolumes == 0 && !warnedVolumes_) {
                logWarning("OptiX: VDB grids present but dense GPU bake failed — volume PT skipped.");
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
                      const RenderMidProgressFn& midProgress) override {
        if (!initialized_ || !scene_ || !stream_) return;
        if (cancel.load(std::memory_order_relaxed)) return;
        try {
            CUDA_CHECK(cudaSetDevice(0));
            if (scene_->settings.integrator != kIntegratorPathTracer && !warnedNonPath_) {
                logWarning("GPU (OptiX) runs the unidirectional path tracer only. "
                           "BDPT / spectral / AO / wireframe / MNEE / polynomial optics stay on CPU Embree.");
                warnedNonPath_ = true;
            }
            const int width = fb.width();
            const int height = fb.height();
            if (width <= 0 || height <= 0) return;

            const size_t pixelCount = size_t(width) * size_t(height);
            if (accumBuffer_.size() != pixelCount * sizeof(Vec4)) {
                destroyGraph();
                accumBuffer_.alloc(pixelCount * sizeof(Vec4));
                CUDA_CHECK(cudaMemsetAsync(accumBuffer_.as<void>(), 0, accumBuffer_.size(), stream_));
            }
            if (pathBuffer_.size() != pixelCount * sizeof(GpuPath) ||
                hitBuffer_.size() != pixelCount * sizeof(GpuHit) ||
                shadowBuffer_.size() != pixelCount * sizeof(GpuShadow)) {
                destroyGraph();
                pathBuffer_.alloc(pixelCount * sizeof(GpuPath));
                hitBuffer_.alloc(pixelCount * sizeof(GpuHit));
                shadowBuffer_.alloc(pixelCount * sizeof(GpuShadow));
            }
            if (sampleIndex == 0) {
                CUDA_CHECK(cudaMemsetAsync(accumBuffer_.as<void>(), 0, accumBuffer_.size(), stream_));
            }

            LaunchParams launchParams{};
            launchParams.scene = deviceScene_;
            launchParams.accumBuffer = accumBuffer_.as<Vec4>();
            launchParams.paths = pathBuffer_.as<GpuPath>();
            launchParams.hits = hitBuffer_.as<GpuHit>();
            launchParams.shadows = shadowBuffer_.as<GpuShadow>();
            launchParams.width = width;
            launchParams.height = height;
            launchParams.sampleIndex = sampleIndex;
            launchParams.frameSeed = unsigned(scene_->settings.seed) * 9781u + unsigned(sampleIndex) * 6271u;
            launchParams.pixelSampler = scene_->settings.pixelSampler;
            launchParams.traversable = static_cast<unsigned long long>(iasHandle_);
            launchParams.volumes = volumeViewBuffer_.as<const GpuVolumeGrid>();
            launchParams.volumeCount = gpuVolumeCount_;

            if (!launchParamsBuffer_.valid()) launchParamsBuffer_.alloc(sizeof(LaunchParams));
            CUDA_CHECK(cudaMemcpyAsync(launchParamsBuffer_.as<void>(), &launchParams, sizeof(LaunchParams),
                                       cudaMemcpyHostToDevice, stream_));

            const unsigned w = unsigned(width);
            const unsigned h = unsigned(height);
            const int maxDepth = scene_->settings.maxDepth > 0 ? scene_->settings.maxDepth : 1;
            const int maxIters = maxDepth + 18;
            ensureGraph(w, h, maxIters);

            const auto wall0 = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaEventRecord(gpuStartEvent_, stream_));
            if (graphExec_) {
                CUDA_CHECK(cudaGraphLaunch(graphExec_, stream_));
            } else {
                launchBounceLoop(w, h, maxIters);
            }
            CUDA_CHECK(cudaEventRecord(gpuStopEvent_, stream_));
            CUDA_CHECK(cudaMemcpyAsync(fb.data(), accumBuffer_.as<Vec4>(), pixelCount * sizeof(Vec4),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaEventSynchronize(gpuStopEvent_));
            float gpuMs = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&gpuMs, gpuStartEvent_, gpuStopEvent_));
            lastGpuSampleMs_ = double(gpuMs);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            const double wallMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();

            if (sampleIndex < 2 || sampleIndex % 32 == 0) {
                std::ostringstream msg;
                msg.setf(std::ios::fixed);
                msg.precision(2);
                msg << "OptiX GPU " << lastGpuSampleMs_ << " ms  wall " << wallMs << " ms  " << deviceName_
                    << "  " << width << "x" << height << (graphExec_ ? "  graph=yes" : "  graph=no");
                logInfo(msg.str());
            }

            fb.markHasData();
            if (midProgress) midProgress();
        } catch (const std::exception& e) {
            lastGpuSampleMs_ = 0.0;
            logError(std::string("OptiX render failed: ") + e.what());
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

private:
    void launchKernel(int raygenIndex, unsigned width, unsigned height) {
        sbt_.raygenRecord =
            raygenRecordBuffer_.device() + sizeof(RayGenRecord) * static_cast<size_t>(raygenIndex);
        OPTIX_CHECK(optixLaunch(pipeline_, stream_, launchParamsBuffer_.device(), sizeof(LaunchParams), &sbt_,
                                width, height, 1));
    }

    void launchBounceLoop(unsigned width, unsigned height, int maxIters) {
        launchKernel(kRgInit, width, height);
        for (int i = 0; i < maxIters; ++i) {
            launchKernel(kRgIntersectClosest, width, height);
            launchKernel(kRgShadeVolume, width, height);
            launchKernel(kRgShadeBackground, width, height);
            launchKernel(kRgShadeSurface, width, height);
            launchKernel(kRgIntersectShadow, width, height);
            launchKernel(kRgShadeShadow, width, height);
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

    bool captureGraph(unsigned width, unsigned height, int maxIters) {
        destroyGraph();
        const cudaError_t begin = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
        if (begin != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cudaGraph_t graph = nullptr;
        try {
            launchBounceLoop(width, height, maxIters);
            const cudaError_t end = cudaStreamEndCapture(stream_, &graph);
            if (end != cudaSuccess || !graph) {
                cudaGetLastError();
                if (graph) cudaGraphDestroy(graph);
                return false;
            }
        } catch (...) {
            cudaGraph_t dumped = nullptr;
            cudaStreamEndCapture(stream_, &dumped);
            if (dumped) cudaGraphDestroy(dumped);
            throw;
        }
        const cudaError_t inst = cudaGraphInstantiate(&graphExec_, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);
        if (inst != cudaSuccess) {
            graphExec_ = nullptr;
            cudaGetLastError();
            return false;
        }
        graphW_ = int(width);
        graphH_ = int(height);
        graphIters_ = maxIters;
        return true;
    }

    void ensureGraph(unsigned width, unsigned height, int maxIters) {
        if (graphExec_ && graphW_ == int(width) && graphH_ == int(height) && graphIters_ == maxIters) return;
        if (graphFailed_) return;
        // Capture requires an idle stream (no pending memcpy / events).
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (captureGraph(width, height, maxIters)) {
            logInfo("OptiX: CUDA graph captured (" + std::to_string(1 + 6 * maxIters) +
                    " launches/spp) so the GPU stays busy instead of waiting on CPU optixLaunch");
            return;
        }
        graphFailed_ = true;
        logWarning("OptiX: CUDA graph capture failed — falling back to one optixLaunch per kernel. "
                   "GPU load in Task Manager 3D will look near zero; watch the HUD ms / CUDA graph.");
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
        moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

        OptixPipelineCompileOptions pipelineOptions{};
        pipelineOptions.usesMotionBlur = 0;
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipelineOptions.numPayloadValues = 6;
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
            solsticeOptixIntersectClosestIr,
            solsticeOptixIntersectShadowIr,
            solsticeOptixShadeSurfaceIr,
            solsticeOptixShadeBackgroundIr,
            solsticeOptixShadeShadowIr,
            solsticeOptixShadeVolumeIr,
            solsticeOptixHitIr,
        };
        const unsigned long long irSize[kModCount] = {
            solsticeOptixInitIrSize,
            solsticeOptixIntersectClosestIrSize,
            solsticeOptixIntersectShadowIrSize,
            solsticeOptixShadeSurfaceIrSize,
            solsticeOptixShadeBackgroundIrSize,
            solsticeOptixShadeShadowIrSize,
            solsticeOptixShadeVolumeIrSize,
            solsticeOptixHitIrSize,
        };
        for (int i = 0; i < kModCount; ++i) loadModule(ir[i], irSize[i], modules_[i]);

        OptixProgramGroupOptions groupOptions{};
        const char* raygenNames[kRgCount] = {
            "__raygen__init_from_camera",     "__raygen__intersect_closest", "__raygen__intersect_shadow",
            "__raygen__shade_surface",        "__raygen__shade_background",  "__raygen__shade_shadow",
            "__raygen__shade_volume",
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

        OptixProgramGroup groups[kRgCount + 4];
        for (int i = 0; i < kRgCount; ++i) groups[i] = raygenGroups_[i];
        groups[kRgCount + 0] = missGroups_[0];
        groups[kRgCount + 1] = missGroups_[1];
        groups[kRgCount + 2] = hitGroups_[0];
        groups[kRgCount + 3] = hitGroups_[1];
        OptixPipelineLinkOptions linkOptions{};
        linkOptions.maxTraceDepth = 1;
        logSize = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(context_, &pipelineOptions, &linkOptions, groups,
                                        unsigned(sizeof(groups) / sizeof(groups[0])), log, &logSize, &pipeline_));

        OptixStackSizes stackSizes{};
        for (OptixProgramGroup group : groups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(group, &stackSizes, pipeline_));
        }
        unsigned int directCallableFromTraversal = 0;
        unsigned int directCallableFromState = 0;
        unsigned int continuationStack = 0;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, linkOptions.maxTraceDepth, 0, 0,
                                               &directCallableFromTraversal, &directCallableFromState,
                                               &continuationStack));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline_, directCallableFromTraversal, directCallableFromState,
                                              continuationStack, 1));

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
        pathBuffer_.free();
        hitBuffer_.free();
        shadowBuffer_.free();
        launchParamsBuffer_.free();
        raygenRecordBuffer_.free();
        missRecordBuffer_.free();
        hitRecordBuffer_.free();
        if (pipeline_) optixPipelineDestroy(pipeline_);
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
        context_ = nullptr;
        initialized_ = false;
    }

    bool initialized_ = false;
    bool warnedNonPath_ = false;
    bool warnedProcedurals_ = false;
    bool warnedOptics_ = false;
    bool warnedVolumes_ = false;
    bool graphFailed_ = false;
    double lastGpuSampleMs_ = 0.0;
    std::string deviceName_;

    OptixDeviceContext context_ = nullptr;
    OptixModule modules_[kModCount] = {};
    OptixProgramGroup raygenGroups_[kRgCount] = {};
    OptixProgramGroup missGroups_[2] = {nullptr, nullptr};
    OptixProgramGroup hitGroups_[2] = {nullptr, nullptr};
    OptixPipeline pipeline_ = nullptr;
    OptixShaderBindingTable sbt_{};

    cudaStream_t stream_ = nullptr;
    cudaEvent_t gpuStartEvent_ = nullptr;
    cudaEvent_t gpuStopEvent_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
    int graphW_ = 0;
    int graphH_ = 0;
    int graphIters_ = 0;

    DeviceBuffer raygenRecordBuffer_, missRecordBuffer_, hitRecordBuffer_;
    DeviceBuffer launchParamsBuffer_, accumBuffer_;
    DeviceBuffer pathBuffer_, hitBuffer_, shadowBuffer_;
    DeviceBuffer meshViewBuffer_, instanceBuffer_, materialBuffer_, lightBuffer_, envViewBuffer_;
    DeviceBuffer textureViewBuffer_;
    DeviceBuffer proceduralBuffer_;
    DeviceBuffer mediaBuffer_;
    DeviceBuffer volumeViewBuffer_;
    std::vector<DeviceBuffer> volumeDensityBuffers_;
    int gpuVolumeCount_ = 0;
    DeviceBuffer instanceDescBuffer_;
    std::vector<DeviceBuffer> geometryBuffers_;
    std::vector<DeviceBuffer> accelBuffers_;
    std::vector<OptixTraversableHandle> gasHandles_;
    OptixTraversableHandle iasHandle_ = 0;

    ScenePtr scene_;
    SceneView deviceScene_;
};

enum class OptixRuntimeState { Unknown, Ok, Fail };

std::mutex gOptixRuntimeMutex;
OptixRuntimeState gOptixRuntimeState = OptixRuntimeState::Unknown;
std::string gOptixRuntimeError;

void setOptixRuntime(bool ok, std::string error) {
    gOptixRuntimeState = ok ? OptixRuntimeState::Ok : OptixRuntimeState::Fail;
    gOptixRuntimeError = std::move(error);
}

// CUDA device enumeration + optixInit only. Do not cudaSetDevice here: the HUD
// probe runs on the UI thread and must not bind a CUDA context there.
bool probeOptixRuntimeUnlocked(std::string& error) {
    int deviceCount = 0;
    const cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
        error = std::string("CUDA: ") + cudaGetErrorString(status);
        return false;
    }
    if (deviceCount <= 0) {
        error = "no CUDA capable device found";
        return false;
    }
    const OptixResult init = optixInit();
    if (init != OPTIX_SUCCESS) {
        error = "optixInit failed (" + std::to_string(int(init)) + ")";
        return false;
    }
    return true;
}

}  // namespace

bool optixBackendCompiledIn() { return true; }

bool optixRuntimeAvailable(std::string* error) {
    std::lock_guard<std::mutex> lock(gOptixRuntimeMutex);
    if (gOptixRuntimeState == OptixRuntimeState::Unknown) {
        std::string err;
        const bool ok = probeOptixRuntimeUnlocked(err);
        setOptixRuntime(ok, err);
        if (ok) logInfo("OptiX runtime probe: available");
        else logWarning("OptiX runtime probe: not available (" + err + ")");
    }
    if (error) *error = gOptixRuntimeError;
    return gOptixRuntimeState == OptixRuntimeState::Ok;
}

RenderDevicePtr createOptixDevice() {
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
