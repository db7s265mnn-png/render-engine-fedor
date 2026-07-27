// GPU path tracing backend built on NVIDIA OptiX.
//
// Geometry acceleration structures are built per mesh and instanced through a
// top level IAS, mirroring the Embree backend so both produce the same image.
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPTIX

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/log.h"
#include "render/optix/launch_params.h"
#include "render/render_device.h"

// Emitted by the build from optix_programs.cu.
extern "C" const unsigned char solsticeOptixIr[];
extern "C" const unsigned long long solsticeOptixIrSize;

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
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyRecord {
    int unused = 0;
};

using RayGenRecord = SbtRecord<EmptyRecord>;
using MissRecord = SbtRecord<EmptyRecord>;
using HitGroupRecord = SbtRecord<EmptyRecord>;

class OptixPathTracer final : public RenderDevice {
public:
    OptixPathTracer() = default;

    ~OptixPathTracer() override { shutdown(); }

    std::string name() const override { return deviceName_.empty() ? "GPU / OptiX" : "GPU / OptiX (" + deviceName_ + ")"; }

    bool isAvailable() const override { return initialized_; }

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

            buildPipeline();
            initialized_ = true;
            logInfo("OptiX backend initialised on " + deviceName_);
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
                DeviceBuffer positions, normals, uvs, indices;
                positions.upload(mesh->positions);
                indices.upload(mesh->indices);
                if (mesh->normals.size() == mesh->positions.size()) normals.upload(mesh->normals);
                if (mesh->uvs.size() == mesh->positions.size()) uvs.upload(mesh->uvs);

                view.positions = positions.as<const Vec3>();
                view.normals = normals.as<const Vec3>();
                view.uvs = uvs.as<const Vec2>();
                view.indices = indices.as<const uint32_t>();
                view.triangleCount = uint32_t(mesh->indices.size() / 3);
                view.vertexCount = uint32_t(mesh->positions.size());
                meshViews.push_back(view);

                gasHandles_[i] = buildTriangleGas(positions, indices, view.vertexCount, view.triangleCount);

                geometryBuffers_.push_back(std::move(positions));
                geometryBuffers_.push_back(std::move(indices));
                geometryBuffers_.push_back(std::move(normals));
                geometryBuffers_.push_back(std::move(uvs));
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

            logInfo("OptiX: uploaded " + std::to_string(scene_->instances.size()) + " instances, " +
                    std::to_string(scene_->totalTriangles()) + " triangles, " +
                    std::to_string(textureViews.size()) + " textures");
            return true;
        } catch (const std::exception& e) {
            error = e.what();
            return false;
        }
    }

    void renderSample(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                      const RenderMidProgressFn& midProgress) override {
        if (!initialized_ || !scene_) return;
        if (cancel.load(std::memory_order_relaxed)) return;
        try {
            const int width = fb.width();
            const int height = fb.height();
            if (width <= 0 || height <= 0) return;

            const size_t pixelCount = size_t(width) * size_t(height);
            if (accumBuffer_.size() != pixelCount * sizeof(Vec4)) {
                accumBuffer_.alloc(pixelCount * sizeof(Vec4));
                accumBuffer_.clear();
            }
            if (sampleIndex == 0) accumBuffer_.clear();

            LaunchParams launchParams{};
            launchParams.scene = deviceScene_;
            launchParams.accumBuffer = accumBuffer_.as<Vec4>();
            launchParams.width = width;
            launchParams.height = height;
            launchParams.sampleIndex = sampleIndex;
            launchParams.frameSeed = unsigned(scene_->settings.seed) * 9781u + unsigned(sampleIndex) * 6271u;
            launchParams.traversable = static_cast<unsigned long long>(iasHandle_);

            if (!launchParamsBuffer_.valid()) launchParamsBuffer_.alloc(sizeof(LaunchParams));
            CUDA_CHECK(cudaMemcpy(launchParamsBuffer_.as<void>(), &launchParams, sizeof(LaunchParams),
                                  cudaMemcpyHostToDevice));

            OPTIX_CHECK(optixLaunch(pipeline_, nullptr, launchParamsBuffer_.device(), sizeof(LaunchParams), &sbt_,
                                    unsigned(width), unsigned(height), 1));
            CUDA_CHECK(cudaDeviceSynchronize());

            // Mirror the accumulation buffer into the host framebuffer.
            accumBuffer_.download(fb.data(), pixelCount);
            fb.markHasData();
            if (midProgress) midProgress();
        } catch (const std::exception& e) {
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
#if OPTIX_VERSION >= 70700
        OPTIX_CHECK(optixModuleCreate(context_, &moduleOptions, &pipelineOptions,
                                      reinterpret_cast<const char*>(solsticeOptixIr),
                                      size_t(solsticeOptixIrSize), log, &logSize, &module_));
#else
        OPTIX_CHECK(optixModuleCreateFromPTX(context_, &moduleOptions, &pipelineOptions,
                                             reinterpret_cast<const char*>(solsticeOptixIr),
                                             size_t(solsticeOptixIrSize), log, &logSize, &module_));
#endif

        OptixProgramGroupOptions groupOptions{};

        OptixProgramGroupDesc raygenDesc{};
        raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenDesc.raygen.module = module_;
        raygenDesc.raygen.entryFunctionName = "__raygen__path";
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(context_, &raygenDesc, 1, &groupOptions, log, &logSize, &raygenGroup_));

        OptixProgramGroupDesc missDesc[2]{};
        missDesc[0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDesc[0].miss.module = module_;
        missDesc[0].miss.entryFunctionName = "__miss__radiance";
        missDesc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDesc[1].miss.module = module_;
        missDesc[1].miss.entryFunctionName = "__miss__shadow";
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(context_, missDesc, 2, &groupOptions, log, &logSize, missGroups_));

        OptixProgramGroupDesc hitDesc[2]{};
        hitDesc[0].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitDesc[0].hitgroup.moduleCH = module_;
        hitDesc[0].hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        hitDesc[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitDesc[1].hitgroup.moduleCH = module_;
        hitDesc[1].hitgroup.entryFunctionNameCH = "__closesthit__shadow";
        logSize = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(context_, hitDesc, 2, &groupOptions, log, &logSize, hitGroups_));

        OptixProgramGroup groups[] = {raygenGroup_, missGroups_[0], missGroups_[1], hitGroups_[0], hitGroups_[1]};
        OptixPipelineLinkOptions linkOptions{};
        linkOptions.maxTraceDepth = 2;
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

        // Shader binding table.
        RayGenRecord raygenRecord{};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenGroup_, &raygenRecord));
        raygenRecordBuffer_.upload(&raygenRecord, 1);

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
        iasHandle_ = 0;
        deviceScene_ = SceneView();
        scene_.reset();
    }

    void shutdown() {
        releaseScene();
        accumBuffer_.free();
        launchParamsBuffer_.free();
        raygenRecordBuffer_.free();
        missRecordBuffer_.free();
        hitRecordBuffer_.free();
        if (pipeline_) optixPipelineDestroy(pipeline_);
        if (raygenGroup_) optixProgramGroupDestroy(raygenGroup_);
        for (OptixProgramGroup& group : missGroups_) {
            if (group) optixProgramGroupDestroy(group);
            group = nullptr;
        }
        for (OptixProgramGroup& group : hitGroups_) {
            if (group) optixProgramGroupDestroy(group);
            group = nullptr;
        }
        if (module_) optixModuleDestroy(module_);
        if (context_) optixDeviceContextDestroy(context_);
        pipeline_ = nullptr;
        raygenGroup_ = nullptr;
        module_ = nullptr;
        context_ = nullptr;
        initialized_ = false;
    }

    bool initialized_ = false;
    std::string deviceName_;

    OptixDeviceContext context_ = nullptr;
    OptixModule module_ = nullptr;
    OptixProgramGroup raygenGroup_ = nullptr;
    OptixProgramGroup missGroups_[2] = {nullptr, nullptr};
    OptixProgramGroup hitGroups_[2] = {nullptr, nullptr};
    OptixPipeline pipeline_ = nullptr;
    OptixShaderBindingTable sbt_{};

    DeviceBuffer raygenRecordBuffer_, missRecordBuffer_, hitRecordBuffer_;
    DeviceBuffer launchParamsBuffer_, accumBuffer_;
    DeviceBuffer meshViewBuffer_, instanceBuffer_, materialBuffer_, lightBuffer_, envViewBuffer_;
    DeviceBuffer textureViewBuffer_;
    DeviceBuffer instanceDescBuffer_;
    std::vector<DeviceBuffer> geometryBuffers_;
    std::vector<DeviceBuffer> accelBuffers_;
    std::vector<OptixTraversableHandle> gasHandles_;
    OptixTraversableHandle iasHandle_ = 0;

    ScenePtr scene_;
    SceneView deviceScene_;
};

}  // namespace

bool optixBackendCompiledIn() { return true; }

RenderDevicePtr createOptixDevice() {
    auto device = std::make_shared<OptixPathTracer>();
    std::string error;
    if (!device->initialize(error)) {
        logWarning("OptiX backend unavailable: " + error);
        return nullptr;
    }
    return device;
}

}  // namespace sol

#endif  // SOLSTICE_HAVE_OPTIX
