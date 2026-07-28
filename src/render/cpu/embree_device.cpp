// CPU path tracing backend built on Intel Embree 4.
#include <embree4/rtcore.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "core/log.h"
#include "core/thread_pool.h"
#include "render/blue_noise.h"
#include "render/cpu/polynomial_optics.h"
#include "render/integrator.h"
#include "render/render_device.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPENPGL
#include "render/cpu/path_guiding.h"
#endif

namespace sol {
namespace {

void embreeErrorCallback(void* /*userPtr*/, RTCError code, const char* message) {
    logError(std::string("Embree error ") + std::to_string(int(code)) + ": " + (message ? message : ""));
}

// Adapter that lets the shared integrator trace against an Embree scene.
struct EmbreeTracer {
    RTCScene scene = nullptr;

    bool intersect(Vec3 origin, Vec3 dir, float tMax, RayHit& out) const {
        RTCRayHit rayhit{};
        rayhit.ray.org_x = origin.x;
        rayhit.ray.org_y = origin.y;
        rayhit.ray.org_z = origin.z;
        rayhit.ray.dir_x = dir.x;
        rayhit.ray.dir_y = dir.y;
        rayhit.ray.dir_z = dir.z;
        rayhit.ray.tnear = 0.0f;
        rayhit.ray.tfar = tMax;
        rayhit.ray.mask = unsigned(kVisAll);
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        rtcIntersect1(scene, &rayhit, nullptr);
        if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) return false;
        out.t = rayhit.ray.tfar;
        out.primIndex = rayhit.hit.primID;
        out.u = rayhit.hit.u;
        out.v = rayhit.hit.v;
        const unsigned int instID = rayhit.hit.instID[0];
        out.instanceIndex = instID == RTC_INVALID_GEOMETRY_ID ? int(rayhit.hit.geomID) : int(instID);
        return true;
    }

    bool occluded(Vec3 origin, Vec3 dir, float tMax) const {
        RTCRay ray{};
        ray.org_x = origin.x;
        ray.org_y = origin.y;
        ray.org_z = origin.z;
        ray.dir_x = dir.x;
        ray.dir_y = dir.y;
        ray.dir_z = dir.z;
        ray.tnear = 0.0f;
        ray.tfar = tMax;
        // Shadow rays skip light proxies that have self-shadows disabled.
        ray.mask = unsigned(kVisShadow);
        rtcOccluded1(scene, &ray, nullptr);
        return ray.tfar < 0.0f;
    }
};

class EmbreeDevice final : public RenderDevice {
public:
    explicit EmbreeDevice(int threadCount) : threadCount_(threadCount) {
        std::string config = "verbose=0";
        if (threadCount > 0) config += ",threads=" + std::to_string(threadCount);
        device_ = rtcNewDevice(config.c_str());
        if (device_) rtcSetDeviceErrorFunction(device_, embreeErrorCallback, nullptr);
        pool_ = std::make_unique<ThreadPool>(threadCount);
#if SOLSTICE_HAVE_OPENPGL
        pathGuiding_ = std::make_unique<PathGuiding>();
#endif
    }

    ~EmbreeDevice() override {
        releaseScene();
        if (device_) rtcReleaseDevice(device_);
    }

    std::string name() const override { return "CPU / Embree 4"; }
    bool isAvailable() const override { return device_ != nullptr; }

    bool buildScene(const ScenePtr& scene, std::string& error) override {
        if (!device_) {
            error = "Embree device could not be created";
            return false;
        }
        releaseScene();
        scene_ = scene;
        if (!scene_) {
            error = "no scene";
            return false;
        }
        view_ = scene_->view();
        polyOptics_.prepare(view_.camera);

        const auto start = std::chrono::steady_clock::now();

        // One Embree scene per mesh so instances can share geometry.
        meshScenes_.assign(scene_->meshes.size(), nullptr);
        for (size_t i = 0; i < scene_->meshes.size(); ++i) {
            const MeshPtr& mesh = scene_->meshes[i];
            if (!mesh || mesh->indices.empty()) continue;
            RTCScene meshScene = rtcNewScene(device_);
            rtcSetSceneBuildQuality(meshScene, RTC_BUILD_QUALITY_HIGH);
            RTCGeometry geom = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);

            const size_t vertexCount = mesh->positions.size();
            const size_t triCount = mesh->indices.size() / 3;
            // Embree owns these buffers so it can apply the required padding.
            float* vertices = static_cast<float*>(rtcSetNewGeometryBuffer(
                geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), vertexCount));
            uint32_t* indices = static_cast<uint32_t*>(rtcSetNewGeometryBuffer(
                geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(uint32_t), triCount));
            if (!vertices || !indices) {
                error = "failed to allocate Embree geometry buffers";
                rtcReleaseGeometry(geom);
                rtcReleaseScene(meshScene);
                return false;
            }
            std::memcpy(vertices, mesh->positions.data(), vertexCount * 3 * sizeof(float));
            std::memcpy(indices, mesh->indices.data(), triCount * 3 * sizeof(uint32_t));

            rtcCommitGeometry(geom);
            rtcAttachGeometry(meshScene, geom);
            rtcReleaseGeometry(geom);
            rtcCommitScene(meshScene);
            meshScenes_[i] = meshScene;
        }

        topScene_ = rtcNewScene(device_);
        rtcSetSceneFlags(topScene_, RTC_SCENE_FLAG_ROBUST);
        rtcSetSceneBuildQuality(topScene_, RTC_BUILD_QUALITY_HIGH);
        for (size_t i = 0; i < scene_->instances.size(); ++i) {
            const InstanceData& inst = scene_->instances[i];
            if (inst.meshIndex < 0 || inst.meshIndex >= int(meshScenes_.size())) continue;
            RTCScene meshScene = meshScenes_[size_t(inst.meshIndex)];
            if (!meshScene) continue;
            RTCGeometry instGeom = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_INSTANCE);
            rtcSetGeometryInstancedScene(instGeom, meshScene);
            rtcSetGeometryTimeStepCount(instGeom, 1);
            // Our matrices are row major with column vectors, so the first
            // twelve floats are exactly Embree's 3x4 row major affine layout.
            rtcSetGeometryTransform(instGeom, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, inst.xform.m);
            rtcSetGeometryMask(instGeom, unsigned(inst.visibilityMask));
            rtcCommitGeometry(instGeom);
            // Geometry ids match instance indices so hits map back directly.
            rtcAttachGeometryByID(topScene_, instGeom, static_cast<unsigned int>(i));
            rtcReleaseGeometry(instGeom);
        }
        rtcCommitScene(topScene_);

#if SOLSTICE_HAVE_OPENPGL
        if (pathGuiding_) {
            const int threads = pool_ ? pool_->threadCount() : threadCount_;
            pathGuiding_->reset(view_.worldBounds, threads);
        }
#endif

        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        logInfo("Embree: built BVH for " + std::to_string(scene_->instances.size()) + " instances, " +
                std::to_string(scene_->totalTriangles()) + " triangles in " + std::to_string(int(ms)) + " ms");
        return true;
    }

    void renderSample(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                      const RenderMidProgressFn& midProgress) override {
        if (!topScene_ || !scene_) return;
        const RenderSettingsData& settings = view_.settings;
        const int width = fb.width();
        const int height = fb.height();
        if (width <= 0 || height <= 0) return;

        const int tileSize = std::clamp(settings.tileSize, 8, 256);
        const int tilesX = (width + tileSize - 1) / tileSize;
        const int tilesY = (height + tileSize - 1) / tileSize;
        const int tileCount = tilesX * tilesY;

        EmbreeTracer tracer{topScene_};
        const SceneView& scene = view_;
        const uint32_t frameSeed = uint32_t(settings.seed) * 9781u + uint32_t(sampleIndex) * 6271u;

#if SOLSTICE_HAVE_OPENPGL
        const bool useGuiding =
            settings.pathGuiding != 0 && pathGuiding_ && pathGuiding_->available() &&
            settings.integrator == kIntegratorPathTracer;
#else
        const bool useGuiding = false;
#endif

        auto shadePixel = [&](int x, int y, int threadId) {
            const uint32_t pixelIndex = uint32_t(y) * uint32_t(width) + uint32_t(x);
            Rng rng(hashCombine(pixelIndex, frameSeed), hashUint(pixelIndex ^ (frameSeed * 2654435761u)));
            float jx = 0.5f, jy = 0.5f;
            blueNoisePixelJitter(x, y, sampleIndex, jx, jy);
            float lensU = 0.5f, lensV = 0.5f;
            blueNoiseLensSample(x, y, sampleIndex, lensU, lensV);
            Vec3 origin, direction;
            int chromaticChannel = -1;
            float lensTau = 1.0f;
            if (polyOptics_.active) {
                float wavelengthNm = -1.0f;
                if (scene.camera.chromaticAberration != 0) {
                    chromaticChannel = int(rng.nextFloat() * 3.0f);
                    if (chromaticChannel > 2) chromaticChannel = 2;
                    wavelengthNm = chromaticWavelengthNm(chromaticChannel);
                }
                if (!generatePolynomialOpticsRay(polyOptics_, scene.camera, float(x) + jx, float(y) + jy,
                                                 width, height, lensU, lensV, rng, origin, direction,
                                                 wavelengthNm, &lensTau)) {
                    fb.addSample(x, y, Vec3(0.0f, 0.0f, 0.0f));
                    return;
                }
            } else {
                generateCameraRay(scene, float(x) + jx, float(y) + jy, lensU, lensV, origin, direction);
            }
#if SOLSTICE_HAVE_OPENPGL
            Vec3 radiance;
            if (useGuiding) {
                PathGuiding::ThreadState& guiding = pathGuiding_->thread(threadId);
                guiding.beginPath();
                radiance = traceRadiance(scene, tracer, origin, direction, rng, &guiding);
                guiding.endPath();
            } else {
                radiance = traceRadiance(scene, tracer, origin, direction, rng);
            }
#else
            (void)threadId;
            (void)useGuiding;
            Vec3 radiance = traceRadiance(scene, tracer, origin, direction, rng);
#endif
            if (chromaticChannel >= 0) {
                // Hero-wavelength RGB: deposit only the sampled channel, scaled by 3 / pdf.
                radiance = radiance * std::max(0.0f, lensTau);
                const float hero = (chromaticChannel == 0   ? radiance.x
                                    : chromaticChannel == 1 ? radiance.y
                                                           : radiance.z) *
                                   3.0f;
                radiance = Vec3(0.0f, 0.0f, 0.0f);
                if (chromaticChannel == 0)
                    radiance.x = hero;
                else if (chromaticChannel == 1)
                    radiance.y = hero;
                else
                    radiance.z = hero;
            }
            fb.addSample(x, y, radiance);
        };

        // First sample: interleaved 4x4 bootstrap so the whole frame appears
        // gradually (with hole-fill in resolveDisplay) instead of black tiles.
        constexpr int kBootstrapStep = 4;
        if (sampleIndex == 0) {
            const int phaseCount = kBootstrapStep * kBootstrapStep;
            for (int phase = 0; phase < phaseCount; ++phase) {
                if (cancel.load(std::memory_order_relaxed)) break;
                pool_->parallelFor(tileCount, [&](int tileIndex, int threadId) {
                    if (cancel.load(std::memory_order_relaxed)) return;
                    const int tx = tileIndex % tilesX;
                    const int ty = tileIndex / tilesX;
                    const int x0 = tx * tileSize;
                    const int y0 = ty * tileSize;
                    const int x1 = std::min(x0 + tileSize, width);
                    const int y1 = std::min(y0 + tileSize, height);
                    for (int y = y0; y < y1; ++y) {
                        if (cancel.load(std::memory_order_relaxed)) return;
                        for (int x = x0; x < x1; ++x) {
                            if (((x % kBootstrapStep) + (y % kBootstrapStep) * kBootstrapStep) != phase)
                                continue;
                            shadePixel(x, y, threadId);
                        }
                    }
                });
                if (midProgress) midProgress();
            }
        } else {
            pool_->parallelFor(tileCount, [&](int tileIndex, int threadId) {
                if (cancel.load(std::memory_order_relaxed)) return;
                const int tx = tileIndex % tilesX;
                const int ty = tileIndex / tilesX;
                const int x0 = tx * tileSize;
                const int y0 = ty * tileSize;
                const int x1 = std::min(x0 + tileSize, width);
                const int y1 = std::min(y0 + tileSize, height);

                for (int y = y0; y < y1; ++y) {
                    if (cancel.load(std::memory_order_relaxed)) return;
                    for (int x = x0; x < x1; ++x) shadePixel(x, y, threadId);
                }
            });
        }

#if SOLSTICE_HAVE_OPENPGL
        if (useGuiding) pathGuiding_->commitSample();
#endif
    }

    void refreshSceneData() override {
        if (scene_) {
            view_ = scene_->view();
            polyOptics_.prepare(view_.camera);
        }
    }

    void release() override { releaseScene(); }

private:
    void releaseScene() {
        if (topScene_) {
            rtcReleaseScene(topScene_);
            topScene_ = nullptr;
        }
        for (RTCScene s : meshScenes_) {
            if (s) rtcReleaseScene(s);
        }
        meshScenes_.clear();
        scene_.reset();
    }

    RTCDevice device_ = nullptr;
    RTCScene topScene_ = nullptr;
    std::vector<RTCScene> meshScenes_;
    ScenePtr scene_;
    SceneView view_;
    PolynomialOpticsCamera polyOptics_;
    std::unique_ptr<ThreadPool> pool_;
    int threadCount_ = 0;
#if SOLSTICE_HAVE_OPENPGL
    std::unique_ptr<PathGuiding> pathGuiding_;
#endif
};

}  // namespace

RenderDevicePtr createEmbreeDevice(int threadCount) {
    auto device = std::make_shared<EmbreeDevice>(threadCount);
    if (!device->isAvailable()) {
        logError("Failed to create the Embree device");
        return nullptr;
    }
    return device;
}

}  // namespace sol
