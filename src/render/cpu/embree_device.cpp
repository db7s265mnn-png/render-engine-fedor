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
#include "render/film_tile.h"
#include "render/sobol.h"
#include "render/cpu/polynomial_optics.h"
#include "render/integrator.h"
#include "render/integrator_base.h"
#include "render/integrator_bdpt.h"
#include "render/integrator_bdpt_spectral.h"
#include "render/integrator_mnee.h"
#include "render/integrator_spectral.h"
#include "render/photon_map.h"
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
    float time = 0.0f;

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
        rayhit.ray.time = time;
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
        out.time = time;
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
        ray.time = time;
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

        const RTCBuildQuality buildQuality =
            scene_->fastRebuild ? RTC_BUILD_QUALITY_LOW : RTC_BUILD_QUALITY_HIGH;

        // One Embree scene per mesh so instances can share geometry.
        meshScenes_.assign(scene_->meshes.size(), nullptr);
        for (size_t i = 0; i < scene_->meshes.size(); ++i) {
            const MeshPtr& mesh = scene_->meshes[i];
            if (!mesh || mesh->indices.empty()) continue;
            RTCScene meshScene = rtcNewScene(device_);
            rtcSetSceneBuildQuality(meshScene, buildQuality);
            RTCGeometry geom = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);

            const size_t vertexCount = mesh->positions.size();
            const size_t triCount = mesh->indices.size() / 3;
            int timeSteps = 1;
            if (!mesh->motionPositions.empty()) {
                bool ok = true;
                for (const auto& key : mesh->motionPositions) {
                    if (key.size() != vertexCount) {
                        ok = false;
                        break;
                    }
                }
                if (ok) timeSteps = int(mesh->motionPositions.size()) + 1;
            }
            rtcSetGeometryTimeStepCount(geom, timeSteps);

            for (int t = 0; t < timeSteps; ++t) {
                float* vertices = static_cast<float*>(rtcSetNewGeometryBuffer(
                    geom, RTC_BUFFER_TYPE_VERTEX, unsigned(t), RTC_FORMAT_FLOAT3, 3 * sizeof(float),
                    vertexCount));
                if (!vertices) {
                    error = "failed to allocate Embree geometry buffers";
                    rtcReleaseGeometry(geom);
                    rtcReleaseScene(meshScene);
                    return false;
                }
                const Vec3* src = (t == 0) ? mesh->positions.data() : mesh->motionPositions[size_t(t - 1)].data();
                std::memcpy(vertices, src, vertexCount * 3 * sizeof(float));
            }

            uint32_t* indices = static_cast<uint32_t*>(rtcSetNewGeometryBuffer(
                geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(uint32_t), triCount));
            if (!indices) {
                error = "failed to allocate Embree geometry buffers";
                rtcReleaseGeometry(geom);
                rtcReleaseScene(meshScene);
                return false;
            }
            std::memcpy(indices, mesh->indices.data(), triCount * 3 * sizeof(uint32_t));

            rtcCommitGeometry(geom);
            rtcAttachGeometry(meshScene, geom);
            rtcReleaseGeometry(geom);
            rtcCommitScene(meshScene);
            meshScenes_[i] = meshScene;
        }

        topScene_ = rtcNewScene(device_);
        rtcSetSceneFlags(topScene_, RTC_SCENE_FLAG_ROBUST);
        rtcSetSceneBuildQuality(topScene_, buildQuality);
        for (size_t i = 0; i < scene_->instances.size(); ++i) {
            const InstanceData& inst = scene_->instances[i];
            if (inst.meshIndex < 0 || inst.meshIndex >= int(meshScenes_.size())) continue;
            RTCScene meshScene = meshScenes_[size_t(inst.meshIndex)];
            if (!meshScene) continue;
            RTCGeometry instGeom = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_INSTANCE);
            rtcSetGeometryInstancedScene(instGeom, meshScene);

            const int keyCount = std::max(1, inst.motionKeyCount);
            const bool hasMotion = keyCount > 1 && !scene_->motionXforms.empty() &&
                                   inst.motionKeyOffset >= 0 &&
                                   inst.motionKeyOffset + keyCount <= int(scene_->motionXforms.size());
            rtcSetGeometryTimeStepCount(instGeom, hasMotion ? keyCount : 1);
            if (hasMotion) {
                for (int k = 0; k < keyCount; ++k) {
                    const Mat4& xform = scene_->motionXforms[size_t(inst.motionKeyOffset + k)];
                    rtcSetGeometryTransform(instGeom, unsigned(k), RTC_FORMAT_FLOAT3X4_ROW_MAJOR, xform.m);
                }
            } else {
                // Our matrices are row major with column vectors, so the first
                // twelve floats are exactly Embree's 3x4 row major affine layout.
                rtcSetGeometryTransform(instGeom, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, inst.xform.m);
            }
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
        if (!scene_->fastRebuild) {
            logInfo("Embree: built BVH for " + std::to_string(scene_->instances.size()) + " instances, " +
                    std::to_string(scene_->totalTriangles()) + " triangles in " + std::to_string(int(ms)) + " ms");
        }
        return true;
    }

    void renderSample(Framebuffer& fb, int sampleIndex, const std::atomic<bool>& cancel,
                      const RenderMidProgressFn& midProgress) override {
        if (!topScene_ || !scene_) return;
        const RenderSettingsData& settings = view_.settings;
        const int width = fb.width();
        const int height = fb.height();
        if (width <= 0 || height <= 0) return;

        const int tileSize = settings.tileSize <= 0
                                 ? chooseFilmTileSize(width, height, pool_->threadCount())
                                 : std::clamp(settings.tileSize, 8, 256);
        const int tilesX = (width + tileSize - 1) / tileSize;
        const int tilesY = (height + tileSize - 1) / tileSize;
        const int tileCount = tilesX * tilesY;

        SceneView scene = view_;  // local copy: carries the progressive pass index
        scene.settings.progressiveSample = sampleIndex;
        const uint32_t frameSeed = uint32_t(settings.seed) * 9781u + uint32_t(sampleIndex) * 6271u;

        const bool pathTracer = settings.integrator == kIntegratorPathTracer;
        const bool useSpectralBdpt = settings.integrator == kIntegratorSpectralBdpt;
        const bool useBdpt = settings.integrator == kIntegratorBdpt || useSpectralBdpt;
        const bool useSpectralPt = settings.integrator == kIntegratorSpectralPath;
        const bool useSpectral = useSpectralPt || useSpectralBdpt;
        // MNEE+Photon routes rough refractive casters to Photon, delta-only to MNEE.
        // PT Spectral has no MNEE/photon; BDPT Spectral keeps BDPT caustic estimators.
        const bool usePhoton = !useSpectralPt && causticsUsePhotonMap(settings, &scene);
        const bool useMnee = pathTracer && causticsUseMnee(settings, &scene);
#if SOLSTICE_HAVE_OPENPGL
        // OpenPGL guides eye-path diffuse sampling on PT and BDPT (RGB + Spectral).
        // Specular / near-spec vertices are recorded as delta (radiance propagates for
        // caustic training) but never guide-sampled; MNEE/photon energy trains
        // diffuse receivers when Indirect Guides is on.
        const bool useGuiding = settings.pathGuiding != 0 && pathGuiding_ && pathGuiding_->available() &&
                                (pathTracer || useBdpt);
#else
        const bool useGuiding = false;
#endif

        if (useSpectral) {
            const int bins = std::clamp(settings.spectralBins, 8, 32);
            if (spectralBins_.width != width || spectralBins_.height != height || spectralBins_.bins != bins)
                spectralBins_.resize(width, height, bins);
            if (sampleIndex == 0) spectralBins_.clear();
        }

        const CausticPhotonMap* photonPtr = nullptr;
        if (usePhoton) {
            // Rebuild each progressive pass (independent estimate averaged in the FB).
            const uint32_t photonSeed =
                hashCombine(uint32_t(settings.seed) * 9176u, uint32_t(sampleIndex) * 2654435761u);
            EmbreeTracer photonTracer{topScene_};
            photonTracer.time = scene.settings.motionBlur ? 0.5f : 0.0f;
            photonMap_.build(scene, photonTracer, srMax(0, settings.photonCount), photonSeed);
            // Always pass the map when Photon engine is active — even if empty
            // (Contribute to Caustics off) — so the integrator suppresses BSDF
            // caustic hits instead of falling back to MNEE/BSDF leakage.
            photonPtr = &photonMap_;
        } else {
            photonMap_.clear();
        }

        if (sampleIndex == 0) {
            if (useSpectralBdpt)
                logInfo(std::string("Integrator: BDPT Spectral (hero λ=") +
                        std::to_string(std::clamp(settings.spectralSamples, 2, 16)) + ", bins=" +
                        std::to_string(std::clamp(settings.spectralBins, 8, 32)) + ")" +
                        (useGuiding ? " + OpenPGL guiding" : "") +
                        (usePhoton ? " + Photon caustics" : " + LT/MNEE caustics"));
            else if (useSpectralPt)
                logInfo(std::string("Integrator: PT Spectral (hero λ=") +
                        std::to_string(std::clamp(settings.spectralSamples, 2, 16)) + ", bins=" +
                        std::to_string(std::clamp(settings.spectralBins, 8, 32)) + ")");
            else if (usePhoton)
                logInfo(std::string("Caustics: Photon map (VCM-style gather, ") +
                        std::to_string(photonMap_.size()) + " photons, r=" +
                        std::to_string(settings.photonRadius) +
                        (settings.causticsEngine == kCausticsEngineAuto ? ", MNEE+Photon→rough)" : ")"));
            else if (useBdpt)
                logInfo(std::string("Caustics: BDPT (bidirectional + light-tracing splats)") +
                        (useGuiding ? " + OpenPGL guiding" : ""));
            else if (useMnee)
                logInfo(std::string("Caustics: MNEE (manifold next-event, refractive)") +
                        (settings.causticsEngine == kCausticsEngineAuto ? " [MNEE+Photon→delta]" : "") +
                        (useGuiding ? " + OpenPGL guiding" : ""));
            else if (pathTracer)
                logInfo("Caustics: off (dark shadows through glass; shadow_opacity fakes)");

            const char* samplerName = "Sobol";
            if (settings.pixelSampler == kPixelSamplerBlueNoise) samplerName = "BlueNoise64";
            else if (settings.pixelSampler == kPixelSamplerWhite) samplerName = "White";
            const char* engineName = "FilmTile";
            if (settings.samplingEngine == kSamplingEngineLegacy) engineName = "Legacy";
            else if (settings.samplingEngine == kSamplingEngineProgressive) engineName = "Progressive";
            std::string engineDetail = engineName;
            if (settings.samplingEngine != kSamplingEngineProgressive) {
                engineDetail += " buckets " + std::to_string(tileSize) + "px";
                if (settings.tileSize <= 0) engineDetail += " (auto)";
            } else {
                engineDetail += " (no buckets, scanlines)";
            }
            logInfo(std::string("Sampling Engine: ") + engineDetail + "; Pixel Sampler: " + samplerName);
            if (settings.samplingDebug != kSamplingDebugOff) {
                static const char* kDiagNames[] = {"Off", "PixelJitter", "PathRng", "Bucket", "PixelHash"};
                const int d = std::clamp(settings.samplingDebug, 0, 4);
                logInfo(std::string("Diagnostic: Sampling Debug = ") + kDiagNames[d]);
            }
        }

        const int dispersionMode = settings.dispersionMode;
        const int dispersionMaxIfaces = srMax(1, settings.dispersionMaxInterfaces);

        const int samplingEngine = std::clamp(settings.samplingEngine, 0, 2);
        const bool legacySeed = samplingEngine == kSamplingEngineLegacy;

        auto makePathRng = [&](int x, int y, uint32_t salt = 0u) -> Rng {
            if (legacySeed) {
                // Pre-book seed: linear pixelIndex + weak hashCombine.
                const uint32_t pixelIndex = uint32_t(y) * uint32_t(width) + uint32_t(x);
                const uint32_t fs = frameSeed ^ (salt * 0x9e3779b9u);
                return Rng(hashCombine(pixelIndex, fs),
                           hashUint(pixelIndex ^ (fs * 2654435761u)));
            }
            return makePixelRng(x, y, sampleIndex, frameSeed, salt);
        };

        auto evaluatePixelSample = [&](int x, int y, int threadId) -> Vec3 {
            EmbreeTracer tracer{topScene_};
            Rng rng = makePathRng(x, y);
            // Camera AA / DoF — selectable Pixel Sampler (path bounce RNG stays PCG).
            float jx = 0.5f, jy = 0.5f;
            float lensU = 0.5f, lensV = 0.5f;
            const int pixelSampler = settings.pixelSampler;
            if (pixelSampler == kPixelSamplerBlueNoise) {
                blueNoisePixelJitter(x, y, sampleIndex, jx, jy);
                blueNoiseLensSample(x, y, sampleIndex, lensU, lensV);
            } else if (pixelSampler == kPixelSamplerWhite) {
                jx = rng.nextFloat();
                jy = rng.nextFloat();
                lensU = rng.nextFloat();
                lensV = rng.nextFloat();
            } else {
                // Default: Owen-scrambled Sobol (no fixed screen-space period).
                pixelSample(x, y, sampleIndex, jx, jy);
                lensSample(x, y, sampleIndex, lensU, lensV);
            }

            // Diagnostic: skip light transport and visualise sampler / seed fields.
            if (settings.samplingDebug != kSamplingDebugOff) {
                switch (settings.samplingDebug) {
                    case kSamplingDebugBucket: {
                        const int tx = x / tileSize;
                        const int ty = y / tileSize;
                        const uint32_t h = hashUint(uint32_t(tx) * 0x9e3779b9u ^ uint32_t(ty));
                        return Vec3(float((h >> 0) & 255u), float((h >> 8) & 255u),
                                    float((h >> 16) & 255u)) *
                               (1.0f / 255.0f);
                    }
                    case kSamplingDebugPixelHash: {
                        if (legacySeed) {
                            const uint32_t pixelIndex = uint32_t(y) * uint32_t(width) + uint32_t(x);
                            const uint32_t h = hashCombine(pixelIndex, frameSeed);
                            return Vec3(float((h >> 0) & 255u), float((h >> 8) & 255u),
                                        float((h >> 16) & 255u)) *
                                   (1.0f / 255.0f);
                        }
                        const uint64_t h = hashPixelSample(x, y, uint32_t(sampleIndex), frameSeed);
                        return Vec3(float((h >> 0) & 255u), float((h >> 8) & 255u),
                                    float((h >> 16) & 255u)) *
                               (1.0f / 255.0f);
                    }
                    case kSamplingDebugPixelJitter:
                        return Vec3(jx, jy, 0.25f);
                    case kSamplingDebugPathRng: {
                        const float u0 = makePathRng(x, y, 0xD1A60001u).nextFloat();
                        return Vec3(u0, u0, u0);
                    }
                    default:
                        return Vec3(0.0f);
                }
            }

            // Light-tracing splats assume the pinhole/thin-lens projection —
            // polynomial optics rays and camera motion blur bypass it.
            const bool allowSplats = !polyOptics_.active && scene.settings.motionBlur == 0;
            auto splatFbFor = [&](DispersionContext*) -> Framebuffer* {
                return allowSplats ? &fb : nullptr;
            };

            auto pickHeroChannel = [&](Rng& r) -> int {
                if (scene.camera.chromaticAberration == 0 && scene.hasDispersion == 0) return -1;
                if (dispersionMode == kDispersionFake) return -1;  // no IOR hero
                if (dispersionMode == kDispersionOptimized) {
                    // Stratified across spp (and lightly by pixel) — п.2.
                    return (sampleIndex + x + 2 * y) % 3;
                }
                if (dispersionMode == kDispersionSpectral3) return 0;  // filled per-channel below
                // Hero (default): random channel.
                int ch = int(r.nextFloat() * 3.0f);
                return ch > 2 ? 2 : ch;
            };

            auto makeDispCtx = [&](int channel) {
                DispersionContext ctx;
                ctx.mode = dispersionMode;
                ctx.heroChannel = channel;
                ctx.maxHits = (dispersionMode == kDispersionOptimized) ? dispersionMaxIfaces : 100000;
                ctx.disperseHits = 0;
                ctx.used = false;
                return ctx;
            };

            auto traceOnce = [&](Rng& r, Vec3 o, Vec3 d, DispersionContext* disp, int px, int py) -> Vec3 {
                IntegratorSampleContext<EmbreeTracer> ctx;
                ctx.scene = &scene;
                ctx.tracer = &tracer;
                ctx.origin = o;
                ctx.direction = d;
                ctx.rng = &r;
                ctx.dispersion = disp;
                ctx.splatFb = splatFbFor(disp);
                ctx.photons = photonPtr;
#if SOLSTICE_HAVE_OPENPGL
                PathGuiding::ThreadState* guidingPtr = nullptr;
                if (useGuiding) {
                    PathGuiding::ThreadState& guiding = pathGuiding_->thread(threadId);
                    guiding.beginPath();
                    guidingPtr = &guiding;
                }
                ctx.guiding = guidingPtr;
                Vec3 radiance(0.0f);
                if (useSpectralBdpt) {
                    SpectralBdptIntegrator<EmbreeTracer> integ;
                    radiance = integ.LiPixel(ctx, px, py, &spectralBins_);
                } else if (useSpectralPt) {
                    SpectralPathIntegrator<EmbreeTracer> integ;
                    radiance = integ.LiPixel(ctx, px, py, &spectralBins_);
                } else if (useBdpt) {
                    radiance = BdptIntegrator<EmbreeTracer>{}.Li(ctx);
                } else if (useMnee || usePhoton) {
                    radiance = PathMneeIntegrator<EmbreeTracer>{}.Li(ctx);
                } else {
                    radiance = PathIntegrator<EmbreeTracer>{}.Li(ctx);
                }
                if (guidingPtr) guidingPtr->endPath();
#else
                (void)threadId;
                (void)useGuiding;
                Vec3 radiance(0.0f);
                if (useSpectralBdpt) {
                    SpectralBdptIntegrator<EmbreeTracer> integ;
                    radiance = integ.LiPixel(ctx, px, py, &spectralBins_);
                } else if (useSpectralPt) {
                    SpectralPathIntegrator<EmbreeTracer> integ;
                    radiance = integ.LiPixel(ctx, px, py, &spectralBins_);
                } else if (useBdpt) {
                    radiance = BdptIntegrator<EmbreeTracer>{}.Li(ctx);
                } else if (useMnee || usePhoton) {
                    radiance = PathMneeIntegrator<EmbreeTracer>{}.Li(ctx);
                } else {
                    radiance = PathIntegrator<EmbreeTracer>{}.Li(ctx);
                }
#endif
                return radiance;
            };

            auto generateRay = [&](Rng& r, int chromaticChannel, Vec3& o, Vec3& d, float& tau,
                                   float shutterTime) -> bool {
                tau = 1.0f;
                tracer.time = shutterTime;
                if (polyOptics_.active) {
                    float wavelengthNm = -1.0f;
                    if (chromaticChannel >= 0 && scene.camera.chromaticAberration != 0)
                        wavelengthNm = chromaticWavelengthNm(chromaticChannel);
                    CameraData cam = scene.camera;
                    cam.cameraToWorld = cameraToWorldAtTime(scene, shutterTime);
                    return generatePolynomialOpticsRay(polyOptics_, cam, float(x) + jx, float(y) + jy, width,
                                                       height, lensU, lensV, r, o, d, wavelengthNm, &tau);
                }
                generateCameraRay(scene, float(x) + jx, float(y) + jy, lensU, lensV, o, d, shutterTime);
                return true;
            };

            auto heroMask = [](Vec3 rad, int ch) {
                const float h = (ch == 0 ? rad.x : (ch == 1 ? rad.y : rad.z)) * 3.0f;
                Vec3 out(0.0f);
                if (ch == 0) out.x = h;
                else if (ch == 1) out.y = h;
                else out.z = h;
                return out;
            };

            auto sampleShutter = [&](Rng& r) -> float {
                if (scene.settings.motionBlur == 0) return 0.0f;
                return r.nextFloat();
            };

            Vec3 radiance(0.0f);
            Vec3 origin, direction;
            float lensTau = 1.0f;

            if (dispersionMode == kDispersionSpectral3 &&
                (scene.hasDispersion != 0 || scene.camera.chromaticAberration != 0)) {
                // п.4: average independent R/G/B hero traces.
                for (int ch = 0; ch < 3; ++ch) {
                    Rng rCh = makePathRng(x, y, uint32_t(ch + 1));
                    DispersionContext ctx = makeDispCtx(ch);
                    const float shutterTime = sampleShutter(rCh);
                    if (!generateRay(rCh, ch, origin, direction, lensTau, shutterTime)) continue;
                    Vec3 r = traceOnce(rCh, origin, direction, &ctx, x, y);
                    r = r * std::max(0.0f, lensTau);
                    if (ctx.used) r = heroMask(r, ch);
                    radiance = radiance + r * (1.0f / 3.0f);
                }
            } else {
                const int chromaticChannel = pickHeroChannel(rng);
                DispersionContext ctx = makeDispCtx(chromaticChannel);
                const float shutterTime = sampleShutter(rng);
                if (!generateRay(rng, chromaticChannel, origin, direction, lensTau, shutterTime)) {
                    return Vec3(0.0f);
                }
                radiance = traceOnce(rng, origin, direction, &ctx, x, y);
                radiance = radiance * std::max(0.0f, lensTau);

                if (chromaticChannel >= 0) {
                    // Hero + Optimized: mask only if this path actually hit dispersing
                    // media (lazy mask). Fake never masks.
                    const bool doMask =
                        (dispersionMode == kDispersionHero || dispersionMode == kDispersionOptimized) &&
                        ctx.used;
                    if (doMask) radiance = heroMask(radiance, chromaticChannel);
                }
            }

            if (useSpectral && settings.filmFalseColor != 0 && spectralBins_.bins > 0) {
                // Replace beauty with this sample's contribution to the selected bin.
                const int bin = std::clamp(settings.filmFalseColorBin, 0, spectralBins_.bins - 1);
                // LiPixel already deposited into bins; approximate per-sample bin energy
                // by using returned RGB luminance (stable progressive display).
                const float lum = 0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z;
                const float span = kSpectrumLambdaMax - kSpectrumLambdaMin;
                const float lam =
                    kSpectrumLambdaMin + (float(bin) + 0.5f) / float(spectralBins_.bins) * span;
                radiance = wavelengthToFalseColor(lam) * lum;
                (void)bin;
            }

            return radiance;
        };

        // --- Sampling Engine dispatch -------------------------------------------------
        // Legacy: tiles + direct addSample + weak seed (pre-PBRT book).
        // FilmTile: PBRT local tile accum + merge + strong seed (current default).
        // Progressive: no buckets — parallel scanlines, direct addSample, strong seed.
        constexpr int kBootstrapStep = 2;

        auto runBootstrapOrFull = [&](auto&& renderPass) {
            if (sampleIndex == 0) {
                const int phaseCount = kBootstrapStep * kBootstrapStep;
                for (int phase = 0; phase < phaseCount; ++phase) {
                    if (cancel.load(std::memory_order_relaxed)) break;
                    renderPass(phase, true);
                    if (midProgress) midProgress();
                }
            } else {
                renderPass(0, false);
            }
        };

        if (samplingEngine == kSamplingEngineProgressive) {
            // True progressive: one work item = one scanline (no FilmTile / no buckets).
            runBootstrapOrFull([&](int bootstrapPhase, bool useBootstrap) {
                pool_->parallelFor(height, [&](int y, int threadId) {
                    if (cancel.load(std::memory_order_relaxed)) return;
                    for (int x = 0; x < width; ++x) {
                        if (useBootstrap) {
                            if (((x % kBootstrapStep) + (y % kBootstrapStep) * kBootstrapStep) !=
                                bootstrapPhase)
                                continue;
                        }
                        fb.addSample(x, y, evaluatePixelSample(x, y, threadId));
                    }
                });
            });
        } else if (samplingEngine == kSamplingEngineLegacy) {
            // Pre-book: same tile schedule, but write straight into the Film.
            runBootstrapOrFull([&](int bootstrapPhase, bool useBootstrap) {
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
                            if (useBootstrap) {
                                if (((x % kBootstrapStep) + (y % kBootstrapStep) * kBootstrapStep) !=
                                    bootstrapPhase)
                                    continue;
                            }
                            fb.addSample(x, y, evaluatePixelSample(x, y, threadId));
                        }
                    }
                });
            });
        } else {
            // FilmTile (PBRT ImageTileIntegrator).
            auto renderOneTile = [&](int tileIndex, int threadId, int bootstrapPhase, bool useBootstrap) {
                if (cancel.load(std::memory_order_relaxed)) return;
                const int tx = tileIndex % tilesX;
                const int ty = tileIndex / tilesX;
                const int x0 = tx * tileSize;
                const int y0 = ty * tileSize;
                const int x1 = std::min(x0 + tileSize, width);
                const int y1 = std::min(y0 + tileSize, height);
                FilmTile tile(x0, y0, x1, y1);
                for (int y = y0; y < y1; ++y) {
                    if (cancel.load(std::memory_order_relaxed)) break;
                    for (int x = x0; x < x1; ++x) {
                        if (useBootstrap) {
                            if (((x % kBootstrapStep) + (y % kBootstrapStep) * kBootstrapStep) !=
                                bootstrapPhase)
                                continue;
                        }
                        tile.addSample(x, y, evaluatePixelSample(x, y, threadId));
                    }
                }
                fb.mergeFilmTile(tile);
            };
            runBootstrapOrFull([&](int bootstrapPhase, bool useBootstrap) {
                pool_->parallelFor(tileCount, [&](int tileIndex, int threadId) {
                    renderOneTile(tileIndex, threadId, bootstrapPhase, useBootstrap);
                });
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

    bool copySpectralBins(int& width, int& height, int& bins, std::vector<float>& accum) const override {
        if (spectralBins_.bins <= 0 || spectralBins_.width <= 0) return false;
        width = spectralBins_.width;
        height = spectralBins_.height;
        bins = spectralBins_.bins;
        accum = spectralBins_.accum;
        return true;
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
        meshScenes_.shrink_to_fit();
        scene_.reset();
        photonMap_.clear();
#if SOLSTICE_HAVE_OPENPGL
        // Drop guiding field/sample storage with the scene — next buildScene resets.
        if (pathGuiding_) pathGuiding_.reset();
        pathGuiding_ = std::make_unique<PathGuiding>();
#endif
    }

    RTCDevice device_ = nullptr;
    RTCScene topScene_ = nullptr;
    std::vector<RTCScene> meshScenes_;
    ScenePtr scene_;
    SceneView view_;
    PolynomialOpticsCamera polyOptics_;
    std::unique_ptr<ThreadPool> pool_;
    int threadCount_ = 0;
    CausticPhotonMap photonMap_;
    SpectralBinBuffer spectralBins_;
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
