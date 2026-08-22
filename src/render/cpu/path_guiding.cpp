#include "render/cpu/path_guiding.h"

#if SOLSTICE_HAVE_OPENPGL

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include <openpgl/cpp/OpenPGL.h>

#include "core/log.h"

namespace sol {
namespace {

pgl_vec3f toPgl(Vec3 v) { return openpgl::cpp::Vector3(v.x, v.y, v.z); }
pgl_point3f toPglPoint(Vec3 v) { return openpgl::cpp::Point3(v.x, v.y, v.z); }
Vec3 fromPgl(const pgl_vec3f& v) { return Vec3(v.x, v.y, v.z); }

// OpenPGL isValid() requires finite unit-ish directions. Degenerate wo/n
// (zero after a volume scatter) used to leave garbage and crash the field.
pgl_vec3f toPglDir(Vec3 v) {
    const float l2 = lengthSquared(v);
    if (!(l2 > 1e-12f) || !std::isfinite(l2)) return openpgl::cpp::Vector3(0.f, 1.f, 0.f);
    v = v * (1.0f / std::sqrt(l2));
    return toPgl(v);
}

PGL_DEVICE_TYPE pickDeviceType() {
#if defined(OPENPGL_SUPPORT_DEVICE_TYPE_CPU_16) || defined(OPENPGL_DEVICE_TYPE_CPU_16)
    return PGL_DEVICE_TYPE_CPU_8;
#else
    return PGL_DEVICE_TYPE_CPU_8;
#endif
}

}  // namespace

struct PathGuiding::ThreadState::Data {
    openpgl::cpp::PathSegmentStorage segments;
    std::unique_ptr<openpgl::cpp::SurfaceSamplingDistribution> surfaceDist;
    std::unique_ptr<openpgl::cpp::VolumeSamplingDistribution> volumeDist;
    openpgl::cpp::SampleStorage* sampleStorage = nullptr;
    openpgl::cpp::Field* field = nullptr;
};

PathGuiding::ThreadState::ThreadState() : data_(std::make_unique<Data>()) {
    data_->segments.Reserve(64);
}

PathGuiding::ThreadState::~ThreadState() = default;

void* PathGuiding::ThreadState::takeSegment() {
    // OpenPGL PathSegmentDataStorage::next() writes index `Reserve` (one past
    // the allocation) before returning nullptr. Stop one slot early.
    if (!data_ || segmentsUsed_ >= segmentReserve_ - 1) {
        currentSegment_ = nullptr;
        return nullptr;
    }
    openpgl::cpp::PathSegment* seg = data_->segments.NextSegment();
    if (!seg) {
        currentSegment_ = nullptr;
        return nullptr;
    }
    ++segmentsUsed_;
    currentSegment_ = seg;
    return seg;
}

void PathGuiding::ThreadState::beginPath() {
    if (!active_ || !data_) return;
    data_->segments.Clear();
    currentSegment_ = nullptr;
    segmentsUsed_ = 0;
    prepared_ = false;
    preparedVolume_ = false;
}

void PathGuiding::ThreadState::endPath() {
    if (!active_ || !data_ || !data_->sampleStorage) return;
    try {
        data_->segments.PropagateSamples(data_->sampleStorage, /*guideDirectLight=*/false,
                                         /*useNEEMiWeights=*/true,
                                         /*rrAffectsDirectContribution=*/true);
    } catch (const std::exception& ex) {
        logError(std::string("OpenPGL propagate failed: ") + ex.what());
    } catch (...) {
        logError("OpenPGL propagate failed");
    }
    data_->segments.Clear();
    currentSegment_ = nullptr;
    segmentsUsed_ = 0;
    prepared_ = false;
    preparedVolume_ = false;
}

bool PathGuiding::ThreadState::prepare(Vec3 p, Vec3 n, Rng& rng) {
    prepared_ = false;
    if (!active_ || !data_ || !data_->field || !data_->surfaceDist) return false;
    if (guideProb_ <= 0.0f) return false;  // field not trained yet
    float u = rng.nextFloat();
    if (!data_->surfaceDist->Init(data_->field, toPglPoint(p), u)) return false;
    if (data_->surfaceDist->SupportsApplyCosineProduct())
        data_->surfaceDist->ApplyCosineProduct(toPgl(n));
    prepared_ = true;
    return true;
}

float PathGuiding::ThreadState::pdf(Vec3 wiWorld) const {
    if (!prepared_ || !data_ || !data_->surfaceDist) return 0.0f;
    return data_->surfaceDist->PDF(toPgl(wiWorld));
}

bool PathGuiding::ThreadState::sample(float u1, float u2, Vec3& wiWorld, float& guidePdf) const {
    if (!prepared_ || !data_ || !data_->surfaceDist) return false;
    pgl_vec3f dir{};
    guidePdf = data_->surfaceDist->SamplePDF(openpgl::cpp::Point2(u1, u2), dir);
    if (!(guidePdf > 0.0f) || !std::isfinite(guidePdf)) return false;
    wiWorld = normalize(fromPgl(dir));
    return lengthSquared(wiWorld) > 0.0f;
}

void PathGuiding::ThreadState::beginSegment(Vec3 p, Vec3 wo) {
    if (!active_ || !data_) return;
    auto* seg = static_cast<openpgl::cpp::PathSegment*>(takeSegment());
    if (!seg) return;
    openpgl::cpp::SetPosition(seg, toPglPoint(p));
    openpgl::cpp::SetDirectionOut(seg, toPglDir(wo));
    openpgl::cpp::SetNormal(seg, toPglDir(wo));
    openpgl::cpp::SetVolumeScatter(seg, false);
    openpgl::cpp::SetScatteredContribution(seg, openpgl::cpp::Vector3(0.f, 0.f, 0.f));
    openpgl::cpp::SetDirectContribution(seg, openpgl::cpp::Vector3(0.f, 0.f, 0.f));
    openpgl::cpp::SetTransmittanceWeight(seg, openpgl::cpp::Vector3(1.f, 1.f, 1.f));
    openpgl::cpp::SetEta(seg, 1.0f);
    prepared_ = false;
    preparedVolume_ = false;
}

void PathGuiding::ThreadState::beginVolumeSegment(Vec3 p, Vec3 wo) {
    if (!active_ || !data_) return;
    auto* seg = static_cast<openpgl::cpp::PathSegment*>(takeSegment());
    if (!seg) return;
    // Volume vertices have no geometric normal; OpenPGL still requires a
    // finite unit direction on every segment (including ones that never
    // reach recordBounce if the path dies).
    openpgl::cpp::SetPosition(seg, toPglPoint(p));
    openpgl::cpp::SetDirectionOut(seg, toPglDir(wo));
    openpgl::cpp::SetNormal(seg, toPglDir(wo));
    openpgl::cpp::SetVolumeScatter(seg, true);
    openpgl::cpp::SetScatteredContribution(seg, openpgl::cpp::Vector3(0.f, 0.f, 0.f));
    openpgl::cpp::SetDirectContribution(seg, openpgl::cpp::Vector3(0.f, 0.f, 0.f));
    openpgl::cpp::SetTransmittanceWeight(seg, openpgl::cpp::Vector3(1.f, 1.f, 1.f));
    openpgl::cpp::SetEta(seg, 1.0f);
    prepared_ = false;
    preparedVolume_ = false;
}

bool PathGuiding::ThreadState::prepareVolume(Vec3 p, Vec3 wo, float g, Rng& rng) {
    preparedVolume_ = false;
    if (!active_ || !data_ || !data_->field || !data_->volumeDist) return false;
    if (guideProb_ <= 0.0f) return false;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) return false;
    const float woLen2 = lengthSquared(wo);
    if (!(woLen2 > 1e-12f) || !std::isfinite(woLen2)) return false;
    wo = wo * (1.0f / std::sqrt(woLen2));
    g = clampf(g, -0.99f, 0.99f);
    try {
        float u = rng.nextFloat();
        if (!data_->volumeDist->Init(data_->field, toPglPoint(p), u)) return false;
        if (!data_->volumeDist->Validate()) return false;
        if (std::fabs(g) > 1e-3f &&
            data_->volumeDist->SupportsApplySingleLobeHenyeyGreensteinProduct()) {
            data_->volumeDist->ApplySingleLobeHenyeyGreensteinProduct(toPgl(wo), g);
            if (!data_->volumeDist->Validate()) return false;
        }
        // Zero-weight HG products leave NaN mixture weights; sampling those
        // underflows uint32 numDistributions-1 and reads 0xFFFFFFFFFFFFFFFF.
        const float probe = data_->volumeDist->PDF(toPgl(wo));
        if (!(probe >= 0.0f) || !std::isfinite(probe)) return false;
        preparedVolume_ = true;
        return true;
    } catch (const std::exception&) {
        preparedVolume_ = false;
        return false;
    } catch (...) {
        preparedVolume_ = false;
        return false;
    }
}

float PathGuiding::ThreadState::pdfVolume(Vec3 wiWorld) const {
    if (!preparedVolume_ || !data_ || !data_->volumeDist) return 0.0f;
    return data_->volumeDist->PDF(toPgl(wiWorld));
}

bool PathGuiding::ThreadState::sampleVolume(float u1, float u2, Vec3& wiWorld, float& guidePdf) const {
    if (!preparedVolume_ || !data_ || !data_->volumeDist) return false;
    pgl_vec3f dir{};
    guidePdf = data_->volumeDist->SamplePDF(openpgl::cpp::Point2(u1, u2), dir);
    if (!(guidePdf > 0.0f) || !std::isfinite(guidePdf)) return false;
    wiWorld = normalize(fromPgl(dir));
    return lengthSquared(wiWorld) > 0.0f;
}

void PathGuiding::ThreadState::recordEmission(Vec3 Le, float misWeight) {
    if (!currentSegment_) return;
    auto* seg = static_cast<openpgl::cpp::PathSegment*>(currentSegment_);
    openpgl::cpp::SetDirectContribution(seg, toPgl(Le));
    openpgl::cpp::SetMiWeight(seg, misWeight);
}

void PathGuiding::ThreadState::addScattered(Vec3 contrib) {
    addScatteredAt(currentSegment_, contrib);
}

void PathGuiding::ThreadState::addScatteredAt(void* segment, Vec3 contrib) {
    if (!segment) return;
    auto* seg = static_cast<openpgl::cpp::PathSegment*>(segment);
    openpgl::cpp::AddScatteredContribution(seg, toPgl(contrib));
}

void PathGuiding::ThreadState::recordBounce(Vec3 n, Vec3 wi, float pdfVal, Vec3 weight, bool delta,
                                            float roughness, float eta, float rrSurvival,
                                            bool volumeScatter) {
    if (!currentSegment_) return;
    if (!(pdfVal >= 0.0f) || !std::isfinite(pdfVal)) pdfVal = 0.0f;
    auto* seg = static_cast<openpgl::cpp::PathSegment*>(currentSegment_);
    openpgl::cpp::SetTransmittanceWeight(seg, openpgl::cpp::Vector3(1.f, 1.f, 1.f));
    openpgl::cpp::SetVolumeScatter(seg, volumeScatter);
    openpgl::cpp::SetNormal(seg, toPglDir(n));
    openpgl::cpp::SetDirectionIn(seg, toPglDir(wi));
    openpgl::cpp::SetPDFDirectionIn(seg, pdfVal);
    openpgl::cpp::SetScatteringWeight(seg, toPgl(weight));
    openpgl::cpp::SetIsDelta(seg, delta);
    openpgl::cpp::SetEta(seg, eta);
    openpgl::cpp::SetRoughness(seg, roughness);
    openpgl::cpp::SetRussianRouletteProbability(seg, rrSurvival);
}

void PathGuiding::ThreadState::setRussianRoulette(float rrSurvival) {
    if (!currentSegment_) return;
    auto* seg = static_cast<openpgl::cpp::PathSegment*>(currentSegment_);
    openpgl::cpp::SetRussianRouletteProbability(seg, rrSurvival);
}

void PathGuiding::ThreadState::recordBackground(Vec3 rayOrigin, Vec3 rayDir, Vec3 Le,
                                                float misWeight) {
    if (!active_ || !data_) return;
    if (segmentsUsed_ >= segmentReserve_ - 1) return;
    openpgl::cpp::PathSegment background;
    const Vec3 p = rayOrigin + rayDir * 1.0e6f;
    openpgl::cpp::SetPosition(&background, toPglPoint(p));
    openpgl::cpp::SetNormal(&background, openpgl::cpp::Vector3(0.f, 0.f, 1.f));
    openpgl::cpp::SetDirectionOut(&background, toPglDir(-rayDir));
    openpgl::cpp::SetDirectContribution(&background, toPgl(Le));
    openpgl::cpp::SetMiWeight(&background, misWeight);
    data_->segments.AddSegment(background);
    ++segmentsUsed_;
}

void PathGuiding::ThreadState::recordLightHit(Vec3 p, Vec3 wo, Vec3 Le, float misWeight) {
    if (!active_ || !data_) return;
    auto* seg = static_cast<openpgl::cpp::PathSegment*>(takeSegment());
    if (!seg) return;
    openpgl::cpp::SetPosition(seg, toPglPoint(p));
    openpgl::cpp::SetDirectionOut(seg, toPglDir(wo));
    openpgl::cpp::SetNormal(seg, toPglDir(wo));
    openpgl::cpp::SetDirectionIn(seg, toPglDir(-wo));
    openpgl::cpp::SetPDFDirectionIn(seg, 1.0f);
    openpgl::cpp::SetVolumeScatter(seg, false);
    openpgl::cpp::SetScatteredContribution(seg, openpgl::cpp::Vector3(0.f, 0.f, 0.f));
    openpgl::cpp::SetDirectContribution(seg, toPgl(Le));
    openpgl::cpp::SetMiWeight(seg, misWeight);
    openpgl::cpp::SetTransmittanceWeight(seg, openpgl::cpp::Vector3(1.f, 1.f, 1.f));
    openpgl::cpp::SetScatteringWeight(seg, openpgl::cpp::Vector3(1.f, 1.f, 1.f));
    openpgl::cpp::SetEta(seg, 1.0f);
}

struct PathGuiding::Impl {
    std::unique_ptr<openpgl::cpp::Device> device;
    std::unique_ptr<openpgl::cpp::Field> field;
    std::unique_ptr<openpgl::cpp::SampleStorage> sampleStorage;
    std::vector<std::unique_ptr<ThreadState>> threads;
    std::atomic<int> trainedIterations{0};
    bool ready = false;
};

PathGuiding::PathGuiding() : impl_(std::make_unique<Impl>()) {}
PathGuiding::~PathGuiding() = default;

bool PathGuiding::available() const { return impl_ && impl_->ready; }

void PathGuiding::reset(const Bounds3& worldBounds, int threadCount, int maxPathDepth) {
    impl_->ready = false;
    impl_->threads.clear();
    impl_->trainedIterations.store(0);
    impl_->field.reset();
    impl_->sampleStorage.reset();
    impl_->device.reset();

    try {
        const int n =
            std::max(1, threadCount > 0 ? threadCount : int(std::thread::hardware_concurrency()));
        impl_->device = std::make_unique<openpgl::cpp::Device>(pickDeviceType(), size_t(n));
        openpgl::cpp::FieldConfig cfg;
        cfg.Init(PGL_SPATIAL_STRUCTURE_KDTREE, PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM,
                 /*deterministic=*/true);
        cfg.SetSpatialStructureArgMaxDepth(16);
        impl_->field = std::make_unique<openpgl::cpp::Field>(impl_->device.get(), cfg);
        if (worldBounds.valid()) {
            const Vec3 pad = vmax(worldBounds.extent() * 0.05f, Vec3(0.1f, 0.1f, 0.1f));
            impl_->field->SetSceneBounds(openpgl::cpp::Box3(
                worldBounds.lo.x - pad.x, worldBounds.lo.y - pad.y, worldBounds.lo.z - pad.z,
                worldBounds.hi.x + pad.x, worldBounds.hi.y + pad.y, worldBounds.hi.z + pad.z));
        }
        impl_->sampleStorage = std::make_unique<openpgl::cpp::SampleStorage>();
        // Preallocate per-thread states — thread() must be lock-free (called per
        // pixel). ThreadPool::parallelFor runs chunks on the calling thread as
        // thread id 0 AND on N workers as ids 1..N, so N+1 states are needed —
        // sharing one state across two threads corrupts OpenPGL's segment storage.
        const int states = n + 1;
        // Volume paths record one segment per scatter. UI maxDepth goes to 4096;
        // OpenPGL's next() also writes one past Reserve(), so keep a pad.
        const int reserve = std::max(256, std::max(1, maxPathDepth) + 32);
        impl_->threads.reserve(size_t(states));
        for (int i = 0; i < states; ++i) {
            auto ts = std::make_unique<ThreadState>();
            ts->data_->field = impl_->field.get();
            ts->data_->sampleStorage = impl_->sampleStorage.get();
            ts->data_->surfaceDist =
                std::make_unique<openpgl::cpp::SurfaceSamplingDistribution>(impl_->field.get());
            ts->data_->volumeDist =
                std::make_unique<openpgl::cpp::VolumeSamplingDistribution>(impl_->field.get());
            ts->data_->segments.Reserve(size_t(reserve));
            ts->segmentReserve_ = reserve;
            ts->segmentsUsed_ = 0;
            ts->active_ = true;
            ts->guideProb_ = 0.0f;  // stays 0 until the field is trained
            impl_->threads.push_back(std::move(ts));
        }
        impl_->ready = true;
        logInfo("OpenPGL: path guiding field ready (threads=" + std::to_string(n) +
                ", segments=" + std::to_string(reserve) + ")");
    } catch (const std::exception& ex) {
        logError(std::string("OpenPGL init failed: ") + ex.what());
        impl_->ready = false;
        impl_->threads.clear();
        impl_->field.reset();
        impl_->sampleStorage.reset();
        impl_->device.reset();
    }
}

void PathGuiding::commitSample() {
    if (!impl_ || !impl_->ready || !impl_->field || !impl_->sampleStorage) return;
    const size_t n = impl_->sampleStorage->GetSizeSurface() + impl_->sampleStorage->GetSizeVolume();
    if (n < 1024) return;
    try {
        impl_->field->Update(*impl_->sampleStorage);
        impl_->sampleStorage->Clear();
        if (!impl_->field->Validate()) {
            logError("OpenPGL field invalid after update — keeping previous guide probability");
            return;
        }
        const int iters = impl_->trainedIterations.fetch_add(1) + 1;
        // Ramp the guided fraction in as the field converges: 0 (untrained) → 0.5.
        const float prob = iters <= 0 ? 0.0f : std::min(0.5f, 0.2f + 0.1f * float(iters));
        for (auto& ts : impl_->threads)
            if (ts) ts->guideProb_ = prob;
    } catch (const std::exception& ex) {
        logError(std::string("OpenPGL update failed: ") + ex.what());
    } catch (...) {
        logError("OpenPGL update failed");
    }
}

int PathGuiding::trainedIterations() const {
    return impl_ ? impl_->trainedIterations.load() : 0;
}

PathGuiding::ThreadState& PathGuiding::thread(int threadId) {
    static ThreadState fallback;
    if (!impl_ || impl_->threads.empty()) return fallback;
    size_t idx = size_t(threadId < 0 ? 0 : threadId);
    if (idx >= impl_->threads.size()) idx = impl_->threads.size() - 1;
    return *impl_->threads[idx];
}

}  // namespace sol

#endif  // SOLSTICE_HAVE_OPENPGL
