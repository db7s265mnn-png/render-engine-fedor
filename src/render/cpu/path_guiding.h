// Host-only OpenPGL path guiding wrapper (CPU / Embree).
// Not included from CUDA translation units.
#pragma once

#include <memory>
#include <vector>

#include "core/math.h"
#include "core/rng.h"
#include "solstice_config.h"

namespace sol {

#if SOLSTICE_HAVE_OPENPGL

class PathGuiding {
public:
    PathGuiding();
    ~PathGuiding();

    PathGuiding(const PathGuiding&) = delete;
    PathGuiding& operator=(const PathGuiding&) = delete;

    bool available() const;

    // (Re)create the guiding field for a new scene.
    void reset(const Bounds3& worldBounds, int threadCount);

    // Call once per progressive sample after all tiles finish.
    void commitSample();

    class ThreadState {
    public:
        ThreadState();
        ~ThreadState();
        ThreadState(const ThreadState&) = delete;
        ThreadState& operator=(const ThreadState&) = delete;

        bool active() const { return active_; }
        float guideProbability() const { return guideProb_; }
        bool prepared() const { return prepared_; }

        void beginPath();
        void endPath();

        // Query / sample the surface guiding distribution at a shading point.
        bool prepare(Vec3 p, Vec3 n, Rng& rng);
        float pdf(Vec3 wiWorld) const;
        bool sample(float u1, float u2, Vec3& wiWorld, float& guidePdf) const;

        void beginSegment(Vec3 p, Vec3 wo);
        void recordEmission(Vec3 Le, float misWeight);
        void addScattered(Vec3 contrib);
        void recordBounce(Vec3 n, Vec3 wi, float pdf, Vec3 weight, bool delta, float roughness,
                          float eta, float rrSurvival);
        void setRussianRoulette(float rrSurvival);
        void recordBackground(Vec3 rayOrigin, Vec3 rayDir, Vec3 Le, float misWeight);
        void recordLightHit(Vec3 p, Vec3 wo, Vec3 Le, float misWeight);

    private:
        friend class PathGuiding;
        struct Data;
        std::unique_ptr<Data> data_;
        bool active_ = false;
        bool prepared_ = false;
        float guideProb_ = 0.5f;
        void* currentSegment_ = nullptr;
    };

    ThreadState& thread(int threadId);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif  // SOLSTICE_HAVE_OPENPGL

}  // namespace sol
