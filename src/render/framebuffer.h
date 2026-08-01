// Accumulation buffer for progressive rendering.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "core/image.h"
#include "scene/types.h"

namespace sol {

class Framebuffer {
public:
    void resize(int width, int height);
    void clear();
    // Drop accumulation storage entirely (Render teardown).
    void release();

    int width() const { return width_; }
    int height() const { return height_; }
    int sampleCount() const { return samples_.load(std::memory_order_relaxed); }
    void setSampleCount(int n) { samples_.store(n, std::memory_order_relaxed); }
    bool hasAccumulatedData() const { return hasData_.load(std::memory_order_relaxed); }
    void markHasData() { hasData_.store(true, std::memory_order_relaxed); }

    // Accumulates one sample. Safe as long as different threads own different
    // pixels, which is how the tile scheduler works. The UI resolves the buffer
    // while workers write to it; a preview frame may therefore mix samples from
    // two passes, which is invisible in practice and avoids a per pixel lock.
    void addSample(int x, int y, Vec3 radiance) {
        Vec4& px = accum_[size_t(y) * size_t(width_) + size_t(x)];
        px.x += radiance.x;
        px.y += radiance.y;
        px.z += radiance.z;
        px.w += 1.0f;
        hasData_.store(true, std::memory_order_relaxed);
    }

    void setPixel(int x, int y, Vec3 radiance, float weight) {
        Vec4& px = accum_[size_t(y) * size_t(width_) + size_t(x)];
        px = Vec4(radiance, weight);
        if (weight > 0.0f) hasData_.store(true, std::memory_order_relaxed);
    }

    // Light-tracing splats (BDPT t=1): any thread may write to any pixel, so the
    // splat plane uses atomic adds. Normalized by the number of traced light
    // paths, not per-pixel sample counts.
    //
    // Both the plane and the path counter are 64 bit on purpose. One path is
    // counted per pixel per pass, so a 960x540 frame overflows a 32-bit counter
    // after ~4100 passes — and `long` is 32 bit on Windows. Single-precision
    // accumulation fails earlier still: splat values carry the camera pdf (~1e6
    // here), so once a pixel's running sum reaches ~1e13 further adds round away
    // and the caustic silently stops converging.
    void addSplat(int x, int y, Vec3 v) {
        if (!splat_ || x < 0 || y < 0 || x >= width_ || y >= height_) return;
        std::atomic<double>* px = splat_.get() + (size_t(y) * size_t(width_) + size_t(x)) * 3;
        for (int c = 0; c < 3; ++c) {
            const double add = c == 0 ? double(v.x) : (c == 1 ? double(v.y) : double(v.z));
            double cur = px[c].load(std::memory_order_relaxed);
            while (!px[c].compare_exchange_weak(cur, cur + add, std::memory_order_relaxed)) {
            }
        }
    }
    void addSplatPath() { splatPaths_.fetch_add(1, std::memory_order_relaxed); }
    // Bulk form used by tests to reach counts that a per-pixel loop cannot.
    void addSplatPaths(int64_t n) { splatPaths_.fetch_add(n, std::memory_order_relaxed); }
    int64_t splatPaths() const { return splatPaths_.load(std::memory_order_relaxed); }

    Vec3 splatPixel(int x, int y, double invPaths) const {
        if (!splat_ || invPaths <= 0.0) return Vec3(0.0f);
        const std::atomic<double>* px = splat_.get() + (size_t(y) * size_t(width_) + size_t(x)) * 3;
        return Vec3(float(px[0].load(std::memory_order_relaxed) * invPaths),
                    float(px[1].load(std::memory_order_relaxed) * invPaths),
                    float(px[2].load(std::memory_order_relaxed) * invPaths));
    }

    // 1 / traced light paths, or 0 when light tracing has not run.
    double invSplatPaths() const {
        const int64_t paths = splatPaths_.load(std::memory_order_relaxed);
        return paths > 0 ? 1.0 / double(paths) : 0.0;
    }

    Vec3 resolvePixel(int x, int y) const {
        const Vec4& px = accum_[size_t(y) * size_t(width_) + size_t(x)];
        const Vec3 base = px.w > 0.0f ? px.xyz() * (1.0f / px.w) : Vec3(0.0f);
        return base + splatPixel(x, y, invSplatPaths());
    }

    Vec4* data() { return accum_.data(); }
    const Vec4* data() const { return accum_.data(); }

    // Linear HDR image with the accumulation weight divided out.
    Image resolveLinear() const;
    // Display ready image: exposure, tone mapping and gamma applied.
    // Incomplete bootstrap pixels are filled from nearby samples so IPR does
    // not flash black tile holes (soft 2×2, not hard 4×4 blocks).
    Image resolveDisplay(const RenderSettingsData& settings) const;

    std::mutex& mutex() { return mutex_; }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<Vec4> accum_;
    std::unique_ptr<std::atomic<double>[]> splat_;  // 3 doubles per pixel
    std::atomic<int64_t> splatPaths_{0};
    std::atomic<int> samples_{0};
    std::atomic<bool> hasData_{false};
    mutable std::mutex mutex_;
};

Vec3 applyToneMap(Vec3 color, const RenderSettingsData& settings);

}  // namespace sol
