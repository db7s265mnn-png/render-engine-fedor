// Accumulation buffer for progressive rendering.
#pragma once

#include <atomic>
#include <mutex>
#include <vector>

#include "core/image.h"
#include "scene/types.h"

namespace sol {

class Framebuffer {
public:
    void resize(int width, int height);
    void clear();

    int width() const { return width_; }
    int height() const { return height_; }
    int sampleCount() const { return samples_.load(std::memory_order_relaxed); }
    void setSampleCount(int n) { samples_.store(n, std::memory_order_relaxed); }

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
    }

    void setPixel(int x, int y, Vec3 radiance, float weight) {
        Vec4& px = accum_[size_t(y) * size_t(width_) + size_t(x)];
        px = Vec4(radiance, weight);
    }

    Vec3 resolvePixel(int x, int y) const {
        const Vec4& px = accum_[size_t(y) * size_t(width_) + size_t(x)];
        return px.w > 0.0f ? px.xyz() * (1.0f / px.w) : Vec3(0.0f);
    }

    Vec4* data() { return accum_.data(); }
    const Vec4* data() const { return accum_.data(); }

    // Linear HDR image with the accumulation weight divided out.
    Image resolveLinear() const;
    // Display ready image: exposure, tone mapping and gamma applied.
    Image resolveDisplay(const RenderSettingsData& settings) const;

    std::mutex& mutex() { return mutex_; }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<Vec4> accum_;
    std::atomic<int> samples_{0};
    mutable std::mutex mutex_;
};

Vec3 applyToneMap(Vec3 color, const RenderSettingsData& settings);

}  // namespace sol
