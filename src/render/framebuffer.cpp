#include "render/framebuffer.h"

#include <algorithm>
#include <cmath>

namespace sol {

void Framebuffer::resize(int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    width_ = width > 0 ? width : 0;
    height_ = height > 0 ? height : 0;
    accum_.assign(size_t(width_) * size_t(height_), Vec4(0.0f, 0.0f, 0.0f, 0.0f));
    const size_t n = size_t(width_) * size_t(height_) * 3;
    splat_ = n > 0 ? std::make_unique<std::atomic<double>[]>(n) : nullptr;
    for (size_t i = 0; i < n; ++i) splat_[i].store(0.0, std::memory_order_relaxed);
    splatPaths_.store(0, std::memory_order_relaxed);
    samples_.store(0, std::memory_order_relaxed);
    hasData_.store(false, std::memory_order_relaxed);
}

void Framebuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(accum_.begin(), accum_.end(), Vec4(0.0f, 0.0f, 0.0f, 0.0f));
    const size_t n = size_t(width_) * size_t(height_) * 3;
    for (size_t i = 0; i < n && splat_; ++i) splat_[i].store(0.0, std::memory_order_relaxed);
    splatPaths_.store(0, std::memory_order_relaxed);
    samples_.store(0, std::memory_order_relaxed);
    hasData_.store(false, std::memory_order_relaxed);
}

void Framebuffer::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    width_ = 0;
    height_ = 0;
    accum_.clear();
    accum_.shrink_to_fit();
    splat_.reset();
    splatPaths_.store(0, std::memory_order_relaxed);
    samples_.store(0, std::memory_order_relaxed);
    hasData_.store(false, std::memory_order_relaxed);
}

Image Framebuffer::resolveLinear() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Image image(width_, height_);
    const double invPaths = invSplatPaths();
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Vec4& px = accum_[size_t(y) * size_t(width_) + size_t(x)];
            Vec3 c = px.w > 0.0f ? px.xyz() * (1.0f / px.w) : Vec3(0.0f);
            c += splatPixel(x, y, invPaths);
            image.setRgb(x, y, c, 1.0f);
        }
    }
    return image;
}

Vec3 applyToneMap(Vec3 color, const RenderSettingsData& settings) {
    Vec3 c = color * std::exp2(settings.exposure);
    switch (settings.toneMapper) {
        case kToneReinhard: c = reinhard(c); break;
        case kToneAces: c = acesFilmic(c); break;
        default: break;
    }
    const float invGamma = 1.0f / (settings.gamma > 0.0f ? settings.gamma : 2.2f);
    return Vec3(std::pow(std::max(0.0f, c.x), invGamma), std::pow(std::max(0.0f, c.y), invGamma),
                std::pow(std::max(0.0f, c.z), invGamma));
}

Image Framebuffer::resolveDisplay(const RenderSettingsData& settings) const {
    std::lock_guard<std::mutex> lock(mutex_);
    Image image(width_, height_);
    const bool bootstrap = samples_.load(std::memory_order_relaxed) == 0 &&
                           hasData_.load(std::memory_order_relaxed);
    // Soft charcoal when a bootstrap cell has no finished neighbor yet — avoids
    // pure-black horizontal tile bars during slow BDPT Spectral phase 0.
    const Vec3 charcoal(0.07f, 0.07f, 0.08f);
    const double invPaths = invSplatPaths();

    auto gatherBootstrap = [&](int x, int y) -> Vec3 {
        // Expand search so unfinished 32px tiles borrow from finished ones.
        static const int kRadii[] = {2, 8, 16, 32};
        for (int radius : kRadii) {
            Vec3 sum(0.0f);
            float wSum = 0.0f;
            const int x0 = std::max(0, x - radius);
            const int y0 = std::max(0, y - radius);
            const int x1 = std::min(width_, x + radius + 1);
            const int y1 = std::min(height_, y + radius + 1);
            for (int sy = y0; sy < y1; ++sy) {
                for (int sx = x0; sx < x1; ++sx) {
                    const Vec4& q = accum_[size_t(sy) * size_t(width_) + size_t(sx)];
                    if (q.w <= 0.0f) continue;
                    const float dx = float(sx - x);
                    const float dy = float(sy - y);
                    const float w = 1.0f / (1.0f + dx * dx + dy * dy);
                    sum += q.xyz() * (w / q.w);
                    wSum += w;
                }
            }
            if (wSum > 0.0f) return sum * (1.0f / wSum) + splatPixel(x, y, invPaths);
        }
        // Splat-only pixel (eye sample not yet there) or empty cell.
        const Vec3 splat = splatPixel(x, y, invPaths);
        if (splat.x > 0.0f || splat.y > 0.0f || splat.z > 0.0f) return splat;
        return charcoal;
    };

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Vec4& px = accum_[size_t(y) * size_t(width_) + size_t(x)];
            Vec3 linearColor(0.0f);
            if (px.w > 0.0f) {
                linearColor = px.xyz() * (1.0f / px.w) + splatPixel(x, y, invPaths);
            } else if (bootstrap) {
                linearColor = gatherBootstrap(x, y);
            } else {
                // After sample 0 every eye pixel should have w>0; still show splats.
                linearColor = splatPixel(x, y, invPaths);
            }
            image.setRgb(x, y, applyToneMap(linearColor, settings), 1.0f);
        }
    }
    return image;
}

}  // namespace sol
