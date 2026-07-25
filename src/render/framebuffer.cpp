#include "render/framebuffer.h"

#include <algorithm>
#include <cmath>

namespace sol {

void Framebuffer::resize(int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    width_ = width > 0 ? width : 0;
    height_ = height > 0 ? height : 0;
    accum_.assign(size_t(width_) * size_t(height_), Vec4(0.0f, 0.0f, 0.0f, 0.0f));
    samples_.store(0, std::memory_order_relaxed);
}

void Framebuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(accum_.begin(), accum_.end(), Vec4(0.0f, 0.0f, 0.0f, 0.0f));
    samples_.store(0, std::memory_order_relaxed);
}

Image Framebuffer::resolveLinear() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Image image(width_, height_);
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Vec4& px = accum_[size_t(y) * size_t(width_) + size_t(x)];
            const Vec3 c = px.w > 0.0f ? px.xyz() * (1.0f / px.w) : Vec3(0.0f);
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
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Vec4& px = accum_[size_t(y) * size_t(width_) + size_t(x)];
            const Vec3 linearColor = px.w > 0.0f ? px.xyz() * (1.0f / px.w) : Vec3(0.0f);
            image.setRgb(x, y, applyToneMap(linearColor, settings), 1.0f);
        }
    }
    return image;
}

}  // namespace sol
