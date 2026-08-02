#include "render/framebuffer.h"
#include "render/film_tile.h"

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

void Framebuffer::mergeFilmTile(const FilmTile& tile) {
    if (tile.empty() || tile.pixels.empty()) return;
    const int tw = tile.storW;
    const int th = tile.storH;
    bool wrote = false;
    for (int ty = 0; ty < th; ++ty) {
        const int y = tile.storY0 + ty;
        if (y < 0 || y >= height_) continue;
        for (int tx = 0; tx < tw; ++tx) {
            const int x = tile.storX0 + tx;
            if (x < 0 || x >= width_) continue;
            const Vec4& src = tile.pixels[size_t(ty) * size_t(tw) + size_t(tx)];
            if (src.w <= 0.0f) continue;
            Vec4& dst = accum_[size_t(y) * size_t(width_) + size_t(x)];
            dst.x += src.x;
            dst.y += src.y;
            dst.z += src.z;
            dst.w += src.w;
            wrote = true;
        }
    }
    if (wrote) hasData_.store(true, std::memory_order_relaxed);
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
    // Keep bootstrap cheap (2×2). The 9851876 multi-radius (…32) hole-fill
    // scanned up to ~65² neighbors per empty pixel on every IPR tick while
    // sampleCount==0 — after every camera clear that made orbit hitch hard.
    // Soft charcoal still avoids pure-black tile bars on sparse phase 0.
    constexpr int kBootstrapStep = 2;
    const Vec3 charcoal(0.07f, 0.07f, 0.08f);
    const double invPaths = invSplatPaths();

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Vec4& px = accum_[size_t(y) * size_t(width_) + size_t(x)];
            Vec3 linearColor(0.0f);
            if (px.w > 0.0f) {
                linearColor = px.xyz() * (1.0f / px.w) + splatPixel(x, y, invPaths);
            } else if (bootstrap) {
                const int x0 = x - (x % kBootstrapStep);
                const int y0 = y - (y % kBootstrapStep);
                const int x1 = std::min(x0 + kBootstrapStep, width_);
                const int y1 = std::min(y0 + kBootstrapStep, height_);
                Vec3 sum(0.0f);
                float wSum = 0.0f;
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
                if (wSum > 0.0f) {
                    linearColor = sum * (1.0f / wSum) + splatPixel(x, y, invPaths);
                } else {
                    const Vec3 splat = splatPixel(x, y, invPaths);
                    linearColor = (splat.x > 0.0f || splat.y > 0.0f || splat.z > 0.0f) ? splat
                                                                                        : charcoal;
                }
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
