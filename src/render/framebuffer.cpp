#include "render/framebuffer.h"
#include "render/film_tile.h"
#include "render/pixel_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "io/ocio_util.h"

namespace sol {

void Framebuffer::resize(int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    width_ = width > 0 ? width : 0;
    height_ = height > 0 ? height : 0;
    accum_.assign(size_t(width_) * size_t(height_), Vec4(0.0f, 0.0f, 0.0f, 0.0f));
    lumSq_.assign(size_t(width_) * size_t(height_), 0.0f);
    skip_.assign(size_t(width_) * size_t(height_), 0);
    skipCount_ = 0;
    noiseDone_ = false;
    const size_t n = size_t(width_) * size_t(height_) * 3;
    splat_ = n > 0 ? std::make_unique<std::atomic<double>[]>(n) : nullptr;
    for (size_t i = 0; i < n; ++i) splat_[i].store(0.0, std::memory_order_relaxed);
    splatPaths_.store(0, std::memory_order_relaxed);
    samples_.store(0, std::memory_order_relaxed);
    hasData_.store(false, std::memory_order_relaxed);
    presentable_.store(false, std::memory_order_relaxed);
}

void Framebuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fill(accum_.begin(), accum_.end(), Vec4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(lumSq_.begin(), lumSq_.end(), 0.0f);
    std::fill(skip_.begin(), skip_.end(), uint8_t(0));
    skipCount_ = 0;
    noiseDone_ = false;
    const size_t n = size_t(width_) * size_t(height_) * 3;
    for (size_t i = 0; i < n && splat_; ++i) splat_[i].store(0.0, std::memory_order_relaxed);
    splatPaths_.store(0, std::memory_order_relaxed);
    samples_.store(0, std::memory_order_relaxed);
    hasData_.store(false, std::memory_order_relaxed);
    presentable_.store(false, std::memory_order_relaxed);
}

void Framebuffer::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    width_ = 0;
    height_ = 0;
    accum_.clear();
    accum_.shrink_to_fit();
    lumSq_.clear();
    lumSq_.shrink_to_fit();
    skip_.clear();
    skip_.shrink_to_fit();
    skipCount_ = 0;
    noiseDone_ = false;
    splat_.reset();
    splatPaths_.store(0, std::memory_order_relaxed);
    samples_.store(0, std::memory_order_relaxed);
    hasData_.store(false, std::memory_order_relaxed);
    presentable_.store(false, std::memory_order_relaxed);
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

void Framebuffer::addNoiseSample(int x, int y, Vec3 radiance, float count) {
    if (x < 0 || y < 0 || x >= width_ || y >= height_ || !(count > 0.0f) || lumSq_.empty()) return;
    const float L = pixelLuminance(radiance.x, radiance.y, radiance.z);
    lumSq_[size_t(y) * size_t(width_) + size_t(x)] += L * L * count;
}

void Framebuffer::copyLumSq(const float* src, size_t count) {
    if (!src || lumSq_.size() != count) return;
    std::memcpy(lumSq_.data(), src, count * sizeof(float));
}

void Framebuffer::addLumSq(const float* src, size_t count) {
    if (!src || lumSq_.size() != count) return;
    for (size_t i = 0; i < count; ++i) lumSq_[i] += src[i];
}

void Framebuffer::copySkipMask(const std::vector<uint8_t>& src) {
    if (src.size() != skip_.size()) return;
    skip_ = src;
    skipCount_ = 0;
    for (uint8_t v : skip_) {
        if (v) ++skipCount_;
    }
}

bool Framebuffer::skipPixel(int x, int y) const {
    if (skip_.empty() || x < 0 || y < 0 || x >= width_ || y >= height_) return false;
    return skip_[size_t(y) * size_t(width_) + size_t(x)] != 0;
}

void Framebuffer::refreshNoiseOracle(float threshold, int sppDone, int maxSpp) {
    noiseDone_ = false;
    const size_t n = size_t(width_) * size_t(height_);
    if (!(threshold > 0.0f) || n == 0 || accum_.size() != n || lumSq_.size() != n) {
        skip_.assign(n, 0);
        skipCount_ = 0;
        return;
    }
    if (skip_.size() != n) skip_.assign(n, 0);
    const int minS = noiseOracleMinSamples(maxSpp);
    if (sppDone < minS) return;

    std::vector<uint8_t> quiet(n, 0);
    int quietCount = 0;
    for (size_t i = 0; i < n; ++i) {
        if (skip_[i]) {
            quiet[i] = 1;
            ++quietCount;
            continue;
        }
        const Vec4& px = accum_[i];
        const int w = int(px.w + 0.5f);
        if (w < minS) continue;
        const float inv = 1.0f / px.w;
        if (noiseOraclePixelQuiet(px.x * inv, px.y * inv, px.z * inv, lumSq_[i], w, threshold)) {
            quiet[i] = 1;
            ++quietCount;
        }
    }
    // Karma: also look at adjacent pixels so edges / isolated noisy pixels keep sampling.
    // Pixels that already stopped stay stopped — restarting them wastes spp.
    int skipped = 0;
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const size_t i = size_t(y) * size_t(width_) + size_t(x);
            if (skip_[i]) {
                ++skipped;
                continue;
            }
            if (!quiet[i]) continue;
            auto neighborNoisy = [&](int nx, int ny) {
                if (nx < 0 || ny < 0 || nx >= width_ || ny >= height_) return false;
                return quiet[size_t(ny) * size_t(width_) + size_t(nx)] == 0;
            };
            if (neighborNoisy(x - 1, y) || neighborNoisy(x + 1, y) || neighborNoisy(x, y - 1) ||
                neighborNoisy(x, y + 1))
                continue;
            skip_[i] = 1;
            ++skipped;
        }
    }
    skipCount_ = skipped;
    noiseDone_ = skipped == int(n) && int(n) > 0;
    (void)quietCount;
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

Vec3 applyDisplayView(Vec3 linearWorking, const RenderSettingsData& settings) {
    Vec3 display = linearWorking;
    if (settings.colorManagement == kColorClassic) {
        display = classicApplyView(linearWorking, settings.workingSpace, settings.viewTransform);
    } else if (!ocioApplyView(linearWorking, settings.workingSpace, settings.viewTransform, display)) {
        // OCIO unavailable — classicApplyView already used inside ocioApplyView fallback.
    }
    return quantizeRgb(display, settings.bitDepth);
}

Vec3 applyToneMap(Vec3 color, const RenderSettingsData& settings) {
    return applyDisplayView(color, settings);
}

Image Framebuffer::resolveDisplay(const RenderSettingsData& settings) const {
    std::lock_guard<std::mutex> lock(mutex_);
    Image image(width_, height_);
    const bool bootstrap = samples_.load(std::memory_order_relaxed) == 0 &&
                           hasData_.load(std::memory_order_relaxed);
    constexpr int kBootstrapStep = 2;
    const Vec3 charcoal(0.07f, 0.07f, 0.08f);
    const double invPaths = invSplatPaths();

    // Prepare Classic or OCIO once per frame (mplay-style Display/View).
    displayPrepareView(settings.workingSpace, settings.colorManagement, settings.viewTransform);

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
                linearColor = splatPixel(x, y, invPaths);
            }
            const Vec3 display = quantizeRgb(ocioApplyViewPrepared(linearColor), settings.bitDepth);
            image.setRgb(x, y, display, 1.0f);
        }
    }
    return image;
}

}  // namespace sol
