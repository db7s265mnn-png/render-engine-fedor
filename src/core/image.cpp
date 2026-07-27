#include "core/image.h"

#include <algorithm>
#include <cmath>

namespace sol {
namespace {

size_t mipPyramidFloatCount(int width, int height, int mipCount) {
    size_t total = 0;
    int w = width;
    int h = height;
    for (int level = 0; level < mipCount; ++level) {
        total += size_t(std::max(1, w)) * size_t(std::max(1, h)) * 4;
        w = std::max(1, w >> 1);
        h = std::max(1, h >> 1);
    }
    return total;
}

int computeFullMipCount(int width, int height) {
    int count = 1;
    int w = width;
    int h = height;
    while (w > 1 || h > 1) {
        w = std::max(1, w >> 1);
        h = std::max(1, h >> 1);
        ++count;
        if (count > 32) break;
    }
    return count;
}

Vec3 sampleLevelBilinear(const float* pixels, int width, int height, float u, float v) {
    if (!pixels || width <= 0 || height <= 0) return Vec3(0.0f);
    u = u - std::floor(u);
    v = clampf(v, 0.0f, 1.0f);
    const float fx = u * float(width) - 0.5f;
    const float fy = v * float(height) - 0.5f;
    const int x0 = int(std::floor(fx));
    const int y0 = int(std::floor(fy));
    const float tx = fx - float(x0);
    const float ty = fy - float(y0);
    auto wrapX = [&](int x) { return ((x % width) + width) % width; };
    auto fetch = [&](int x, int y) {
        x = wrapX(x);
        y = std::clamp(y, 0, height - 1);
        const size_t idx = (size_t(y) * size_t(width) + size_t(x)) * 4;
        return Vec3(pixels[idx + 0], pixels[idx + 1], pixels[idx + 2]);
    };
    return lerp(lerp(fetch(x0, y0), fetch(x0 + 1, y0), tx), lerp(fetch(x0, y0 + 1), fetch(x0 + 1, y0 + 1), tx), ty);
}

}  // namespace

Image::Image(int width, int height, Vec4 fill) { resize(width, height, fill); }

void Image::resize(int width, int height, Vec4 fill) {
    width_ = std::max(0, width);
    height_ = std::max(0, height);
    mipCount_ = 1;
    pixels_.assign(size_t(width_) * size_t(height_) * 4, 0.0f);
    clear(fill);
}

void Image::clear(Vec4 value) {
    // Clear only level 0 — generated / loaded mips are rebuilt by the caller.
    const size_t level0 = level0Floats();
    for (size_t i = 0; i + 3 < level0; i += 4) {
        pixels_[i + 0] = value.x;
        pixels_[i + 1] = value.y;
        pixels_[i + 2] = value.z;
        pixels_[i + 3] = value.w;
    }
}

int Image::mipWidth(int level) const {
    if (empty() || level < 0 || level >= mipCount_) return 0;
    return std::max(1, width_ >> level);
}

int Image::mipHeight(int level) const {
    if (empty() || level < 0 || level >= mipCount_) return 0;
    return std::max(1, height_ >> level);
}

const float* Image::mipData(int level) const {
    if (empty() || level < 0 || level >= mipCount_) return nullptr;
    size_t offset = 0;
    int w = width_;
    int h = height_;
    for (int i = 0; i < level; ++i) {
        offset += size_t(std::max(1, w)) * size_t(std::max(1, h)) * 4;
        w = std::max(1, w >> 1);
        h = std::max(1, h >> 1);
    }
    return pixels_.data() + offset;
}

float* Image::mipData(int level) {
    return const_cast<float*>(static_cast<const Image*>(this)->mipData(level));
}

void Image::setMipPyramid(std::vector<float> packedPixels, int width, int height, int mipCount) {
    width_ = std::max(0, width);
    height_ = std::max(0, height);
    mipCount_ = std::max(1, mipCount);
    udimGridU_ = 0;
    udimGridV_ = 0;
    pixels_ = std::move(packedPixels);
    const size_t expected = mipPyramidFloatCount(width_, height_, mipCount_);
    if (pixels_.size() < expected) pixels_.resize(expected, 0.0f);
}

void Image::generateMipChain() {
    if (empty() || isUdimAtlas()) {
        mipCount_ = 1;
        pixels_.resize(level0Floats());
        return;
    }
    const int levels = computeFullMipCount(width_, height_);
    std::vector<float> packed(mipPyramidFloatCount(width_, height_, levels), 0.0f);
    // Copy level 0.
    const size_t level0 = level0Floats();
    std::copy_n(pixels_.data(), std::min(level0, packed.size()), packed.begin());

    int srcW = width_;
    int srcH = height_;
    size_t srcOffset = 0;
    for (int level = 1; level < levels; ++level) {
        const int dstW = std::max(1, srcW >> 1);
        const int dstH = std::max(1, srcH >> 1);
        const size_t dstOffset = srcOffset + size_t(srcW) * size_t(srcH) * 4;
        const float* src = packed.data() + srcOffset;
        float* dst = packed.data() + dstOffset;
        for (int y = 0; y < dstH; ++y) {
            const int y0 = std::min(y * 2, srcH - 1);
            const int y1 = std::min(y0 + 1, srcH - 1);
            for (int x = 0; x < dstW; ++x) {
                const int x0 = std::min(x * 2, srcW - 1);
                const int x1 = std::min(x0 + 1, srcW - 1);
                auto fetch = [&](int ix, int iy) {
                    const size_t idx = (size_t(iy) * size_t(srcW) + size_t(ix)) * 4;
                    return Vec4(src[idx + 0], src[idx + 1], src[idx + 2], src[idx + 3]);
                };
                const Vec4 c = (fetch(x0, y0) + fetch(x1, y0) + fetch(x0, y1) + fetch(x1, y1)) * 0.25f;
                const size_t out = (size_t(y) * size_t(dstW) + size_t(x)) * 4;
                dst[out + 0] = c.x;
                dst[out + 1] = c.y;
                dst[out + 2] = c.z;
                dst[out + 3] = c.w;
            }
        }
        srcOffset = dstOffset;
        srcW = dstW;
        srcH = dstH;
    }
    pixels_ = std::move(packed);
    mipCount_ = levels;
}

Vec3 Image::texel(int x, int y) const {
    if (empty()) return Vec3(0.0f);
    x = std::clamp(x, 0, width_ - 1);
    y = std::clamp(y, 0, height_ - 1);
    return rgb(x, y);
}

Vec3 Image::sampleBilinear(float u, float v) const {
    if (empty()) return Vec3(0.0f);
    return sampleLevelBilinear(pixels_.data(), width_, height_, u, v);
}

Vec3 Image::sampleTrilinear(float u, float v, float lod) const {
    if (empty()) return Vec3(0.0f);
    if (mipCount_ <= 1) return sampleBilinear(u, v);
    const float maxLod = float(mipCount_ - 1);
    lod = clampf(lod, 0.0f, maxLod);
    const int level0 = int(std::floor(lod));
    const int level1 = std::min(level0 + 1, mipCount_ - 1);
    const float frac = lod - float(level0);
    const Vec3 c0 = sampleLevelBilinear(mipData(level0), mipWidth(level0), mipHeight(level0), u, v);
    if (frac < 1e-4f || level0 == level1) return c0;
    const Vec3 c1 = sampleLevelBilinear(mipData(level1), mipWidth(level1), mipHeight(level1), u, v);
    return lerp(c0, c1, frac);
}

void Image::scaleRgb(float s) {
    for (size_t i = 0; i + 3 < pixels_.size(); i += 4) {
        pixels_[i + 0] *= s;
        pixels_[i + 1] *= s;
        pixels_[i + 2] *= s;
    }
}

// ---------------------------------------------------------------------------

void Distribution2D::build(const std::vector<float>& funcValues, int width, int height) {
    width_ = width;
    height_ = height;
    func_ = funcValues;
    conditionalCdf_.assign(size_t(width + 1) * height, 0.0f);
    conditionalIntegral_.assign(height, 0.0f);
    marginalFunc_.assign(height, 0.0f);
    marginalCdf_.assign(height + 1, 0.0f);

    if (width <= 0 || height <= 0 || func_.size() != size_t(width) * size_t(height)) {
        integral_ = 0.0f;
        return;
    }

    for (int y = 0; y < height; ++y) {
        float* cdf = &conditionalCdf_[size_t(y) * (width + 1)];
        cdf[0] = 0.0f;
        for (int x = 0; x < width; ++x) {
            const float f = std::max(0.0f, func_[size_t(y) * width + x]);
            cdf[x + 1] = cdf[x] + f / float(width);
        }
        const float rowIntegral = cdf[width];
        conditionalIntegral_[y] = rowIntegral;
        if (rowIntegral > 0.0f) {
            for (int x = 1; x <= width; ++x) cdf[x] /= rowIntegral;
        } else {
            for (int x = 1; x <= width; ++x) cdf[x] = float(x) / float(width);
        }
        marginalFunc_[y] = rowIntegral;
    }

    marginalCdf_[0] = 0.0f;
    for (int y = 0; y < height; ++y) marginalCdf_[y + 1] = marginalCdf_[y] + marginalFunc_[y] / float(height);
    marginalIntegral_ = marginalCdf_[height];
    if (marginalIntegral_ > 0.0f) {
        for (int y = 1; y <= height; ++y) marginalCdf_[y] /= marginalIntegral_;
    } else {
        for (int y = 1; y <= height; ++y) marginalCdf_[y] = float(y) / float(height);
    }
    integral_ = marginalIntegral_;
}

namespace {
int findInterval(const float* cdf, int size, float u) {
    int first = 0;
    int len = size;
    while (len > 0) {
        const int half = len >> 1;
        const int middle = first + half;
        if (cdf[middle] <= u) {
            first = middle + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    return std::clamp(first - 1, 0, size - 2);
}
}  // namespace

Vec2 Distribution2D::sample(float u1, float u2, float& pdfOut) const {
    pdfOut = 0.0f;
    if (!valid()) return Vec2(u1, u2);

    const int y = findInterval(marginalCdf_.data(), height_ + 1, u2);
    const float dyDen = marginalCdf_[y + 1] - marginalCdf_[y];
    const float dy = dyDen > 0.0f ? (u2 - marginalCdf_[y]) / dyDen : 0.0f;
    const float marginalPdf = marginalIntegral_ > 0.0f ? marginalFunc_[y] / marginalIntegral_ : 0.0f;

    const float* cdf = &conditionalCdf_[size_t(y) * (width_ + 1)];
    const int x = findInterval(cdf, width_ + 1, u1);
    const float dxDen = cdf[x + 1] - cdf[x];
    const float dx = dxDen > 0.0f ? (u1 - cdf[x]) / dxDen : 0.0f;
    const float rowIntegral = conditionalIntegral_[y];
    const float conditionalPdf = rowIntegral > 0.0f ? func_[size_t(y) * width_ + x] / rowIntegral : 0.0f;

    pdfOut = marginalPdf * conditionalPdf;
    return Vec2((float(x) + dx) / float(width_), (float(y) + dy) / float(height_));
}

float Distribution2D::pdf(float u, float v) const {
    if (!valid()) return 0.0f;
    const int x = std::clamp(int(u * float(width_)), 0, width_ - 1);
    const int y = std::clamp(int(v * float(height_)), 0, height_ - 1);
    return func_[size_t(y) * width_ + x] / integral_;
}

}  // namespace sol
