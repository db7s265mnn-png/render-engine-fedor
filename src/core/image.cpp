#include "core/image.h"

#include <algorithm>
#include <cmath>

namespace sol {

Image::Image(int width, int height, Vec4 fill) { resize(width, height, fill); }

void Image::resize(int width, int height, Vec4 fill) {
    width_ = std::max(0, width);
    height_ = std::max(0, height);
    pixels_.assign(size_t(width_) * size_t(height_) * 4, 0.0f);
    clear(fill);
}

void Image::clear(Vec4 value) {
    for (size_t i = 0; i + 3 < pixels_.size(); i += 4) {
        pixels_[i + 0] = value.x;
        pixels_[i + 1] = value.y;
        pixels_[i + 2] = value.z;
        pixels_[i + 3] = value.w;
    }
}

Vec3 Image::texel(int x, int y) const {
    if (empty()) return Vec3(0.0f);
    x = std::clamp(x, 0, width_ - 1);
    y = std::clamp(y, 0, height_ - 1);
    return rgb(x, y);
}

Vec3 Image::sampleBilinear(float u, float v) const {
    if (empty()) return Vec3(0.0f);
    u = u - std::floor(u);
    v = clampf(v, 0.0f, 1.0f);
    const float fx = u * float(width_) - 0.5f;
    const float fy = v * float(height_) - 0.5f;
    const int x0 = int(std::floor(fx));
    const int y0 = int(std::floor(fy));
    const float tx = fx - float(x0);
    const float ty = fy - float(y0);
    auto wrapX = [&](int x) { return ((x % width_) + width_) % width_; };
    const int x1 = wrapX(x0 + 1);
    const int xa = wrapX(x0);
    const int ya = std::clamp(y0, 0, height_ - 1);
    const int yb = std::clamp(y0 + 1, 0, height_ - 1);
    const Vec3 c00 = rgb(xa, ya);
    const Vec3 c10 = rgb(x1, ya);
    const Vec3 c01 = rgb(xa, yb);
    const Vec3 c11 = rgb(x1, yb);
    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
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
