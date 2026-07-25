// Floating point RGBA image buffer plus a 2D distribution used for
// importance sampling environment maps.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/math.h"

namespace sol {

class Image {
public:
    Image() = default;
    Image(int width, int height, Vec4 fill = Vec4(0.0f, 0.0f, 0.0f, 1.0f));

    void resize(int width, int height, Vec4 fill = Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    void clear(Vec4 value = Vec4(0.0f, 0.0f, 0.0f, 1.0f));

    int width() const { return width_; }
    int height() const { return height_; }
    bool empty() const { return width_ <= 0 || height_ <= 0; }

    float* data() { return pixels_.data(); }
    const float* data() const { return pixels_.data(); }
    size_t sizeInFloats() const { return pixels_.size(); }

    Vec4& at(int x, int y) { return *reinterpret_cast<Vec4*>(&pixels_[(size_t(y) * width_ + x) * 4]); }
    const Vec4& at(int x, int y) const {
        return *reinterpret_cast<const Vec4*>(&pixels_[(size_t(y) * width_ + x) * 4]);
    }

    Vec3 rgb(int x, int y) const { return at(x, y).xyz(); }
    void setRgb(int x, int y, Vec3 c, float alpha = 1.0f) { at(x, y) = Vec4(c, alpha); }

    // Bilinear lookup with wrapping in u and clamping in v (equirect friendly).
    Vec3 sampleBilinear(float u, float v) const;
    // Nearest texel fetch with clamping.
    Vec3 texel(int x, int y) const;

    void scaleRgb(float s);

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<float> pixels_;
};

// Piecewise-constant 2D distribution (Pharr et al.) for env map sampling.
class Distribution2D {
public:
    Distribution2D() = default;
    // funcValues is a width*height array of non-negative values (row major).
    void build(const std::vector<float>& funcValues, int width, int height);

    bool valid() const { return width_ > 0 && height_ > 0 && integral_ > 0.0f; }
    // Returns (u,v) in [0,1)^2 and the pdf with respect to the unit square.
    Vec2 sample(float u1, float u2, float& pdf) const;
    // pdf with respect to the unit square for a given (u,v).
    float pdf(float u, float v) const;
    float integral() const { return integral_; }

    int width() const { return width_; }
    int height() const { return height_; }
    // Raw tables, uploaded verbatim to the GPU by the OptiX backend.
    const float* funcData() const { return func_.data(); }
    const float* conditionalCdfData() const { return conditionalCdf_.data(); }
    const float* conditionalIntegralData() const { return conditionalIntegral_.data(); }
    const float* marginalCdfData() const { return marginalCdf_.data(); }
    const float* marginalFuncData() const { return marginalFunc_.data(); }

private:
    int width_ = 0;
    int height_ = 0;
    float integral_ = 0.0f;
    std::vector<float> func_;          // width*height
    std::vector<float> conditionalCdf_;  // (width+1)*height
    std::vector<float> marginalCdf_;     // height+1
    std::vector<float> marginalFunc_;    // height
    float marginalIntegral_ = 0.0f;
    std::vector<float> conditionalIntegral_;  // height
};

}  // namespace sol
