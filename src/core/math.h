// Vector / matrix / sampling math shared by the CPU (Embree) and GPU (OptiX) backends.
// Everything here must compile both with a host C++ compiler and with nvcc.
#pragma once

#if defined(__CUDACC__)
#  define SR_HD __host__ __device__
// __forceinline__ on the whole integrator makes cicc optimize one megakernel
// for hours (seen: 3h+ on a 14900K). Keep it optional.
#  if defined(SOLSTICE_CUDA_FORCEINLINE)
#    define SR_INL __forceinline__
#  else
#    define SR_INL inline
#  endif
#else
#  define SR_HD
#  define SR_INL inline
#endif

#include <cmath>
#include <cstdint>

namespace sol {

SR_INL SR_HD float srMin(float a, float b) { return a < b ? a : b; }
SR_INL SR_HD float srMax(float a, float b) { return a > b ? a : b; }
SR_INL SR_HD float clampf(float v, float lo, float hi) { return srMin(srMax(v, lo), hi); }
SR_INL SR_HD float lerpf(float a, float b, float t) { return a + (b - a) * t; }
SR_INL SR_HD float saturatef(float v) { return clampf(v, 0.0f, 1.0f); }
SR_INL SR_HD float sqr(float v) { return v * v; }

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kInvPi = 0.31830988618379067154f;
constexpr float kInv2Pi = 0.15915494309189533577f;
constexpr float kInv4Pi = 0.07957747154594766788f;
constexpr float kRayEpsilon = 1e-4f;
constexpr float kFloatMax = 3.402823466e+38f;

SR_INL SR_HD float radians(float deg) { return deg * (kPi / 180.0f); }
SR_INL SR_HD float degrees(float rad) { return rad * (180.0f / kPi); }

// ---------------------------------------------------------------------------
// Vec2
// ---------------------------------------------------------------------------
struct Vec2 {
    float x = 0.0f, y = 0.0f;
    Vec2() = default;
    SR_HD Vec2(float x_, float y_) : x(x_), y(y_) {}
    explicit SR_HD Vec2(float s) : x(s), y(s) {}
};

SR_INL SR_HD Vec2 operator+(Vec2 a, Vec2 b) { return Vec2(a.x + b.x, a.y + b.y); }
SR_INL SR_HD Vec2 operator-(Vec2 a, Vec2 b) { return Vec2(a.x - b.x, a.y - b.y); }
SR_INL SR_HD Vec2 operator*(Vec2 a, float s) { return Vec2(a.x * s, a.y * s); }
SR_INL SR_HD Vec2 operator*(float s, Vec2 a) { return a * s; }

// ---------------------------------------------------------------------------
// Vec3 (also used as an RGB colour)
// ---------------------------------------------------------------------------
struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    Vec3() = default;
    SR_HD Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    explicit SR_HD Vec3(float s) : x(s), y(s), z(s) {}
    SR_HD float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    SR_HD float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
};

SR_INL SR_HD Vec3 operator+(Vec3 a, Vec3 b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
SR_INL SR_HD Vec3 operator-(Vec3 a, Vec3 b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
SR_INL SR_HD Vec3 operator-(Vec3 a) { return Vec3(-a.x, -a.y, -a.z); }
SR_INL SR_HD Vec3 operator*(Vec3 a, Vec3 b) { return Vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
SR_INL SR_HD Vec3 operator*(Vec3 a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
SR_INL SR_HD Vec3 operator*(float s, Vec3 a) { return a * s; }
SR_INL SR_HD Vec3 operator/(Vec3 a, Vec3 b) { return Vec3(a.x / b.x, a.y / b.y, a.z / b.z); }
SR_INL SR_HD Vec3 operator/(Vec3 a, float s) { return a * (1.0f / s); }
SR_INL SR_HD Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }
SR_INL SR_HD Vec3& operator-=(Vec3& a, Vec3 b) { a = a - b; return a; }
SR_INL SR_HD Vec3& operator*=(Vec3& a, Vec3 b) { a = a * b; return a; }
SR_INL SR_HD Vec3& operator*=(Vec3& a, float s) { a = a * s; return a; }
SR_INL SR_HD Vec3& operator/=(Vec3& a, float s) { a = a / s; return a; }
SR_INL SR_HD bool operator==(Vec3 a, Vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
SR_INL SR_HD bool operator!=(Vec3 a, Vec3 b) { return !(a == b); }

SR_INL SR_HD float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
SR_INL SR_HD float absDot(Vec3 a, Vec3 b) { return fabsf(dot(a, b)); }
SR_INL SR_HD Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
SR_INL SR_HD float lengthSquared(Vec3 a) { return dot(a, a); }
SR_INL SR_HD float length(Vec3 a) { return sqrtf(dot(a, a)); }
SR_INL SR_HD Vec3 normalize(Vec3 a) {
    const float l = length(a);
    return l > 0.0f ? a * (1.0f / l) : Vec3(0.0f, 0.0f, 0.0f);
}
SR_INL SR_HD Vec3 vmin(Vec3 a, Vec3 b) { return Vec3(srMin(a.x, b.x), srMin(a.y, b.y), srMin(a.z, b.z)); }
SR_INL SR_HD Vec3 vmax(Vec3 a, Vec3 b) { return Vec3(srMax(a.x, b.x), srMax(a.y, b.y), srMax(a.z, b.z)); }
SR_INL SR_HD Vec3 vabs(Vec3 a) { return Vec3(fabsf(a.x), fabsf(a.y), fabsf(a.z)); }
SR_INL SR_HD Vec3 lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }
SR_INL SR_HD float maxComponent(Vec3 a) { return srMax(a.x, srMax(a.y, a.z)); }
SR_INL SR_HD float minComponent(Vec3 a) { return srMin(a.x, srMin(a.y, a.z)); }
SR_INL SR_HD float average(Vec3 a) { return (a.x + a.y + a.z) * (1.0f / 3.0f); }
SR_INL SR_HD float luminance(Vec3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }
SR_INL SR_HD bool isBlack(Vec3 c) { return c.x <= 0.0f && c.y <= 0.0f && c.z <= 0.0f; }
SR_INL SR_HD bool srIsFinite(float v) {
#if defined(__CUDACC__)
    return ::isfinite(v);
#else
    return std::isfinite(v);
#endif
}
SR_INL SR_HD bool isFinite(Vec3 c) { return srIsFinite(c.x) && srIsFinite(c.y) && srIsFinite(c.z); }
SR_INL SR_HD Vec3 reflect(Vec3 w, Vec3 n) { return n * (2.0f * dot(w, n)) - w; }
SR_INL SR_HD Vec3 faceforward(Vec3 n, Vec3 v) { return dot(n, v) < 0.0f ? -n : n; }

// ---------------------------------------------------------------------------
// Vec4
// ---------------------------------------------------------------------------
struct Vec4 {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;
    Vec4() = default;
    SR_HD Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    SR_HD Vec4(Vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
    SR_HD Vec3 xyz() const { return Vec3(x, y, z); }
};

SR_INL SR_HD Vec4 operator+(Vec4 a, Vec4 b) { return Vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
SR_INL SR_HD Vec4 operator*(Vec4 a, float s) { return Vec4(a.x * s, a.y * s, a.z * s, a.w * s); }
SR_INL SR_HD Vec4& operator+=(Vec4& a, Vec4 b) { a = a + b; return a; }

// ---------------------------------------------------------------------------
// Mat4 - row major, column vectors: p' = M * p
// ---------------------------------------------------------------------------
struct Mat4 {
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    Mat4() = default;
    SR_HD float& at(int r, int c) { return m[r * 4 + c]; }
    SR_HD float at(int r, int c) const { return m[r * 4 + c]; }

    static SR_HD Mat4 identity() { return Mat4(); }

    static SR_HD Mat4 translate(Vec3 t) {
        Mat4 r;
        r.at(0, 3) = t.x; r.at(1, 3) = t.y; r.at(2, 3) = t.z;
        return r;
    }
    static SR_HD Mat4 scale(Vec3 s) {
        Mat4 r;
        r.at(0, 0) = s.x; r.at(1, 1) = s.y; r.at(2, 2) = s.z;
        return r;
    }
    static SR_HD Mat4 rotateX(float deg) {
        const float a = radians(deg), c = cosf(a), s = sinf(a);
        Mat4 r;
        r.at(1, 1) = c; r.at(1, 2) = -s; r.at(2, 1) = s; r.at(2, 2) = c;
        return r;
    }
    static SR_HD Mat4 rotateY(float deg) {
        const float a = radians(deg), c = cosf(a), s = sinf(a);
        Mat4 r;
        r.at(0, 0) = c; r.at(0, 2) = s; r.at(2, 0) = -s; r.at(2, 2) = c;
        return r;
    }
    static SR_HD Mat4 rotateZ(float deg) {
        const float a = radians(deg), c = cosf(a), s = sinf(a);
        Mat4 r;
        r.at(0, 0) = c; r.at(0, 1) = -s; r.at(1, 0) = s; r.at(1, 1) = c;
        return r;
    }
};

SR_INL SR_HD Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a.at(i, k) * b.at(k, j);
            r.at(i, j) = s;
        }
    }
    return r;
}

SR_INL SR_HD Vec3 transformPoint(const Mat4& t, Vec3 p) {
    const float x = t.at(0, 0) * p.x + t.at(0, 1) * p.y + t.at(0, 2) * p.z + t.at(0, 3);
    const float y = t.at(1, 0) * p.x + t.at(1, 1) * p.y + t.at(1, 2) * p.z + t.at(1, 3);
    const float z = t.at(2, 0) * p.x + t.at(2, 1) * p.y + t.at(2, 2) * p.z + t.at(2, 3);
    const float w = t.at(3, 0) * p.x + t.at(3, 1) * p.y + t.at(3, 2) * p.z + t.at(3, 3);
    if (w != 1.0f && w != 0.0f) return Vec3(x / w, y / w, z / w);
    return Vec3(x, y, z);
}

SR_INL SR_HD Vec3 transformVector(const Mat4& t, Vec3 v) {
    return Vec3(t.at(0, 0) * v.x + t.at(0, 1) * v.y + t.at(0, 2) * v.z,
                t.at(1, 0) * v.x + t.at(1, 1) * v.y + t.at(1, 2) * v.z,
                t.at(2, 0) * v.x + t.at(2, 1) * v.y + t.at(2, 2) * v.z);
}

// Normals need the inverse transpose; pass the already inverted matrix.
SR_INL SR_HD Vec3 transformNormalWithInverse(const Mat4& inv, Vec3 n) {
    return Vec3(inv.at(0, 0) * n.x + inv.at(1, 0) * n.y + inv.at(2, 0) * n.z,
                inv.at(0, 1) * n.x + inv.at(1, 1) * n.y + inv.at(2, 1) * n.z,
                inv.at(0, 2) * n.x + inv.at(1, 2) * n.y + inv.at(2, 2) * n.z);
}

SR_INL SR_HD Mat4 transpose(const Mat4& a) {
    Mat4 r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) r.at(i, j) = a.at(j, i);
    return r;
}

SR_INL SR_HD Mat4 inverse(const Mat4& src) {
    // Gauss-Jordan elimination with partial pivoting.
    float a[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) a[i][j] = src.at(i, j);
        for (int j = 0; j < 4; ++j) a[i][4 + j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 4; ++row)
            if (fabsf(a[row][col]) > fabsf(a[pivot][col])) pivot = row;
        if (fabsf(a[pivot][col]) < 1e-20f) return Mat4();  // singular
        if (pivot != col)
            for (int j = 0; j < 8; ++j) {
                const float tmp = a[col][j];
                a[col][j] = a[pivot][j];
                a[pivot][j] = tmp;
            }
        const float inv = 1.0f / a[col][col];
        for (int j = 0; j < 8; ++j) a[col][j] *= inv;
        for (int row = 0; row < 4; ++row) {
            if (row == col) continue;
            const float f = a[row][col];
            if (f == 0.0f) continue;
            for (int j = 0; j < 8; ++j) a[row][j] -= f * a[col][j];
        }
    }
    Mat4 r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) r.at(i, j) = a[i][4 + j];
    return r;
}

// Houdini-style transform order: scale, then rotate (XYZ), then translate.
SR_INL SR_HD Mat4 composeTRS(Vec3 translate, Vec3 rotateDeg, Vec3 scale) {
    return Mat4::translate(translate) * Mat4::rotateZ(rotateDeg.z) * Mat4::rotateY(rotateDeg.y) *
           Mat4::rotateX(rotateDeg.x) * Mat4::scale(scale);
}

SR_INL SR_HD Mat4 lookAtMatrix(Vec3 eye, Vec3 target, Vec3 up) {
    // Camera-to-world matrix with -Z forward (OpenGL/USD convention).
    Vec3 forward = normalize(target - eye);
    if (lengthSquared(forward) < 1e-12f) forward = Vec3(0.0f, 0.0f, -1.0f);
    Vec3 right = cross(forward, up);
    if (lengthSquared(right) < 1e-12f) right = cross(forward, Vec3(0.0f, 0.0f, 1.0f));
    right = normalize(right);
    const Vec3 trueUp = cross(right, forward);
    Mat4 r;
    r.at(0, 0) = right.x;   r.at(0, 1) = trueUp.x;  r.at(0, 2) = -forward.x;  r.at(0, 3) = eye.x;
    r.at(1, 0) = right.y;   r.at(1, 1) = trueUp.y;  r.at(1, 2) = -forward.y;  r.at(1, 3) = eye.y;
    r.at(2, 0) = right.z;   r.at(2, 1) = trueUp.z;  r.at(2, 2) = -forward.z;  r.at(2, 3) = eye.z;
    return r;
}

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------
struct Bounds3 {
    Vec3 lo{kFloatMax, kFloatMax, kFloatMax};
    Vec3 hi{-kFloatMax, -kFloatMax, -kFloatMax};

    SR_HD void extend(Vec3 p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    SR_HD void extend(const Bounds3& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
    SR_HD bool valid() const { return hi.x >= lo.x && hi.y >= lo.y && hi.z >= lo.z; }
    SR_HD Vec3 center() const { return (lo + hi) * 0.5f; }
    SR_HD Vec3 extent() const { return valid() ? hi - lo : Vec3(0.0f); }
    SR_HD float radius() const { return valid() ? length(hi - lo) * 0.5f : 0.0f; }
};

SR_INL SR_HD Bounds3 transformBounds(const Mat4& xf, const Bounds3& b) {
    if (!b.valid()) return b;
    Bounds3 r;
    for (int i = 0; i < 8; ++i) {
        const Vec3 corner((i & 1) ? b.hi.x : b.lo.x, (i & 2) ? b.hi.y : b.lo.y, (i & 4) ? b.hi.z : b.lo.z);
        r.extend(transformPoint(xf, corner));
    }
    return r;
}

// ---------------------------------------------------------------------------
// Orthonormal basis and direction helpers
// ---------------------------------------------------------------------------
// Duff et al. "Building an Orthonormal Basis, Revisited".
SR_INL SR_HD void branchlessOnb(Vec3 n, Vec3& t, Vec3& b) {
    const float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (sign + n.z);
    const float bb = n.x * n.y * a;
    t = Vec3(1.0f + sign * n.x * n.x * a, sign * bb, -sign * n.x);
    b = Vec3(bb, sign + n.y * n.y * a, -n.y);
}

struct Frame {
    Vec3 t, b, n;
    Frame() = default;
    explicit SR_HD Frame(Vec3 normal) : n(normal) { branchlessOnb(n, t, b); }
    SR_HD Vec3 toLocal(Vec3 v) const { return Vec3(dot(v, t), dot(v, b), dot(v, n)); }
    SR_HD Vec3 toWorld(Vec3 v) const { return t * v.x + b * v.y + n * v.z; }
};

SR_INL SR_HD Vec3 sphericalDirection(float sinTheta, float cosTheta, float phi) {
    return Vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
}

// Equirectangular mapping used for dome lights: +Y up, -Z at u = 0.5.
SR_INL SR_HD Vec2 directionToEquirect(Vec3 d) {
    const float u = atan2f(d.x, -d.z) * kInv2Pi + 0.5f;
    const float v = acosf(clampf(d.y, -1.0f, 1.0f)) * kInvPi;
    return Vec2(u, v);
}

SR_INL SR_HD Vec3 equirectToDirection(float u, float v) {
    const float phi = (u - 0.5f) * kTwoPi;
    const float theta = v * kPi;
    const float sinTheta = sinf(theta);
    return Vec3(sinTheta * sinf(phi), cosf(theta), -sinTheta * cosf(phi));
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------
SR_INL SR_HD Vec2 sampleConcentricDisk(float u1, float u2) {
    const float ox = 2.0f * u1 - 1.0f;
    const float oy = 2.0f * u2 - 1.0f;
    if (ox == 0.0f && oy == 0.0f) return Vec2(0.0f, 0.0f);
    float r, theta;
    if (fabsf(ox) > fabsf(oy)) {
        r = ox;
        theta = (kPi * 0.25f) * (oy / ox);
    } else {
        r = oy;
        theta = (kPi * 0.5f) - (kPi * 0.25f) * (ox / oy);
    }
    return Vec2(r * cosf(theta), r * sinf(theta));
}

SR_INL SR_HD Vec3 sampleCosineHemisphere(float u1, float u2) {
    const Vec2 d = sampleConcentricDisk(u1, u2);
    const float z = sqrtf(srMax(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return Vec3(d.x, d.y, z);
}

SR_INL SR_HD Vec3 sampleUniformSphere(float u1, float u2) {
    const float z = 1.0f - 2.0f * u1;
    const float r = sqrtf(srMax(0.0f, 1.0f - z * z));
    const float phi = kTwoPi * u2;
    return Vec3(r * cosf(phi), r * sinf(phi), z);
}

SR_INL SR_HD Vec3 sampleUniformCone(float u1, float u2, float cosThetaMax) {
    const float cosTheta = (1.0f - u1) + u1 * cosThetaMax;
    const float sinTheta = sqrtf(srMax(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = kTwoPi * u2;
    return sphericalDirection(sinTheta, cosTheta, phi);
}

SR_INL SR_HD Vec2 sampleUniformTriangle(float u1, float u2) {
    const float su = sqrtf(u1);
    return Vec2(1.0f - su, u2 * su);
}

SR_INL SR_HD float powerHeuristic(float nf, float fPdf, float ng, float gPdf) {
    const float f = nf * fPdf;
    const float g = ng * gPdf;
    const float d = f * f + g * g;
    return d > 0.0f ? (f * f) / d : 0.0f;
}

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------
SR_INL SR_HD float linearToSrgb(float c) {
    c = saturatef(c);
    return c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

SR_INL SR_HD float srgbToLinear(float c) {
    c = saturatef(c);
    return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}

// Same EOTF without the 0–1 clamp so HDR sRGB-encoded values can convert to ACEScg.
SR_INL SR_HD float srgbToLinearUnclamped(float c) {
    if (c <= 0.04045f) return c / 12.92f;
    return powf((c + 0.055f) / 1.055f, 2.4f);
}

// Narkowicz's ACES filmic approximation (good stand-in for ACES Output → sRGB).
SR_INL SR_HD Vec3 acesFilmic(Vec3 c) {
    const Vec3 a = c * 2.51f + Vec3(0.03f);
    const Vec3 b = c * 2.43f + Vec3(0.59f);
    const Vec3 num = c * a;
    const Vec3 den = c * b + Vec3(0.14f);
    return Vec3(saturatef(num.x / den.x), saturatef(num.y / den.y), saturatef(num.z / den.z));
}

SR_INL SR_HD Vec3 reinhard(Vec3 c) {
    return Vec3(c.x / (1.0f + c.x), c.y / (1.0f + c.y), c.z / (1.0f + c.z));
}

SR_INL SR_HD Vec3 linearToSrgbVec(Vec3 c) {
    return Vec3(linearToSrgb(c.x), linearToSrgb(c.y), linearToSrgb(c.z));
}

// ACEScg (AP1) → linear Rec.709 / sRGB primaries (OpenColorIO / ACES utility).
SR_INL SR_HD Vec3 acescgToLinearSrgb(Vec3 c) {
    return Vec3(1.7050509927f * c.x + -0.6217921207f * c.y + -0.0832588720f * c.z,
                -0.1302564175f * c.x + 1.1408047365f * c.y + -0.0105483191f * c.z,
                -0.0240033472f * c.x + -0.1289689761f * c.y + 1.1529723230f * c.z);
}

// Linear Rec.709 / sRGB primaries → ACEScg (AP1). Inverse of acescgToLinearSrgb.
SR_INL SR_HD Vec3 linearSrgbToAcescg(Vec3 c) {
    return Vec3(0.6130974024f * c.x + 0.3395231462f * c.y + 0.0473794515f * c.z,
                0.0701937225f * c.x + 0.9163538791f * c.y + 0.0134523985f * c.z,
                0.0206155929f * c.x + 0.1095697729f * c.y + 0.8698146342f * c.z);
}

// Quantize a display-referred channel to the configured output bit depth.
SR_INL float quantizeChannel(float c, int bitDepth) {
    c = saturatef(c);
    if (bitDepth <= 8) {
        const float q = 255.0f;
        return std::floor(c * q + 0.5f) / q;
    }
    if (bitDepth <= 16) {
        const float q = 65535.0f;
        return std::floor(c * q + 0.5f) / q;
    }
    return c;  // 32-bit: leave float
}

SR_INL Vec3 quantizeRgb(Vec3 c, int bitDepth) {
    return Vec3(quantizeChannel(c.x, bitDepth), quantizeChannel(c.y, bitDepth),
                quantizeChannel(c.z, bitDepth));
}

}  // namespace sol
