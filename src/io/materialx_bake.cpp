#include "io/materialx_bake.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "core/log.h"
#include "io/image_io.h"

#if SOLSTICE_HAVE_MATERIALX
#  include <MaterialXCore/Interface.h>
#  include <MaterialXCore/Value.h>
#endif

namespace sol {
#if SOLSTICE_HAVE_MATERIALX
namespace mx = MaterialX;
namespace {

// ---------------------------------------------------------------------------
// Compact Perlin (MaterialX / OSL style: roughly -1..1)
// ---------------------------------------------------------------------------
uint32_t hashU32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float hashToFloat(uint32_t h) { return float(h >> 8) / float(0x00ffffffu); }

float fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

float grad2(uint32_t h, float x, float y) {
    const uint32_t g = h & 7u;
    const float u = (g < 4u) ? x : y;
    const float v = 2.0f * ((g < 4u) ? y : x);
    return ((g & 1u) ? -u : u) + ((g & 2u) ? -v : v);
}

float grad3(uint32_t h, float x, float y, float z) {
    const uint32_t g = h & 15u;
    const float u = (g < 8u) ? x : y;
    const float v = (g < 4u) ? y : ((g == 12u || g == 14u) ? x : z);
    return ((g & 1u) ? -u : u) + ((g & 2u) ? -v : v);
}

float perlin2(float x, float y) {
    // Non-finite / extreme coords → int(floor) is UB (INT_MIN) and crashes MSVC builds.
    if (!std::isfinite(x) || !std::isfinite(y)) return 0.0f;
    x = std::clamp(x, -1.0e5f, 1.0e5f);
    y = std::clamp(y, -1.0e5f, 1.0e5f);
    const int x0 = int(std::floor(x));
    const int y0 = int(std::floor(y));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    const float u = fade(fx);
    const float v = fade(fy);
    const uint32_t h00 = hashU32(uint32_t(x0) * 374761393u + uint32_t(y0) * 668265263u);
    const uint32_t h10 = hashU32(uint32_t(x0 + 1) * 374761393u + uint32_t(y0) * 668265263u);
    const uint32_t h01 = hashU32(uint32_t(x0) * 374761393u + uint32_t(y0 + 1) * 668265263u);
    const uint32_t h11 = hashU32(uint32_t(x0 + 1) * 374761393u + uint32_t(y0 + 1) * 668265263u);
    const float n00 = grad2(h00, fx, fy);
    const float n10 = grad2(h10, fx - 1.0f, fy);
    const float n01 = grad2(h01, fx, fy - 1.0f);
    const float n11 = grad2(h11, fx - 1.0f, fy - 1.0f);
    const float nx0 = n00 * (1.0f - u) + n10 * u;
    const float nx1 = n01 * (1.0f - u) + n11 * u;
    return 0.7071f * (nx0 * (1.0f - v) + nx1 * v);
}

float perlin3(float x, float y, float z) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return 0.0f;
    x = std::clamp(x, -1.0e5f, 1.0e5f);
    y = std::clamp(y, -1.0e5f, 1.0e5f);
    z = std::clamp(z, -1.0e5f, 1.0e5f);
    const int x0 = int(std::floor(x));
    const int y0 = int(std::floor(y));
    const int z0 = int(std::floor(z));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    const float fz = z - float(z0);
    const float u = fade(fx);
    const float v = fade(fy);
    const float w = fade(fz);
    auto H = [](int ix, int iy, int iz) {
        return hashU32(uint32_t(ix) * 374761393u + uint32_t(iy) * 668265263u + uint32_t(iz) * 2147483647u);
    };
    float n = 0.0f;
    for (int dz = 0; dz <= 1; ++dz) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dx = 0; dx <= 1; ++dx) {
                const float g = grad3(H(x0 + dx, y0 + dy, z0 + dz), fx - float(dx), fy - float(dy), fz - float(dz));
                const float wx = dx ? u : (1.0f - u);
                const float wy = dy ? v : (1.0f - v);
                const float wz = dz ? w : (1.0f - w);
                n += g * wx * wy * wz;
            }
        }
    }
    return 0.866f * n;
}

Vec3 perlin2Vec3(float x, float y) {
    return Vec3(perlin2(x, y), perlin2(x + 17.1f, y - 9.3f), perlin2(x - 5.7f, y + 23.4f));
}

Vec3 perlin3Vec3(float x, float y, float z) {
    return Vec3(perlin3(x, y, z), perlin3(x + 17.1f, y - 9.3f, z + 3.2f),
                perlin3(x - 5.7f, y + 23.4f, z - 11.8f));
}

float cellNoise2(float x, float y) {
    if (!std::isfinite(x) || !std::isfinite(y)) return 0.0f;
    x = std::clamp(x, -1.0e5f, 1.0e5f);
    y = std::clamp(y, -1.0e5f, 1.0e5f);
    const int xi = int(std::floor(x));
    const int yi = int(std::floor(y));
    return hashToFloat(hashU32(uint32_t(xi) * 374761393u + uint32_t(yi) * 668265263u)) * 2.0f - 1.0f;
}

float cellNoise3(float x, float y, float z) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return 0.0f;
    x = std::clamp(x, -1.0e5f, 1.0e5f);
    y = std::clamp(y, -1.0e5f, 1.0e5f);
    z = std::clamp(z, -1.0e5f, 1.0e5f);
    const int xi = int(std::floor(x));
    const int yi = int(std::floor(y));
    const int zi = int(std::floor(z));
    return hashToFloat(hashU32(uint32_t(xi) * 374761393u + uint32_t(yi) * 668265263u +
                               uint32_t(zi) * 2147483647u)) *
               2.0f -
           1.0f;
}

float fractal3(float x, float y, float z, int octaves, float lacunarity, float diminish) {
    float amp = 1.0f;
    float sum = 0.0f;
    float maxAmp = 0.0f;
    float px = x, py = y, pz = z;
    octaves = std::clamp(octaves, 1, 16);
    for (int i = 0; i < octaves; ++i) {
        sum += amp * perlin3(px, py, pz);
        maxAmp += amp;
        amp *= diminish;
        px *= lacunarity;
        py *= lacunarity;
        pz *= lacunarity;
    }
    return maxAmp > 0.0f ? sum / maxAmp : 0.0f;
}

Vec3 fractal3Vec3(float x, float y, float z, int octaves, float lacunarity, float diminish) {
    return Vec3(fractal3(x, y, z, octaves, lacunarity, diminish),
                fractal3(x + 19.1f, y - 7.4f, z + 2.3f, octaves, lacunarity, diminish),
                fractal3(x - 4.2f, y + 13.8f, z - 9.6f, octaves, lacunarity, diminish));
}

// ---------------------------------------------------------------------------
// Value helpers
// ---------------------------------------------------------------------------
bool parseFloats(const std::string& value, float* out, int count) {
    if (!out || count <= 0) return false;
    float a = 0, b = 0, c = 0, d = 0;
    int n = 0;
    if (count >= 4)
        n = std::sscanf(value.c_str(), "%f,%f,%f,%f", &a, &b, &c, &d);
    if (n < count) n = std::sscanf(value.c_str(), "%f %f %f %f", &a, &b, &c, &d);
    if (n < 1) n = std::sscanf(value.c_str(), "%f", &a);
    if (n < 1) return false;
    out[0] = a;
    if (count > 1) out[1] = n > 1 ? b : a;
    if (count > 2) out[2] = n > 2 ? c : a;
    if (count > 3) out[3] = n > 3 ? d : 1.0f;
    return true;
}

std::string inputValueString(const mx::NodePtr& node, const std::string& inputName) {
    if (!node) return {};
    mx::InputPtr input = node->getInput(inputName);
    if (!input) return {};
    if (input->hasValueString()) return input->getValueString();
    mx::ValuePtr value = input->getValue();
    return value ? value->getValueString() : std::string();
}

mx::NodePtr connected(const mx::NodePtr& node, const std::string& inputName) {
    if (!node) return nullptr;
    mx::InputPtr input = node->getInput(inputName);
    if (!input) return nullptr;
    // Only follow explicit wires. defaultgeomprop (Pobject etc.) must not invent
    // phantom connections — that path has been a source of cook crashes.
    if (!input->hasNodeName() || input->getNodeName().empty()) return nullptr;
    return input->getConnectedNode();
}

Vec4 asColor(const Vec4& v, const std::string& type) {
    if (type == "float" || type == "integer" || type == "int" || type == "boolean")
        return Vec4(v.x, v.x, v.x, 1.0f);
    if (type == "vector2") return Vec4(v.x, v.y, 0.0f, 1.0f);
    if (type == "color4" || type == "vector4") return v;
    return Vec4(v.x, v.y, v.z, 1.0f);
}

Vec4 splat4(float x) { return Vec4(x, x, x, 1.0f); }

Vec4 sampleImageUV(const Image& image, float u, float v) {
    if (image.empty()) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    // Wrap U, clamp V — same convention as shading.h for regular maps.
    u = u - std::floor(u);
    v = std::clamp(v, 0.0f, 1.0f);
    const Vec3 c = image.sampleBilinear(u, v);
    return Vec4(c.x, c.y, c.z, 1.0f);
}

struct BakeState {
    QString searchDirectory;
    std::vector<int> udimSet;
    std::map<const mx::Node*, std::shared_ptr<Image>> imageCache;
    int depth = 0;
};

std::shared_ptr<Image> loadImageCached(BakeState& state, const mx::NodePtr& imageNode) {
    if (!imageNode) return nullptr;
    auto it = state.imageCache.find(imageNode.get());
    if (it != state.imageCache.end()) return it->second;
    const std::string cat = imageNode->getCategory();
    if (cat != "image" && cat != "tiledimage") return nullptr;
    std::string file = inputValueString(imageNode, "file");
    if (file.empty()) return nullptr;
    std::string error;
    std::shared_ptr<Image> image;
    QString pattern;
    std::vector<int> discovered;
    const QString fileQ = QString::fromStdString(file);
    if (resolveUdimPattern(fileQ, state.searchDirectory, pattern, discovered)) {
        std::vector<int> tiles = state.udimSet.empty() ? discovered : state.udimSet;
        image = loadImageOrUdim(pattern, state.searchDirectory, error, tiles);
    } else {
        image = loadImageOrUdim(fileQ, state.searchDirectory, error, state.udimSet);
    }
    if (!image && !error.empty()) logWarning("MaterialX bake: " + error);
    state.imageCache[imageNode.get()] = image;
    return image;
}

Vec4 evalNode(const mx::NodePtr& node, float u, float v, BakeState& state);

Vec4 evalInput(const mx::NodePtr& node, const std::string& inputName, float u, float v, BakeState& state,
               Vec4 fallback) {
    if (!node) return fallback;
    if (mx::NodePtr child = connected(node, inputName)) return evalNode(child, u, v, state);
    float vals[4] = {fallback.x, fallback.y, fallback.z, fallback.w};
    const std::string raw = inputValueString(node, inputName);
    if (!raw.empty()) parseFloats(raw, vals, 4);
    return Vec4(vals[0], vals[1], vals[2], vals[3]);
}

float evalFloat(const mx::NodePtr& node, const std::string& inputName, float u, float v, BakeState& state,
                float fallback) {
    return evalInput(node, inputName, u, v, state, splat4(fallback)).x;
}

Vec3 evalVec3(const mx::NodePtr& node, const std::string& inputName, float u, float v, BakeState& state,
              Vec3 fallback) {
    const Vec4 c = evalInput(node, inputName, u, v, state, Vec4(fallback.x, fallback.y, fallback.z, 1.0f));
    return Vec3(c.x, c.y, c.z);
}

bool isBakableCategory(const std::string& cat) {
    return cat == "noise2d" || cat == "noise3d" || cat == "fractal2d" || cat == "fractal3d" ||
           cat == "cellnoise2d" || cat == "cellnoise3d" || cat == "worleynoise2d" || cat == "worleynoise3d" ||
           cat == "unifiednoise2d" || cat == "unifiednoise3d" || cat == "constant" || cat == "image" ||
           cat == "tiledimage" || cat == "triplanarprojection" || cat == "multiply" || cat == "mix" ||
           cat == "add" || cat == "subtract" || cat == "divide" || cat == "clamp" || cat == "saturate" ||
           cat == "invert" || cat == "absval" || cat == "power" || cat == "convert" || cat == "swizzle" ||
           cat == "combine2" || cat == "combine3" || cat == "combine4" || cat == "extract" ||
           cat == "dotproduct" || cat == "magnitude" || cat == "normalize" || cat == "texcoord" ||
           cat == "position" || cat == "normal" || cat == "tangent" || cat == "place2d" || cat == "ramplr" ||
           cat == "ramptb" || cat == "uniform" || cat == "checkerboard";
}

struct BakeDepthGuard {
    int& depth;
    explicit BakeDepthGuard(int& d) : depth(d) { ++depth; }
    ~BakeDepthGuard() { --depth; }
};

Vec4 evalNode(const mx::NodePtr& node, float u, float v, BakeState& state) {
    if (!node) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    if (state.depth > 48) return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    BakeDepthGuard depthGuard(state.depth);

    const std::string cat = node->getCategory();
    const std::string type = node->getType();
    Vec4 result(0.0f, 0.0f, 0.0f, 1.0f);

    if (cat == "texcoord") {
        result = Vec4(u, v, 0.0f, 1.0f);
    } else if (cat == "position") {
        // UV bake stand-in for object-space P (defaultgeomprop Pobject).
        result = Vec4(u * 2.0f - 1.0f, v * 2.0f - 1.0f, 0.0f, 1.0f);
    } else if (cat == "normal") {
        result = Vec4(0.0f, 0.0f, 1.0f, 1.0f);
    } else if (cat == "tangent") {
        result = Vec4(1.0f, 0.0f, 0.0f, 1.0f);
    } else if (cat == "constant" || cat == "uniform") {
        result = evalInput(node, "value", u, v, state, splat4(0.0f));
        result = asColor(result, type);
    } else if (cat == "image" || cat == "tiledimage") {
        Vec2 uv(u, v);
        if (mx::NodePtr tc = connected(node, "texcoord")) {
            const Vec4 t = evalNode(tc, u, v, state);
            uv = Vec2(t.x, t.y);
        }
        if (std::shared_ptr<Image> image = loadImageCached(state, node)) {
            result = sampleImageUV(*image, uv.x, uv.y);
        } else {
            result = evalInput(node, "default", u, v, state, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            result = asColor(result, type);
        }
    } else if (cat == "noise2d") {
        Vec2 tc(u, v);
        if (mx::NodePtr tcn = connected(node, "texcoord")) {
            const Vec4 t = evalNode(tcn, u, v, state);
            tc = Vec2(t.x, t.y);
        }
        // Scale UV into a useful default frequency (MaterialX uses raw texcoord units).
        const float sx = tc.x * 8.0f;
        const float sy = tc.y * 8.0f;
        const float pivot = evalFloat(node, "pivot", u, v, state, 0.0f);
        if (type == "float") {
            const float amp = evalFloat(node, "amplitude", u, v, state, 1.0f);
            const float n = perlin2(sx, sy);
            result = splat4(n * amp + pivot);
        } else {
            const Vec3 amp = evalVec3(node, "amplitude", u, v, state, Vec3(1.0f));
            const Vec3 n = perlin2Vec3(sx, sy);
            result = Vec4(n.x * amp.x + pivot, n.y * amp.y + pivot, n.z * amp.z + pivot, 1.0f);
        }
        result = asColor(result, type);
    } else if (cat == "noise3d" || cat == "unifiednoise3d") {
        Vec3 pos(u * 8.0f, v * 8.0f, 0.0f);
        if (mx::NodePtr pn = connected(node, "position")) {
            const Vec4 p = evalNode(pn, u, v, state);
            pos = Vec3(p.x, p.y, p.z);
            if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z))
                pos = Vec3(u * 8.0f, v * 8.0f, 0.0f);
        }
        const float pivot = evalFloat(node, "pivot", u, v, state, 0.0f);
        if (type == "float") {
            const float amp = evalFloat(node, "amplitude", u, v, state, 1.0f);
            result = splat4(perlin3(pos.x, pos.y, pos.z) * amp + pivot);
        } else {
            const Vec3 amp = evalVec3(node, "amplitude", u, v, state, Vec3(1.0f));
            const Vec3 n = perlin3Vec3(pos.x, pos.y, pos.z);
            result = Vec4(n.x * amp.x + pivot, n.y * amp.y + pivot, n.z * amp.z + pivot, 1.0f);
        }
        result = asColor(result, type);
    } else if (cat == "fractal3d" || cat == "fractal2d" || cat == "unifiednoise2d") {
        Vec3 pos(u * 4.0f, v * 4.0f, 0.0f);
        if (mx::NodePtr pn = connected(node, "position")) {
            const Vec4 p = evalNode(pn, u, v, state);
            pos = Vec3(p.x, p.y, p.z);
        } else if (mx::NodePtr tcn = connected(node, "texcoord")) {
            const Vec4 t = evalNode(tcn, u, v, state);
            pos = Vec3(t.x * 4.0f, t.y * 4.0f, 0.0f);
        }
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z))
            pos = Vec3(u * 4.0f, v * 4.0f, 0.0f);
        const int octaves = int(std::lround(evalFloat(node, "octaves", u, v, state, 3.0f)));
        const float lacunarity = evalFloat(node, "lacunarity", u, v, state, 2.0f);
        const float diminish = evalFloat(node, "diminish", u, v, state, 0.5f);
        if (type == "float") {
            const float amp = evalFloat(node, "amplitude", u, v, state, 1.0f);
            result = splat4(fractal3(pos.x, pos.y, pos.z, octaves, lacunarity, diminish) * amp);
        } else {
            const Vec3 amp = evalVec3(node, "amplitude", u, v, state, Vec3(1.0f));
            const Vec3 n = fractal3Vec3(pos.x, pos.y, pos.z, octaves, lacunarity, diminish);
            result = Vec4(n.x * amp.x, n.y * amp.y, n.z * amp.z, 1.0f);
        }
        result = asColor(result, type);
    } else if (cat == "cellnoise2d") {
        Vec2 tc(u * 8.0f, v * 8.0f);
        if (mx::NodePtr tcn = connected(node, "texcoord")) {
            const Vec4 t = evalNode(tcn, u, v, state);
            tc = Vec2(t.x * 8.0f, t.y * 8.0f);
        }
        result = asColor(splat4(cellNoise2(tc.x, tc.y)), type);
    } else if (cat == "cellnoise3d") {
        Vec3 pos(u * 8.0f, v * 8.0f, 0.0f);
        if (mx::NodePtr pn = connected(node, "position")) {
            const Vec4 p = evalNode(pn, u, v, state);
            pos = Vec3(p.x, p.y, p.z);
        }
        result = asColor(splat4(cellNoise3(pos.x, pos.y, pos.z)), type);
    } else if (cat == "worleynoise2d" || cat == "worleynoise3d") {
        // Cheap stand-in: cell noise is enough for a stable preview bake.
        if (cat == "worleynoise2d")
            result = asColor(splat4(cellNoise2(u * 8.0f, v * 8.0f)), type);
        else
            result = asColor(splat4(cellNoise3(u * 8.0f, v * 8.0f, 0.0f)), type);
    } else if (cat == "triplanarprojection") {
        auto sampleAxis = [&](const char* fileInput) -> Vec4 {
            const std::string file = inputValueString(node, fileInput);
            if (file.empty()) return evalInput(node, "default", u, v, state, splat4(0.0f));
            // Build a transient image node-like load via cache key on filename.
            std::string error;
            auto image = loadImageOrUdim(QString::fromStdString(file), state.searchDirectory, error, state.udimSet);
            if (!image) return evalInput(node, "default", u, v, state, splat4(0.0f));
            return sampleImageUV(*image, u, v);
        };
        const Vec4 cx = sampleAxis("filex");
        const Vec4 cy = sampleAxis("filey");
        const Vec4 cz = sampleAxis("filez");
        // UV-space bake has no mesh normal — use a mild Y-weighted blend so all
        // three maps contribute and the node is visibly "working".
        const float blend = std::max(0.01f, evalFloat(node, "blend", u, v, state, 1.0f));
        Vec3 w(0.25f, 0.5f, 0.25f);
        w = Vec3(std::pow(w.x, blend), std::pow(w.y, blend), std::pow(w.z, blend));
        const float s = w.x + w.y + w.z;
        w = w * (1.0f / s);
        result = Vec4(cx.x * w.x + cy.x * w.y + cz.x * w.z, cx.y * w.x + cy.y * w.y + cz.y * w.z,
                      cx.z * w.x + cy.z * w.y + cz.z * w.z, 1.0f);
        result = asColor(result, type);
    } else if (cat == "multiply") {
        const Vec4 a = evalInput(node, "in1", u, v, state, splat4(1.0f));
        const Vec4 b = evalInput(node, "in2", u, v, state, splat4(1.0f));
        result = Vec4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
        result = asColor(result, type);
    } else if (cat == "add") {
        const Vec4 a = evalInput(node, "in1", u, v, state, splat4(0.0f));
        const Vec4 b = evalInput(node, "in2", u, v, state, splat4(0.0f));
        result = Vec4(a.x + b.x, a.y + b.y, a.z + b.z, 1.0f);
        result = asColor(result, type);
    } else if (cat == "subtract") {
        const Vec4 a = evalInput(node, "in1", u, v, state, splat4(0.0f));
        const Vec4 b = evalInput(node, "in2", u, v, state, splat4(0.0f));
        result = Vec4(a.x - b.x, a.y - b.y, a.z - b.z, 1.0f);
        result = asColor(result, type);
    } else if (cat == "divide") {
        const Vec4 a = evalInput(node, "in1", u, v, state, splat4(0.0f));
        const Vec4 b = evalInput(node, "in2", u, v, state, splat4(1.0f));
        result = Vec4(b.x != 0.0f ? a.x / b.x : 0.0f, b.y != 0.0f ? a.y / b.y : 0.0f,
                      b.z != 0.0f ? a.z / b.z : 0.0f, 1.0f);
        result = asColor(result, type);
    } else if (cat == "mix") {
        const Vec4 bg = evalInput(node, "bg", u, v, state, splat4(0.0f));
        const Vec4 fg = evalInput(node, "fg", u, v, state, splat4(1.0f));
        const float m = std::clamp(evalFloat(node, "mix", u, v, state, 0.5f), 0.0f, 1.0f);
        result = Vec4(bg.x * (1.0f - m) + fg.x * m, bg.y * (1.0f - m) + fg.y * m,
                      bg.z * (1.0f - m) + fg.z * m, 1.0f);
        result = asColor(result, type);
    } else if (cat == "clamp") {
        const Vec4 in = evalInput(node, "in", u, v, state, splat4(0.0f));
        const float lo = evalFloat(node, "low", u, v, state, 0.0f);
        const float hi = evalFloat(node, "high", u, v, state, 1.0f);
        result = Vec4(std::clamp(in.x, lo, hi), std::clamp(in.y, lo, hi), std::clamp(in.z, lo, hi), 1.0f);
        result = asColor(result, type);
    } else if (cat == "saturate") {
        const Vec4 in = evalInput(node, "in", u, v, state, splat4(0.0f));
        result = Vec4(std::clamp(in.x, 0.0f, 1.0f), std::clamp(in.y, 0.0f, 1.0f), std::clamp(in.z, 0.0f, 1.0f),
                      1.0f);
    } else if (cat == "invert") {
        const Vec4 in = evalInput(node, "in", u, v, state, splat4(0.0f));
        const float amount = evalFloat(node, "amount", u, v, state, 1.0f);
        result = Vec4(in.x * (1.0f - amount) + (1.0f - in.x) * amount,
                      in.y * (1.0f - amount) + (1.0f - in.y) * amount,
                      in.z * (1.0f - amount) + (1.0f - in.z) * amount, 1.0f);
    } else if (cat == "absval") {
        const Vec4 in = evalInput(node, "in", u, v, state, splat4(0.0f));
        result = Vec4(std::fabs(in.x), std::fabs(in.y), std::fabs(in.z), 1.0f);
        result = asColor(result, type);
    } else if (cat == "power") {
        const Vec4 in = evalInput(node, "in", u, v, state, splat4(0.0f));
        const float exp = evalFloat(node, "amount", u, v, state, 2.0f);
        result = Vec4(std::pow(std::max(0.0f, in.x), exp), std::pow(std::max(0.0f, in.y), exp),
                      std::pow(std::max(0.0f, in.z), exp), 1.0f);
        result = asColor(result, type);
    } else if (cat == "convert" || cat == "swizzle") {
        result = asColor(evalInput(node, "in", u, v, state, splat4(0.0f)), type);
    } else if (cat == "combine3" || cat == "combine2" || cat == "combine4") {
        const float x = evalFloat(node, "in1", u, v, state, 0.0f);
        const float y = evalFloat(node, "in2", u, v, state, 0.0f);
        const float z = evalFloat(node, "in3", u, v, state, 0.0f);
        const float w = evalFloat(node, "in4", u, v, state, 1.0f);
        result = asColor(Vec4(x, y, z, w), type);
    } else if (cat == "extract") {
        const Vec4 in = evalInput(node, "in", u, v, state, splat4(0.0f));
        const int index = int(std::lround(evalFloat(node, "index", u, v, state, 0.0f)));
        const float channel = index <= 0 ? in.x : (index == 1 ? in.y : (index == 2 ? in.z : in.w));
        result = asColor(splat4(channel), type);
    } else if (cat == "ramplr") {
        const Vec4 a = evalInput(node, "valuel", u, v, state, splat4(0.0f));
        const Vec4 b = evalInput(node, "valuer", u, v, state, splat4(1.0f));
        const float t = std::clamp(u, 0.0f, 1.0f);
        result = Vec4(a.x * (1.0f - t) + b.x * t, a.y * (1.0f - t) + b.y * t, a.z * (1.0f - t) + b.z * t, 1.0f);
        result = asColor(result, type);
    } else if (cat == "ramptb") {
        const Vec4 a = evalInput(node, "valuet", u, v, state, splat4(0.0f));
        const Vec4 b = evalInput(node, "valueb", u, v, state, splat4(1.0f));
        const float t = std::clamp(v, 0.0f, 1.0f);
        result = Vec4(a.x * (1.0f - t) + b.x * t, a.y * (1.0f - t) + b.y * t, a.z * (1.0f - t) + b.z * t, 1.0f);
        result = asColor(result, type);
    } else if (cat == "checkerboard") {
        const float xf = std::floor(u * 8.0f);
        const float yf = std::floor(v * 8.0f);
        const bool on = (int(xf) + int(yf)) & 1;
        result = on ? splat4(1.0f) : Vec4(0.0f, 0.0f, 0.0f, 1.0f);
        result = asColor(result, type);
    } else {
        // Unknown node: try to pass through a single "in" / "in1" connection.
        if (mx::NodePtr child = connected(node, "in"))
            result = asColor(evalNode(child, u, v, state), type);
        else if (mx::NodePtr child = connected(node, "in1"))
            result = asColor(evalNode(child, u, v, state), type);
        else
            result = asColor(Vec4(0.5f, 0.5f, 0.5f, 1.0f), type);
    }

    return result;
}

bool subtreeHasBakable(const mx::NodePtr& node, int depth) {
    if (!node || depth > 32) return false;
    if (isBakableCategory(node->getCategory())) return true;
    for (const mx::InputPtr& input : node->getInputs()) {
        if (!input) continue;
        if (!input->hasNodeName() || input->getNodeName().empty()) continue;
        if (subtreeHasBakable(input->getConnectedNode(), depth + 1)) return true;
    }
    return false;
}

}  // namespace

bool materialXNodeIsBakable(mx::NodePtr node) { return subtreeHasBakable(node, 0); }

std::shared_ptr<Image> bakeMaterialXNodeToTexture(mx::NodePtr root, const QString& searchDirectory,
                                                   const std::vector<int>& udimSet, int resolution,
                                                   std::string& error) {
    error.clear();
    if (!root) {
        error = "no node to bake";
        return nullptr;
    }
    if (!materialXNodeIsBakable(root)) {
        error = "unsupported MaterialX node for bake: " + root->getCategory();
        return nullptr;
    }
    resolution = std::clamp(resolution, 64, 2048);
    try {
        BakeState state;
        state.searchDirectory = searchDirectory;
        state.udimSet = udimSet;
        auto image = std::make_shared<Image>(resolution, resolution);
        for (int y = 0; y < resolution; ++y) {
            const float v = (float(y) + 0.5f) / float(resolution);
            for (int x = 0; x < resolution; ++x) {
                const float u = (float(x) + 0.5f) / float(resolution);
                state.depth = 0;
                Vec4 c = evalNode(root, u, v, state);
                // Signed noise (pivot 0) → remap into 0..1 so albedo/roughness maps look right.
                if (c.x < 0.0f || c.y < 0.0f || c.z < 0.0f) {
                    c.x = c.x * 0.5f + 0.5f;
                    c.y = c.y * 0.5f + 0.5f;
                    c.z = c.z * 0.5f + 0.5f;
                }
                c.x = std::max(0.0f, c.x);
                c.y = std::max(0.0f, c.y);
                c.z = std::max(0.0f, c.z);
                image->setRgb(x, y, Vec3(c.x, c.y, c.z), 1.0f);
            }
        }
        return image;
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    } catch (...) {
        error = "unknown bake failure";
        return nullptr;
    }
}

#endif  // SOLSTICE_HAVE_MATERIALX
}  // namespace sol
