#include "io/usd_loader.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "core/log.h"
#include "io/alembic_loader.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_TINYUSDZ
#include "tinyusdz.hh"
#include "pprinter.hh"
#include "value-pprint.hh"
#endif

namespace sol {
namespace {

std::string trim(const std::string& text) {
    size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) ++start;
    size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return text.substr(start, end - start);
}

bool readFileText(const std::string& path, std::string& out) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    out.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    return true;
}

bool isUsdaText(const std::string& data) {
    if (data.size() >= 5 && data.compare(0, 5, "#usda") == 0) return true;
    if (data.find("def ") != std::string::npos || data.find("over ") != std::string::npos) return true;
    return false;
}

bool isUsdcBinary(const std::string& data) {
    return data.size() >= 8 && data.compare(0, 8, "PXR-USDC") == 0;
}

bool isUsdzArchive(const std::string& data) {
    // ZIP local file header (USDZ is a zip package).
    return data.size() >= 4 && data[0] == 'P' && data[1] == 'K' &&
           (unsigned char)data[2] <= 0x05;
}

bool parseFloatList(const std::string& text, std::vector<float>& values) {
    values.clear();
    std::string token;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '(' || c == ')' || c == '[' || c == ']' || c == ',' || c == ' ') {
            if (!token.empty()) {
                try {
                    values.push_back(std::stof(token));
                } catch (...) {
                    return false;
                }
                token.clear();
            }
            continue;
        }
        token.push_back(c);
    }
    if (!token.empty()) {
        try {
            values.push_back(std::stof(token));
        } catch (...) {
            return false;
        }
    }
    return !values.empty();
}

bool parseIntList(const std::string& text, std::vector<int>& values) {
    values.clear();
    std::string token;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '(' || c == ')' || c == '[' || c == ']' || c == ',' || c == ' ') {
            if (!token.empty()) {
                try {
                    values.push_back(std::stoi(token));
                } catch (...) {
                    return false;
                }
                token.clear();
            }
            continue;
        }
        token.push_back(c);
    }
    if (!token.empty()) {
        try {
            values.push_back(std::stoi(token));
        } catch (...) {
            return false;
        }
    }
    return !values.empty();
}

Mat4 matrixFromUsdRowMajor(const std::vector<float>& m) {
    if (m.size() < 16) return Mat4::identity();
    Mat4 result = Mat4::identity();
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) result.at(r, c) = m[size_t(r * 4 + c)];
    return result;
}

struct UsdBlock {
    std::string type;
    std::string name;
    std::string path;
    Mat4 localXform = Mat4::identity();
    std::vector<float> points;
    std::vector<int> faceVertexCounts;
    std::vector<int> faceVertexIndices;
    CameraData camera;
    LightData light;
    bool hasCamera = false;
    bool hasLight = false;
};

LightType lightTypeFromUsd(const std::string& type) {
    if (type == "DistantLight") return kLightDistant;
    if (type == "RectLight") return kLightRect;
    if (type == "DiskLight") return kLightDisk;
    if (type == "SphereLight") return kLightSphere;
    if (type == "DomeLight") return kLightDome;
    return kLightDistant;
}

void triangulateFaces(const std::vector<float>& points, const std::vector<int>& faceVertexCounts,
                      const std::vector<int>& faceVertexIndices, MeshPtr& mesh) {
    if (points.size() < 9 || faceVertexCounts.empty() || faceVertexIndices.empty()) return;
    mesh = std::make_shared<Mesh>();
    const int vertexCount = int(points.size() / 3);
    mesh->positions.resize(vertexCount);
    for (int i = 0; i < vertexCount; ++i)
        mesh->positions[i] = Vec3(points[size_t(i * 3)], points[size_t(i * 3 + 1)], points[size_t(i * 3 + 2)]);

    size_t cursor = 0;
    for (int count : faceVertexCounts) {
        if (count < 3 || cursor + size_t(count) > faceVertexIndices.size()) {
            cursor += size_t(std::max(count, 0));
            continue;
        }
        const int i0 = faceVertexIndices[cursor];
        for (int i = 1; i < count - 1; ++i) {
            mesh->indices.push_back(uint32_t(i0));
            mesh->indices.push_back(uint32_t(faceVertexIndices[cursor + size_t(i)]));
            mesh->indices.push_back(uint32_t(faceVertexIndices[cursor + size_t(i + 1)]));
        }
        cursor += size_t(count);
    }
    mesh->computeNormalsIfMissing();
}

void applyAttribute(UsdBlock& block, const std::string& key, const std::string& valueRaw) {
    const std::string value = trim(valueRaw);
    if (key.find("xformOp:transform") != std::string::npos) {
        std::vector<float> m;
        if (parseFloatList(value, m)) block.localXform = matrixFromUsdRowMajor(m);
        return;
    }
    if (key.find("xformOp:translate") != std::string::npos) {
        std::vector<float> t;
        if (parseFloatList(value, t) && t.size() >= 3)
            block.localXform = block.localXform * Mat4::translate(Vec3(t[0], t[1], t[2]));
        return;
    }
    if (key.find("xformOp:rotateXYZ") != std::string::npos) {
        std::vector<float> r;
        if (parseFloatList(value, r) && r.size() >= 3)
            block.localXform = block.localXform * Mat4::rotateY(radians(r[1])) * Mat4::rotateX(radians(r[0])) *
                               Mat4::rotateZ(radians(r[2]));
        return;
    }
    if (key.find("xformOp:scale") != std::string::npos) {
        std::vector<float> s;
        if (parseFloatList(value, s) && s.size() >= 3)
            block.localXform = block.localXform * Mat4::scale(Vec3(s[0], s[1], s[2]));
        return;
    }

    if (block.type == "Mesh") {
        if (key.find("point3f[] points") != std::string::npos || key.find("points") != std::string::npos) {
            parseFloatList(value, block.points);
            return;
        }
        if (key.find("faceVertexCounts") != std::string::npos) {
            parseIntList(value, block.faceVertexCounts);
            return;
        }
        if (key.find("faceVertexIndices") != std::string::npos) {
            parseIntList(value, block.faceVertexIndices);
            return;
        }
    }

    if (block.type == "Camera") {
        std::vector<float> nums;
        if (key.find("focalLength") != std::string::npos && parseFloatList(value, nums))
            block.camera.focalLength = nums[0];
        if (key.find("horizontalAperture") != std::string::npos && parseFloatList(value, nums))
            block.camera.sensorWidth = nums[0];
        if (key.find("focusDistance") != std::string::npos && parseFloatList(value, nums))
            block.camera.focusDistance = nums[0];
        if (key.find("fStop") != std::string::npos && parseFloatList(value, nums)) block.camera.fStop = nums[0];
        block.hasCamera = true;
        return;
    }

    if (block.type == "DistantLight" || block.type == "RectLight" || block.type == "DiskLight" ||
        block.type == "SphereLight" || block.type == "DomeLight") {
        std::vector<float> nums;
        block.light.type = lightTypeFromUsd(block.type);
        block.hasLight = true;
        if (key.find("intensity") != std::string::npos && parseFloatList(value, nums))
            block.light.intensity = nums[0];
        if (key.find("exposure") != std::string::npos && parseFloatList(value, nums)) block.light.exposure = nums[0];
        if ((key.find("color") != std::string::npos || key.find("inputs:color") != std::string::npos) &&
            parseFloatList(value, nums) && nums.size() >= 3)
            block.light.color = Vec3(nums[0], nums[1], nums[2]);
        if (key.find("width") != std::string::npos && parseFloatList(value, nums)) block.light.width = nums[0];
        if (key.find("height") != std::string::npos && parseFloatList(value, nums)) block.light.height = nums[0];
        if (key.find("radius") != std::string::npos && parseFloatList(value, nums)) block.light.radius = nums[0];
        if (key.find("angle") != std::string::npos && parseFloatList(value, nums)) block.light.angle = nums[0];
    }
}

void emitPrim(const UsdBlock& block, const Mat4& worldXform, UsdContents& out, const UsdLoadOptions& options) {
    if (!options.pathFilter.empty() && !globMatch(options.pathFilter, block.path)) return;

    if (block.type == "Mesh") {
        MeshPtr mesh;
        triangulateFaces(block.points, block.faceVertexCounts, block.faceVertexIndices, mesh);
        if (!mesh || mesh->indices.empty()) return;
        if (options.scale != 1.0f) {
            for (Vec3& p : mesh->positions) p *= options.scale;
        }
        UsdPrim prim;
        prim.type = UsdPrim::Type::Mesh;
        prim.path = block.path;
        prim.mesh = std::move(mesh);
        prim.transform = worldXform;
        out.prims.push_back(std::move(prim));
        return;
    }

    if (block.type == "Camera" && block.hasCamera) {
        UsdPrim prim;
        prim.type = UsdPrim::Type::Camera;
        prim.path = block.path;
        prim.transform = worldXform;
        prim.camera = block.camera;
        prim.hasCamera = true;
        out.prims.push_back(std::move(prim));
        return;
    }

    if (block.hasLight) {
        UsdPrim prim;
        prim.type = UsdPrim::Type::Light;
        prim.path = block.path;
        prim.transform = worldXform;
        prim.light = block.light;
        prim.hasLight = true;
        out.prims.push_back(std::move(prim));
    }
}

void parseUsda(const std::string& data, UsdContents& out, const UsdLoadOptions& options) {
    struct Scope {
        std::string path;
        Mat4 parentXform = Mat4::identity();
        Mat4 localXform = Mat4::identity();
    };
    std::vector<Scope> scopes{{"", Mat4::identity(), Mat4::identity()}};
    UsdBlock current;
    int braceDepth = 0;
    bool inBlock = false;

    auto parentXformForChildren = [&]() -> Mat4 {
        if (scopes.size() <= 1) return Mat4::identity();
        const Scope& scope = scopes.back();
        return scope.parentXform * scope.localXform;
    };

    auto beginBlock = [&](const std::string& trimmed) {
        std::string rest = trim(trimmed.substr(trimmed.find(' ') + 1));
        current = UsdBlock{};
        current.localXform = Mat4::identity();

        const size_t quoteStart = rest.find('"');
        const size_t quoteEnd = quoteStart != std::string::npos ? rest.find('"', quoteStart + 1) : std::string::npos;
        if (quoteStart != std::string::npos && quoteEnd != std::string::npos) {
            current.type = trim(rest.substr(0, quoteStart));
            current.name = rest.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
        } else {
            const size_t space = rest.find(' ');
            current.type = space == std::string::npos ? rest : trim(rest.substr(0, space));
            current.name = space == std::string::npos ? rest : trim(rest.substr(space + 1));
        }

        std::string parentPath = scopes.back().path;
        if (!parentPath.empty() && parentPath.back() != '/') parentPath += "/";
        current.path = parentPath + current.name;
        inBlock = true;
        braceDepth = 0;
        if (trimmed.find('{') != std::string::npos) {
            braceDepth = 1;
            if (current.type == "Xform" || current.type == "Scope") {
                Scope scope;
                scope.path = current.path;
                scope.parentXform = parentXformForChildren();
                scopes.push_back(scope);
            }
        }
    };

    std::istringstream stream(data);
    std::string line;
    while (std::getline(stream, line)) {
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed.rfind("//", 0) == 0) continue;
        if (trimmed.rfind("#", 0) == 0 && trimmed.rfind("#usda", 0) != 0) continue;

        if (trimmed.rfind("def ", 0) == 0 || trimmed.rfind("over ", 0) == 0) {
            beginBlock(trimmed);
            continue;
        }

        if (!inBlock) continue;

        if (trimmed.find('{') != std::string::npos) {
            const int opens = int(std::count(trimmed.begin(), trimmed.end(), '{'));
            if (braceDepth == 0 && opens > 0 && (current.type == "Xform" || current.type == "Scope")) {
                Scope scope;
                scope.path = current.path;
                scope.parentXform = parentXformForChildren();
                scopes.push_back(scope);
            }
            braceDepth += opens;
        }
        if (trimmed.find('}') != std::string::npos) braceDepth -= int(std::count(trimmed.begin(), trimmed.end(), '}'));

        const size_t eq = trimmed.find('=');
        if (eq != std::string::npos && trimmed.find('}') == std::string::npos) {
            const std::string key = trim(trimmed.substr(0, eq));
            std::string value = trim(trimmed.substr(eq + 1));
            if (!value.empty() && value.back() == ')') {
                const size_t paren = value.rfind('(');
                if (paren != std::string::npos) value = value.substr(paren);
            }
            applyAttribute(current, key, value);
            if (!scopes.empty() && scopes.back().path == current.path &&
                (current.type == "Xform" || current.type == "Scope")) {
                scopes.back().localXform = current.localXform;
            }
        }

        if (braceDepth <= 0) {
            const Mat4 world = parentXformForChildren() * current.localXform;
            emitPrim(current, world, out, options);
            if ((current.type == "Xform" || current.type == "Scope") && scopes.size() > 1 &&
                scopes.back().path == current.path) {
                scopes.pop_back();
            }
            inBlock = false;
        }
    }
}

}  // namespace

bool usdSupportAvailable() { return true; }

#if SOLSTICE_HAVE_TINYUSDZ
namespace {

bool loadUsdViaTinyusdz(const std::string& filePath, const UsdLoadOptions& options, UsdContents& out,
                        std::string& error) {
    tinyusdz::Stage stage;
    std::string warn;
    std::string err;
    if (!tinyusdz::LoadUSDFromFile(filePath, &stage, &warn, &err)) {
        error = err.empty() ? ("TinyUSDZ failed to load " + filePath) : err;
        return false;
    }
    if (!warn.empty()) logInfo("USD: TinyUSDZ warn: " + warn);

    // ExportToString() can omit mesh topology; to_string() dumps full USDA.
    std::string usda = tinyusdz::to_string(stage);
    if (usda.empty()) {
        error = "TinyUSDZ produced empty USDA for " + filePath;
        return false;
    }

    out = UsdContents{};
    parseUsda(usda, out, options);
    out.archiveInfo = "USDC/USDZ via TinyUSDZ " + filePath;
    logInfo("USD: loaded " + std::to_string(out.prims.size()) + " prims from binary USD " + filePath);
    if (out.prims.empty()) {
        error = "no supported geometry, camera or light prims found in " + filePath;
        return false;
    }
    return true;
}

}  // namespace
#endif

bool loadUsd(const std::string& filePath, const UsdLoadOptions& options, UsdContents& out, std::string& error) {
    std::string data;
    if (!readFileText(filePath, data)) {
        error = "cannot read USD file " + filePath;
        return false;
    }

    // Prefer magic-number detection so binary crates are never misread as USDA.
    if (isUsdcBinary(data) || isUsdzArchive(data)) {
#if SOLSTICE_HAVE_TINYUSDZ
        return loadUsdViaTinyusdz(filePath, options, out, error);
#else
        error = "binary USD (.usdc/.usdz) support requires TinyUSDZ; rebuild with SOLSTICE_ENABLE_TINYUSDZ=ON";
        return false;
#endif
    }

    if (isUsdaText(data)) {
        out = UsdContents{};
        parseUsda(data, out, options);
        out.archiveInfo = "USDA " + filePath;
        logInfo("USD: loaded " + std::to_string(out.prims.size()) + " prims from " + filePath);
        if (out.prims.empty()) {
            error = "no supported geometry, camera or light prims found in " + filePath;
            return false;
        }
        return true;
    }

#if SOLSTICE_HAVE_TINYUSDZ
    // Some exporters write crate USD without a clear USDA header; try TinyUSDZ.
    return loadUsdViaTinyusdz(filePath, options, out, error);
#else
    error = "unrecognized USD file format (expected USDA text or USDC binary)";
    return false;
#endif
}

}  // namespace sol
