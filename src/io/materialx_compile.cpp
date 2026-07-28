#include "io/materialx_compile.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

bool parseFloats(const std::string& value, float* out, int count) {
    if (!out || count <= 0) return false;
    float a = 0, b = 0, c = 0, d = 0;
    int n = 0;
    if (count >= 4) n = std::sscanf(value.c_str(), "%f,%f,%f,%f", &a, &b, &c, &d);
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
    if (!input->hasNodeName() || input->getNodeName().empty()) return nullptr;
    return input->getConnectedNode();
}

int channelsForType(const std::string& type) {
    if (type == "float" || type == "integer" || type == "int" || type == "boolean") return 1;
    if (type == "vector2") return 2;
    if (type == "color4" || type == "vector4") return 4;
    return 3;
}

bool isProceduralCategory(const std::string& cat) {
    return cat == "noise2d" || cat == "noise3d" || cat == "fractal2d" || cat == "fractal3d" ||
           cat == "cellnoise2d" || cat == "cellnoise3d" || cat == "worleynoise2d" || cat == "worleynoise3d" ||
           cat == "unifiednoise2d" || cat == "unifiednoise3d" || cat == "constant" || cat == "uniform" ||
           cat == "image" || cat == "tiledimage" || cat == "triplanarprojection" || cat == "multiply" ||
           cat == "mix" || cat == "add" || cat == "subtract" || cat == "divide" || cat == "clamp" ||
           cat == "saturate" || cat == "invert" || cat == "absval" || cat == "power" || cat == "convert" ||
           cat == "swizzle" || cat == "combine2" || cat == "combine3" || cat == "combine4" || cat == "extract" ||
           cat == "texcoord" || cat == "position" || cat == "normal" || cat == "tangent" || cat == "ramplr" ||
           cat == "ramptb" || cat == "checkerboard";
}

bool subtreeProcedural(const mx::NodePtr& node, int depth) {
    if (!node || depth > 32) return false;
    if (isProceduralCategory(node->getCategory())) return true;
    for (const mx::InputPtr& input : node->getInputs()) {
        if (!input) continue;
        if (!input->hasNodeName() || input->getNodeName().empty()) continue;
        if (subtreeProcedural(input->getConnectedNode(), depth + 1)) return true;
    }
    return false;
}

Vec4 readVec4(const mx::NodePtr& node, const std::string& name, Vec4 fallback) {
    float vals[4] = {fallback.x, fallback.y, fallback.z, fallback.w};
    const std::string raw = inputValueString(node, name);
    if (!raw.empty()) parseFloats(raw, vals, 4);
    return Vec4(vals[0], vals[1], vals[2], vals[3]);
}

float readFloat(const mx::NodePtr& node, const std::string& name, float fallback) {
    return readVec4(node, name, Vec4(fallback, fallback, fallback, fallback)).x;
}

bool readBool(const mx::NodePtr& node, const std::string& name, bool fallback) {
    const std::string raw = inputValueString(node, name);
    if (raw.empty()) return fallback;
    return raw == "true" || raw == "1" || raw == "True";
}

struct CompileState {
    QString searchDirectory;
    std::vector<int> udimSet;
    std::vector<ProceduralNode>* nodes = nullptr;
    std::vector<std::shared_ptr<Image>>* images = nullptr;
    std::map<const mx::Node*, int> cache;
    std::map<std::string, int> imageFileCache;
};

// Named pushNode — Qt signal macro as an empty macro.
int pushNode(CompileState& state, ProceduralNode node) {
    state.nodes->push_back(node);
    return int(state.nodes->size()) - 1;
}

int compileNode(const mx::NodePtr& node, CompileState& state, int depth);

int compileConnected(const mx::NodePtr& node, const std::string& inputName, CompileState& state, int depth) {
    mx::NodePtr child = connected(node, inputName);
    if (!child) return -1;
    return compileNode(child, state, depth + 1);
}

int loadImageIndex(CompileState& state, const std::string& file) {
    if (file.empty()) return -1;
    auto it = state.imageFileCache.find(file);
    if (it != state.imageFileCache.end()) return it->second;
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
    if (!image) {
        if (!error.empty()) logWarning("MaterialX procedural image: " + error);
        state.imageFileCache[file] = -1;
        return -1;
    }
    const int idx = int(state.images->size());
    state.images->push_back(image);
    state.imageFileCache[file] = idx;
    return idx;
}

int compileNode(const mx::NodePtr& node, CompileState& state, int depth) {
    if (!node || depth > 48) return -1;
    auto cached = state.cache.find(node.get());
    if (cached != state.cache.end()) return cached->second;

    const std::string cat = node->getCategory();
    const int channels = channelsForType(node->getType());
    ProceduralNode n;
    n.channels = channels;
    int result = -1;

    if (cat == "texcoord") {
        n.op = kProcUv;
        result = pushNode(state, n);
    } else if (cat == "position") {
        n.op = kProcPosition;
        result = pushNode(state, n);
    } else if (cat == "normal" || cat == "tangent") {
        n.op = kProcNormal;
        result = pushNode(state, n);
    } else if (cat == "constant" || cat == "uniform") {
        n.op = kProcConst;
        n.p0 = readVec4(node, "value", Vec4(0.0f, 0.0f, 0.0f, 1.0f));
        result = pushNode(state, n);
    } else if (cat == "image" || cat == "tiledimage") {
        n.op = kProcImage;
        n.in0 = loadImageIndex(state, inputValueString(node, "file"));
        n.in1 = compileConnected(node, "texcoord", state, depth);
        n.p0 = readVec4(node, "default", Vec4(0.0f, 0.0f, 0.0f, 1.0f));
        result = pushNode(state, n);
    } else if (cat == "noise2d") {
        n.op = kProcNoise2d;
        n.p0 = readVec4(node, "amplitude", Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        n.p1.x = readFloat(node, "pivot", 0.0f);
        n.in0 = compileConnected(node, "texcoord", state, depth);
        result = pushNode(state, n);
    } else if (cat == "noise3d") {
        n.op = kProcNoise3d;
        n.p0 = readVec4(node, "amplitude", Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        n.p1.x = readFloat(node, "pivot", 0.0f);
        n.in0 = compileConnected(node, "position", state, depth);
        result = pushNode(state, n);
    } else if (cat == "fractal2d" || cat == "fractal3d") {
        n.op = kProcFractal;
        n.p0 = readVec4(node, "amplitude", Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        n.s0 = readFloat(node, "octaves", 3.0f);
        n.s1 = readFloat(node, "lacunarity", 2.0f);
        n.s2 = readFloat(node, "diminish", 0.5f);
        n.in0 = compileConnected(node, "position", state, depth);
        if (n.in0 < 0) n.in0 = compileConnected(node, "texcoord", state, depth);
        result = pushNode(state, n);
    } else if (cat == "cellnoise2d" || cat == "worleynoise2d") {
        n.op = kProcCell2d;
        n.in0 = compileConnected(node, "texcoord", state, depth);
        result = pushNode(state, n);
    } else if (cat == "cellnoise3d" || cat == "worleynoise3d") {
        n.op = kProcCell3d;
        n.in0 = compileConnected(node, "position", state, depth);
        result = pushNode(state, n);
    } else if (cat == "unifiednoise2d") {
        n.op = kProcUnified2d;
        n.p0 = readVec4(node, "freq", Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        n.p1 = readVec4(node, "offset", Vec4(0.0f, 0.0f, 0.0f, 0.0f));
        n.p2.x = readFloat(node, "outmin", 0.0f);
        n.p2.y = readFloat(node, "outmax", 1.0f);
        n.p2.z = readFloat(node, "type", 0.0f);
        n.p2.w = readFloat(node, "style", 0.0f);
        n.s0 = readFloat(node, "octaves", 3.0f);
        n.s1 = readFloat(node, "lacunarity", 2.0f);
        n.s2 = readFloat(node, "diminish", 0.5f);
        n.s3 = readBool(node, "clampoutput", true) ? 1.0f : 0.0f;
        n.in0 = compileConnected(node, "texcoord", state, depth);
        result = pushNode(state, n);
    } else if (cat == "unifiednoise3d") {
        n.op = kProcUnified3d;
        n.p0 = readVec4(node, "freq", Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        n.p1 = readVec4(node, "offset", Vec4(0.0f, 0.0f, 0.0f, 0.0f));
        n.p2.x = readFloat(node, "outmin", 0.0f);
        n.p2.y = readFloat(node, "outmax", 1.0f);
        n.p2.z = readFloat(node, "type", 0.0f);
        n.p2.w = readFloat(node, "style", 0.0f);
        n.s0 = readFloat(node, "octaves", 3.0f);
        n.s1 = readFloat(node, "lacunarity", 2.0f);
        n.s2 = readFloat(node, "diminish", 0.5f);
        n.s3 = readBool(node, "clampoutput", true) ? 1.0f : 0.0f;
        n.in0 = compileConnected(node, "position", state, depth);
        result = pushNode(state, n);
    } else if (cat == "triplanarprojection") {
        // Arnold-style: shared `file` by default; `input_per_axis` unlocks filex/y/z.
        n.op = kProcTriplanar;
        const bool perAxis = readBool(node, "input_per_axis", false);
        const std::string shared = inputValueString(node, "file");
        std::string fx = inputValueString(node, "filex");
        std::string fy = inputValueString(node, "filey");
        std::string fz = inputValueString(node, "filez");
        auto firstNonEmpty = [](const std::string& a, const std::string& b, const std::string& c,
                                const std::string& d) -> std::string {
            if (!a.empty()) return a;
            if (!b.empty()) return b;
            if (!c.empty()) return c;
            return d;
        };
        if (!perAxis) {
            const std::string src = firstNonEmpty(shared, fx, fy, fz);
            fx = fy = fz = src;
        } else {
            if (fx.empty()) fx = shared;
            if (fy.empty()) fy = !shared.empty() ? shared : fx;
            if (fz.empty()) fz = !shared.empty() ? shared : fx;
        }
        n.in0 = loadImageIndex(state, fx);
        n.in1 = loadImageIndex(state, fy);
        n.in2 = loadImageIndex(state, fz);
        n.p0 = readVec4(node, "default", Vec4(0.2f, 0.5f, 0.8f, 1.0f));
        n.p1 = readVec4(node, "scale", Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        n.p2 = readVec4(node, "offset", Vec4(0.0f, 0.0f, 0.0f, 0.0f));
        n.s0 = readFloat(node, "blend", 0.1f);
        n.s1 = readFloat(node, "rotate", 0.0f);
        result = pushNode(state, n);
    } else if (cat == "multiply") {
        n.op = kProcMul;
        n.in0 = compileConnected(node, "in1", state, depth);
        n.in1 = compileConnected(node, "in2", state, depth);
        n.p0 = readVec4(node, "in2", Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        result = pushNode(state, n);
    } else if (cat == "add") {
        n.op = kProcAdd;
        n.in0 = compileConnected(node, "in1", state, depth);
        n.in1 = compileConnected(node, "in2", state, depth);
        n.p0 = readVec4(node, "in2", Vec4(0.0f, 0.0f, 0.0f, 1.0f));
        result = pushNode(state, n);
    } else if (cat == "subtract") {
        n.op = kProcSub;
        n.in0 = compileConnected(node, "in1", state, depth);
        n.in1 = compileConnected(node, "in2", state, depth);
        n.p0 = readVec4(node, "in2", Vec4(0.0f, 0.0f, 0.0f, 1.0f));
        result = pushNode(state, n);
    } else if (cat == "divide") {
        n.op = kProcDiv;
        n.in0 = compileConnected(node, "in1", state, depth);
        n.in1 = compileConnected(node, "in2", state, depth);
        n.p0 = readVec4(node, "in2", Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        result = pushNode(state, n);
    } else if (cat == "mix") {
        n.op = kProcMix;
        n.in0 = compileConnected(node, "bg", state, depth);
        n.in1 = compileConnected(node, "fg", state, depth);
        n.in2 = compileConnected(node, "mix", state, depth);
        n.p0.x = readFloat(node, "mix", 0.5f);
        result = pushNode(state, n);
    } else if (cat == "clamp") {
        n.op = kProcClamp;
        n.in0 = compileConnected(node, "in", state, depth);
        n.s1 = readFloat(node, "low", 0.0f);
        n.s2 = readFloat(node, "high", 1.0f);
        result = pushNode(state, n);
    } else if (cat == "saturate") {
        n.op = kProcSaturate;
        n.in0 = compileConnected(node, "in", state, depth);
        result = pushNode(state, n);
    } else if (cat == "invert") {
        n.op = kProcInvert;
        n.in0 = compileConnected(node, "in", state, depth);
        n.s0 = readFloat(node, "amount", 1.0f);
        result = pushNode(state, n);
    } else if (cat == "absval") {
        n.op = kProcAbs;
        n.in0 = compileConnected(node, "in", state, depth);
        result = pushNode(state, n);
    } else if (cat == "power") {
        n.op = kProcPower;
        n.in0 = compileConnected(node, "in", state, depth);
        n.s0 = readFloat(node, "amount", 2.0f);
        result = pushNode(state, n);
    } else if (cat == "convert" || cat == "swizzle") {
        int child = compileConnected(node, "in", state, depth);
        if (child >= 0) {
            result = child;
        } else {
            n.op = kProcConst;
            n.p0 = readVec4(node, "in", Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            result = pushNode(state, n);
        }
    } else if (cat == "combine2" || cat == "combine3" || cat == "combine4") {
        n.op = kProcCombine;
        n.in0 = compileConnected(node, "in1", state, depth);
        n.in1 = compileConnected(node, "in2", state, depth);
        n.in2 = compileConnected(node, "in3", state, depth);
        n.in3 = compileConnected(node, "in4", state, depth);
        n.p0 = Vec4(readFloat(node, "in1", 0.0f), readFloat(node, "in2", 0.0f), readFloat(node, "in3", 0.0f),
                    readFloat(node, "in4", 1.0f));
        result = pushNode(state, n);
    } else if (cat == "extract") {
        n.op = kProcExtract;
        n.in0 = compileConnected(node, "in", state, depth);
        n.s0 = readFloat(node, "index", 0.0f);
        result = pushNode(state, n);
    } else if (cat == "ramplr") {
        n.op = kProcRampLR;
        n.in0 = compileConnected(node, "valuel", state, depth);
        n.in1 = compileConnected(node, "valuer", state, depth);
        result = pushNode(state, n);
    } else if (cat == "ramptb") {
        n.op = kProcRampTB;
        n.in0 = compileConnected(node, "valuet", state, depth);
        n.in1 = compileConnected(node, "valueb", state, depth);
        result = pushNode(state, n);
    } else if (cat == "checkerboard") {
        n.op = kProcChecker;
        result = pushNode(state, n);
    } else {
        int child = compileConnected(node, "in", state, depth);
        if (child < 0) child = compileConnected(node, "in1", state, depth);
        if (child >= 0) {
            result = child;
        } else {
            n.op = kProcConst;
            n.p0 = Vec4(0.5f, 0.5f, 0.5f, 1.0f);
            result = pushNode(state, n);
        }
    }

    state.cache[node.get()] = result;
    return result;
}

}  // namespace

bool materialXNodeIsProcedural(mx::NodePtr node) { return subtreeProcedural(node, 0); }

int compileMaterialXNode(mx::NodePtr root, const QString& searchDirectory, const std::vector<int>& udimSet,
                         std::vector<ProceduralNode>& outNodes, std::vector<std::shared_ptr<Image>>& outImages,
                         std::string& error) {
    error.clear();
    if (!root) {
        error = "no node to compile";
        return -1;
    }
    if (!materialXNodeIsProcedural(root)) {
        error = "unsupported MaterialX node for procedural compile: " + root->getCategory();
        return -1;
    }
    try {
        CompileState state;
        state.searchDirectory = searchDirectory;
        state.udimSet = udimSet;
        state.nodes = &outNodes;
        state.images = &outImages;
        const int rootIndex = compileNode(root, state, 0);
        if (rootIndex < 0) {
            error = "procedural compile produced empty graph";
            return -1;
        }
        return rootIndex;
    } catch (const std::exception& e) {
        error = e.what();
        return -1;
    } catch (...) {
        error = "unknown procedural compile failure";
        return -1;
    }
}

#endif  // SOLSTICE_HAVE_MATERIALX
}  // namespace sol
