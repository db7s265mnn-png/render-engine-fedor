// Minimal OBJ -> Alembic (Ogawa) converter for shipping demo assets.
// Preserves UVs (including UDIM ranges) as face-varying st.
#include <Alembic/Abc/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace Alembic::AbcGeom;

namespace {

struct FaceVert {
    int v = -1;
    int vt = -1;
};

bool parseFaceVert(const std::string& tok, FaceVert& out) {
    // v / v/vt / v/vt/vn / v//vn
    int v = 0, vt = 0, vn = 0;
    if (std::sscanf(tok.c_str(), "%d/%d/%d", &v, &vt, &vn) == 3 ||
        std::sscanf(tok.c_str(), "%d/%d", &v, &vt) == 2) {
        out.v = v;
        out.vt = vt;
        return true;
    }
    if (std::sscanf(tok.c_str(), "%d//%d", &v, &vn) == 2 || std::sscanf(tok.c_str(), "%d", &v) == 1) {
        out.v = v;
        out.vt = -1;
        return true;
    }
    return false;
}

int resolveIndex(int idx, int count) {
    if (idx < 0) return count + idx;
    return idx - 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s input.obj output.abc [name]\n", argv[0]);
        return 1;
    }
    const std::string inPath = argv[1];
    const std::string outPath = argv[2];
    const std::string name = argc > 3 ? argv[3] : "mesh";

    std::ifstream in(inPath);
    if (!in) {
        std::fprintf(stderr, "failed to open %s\n", inPath.c_str());
        return 1;
    }

    std::vector<Imath::V3f> positions;
    std::vector<Imath::V2f> texcoords;
    // Expanded per-corner arrays so UVs survive (OBJ indexes v/vt independently).
    std::vector<Imath::V3f> outPositions;
    std::vector<Imath::V2f> outUvs;
    std::vector<int32_t> faceCounts;
    std::vector<int32_t> faceIndices;
    bool hasUvs = false;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            float x = 0, y = 0, z = 0;
            ss >> x >> y >> z;
            positions.emplace_back(x, y, z);
        } else if (tag == "vt") {
            float u = 0, v = 0;
            ss >> u >> v;
            texcoords.emplace_back(u, v);
        } else if (tag == "f") {
            std::vector<FaceVert> face;
            std::string tok;
            while (ss >> tok) {
                FaceVert fv;
                if (!parseFaceVert(tok, fv)) continue;
                fv.v = resolveIndex(fv.v, int(positions.size()));
                if (fv.vt >= 0) {
                    fv.vt = resolveIndex(fv.vt, int(texcoords.size()));
                    hasUvs = true;
                }
                if (fv.v < 0 || fv.v >= int(positions.size())) continue;
                face.push_back(fv);
            }
            if (face.size() < 3) continue;
            // Triangulate fan; expand corners so each index has its own UV.
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                const FaceVert tri[3] = {face[0], face[i], face[i + 1]};
                faceCounts.push_back(3);
                for (const FaceVert& corner : tri) {
                    faceIndices.push_back(int32_t(outPositions.size()));
                    outPositions.push_back(positions[size_t(corner.v)]);
                    if (hasUvs && corner.vt >= 0 && corner.vt < int(texcoords.size()))
                        outUvs.push_back(texcoords[size_t(corner.vt)]);
                    else
                        outUvs.emplace_back(0.0f, 0.0f);
                }
            }
        }
    }
    if (outPositions.empty() || faceCounts.empty()) {
        std::fprintf(stderr, "no mesh data in %s\n", inPath.c_str());
        return 1;
    }

    // Normalize to sit on the ground roughly 2 units tall.
    Imath::V3f bmin = outPositions[0], bmax = outPositions[0];
    for (const auto& p : outPositions) {
        bmin.x = std::min(bmin.x, p.x);
        bmin.y = std::min(bmin.y, p.y);
        bmin.z = std::min(bmin.z, p.z);
        bmax.x = std::max(bmax.x, p.x);
        bmax.y = std::max(bmax.y, p.y);
        bmax.z = std::max(bmax.z, p.z);
    }
    const Imath::V3f size = bmax - bmin;
    const float scale = 2.0f / std::max(size.y, 1e-6f);
    const Imath::V3f center((bmin.x + bmax.x) * 0.5f, bmin.y, (bmin.z + bmax.z) * 0.5f);
    for (auto& p : outPositions) {
        p = (p - center) * scale;
    }

    OArchive archive(Alembic::AbcCoreOgawa::WriteArchive(), outPath);
    OObject top = archive.getTop();
    OXform xform(top, name + "_xform");
    OPolyMesh meshObj(xform, name);
    OPolyMeshSchema& schema = meshObj.getSchema();

    OV2fGeomParam::Sample uvSample;
    if (hasUvs && outUvs.size() == outPositions.size()) {
        uvSample = OV2fGeomParam::Sample(V2fArraySample(outUvs), kFacevaryingScope);
    }

    OPolyMeshSchema::Sample sample{V3fArraySample(outPositions), Int32ArraySample(faceIndices),
                                   Int32ArraySample(faceCounts)};
    if (hasUvs) sample.setUVs(uvSample);
    schema.set(sample);

    if (hasUvs && !outUvs.empty()) {
        float minu = outUvs[0].x, maxu = outUvs[0].x, minv = outUvs[0].y, maxv = outUvs[0].y;
        for (const auto& uv : outUvs) {
            minu = std::min(minu, uv.x);
            maxu = std::max(maxu, uv.x);
            minv = std::min(minv, uv.y);
            maxv = std::max(maxv, uv.y);
        }
        std::printf("wrote %s (%zu verts, %zu faces, uv=[%.3f,%.3f]x[%.3f,%.3f])\n", outPath.c_str(),
                    outPositions.size(), faceCounts.size(), minu, maxu, minv, maxv);
    } else {
        std::printf("wrote %s (%zu verts, %zu faces, no uvs)\n", outPath.c_str(), outPositions.size(),
                    faceCounts.size());
    }
    return 0;
}
