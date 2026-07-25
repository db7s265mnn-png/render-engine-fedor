// Minimal OBJ -> Alembic (Ogawa) converter for shipping demo assets.
#include <Alembic/Abc/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace Alembic::AbcGeom;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s input.obj output.abc [name]\n", argv[0]);
        return 1;
    }
    const std::string inPath = argv[1];
    const std::string outPath = argv[2];
    const std::string name = argc > 3 ? argv[3] : "buddha";

    std::ifstream in(inPath);
    if (!in) {
        std::fprintf(stderr, "failed to open %s\n", inPath.c_str());
        return 1;
    }

    std::vector<Imath::V3f> positions;
    std::vector<int32_t> faceCounts;
    std::vector<int32_t> faceIndices;
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
        } else if (tag == "f") {
            std::vector<int32_t> face;
            std::string tok;
            while (ss >> tok) {
                const auto slash = tok.find('/');
                const std::string idx = slash == std::string::npos ? tok : tok.substr(0, slash);
                int v = std::stoi(idx);
                if (v < 0) v = int(positions.size()) + v + 1;
                face.push_back(v - 1);
            }
            if (face.size() < 3) continue;
            // Triangulate fan.
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                faceCounts.push_back(3);
                faceIndices.push_back(face[0]);
                faceIndices.push_back(face[i]);
                faceIndices.push_back(face[i + 1]);
            }
        }
    }
    if (positions.empty() || faceCounts.empty()) {
        std::fprintf(stderr, "no mesh data in %s\n", inPath.c_str());
        return 1;
    }

    // Normalize to sit on the ground roughly 2 units tall.
    Imath::V3f bmin = positions[0], bmax = positions[0];
    for (const auto& p : positions) {
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
    for (auto& p : positions) {
        p = (p - center) * scale;
    }

    OArchive archive(Alembic::AbcCoreOgawa::WriteArchive(), outPath);
    OObject top = archive.getTop();
    OXform xform(top, name + "_xform");
    OPolyMesh meshObj(xform, name);
    OPolyMeshSchema& schema = meshObj.getSchema();
    OPolyMeshSchema::Sample sample{V3fArraySample(positions), Int32ArraySample(faceIndices),
                                   Int32ArraySample(faceCounts)};
    schema.set(sample);
    std::printf("wrote %s (%zu verts, %zu faces)\n", outPath.c_str(), positions.size(),
                faceCounts.size());
    return 0;
}
