// Writes a small Alembic archive used to exercise the importer:
// a transformed cube, a subdivision plane and a UV sphere under an xform tree.
#include <Alembic/Abc/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace Alembic::AbcGeom;

namespace {

struct PolyData {
    std::vector<Imath::V3f> positions;
    std::vector<int32_t> faceCounts;
    std::vector<int32_t> faceIndices;
    std::vector<Imath::V2f> uvs;  // per face corner
};

PolyData makeCube(float size) {
    PolyData data;
    const float h = size * 0.5f;
    data.positions = {{-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {-h, -h, h},
                      {-h, h, -h},  {h, h, -h},  {h, h, h},  {-h, h, h}};
    const int faces[6][4] = {{0, 1, 2, 3}, {4, 7, 6, 5}, {0, 4, 5, 1}, {1, 5, 6, 2}, {2, 6, 7, 3}, {3, 7, 4, 0}};
    for (const auto& face : faces) {
        data.faceCounts.push_back(4);
        // Alembic expects clockwise winding when seen from the front.
        for (int i = 3; i >= 0; --i) data.faceIndices.push_back(face[i]);
        data.uvs.insert(data.uvs.end(), {{0, 0}, {1, 0}, {1, 1}, {0, 1}});
    }
    return data;
}

PolyData makeGrid(float size, int divisions) {
    PolyData data;
    for (int z = 0; z <= divisions; ++z) {
        for (int x = 0; x <= divisions; ++x) {
            const float tx = float(x) / float(divisions);
            const float tz = float(z) / float(divisions);
            data.positions.emplace_back((tx - 0.5f) * size, 0.0f, (tz - 0.5f) * size);
        }
    }
    const int stride = divisions + 1;
    for (int z = 0; z < divisions; ++z) {
        for (int x = 0; x < divisions; ++x) {
            data.faceCounts.push_back(4);
            const int32_t i0 = z * stride + x;
            const int32_t i1 = z * stride + x + 1;
            const int32_t i2 = (z + 1) * stride + x + 1;
            const int32_t i3 = (z + 1) * stride + x;
            data.faceIndices.insert(data.faceIndices.end(), {i3, i2, i1, i0});
            data.uvs.insert(data.uvs.end(), {{0, 0}, {1, 0}, {1, 1}, {0, 1}});
        }
    }
    return data;
}

PolyData makeSphere(float radius, int segmentsU, int segmentsV) {
    PolyData data;
    for (int v = 0; v <= segmentsV; ++v) {
        const float theta = float(v) / float(segmentsV) * float(M_PI);
        for (int u = 0; u < segmentsU; ++u) {
            const float phi = float(u) / float(segmentsU) * 2.0f * float(M_PI);
            data.positions.emplace_back(std::sin(theta) * std::cos(phi) * radius, std::cos(theta) * radius,
                                        std::sin(theta) * std::sin(phi) * radius);
        }
    }
    for (int v = 0; v < segmentsV; ++v) {
        for (int u = 0; u < segmentsU; ++u) {
            const int32_t i0 = v * segmentsU + u;
            const int32_t i1 = v * segmentsU + (u + 1) % segmentsU;
            const int32_t i2 = (v + 1) * segmentsU + (u + 1) % segmentsU;
            const int32_t i3 = (v + 1) * segmentsU + u;
            data.faceCounts.push_back(4);
            data.faceIndices.insert(data.faceIndices.end(), {i3, i2, i1, i0});
            data.uvs.insert(data.uvs.end(), {{0, 0}, {1, 0}, {1, 1}, {0, 1}});
        }
    }
    return data;
}

void writePolyMesh(OObject& parent, const std::string& name, const PolyData& data) {
    OPolyMesh mesh(parent, name);
    OPolyMeshSchema& schema = mesh.getSchema();
    OV2fGeomParam::Sample uvSample(V2fArraySample(data.uvs.data(), data.uvs.size()), kFacevaryingScope);
    OPolyMeshSchema::Sample sample(V3fArraySample(data.positions.data(), data.positions.size()),
                                   Int32ArraySample(data.faceIndices.data(), data.faceIndices.size()),
                                   Int32ArraySample(data.faceCounts.data(), data.faceCounts.size()), uvSample);
    schema.set(sample);
}

void writeSubD(OObject& parent, const std::string& name, const PolyData& data) {
    OSubD subd(parent, name);
    OSubDSchema& schema = subd.getSchema();
    OSubDSchema::Sample sample(V3fArraySample(data.positions.data(), data.positions.size()),
                               Int32ArraySample(data.faceIndices.data(), data.faceIndices.size()),
                               Int32ArraySample(data.faceCounts.data(), data.faceCounts.size()));
    schema.set(sample);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "test_scene.abc";

    OArchive archive(Alembic::AbcCoreOgawa::WriteArchive(), path);
    OObject top = archive.getTop();

    OXform rootXform(top, "root");
    OObject rootObject = rootXform;
    {
        XformSample sample;
        rootXform.getSchema().set(sample);
    }

    OXform cubeXform(rootObject, "cube_xform");
    {
        XformSample sample;
        sample.setTranslation(Imath::V3d(-1.6, 0.75, 0.0));
        sample.setRotation(Imath::V3d(0.0, 1.0, 0.0), 25.0);
        cubeXform.getSchema().set(sample);
    }
    OObject cubeObject = cubeXform;
    writePolyMesh(cubeObject, "cube", makeCube(1.5f));

    OXform sphereXform(rootObject, "sphere_xform");
    {
        XformSample sample;
        sample.setTranslation(Imath::V3d(1.5, 1.0, 0.3));
        sphereXform.getSchema().set(sample);
    }
    OObject sphereObject = sphereXform;
    writePolyMesh(sphereObject, "sphere", makeSphere(1.0f, 48, 24));

    writeSubD(rootObject, "ground", makeGrid(12.0f, 4));

    std::printf("Wrote %s\n", path.c_str());
    return 0;
}
