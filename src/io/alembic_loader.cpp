#include "io/alembic_loader.h"

#include <algorithm>

#include "core/log.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_ALEMBIC
#  include <Alembic/Abc/All.h>
#  include <Alembic/AbcCoreFactory/All.h>
#  include <Alembic/AbcGeom/All.h>
#endif

namespace sol {

bool globMatch(const std::string& pattern, const std::string& text) {
    if (pattern.empty()) return true;
    // Iterative wildcard matcher supporting '*' and '?'.
    size_t p = 0, t = 0, starP = std::string::npos, starT = 0;
    while (t < text.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
            ++p;
            ++t;
        } else if (p < pattern.size() && pattern[p] == '*') {
            starP = p++;
            starT = t;
        } else if (starP != std::string::npos) {
            p = starP + 1;
            t = ++starT;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') ++p;
    return p == pattern.size();
}

#if !SOLSTICE_HAVE_ALEMBIC

bool alembicSupportAvailable() { return false; }

bool loadAlembic(const std::string&, const AlembicLoadOptions&, AlembicContents&, std::string& error) {
    error = "this build was compiled without Alembic support";
    return false;
}

#else

namespace {

using namespace Alembic::AbcGeom;

// Imath matrices use row vectors, ours use column vectors.
Mat4 fromImath(const Imath::M44d& m) {
    Mat4 r;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col) r.at(row, col) = float(m.x[col][row]);
    return r;
}

struct LoadContext {
    AlembicLoadOptions options;
    AlembicContents* out = nullptr;
    ISampleSelector selector;
    int skipped = 0;
};

void updateTimeRange(AlembicContents& contents, const TimeSamplingPtr& ts, size_t numSamples) {
    if (!ts || numSamples <= 1) return;
    const double start = ts->getSampleTime(0);
    const double end = ts->getSampleTime(numSamples - 1);
    if (!contents.animated) {
        contents.startTime = start;
        contents.endTime = end;
        contents.animated = true;
    } else {
        contents.startTime = std::min(contents.startTime, start);
        contents.endTime = std::max(contents.endTime, end);
    }
}

// Reads a per-face-vertex or per-vertex vec3 parameter into a corner indexed
// array. Returns false when the parameter is unusable.
template <typename ParamT, typename ValueT>
bool expandGeomParam(const ParamT& param, const ISampleSelector& selector, size_t vertexCount,
                     const Int32ArraySamplePtr& faceIndices, std::vector<ValueT>& perCorner) {
    if (!param.valid()) return false;
    typename ParamT::Sample sample;
    param.getIndexed(sample, selector);
    auto values = sample.getVals();
    auto indices = sample.getIndices();
    if (!values || values->size() == 0) return false;

    const size_t cornerCount = faceIndices->size();
    perCorner.resize(cornerCount);
    const GeometryScope scope = param.getScope();

    for (size_t c = 0; c < cornerCount; ++c) {
        size_t sourceIndex = 0;
        if (scope == kFacevaryingScope || scope == kVaryingScope) {
            sourceIndex = indices && indices->size() == cornerCount ? size_t((*indices)[c]) : c;
        } else if (scope == kVertexScope) {
            const size_t vertex = size_t((*faceIndices)[c]);
            sourceIndex = indices && indices->size() == vertexCount ? size_t((*indices)[vertex]) : vertex;
        } else if (scope == kConstantScope) {
            sourceIndex = 0;
        } else {
            return false;
        }
        if (sourceIndex >= values->size()) sourceIndex = values->size() - 1;
        perCorner[c] = (*values)[sourceIndex];
    }
    return true;
}

MeshPtr buildMesh(const P3fArraySamplePtr& positions, const Int32ArraySamplePtr& faceCounts,
                  const Int32ArraySamplePtr& faceIndices, const std::vector<Imath::V3f>& cornerNormals,
                  const std::vector<Imath::V2f>& cornerUvs, float scale, bool hasNormals, bool hasUvs) {
    auto mesh = std::make_shared<Mesh>();
    const size_t vertexCount = positions->size();
    const size_t faceCount = faceCounts->size();

    if (!hasNormals && !hasUvs) {
        // Shared vertices; normals are computed later.
        mesh->positions.reserve(vertexCount);
        for (size_t i = 0; i < vertexCount; ++i) {
            const Imath::V3f& p = (*positions)[i];
            mesh->positions.emplace_back(p.x * scale, p.y * scale, p.z * scale);
        }
        size_t corner = 0;
        for (size_t f = 0; f < faceCount; ++f) {
            const int count = (*faceCounts)[f];
            if (count < 3) {
                corner += size_t(std::max(0, count));
                continue;
            }
            // Alembic stores faces clockwise when viewed from the front, so the
            // fan is emitted in reverse to get counter-clockwise triangles.
            for (int i = 1; i + 1 < count; ++i) {
                mesh->indices.push_back(uint32_t((*faceIndices)[corner]));
                mesh->indices.push_back(uint32_t((*faceIndices)[corner + i + 1]));
                mesh->indices.push_back(uint32_t((*faceIndices)[corner + i]));
            }
            corner += size_t(count);
        }
        return mesh;
    }

    // Per-corner attributes require unique vertices per face corner.
    size_t corner = 0;
    for (size_t f = 0; f < faceCount; ++f) {
        const int count = (*faceCounts)[f];
        if (count < 3) {
            corner += size_t(std::max(0, count));
            continue;
        }
        const uint32_t base = uint32_t(mesh->positions.size());
        for (int i = 0; i < count; ++i) {
            const size_t c = corner + size_t(i);
            const size_t vertexIndex = size_t((*faceIndices)[c]);
            const Imath::V3f& p = (*positions)[std::min(vertexIndex, vertexCount - 1)];
            mesh->positions.emplace_back(p.x * scale, p.y * scale, p.z * scale);
            if (hasNormals && c < cornerNormals.size()) {
                const Imath::V3f& n = cornerNormals[c];
                mesh->normals.emplace_back(n.x, n.y, n.z);
            }
            if (hasUvs && c < cornerUvs.size()) {
                const Imath::V2f& uv = cornerUvs[c];
                mesh->uvs.emplace_back(uv.x, uv.y);
            }
        }
        for (int i = 1; i + 1 < count; ++i) {
            mesh->indices.push_back(base);
            mesh->indices.push_back(base + uint32_t(i + 1));
            mesh->indices.push_back(base + uint32_t(i));
        }
        corner += size_t(count);
    }
    if (mesh->normals.size() != mesh->positions.size()) mesh->normals.clear();
    if (mesh->uvs.size() != mesh->positions.size()) mesh->uvs.clear();
    return mesh;
}

template <typename SchemaT>
MeshPtr readPolygonSchema(SchemaT& schema, LoadContext& context, bool isSubd) {
    typename SchemaT::Sample sample;
    schema.get(sample, context.selector);
    const P3fArraySamplePtr positions = sample.getPositions();
    const Int32ArraySamplePtr faceCounts = sample.getFaceCounts();
    const Int32ArraySamplePtr faceIndices = sample.getFaceIndices();
    if (!positions || !faceCounts || !faceIndices || positions->size() == 0 || faceIndices->size() == 0)
        return nullptr;

    updateTimeRange(*context.out, schema.getTimeSampling(), schema.getNumSamples());

    std::vector<Imath::V3f> cornerNormals;
    std::vector<Imath::V2f> cornerUvs;
    bool hasNormals = false;
    bool hasUvs = false;

    if (context.options.importNormals && !isSubd) {
        if constexpr (std::is_same_v<SchemaT, IPolyMeshSchema>) {
            IN3fGeomParam normalsParam = schema.getNormalsParam();
            hasNormals = expandGeomParam<IN3fGeomParam, Imath::V3f>(normalsParam, context.selector,
                                                                    positions->size(), faceIndices, cornerNormals);
        }
    }
    if (context.options.importUvs) {
        IV2fGeomParam uvParam = schema.getUVsParam();
        hasUvs = expandGeomParam<IV2fGeomParam, Imath::V2f>(uvParam, context.selector, positions->size(),
                                                            faceIndices, cornerUvs);
    }

    return buildMesh(positions, faceCounts, faceIndices, cornerNormals, cornerUvs, context.options.scale,
                     hasNormals, hasUvs);
}

void traverse(const IObject& object, const Mat4& parentTransform, LoadContext& context) {
    const size_t childCount = object.getNumChildren();
    for (size_t i = 0; i < childCount; ++i) {
        IObject child(object, object.getChildHeader(i).getName());
        if (!child.valid()) continue;

        Mat4 transform = parentTransform;
        const ObjectHeader& header = child.getHeader();
        const std::string path = child.getFullName();

        if (IXform::matches(header)) {
            IXform xform(child, kWrapExisting);
            IXformSchema& schema = xform.getSchema();
            updateTimeRange(*context.out, schema.getTimeSampling(), schema.getNumSamples());
            const XformSample sample = schema.getValue(context.selector);
            const Mat4 local = fromImath(sample.getMatrix());
            transform = schema.getInheritsXforms() ? parentTransform * local : local;
        } else if (IPolyMesh::matches(header)) {
            IPolyMesh polyMesh(child, kWrapExisting);
            IPolyMeshSchema& schema = polyMesh.getSchema();
            if (context.options.pathFilter.empty() || globMatch(context.options.pathFilter, path)) {
                MeshPtr mesh = readPolygonSchema(schema, context, false);
                if (mesh) {
                    mesh->name = child.getName();
                    mesh->validate();
                    AlembicPrim prim;
                    prim.path = path;
                    prim.name = child.getName();
                    prim.mesh = std::move(mesh);
                    prim.transform = transform;
                    context.out->prims.push_back(std::move(prim));
                } else {
                    ++context.skipped;
                }
            }
        } else if (ISubD::matches(header)) {
            ISubD subd(child, kWrapExisting);
            ISubDSchema& schema = subd.getSchema();
            if (context.options.pathFilter.empty() || globMatch(context.options.pathFilter, path)) {
                MeshPtr mesh = readPolygonSchema(schema, context, true);
                if (mesh) {
                    mesh->name = child.getName();
                    mesh->validate();
                    AlembicPrim prim;
                    prim.path = path;
                    prim.name = child.getName();
                    prim.mesh = std::move(mesh);
                    prim.transform = transform;
                    prim.isSubd = true;
                    context.out->prims.push_back(std::move(prim));
                } else {
                    ++context.skipped;
                }
            }
        } else if (ICurves::matches(header) || IPoints::matches(header) || INuPatch::matches(header)) {
            logWarning("Alembic: skipping unsupported prim " + path);
            ++context.skipped;
        }

        traverse(child, transform, context);
    }
}

}  // namespace

bool alembicSupportAvailable() { return true; }

bool loadAlembic(const std::string& filePath, const AlembicLoadOptions& options, AlembicContents& out,
                 std::string& error) {
    try {
        Alembic::AbcCoreFactory::IFactory factory;
        factory.setPolicy(Alembic::Abc::ErrorHandler::kThrowPolicy);
        Alembic::AbcCoreFactory::IFactory::CoreType coreType;
        IArchive archive = factory.getArchive(filePath, coreType);
        if (!archive.valid()) {
            error = "cannot open Alembic archive " + filePath;
            return false;
        }

        LoadContext context;
        context.options = options;
        context.out = &out;
        context.selector = ISampleSelector(options.time, ISampleSelector::kNearIndex);

        out.archiveInfo = archive.getName();
        traverse(archive.getTop(), Mat4::identity(), context);

        if (out.prims.empty()) {
            error = "no polygonal geometry found in " + filePath;
            return false;
        }
        logInfo("Alembic: loaded " + std::to_string(out.prims.size()) + " prims from " + filePath +
                (context.skipped ? " (" + std::to_string(context.skipped) + " skipped)" : ""));
        return true;
    } catch (const std::exception& e) {
        error = std::string("Alembic: ") + e.what();
        return false;
    }
}

#endif  // SOLSTICE_HAVE_ALEMBIC

}  // namespace sol
