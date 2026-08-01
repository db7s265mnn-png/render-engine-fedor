#include "io/alembic_loader.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_map>

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

Mat4 lerpMat4(const Mat4& a, const Mat4& b, float t) {
    Mat4 r;
    for (int i = 0; i < 16; ++i) r.m[i] = lerpf(a.m[i], b.m[i], t);
    return r;
}

struct SampleBlend {
    index_t i0 = 0;
    index_t i1 = 0;
    float alpha = 0.0f;  // 0 = i0, 1 = i1
};

bool sampleBlendAtTime(const TimeSamplingPtr& ts, size_t numSamples, double time, SampleBlend& out) {
    if (!ts || numSamples == 0) return false;
    if (numSamples == 1) {
        out = {0, 0, 0.0f};
        return true;
    }
    const auto floor = ts->getFloorIndex(time, index_t(numSamples));
    const auto ceil = ts->getCeilIndex(time, index_t(numSamples));
    out.i0 = floor.first;
    out.i1 = ceil.first;
    if (out.i0 == out.i1) {
        out.alpha = 0.0f;
        return true;
    }
    const double t0 = floor.second;
    const double t1 = ceil.second;
    if (t1 <= t0) {
        out.alpha = 0.0f;
        return true;
    }
    out.alpha = float(std::clamp((time - t0) / (t1 - t0), 0.0, 1.0));
    return true;
}

struct LoadContext {
    AlembicLoadOptions options;
    AlembicContents* out = nullptr;
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

MeshPtr buildMesh(const Imath::V3f* positions, size_t positionCount, const Int32ArraySamplePtr& faceCounts,
                  const Int32ArraySamplePtr& faceIndices, const std::vector<Imath::V3f>& cornerNormals,
                  const std::vector<Imath::V2f>& cornerUvs, float scale, bool hasNormals, bool hasUvs) {
    if (!positions || !faceCounts || !faceIndices || positionCount == 0 || faceIndices->size() == 0)
        return nullptr;

    auto mesh = std::make_shared<Mesh>();
    const size_t vertexCount = positionCount;
    size_t corner = 0;
    for (size_t f = 0; f < faceCounts->size(); ++f) {
        const int count = (*faceCounts)[f];
        if (count < 3) {
            corner += size_t(std::max(0, count));
            continue;
        }
        if (corner + size_t(count) > faceIndices->size()) break;
        const uint32_t base = uint32_t(mesh->positions.size());
        for (int i = 0; i < count; ++i) {
            const size_t c = corner + size_t(i);
            const size_t vertexIndex = size_t((*faceIndices)[c]);
            const Imath::V3f& p = positions[std::min(vertexIndex, vertexCount - 1)];
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
    const size_t numSamples = schema.getNumSamples();
    if (numSamples == 0) return nullptr;

    updateTimeRange(*context.out, schema.getTimeSampling(), numSamples);

    SampleBlend blend;
    if (!sampleBlendAtTime(schema.getTimeSampling(), numSamples, context.options.time, blend))
        return nullptr;

    typename SchemaT::Sample sample0;
    schema.get(sample0, ISampleSelector(blend.i0));
    const P3fArraySamplePtr positions0 = sample0.getPositions();
    const Int32ArraySamplePtr faceCounts = sample0.getFaceCounts();
    const Int32ArraySamplePtr faceIndices = sample0.getFaceIndices();
    if (!positions0 || !faceCounts || !faceIndices || positions0->size() == 0 || faceIndices->size() == 0)
        return nullptr;

    std::vector<Imath::V3f> lerpedPositions;
    const Imath::V3f* positionPtr = &(*positions0)[0];
    size_t positionCount = positions0->size();

    if (blend.alpha > 1e-5f && blend.i0 != blend.i1) {
        typename SchemaT::Sample sample1;
        schema.get(sample1, ISampleSelector(blend.i1));
        const P3fArraySamplePtr positions1 = sample1.getPositions();
        if (positions1 && positions1->size() == positions0->size()) {
            lerpedPositions.resize(positions0->size());
            for (size_t i = 0; i < positions0->size(); ++i) {
                const Imath::V3f& a = (*positions0)[i];
                const Imath::V3f& b = (*positions1)[i];
                lerpedPositions[i] = a + (b - a) * blend.alpha;
            }
            positionPtr = lerpedPositions.data();
            positionCount = lerpedPositions.size();
        }
    }

    std::vector<Imath::V3f> cornerNormals;
    std::vector<Imath::V2f> cornerUvs;
    bool hasNormals = false;
    bool hasUvs = false;
    // Topology / attributes follow the floor sample (stable under sub-frame blends).
    const ISampleSelector attrSelector(blend.i0);

    if (context.options.importNormals && !isSubd) {
        if constexpr (std::is_same_v<SchemaT, IPolyMeshSchema>) {
            IN3fGeomParam normalsParam = schema.getNormalsParam();
            hasNormals = expandGeomParam<IN3fGeomParam, Imath::V3f>(normalsParam, attrSelector,
                                                                    positionCount, faceIndices, cornerNormals);
        }
    }
    if (context.options.importUvs) {
        IV2fGeomParam uvParam = schema.getUVsParam();
        hasUvs = expandGeomParam<IV2fGeomParam, Imath::V2f>(uvParam, attrSelector, positionCount, faceIndices,
                                                            cornerUvs);
    }

    return buildMesh(positionPtr, positionCount, faceCounts, faceIndices, cornerNormals, cornerUvs,
                     context.options.scale, hasNormals, hasUvs);
}

Mat4 interpolatedLocalXform(IXformSchema& schema, double time) {
    const size_t numSamples = schema.getNumSamples();
    if (numSamples == 0) return Mat4::identity();
    SampleBlend blend;
    if (!sampleBlendAtTime(schema.getTimeSampling(), numSamples, time, blend))
        return fromImath(schema.getValue(ISampleSelector(index_t(0))).getMatrix());
    const Mat4 a = fromImath(schema.getValue(ISampleSelector(blend.i0)).getMatrix());
    if (blend.alpha <= 1e-5f || blend.i0 == blend.i1) return a;
    const Mat4 b = fromImath(schema.getValue(ISampleSelector(blend.i1)).getMatrix());
    return lerpMat4(a, b, blend.alpha);
}

void traverse(const IObject& object, const Mat4& parentTransform, bool parentXformAnimated,
              LoadContext& context) {
    const size_t childCount = object.getNumChildren();
    for (size_t i = 0; i < childCount; ++i) {
        IObject child(object, object.getChildHeader(i).getName());
        if (!child.valid()) continue;

        Mat4 transform = parentTransform;
        bool xformAnimated = parentXformAnimated;
        const ObjectHeader& header = child.getHeader();
        const std::string path = child.getFullName();

        if (IXform::matches(header)) {
            IXform xform(child, kWrapExisting);
            IXformSchema& schema = xform.getSchema();
            updateTimeRange(*context.out, schema.getTimeSampling(), schema.getNumSamples());
            if (schema.getNumSamples() > 1) xformAnimated = true;
            const Mat4 local = interpolatedLocalXform(schema, context.options.time);
            transform = schema.getInheritsXforms() ? parentTransform * local : local;
        } else if (IPolyMesh::matches(header)) {
            IPolyMesh polyMesh(child, kWrapExisting);
            IPolyMeshSchema& schema = polyMesh.getSchema();
            if (context.options.pathFilter.empty() || globMatch(context.options.pathFilter, path)) {
                const bool meshAnimated = schema.getNumSamples() > 1;
                MeshPtr mesh = readPolygonSchema(schema, context, false);
                if (mesh) {
                    mesh->name = child.getName();
                    mesh->timeDependent = meshAnimated || xformAnimated;
                    mesh->validate();
                    AlembicPrim prim;
                    prim.path = path;
                    prim.name = child.getName();
                    prim.mesh = std::move(mesh);
                    prim.transform = transform;
                    prim.timeDependent = meshAnimated || xformAnimated;
                    context.out->prims.push_back(std::move(prim));
                } else {
                    ++context.skipped;
                }
            }
        } else if (ISubD::matches(header)) {
            ISubD subd(child, kWrapExisting);
            ISubDSchema& schema = subd.getSchema();
            if (context.options.pathFilter.empty() || globMatch(context.options.pathFilter, path)) {
                const bool meshAnimated = schema.getNumSamples() > 1;
                MeshPtr mesh = readPolygonSchema(schema, context, true);
                if (mesh) {
                    mesh->name = child.getName();
                    mesh->timeDependent = meshAnimated || xformAnimated;
                    mesh->validate();
                    AlembicPrim prim;
                    prim.path = path;
                    prim.name = child.getName();
                    prim.mesh = std::move(mesh);
                    prim.transform = transform;
                    prim.isSubd = true;
                    prim.timeDependent = meshAnimated || xformAnimated;
                    context.out->prims.push_back(std::move(prim));
                } else {
                    ++context.skipped;
                }
            }
        } else if (ICurves::matches(header) || IPoints::matches(header) || INuPatch::matches(header)) {
            logWarning("Alembic: skipping unsupported prim " + path);
            ++context.skipped;
        }

        traverse(child, transform, xformAnimated, context);
    }
}

}  // namespace

bool alembicSupportAvailable() { return true; }

bool loadAlembic(const std::string& filePath, const AlembicLoadOptions& options, AlembicContents& out,
                 std::string& error) {
    try {
        // Keep archives open so timeline scrubbing does not reopen the file every frame.
        static std::mutex archiveMutex;
        static std::unordered_map<std::string, IArchive> archives;

        IArchive archive;
        {
            std::lock_guard<std::mutex> lock(archiveMutex);
            auto it = archives.find(filePath);
            if (it != archives.end() && it->second.valid()) {
                archive = it->second;
            } else {
                Alembic::AbcCoreFactory::IFactory factory;
                factory.setPolicy(Alembic::Abc::ErrorHandler::kThrowPolicy);
                Alembic::AbcCoreFactory::IFactory::CoreType coreType;
                archive = factory.getArchive(filePath, coreType);
                if (!archive.valid()) {
                    error = "cannot open Alembic archive " + filePath;
                    return false;
                }
                archives[filePath] = archive;
            }
        }

        LoadContext context;
        context.options = options;
        context.out = &out;

        out.archiveInfo = archive.getName();
        traverse(archive.getTop(), Mat4::identity(), false, context);

        if (out.prims.empty()) {
            error = "no polygonal geometry found in " + filePath;
            return false;
        }
        static thread_local std::string lastLogged;
        const std::string logKey = filePath + "|" + std::to_string(out.prims.size());
        if (lastLogged != logKey) {
            lastLogged = logKey;
            logInfo("Alembic: loaded " + std::to_string(out.prims.size()) + " prims from " + filePath +
                    (context.skipped ? " (" + std::to_string(context.skipped) + " skipped)" : ""));
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("Alembic: ") + e.what();
        return false;
    }
}

#endif  // SOLSTICE_HAVE_ALEMBIC

}  // namespace sol
