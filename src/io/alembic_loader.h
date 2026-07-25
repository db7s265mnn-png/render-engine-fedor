// Alembic (.abc) geometry import.
#pragma once

#include <string>
#include <vector>

#include "scene/scene.h"

namespace sol {

struct AlembicLoadOptions {
    double time = 0.0;            // seconds; snapped to the nearest sample
    bool useSubdivision = false;  // ISubD is loaded as its polygon cage
    bool importNormals = true;
    bool importUvs = true;
    float scale = 1.0f;
    std::string pathFilter;  // simple glob, empty means everything
};

struct AlembicPrim {
    std::string path;    // e.g. /root/geo/pigShape
    std::string name;
    MeshPtr mesh;
    Mat4 transform;      // world transform at the requested time
    bool isSubd = false;
};

struct AlembicContents {
    std::vector<AlembicPrim> prims;
    double startTime = 0.0;
    double endTime = 0.0;
    bool animated = false;
    std::string archiveInfo;
};

// Returns false and fills `error` when the archive cannot be read. Individual
// unsupported prims are skipped with a warning.
bool loadAlembic(const std::string& filePath, const AlembicLoadOptions& options, AlembicContents& out,
                 std::string& error);

// True when this build links against Alembic.
bool alembicSupportAvailable();

// Simple '*' / '?' glob matching used for prim path filters everywhere in the
// node graph.
bool globMatch(const std::string& pattern, const std::string& text);

}  // namespace sol
