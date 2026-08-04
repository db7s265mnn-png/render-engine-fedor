// Command line rendering, used for batch renders and by the test suite.
#pragma once

#include <QString>

namespace sol {

struct HeadlessOptions {
    QString scenePath;      // .scene network, optional
    QString alembicPath;    // used when no scene file is given
    QString hdriPath;       // dome light texture for the generated network
    QString outputPath = "render.png";
    QString saveScenePath;  // writes the resulting network as a .scene file
    bool renderImage = true;
    int samples = 0;        // 0 keeps the value from the render settings node
    int width = 0;
    int height = 0;
    int backend = -1;       // -1 keeps the scene value, 0 CPU, 1 GPU
    int threads = -1;
    bool verbose = false;
};

// Returns a process exit code.
int runHeadless(const HeadlessOptions& options);

}  // namespace sol
