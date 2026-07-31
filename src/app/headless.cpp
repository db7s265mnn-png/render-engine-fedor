#include "app/headless.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QTextStream>
#include <cstdio>

#include "app/default_scene.h"
#include "app/document.h"
#include "core/log.h"
#include "io/image_io.h"
#include "nodes/node_registry.h"
#include "render/motion_blur.h"
#include "render/render_session.h"
#include "scene/tessellate.h"

namespace sol {

int runHeadless(const HeadlessOptions& options) {
    registerBuiltinNodes();
    setLogLevel(options.verbose ? LogLevel::Debug : LogLevel::Info);

    NodeGraph graph;
    if (!options.scenePath.isEmpty()) {
        QString error;
        if (!loadGraphFromFile(graph, options.scenePath, error)) {
            std::fprintf(stderr, "error: %s\n", error.toUtf8().constData());
            return 2;
        }
    } else if (!options.alembicPath.isEmpty()) {
        buildAlembicGraph(graph, options.alembicPath, options.hdriPath);
    } else {
        buildDefaultGraph(graph);
    }

    if (!options.saveScenePath.isEmpty()) {
        QString error;
        if (!saveGraphToFile(graph, options.saveScenePath, error)) {
            std::fprintf(stderr, "error: %s\n", error.toUtf8().constData());
            return 6;
        }
        std::fprintf(stderr, "Wrote %s\n", options.saveScenePath.toUtf8().constData());
        if (!options.renderImage) return 0;
    }

    CookContext context;
    if (!options.scenePath.isEmpty())
        context.sceneDirectory = QFileInfo(options.scenePath).absolutePath();
    StagePtr stage = graph.cookDisplay(context);
    if (!stage) {
        std::fprintf(stderr, "error: nothing to render\n");
        return 3;
    }
    for (const QString& warning : context.warnings) std::fprintf(stderr, "warning: %s\n", warning.toUtf8().constData());
    for (const QString& error : context.errors) std::fprintf(stderr, "error: %s\n", error.toUtf8().constData());

    ScenePtr scene = stage->toScene();
    if (options.width > 0) scene->settings.resolutionX = options.width;
    if (options.height > 0) scene->settings.resolutionY = options.height;
    if (options.samples > 0) scene->settings.samplesPerPixel = options.samples;
    if (options.backend >= 0) scene->settings.backend = options.backend;
    if (options.threads >= 0) scene->settings.threads = options.threads;

    if (scene->settings.motionBlur) attachMotionBlurKeys(graph, context, *scene);

    {
        CameraData diceCam = scene->camera;
        if (scene->settings.dicingCameraMode == kDicingCameraCustom &&
            !stage->dicingCameraPath.isEmpty()) {
            for (const StagePrim& prim : stage->prims) {
                if (prim.type != PrimType::Camera || prim.path != stage->dicingCameraPath) continue;
                diceCam = prim.camera;
                diceCam.cameraToWorld = prim.xform;
                break;
            }
        }
        tessellateSceneForRender(*scene, diceCam);
    }

    // tessellateSceneForRender calls finalize(); keep a defensive refresh if
    // tessellation was skipped (empty scene) so views still match meshes.
    scene->refreshMeshViews();

    std::fprintf(stderr, "Rendering %dx%d, %d spp, %zu triangles, %zu lights\n", scene->settings.resolutionX,
                 scene->settings.resolutionY, scene->settings.samplesPerPixel, scene->totalTriangles(),
                 scene->lights.size());

    RenderSession session;
    session.setScene(scene);

    QElapsedTimer timer;
    timer.start();
    int lastReported = -1;
    session.setUpdateCallback([&] {
        const RenderProgress progress = session.progress();
        const int percent = progress.samplesTarget > 0
                                ? int(100.0 * double(progress.samplesDone) / double(progress.samplesTarget))
                                : 0;
        if (percent / 5 != lastReported / 5) {
            lastReported = percent;
            std::fprintf(stderr, "\r  %3d%%  %d/%d samples  %.2f spp/s", percent, progress.samplesDone,
                         progress.samplesTarget, progress.samplesPerSecond);
            std::fflush(stderr);
        }
    });

    session.start();
    session.waitForCompletion();
    std::fprintf(stderr, "\r  100%%  done in %.2f s%20s\n", timer.elapsed() / 1000.0, "");

    const Image linear = session.linearImage();
    if (linear.empty()) {
        std::fprintf(stderr, "error: empty framebuffer\n");
        return 4;
    }
    std::string error;
    if (!saveImageAuto(options.outputPath.toStdString(), linear, scene->settings, error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 5;
    }
    std::fprintf(stderr, "Wrote %s\n", options.outputPath.toUtf8().constData());
    return 0;
}

}  // namespace sol
