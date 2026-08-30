// Main application window: network editor, render view, parameters, scene
// graph tree and log, wired to the cook and render pipeline.
#pragma once

#include <QMainWindow>
#include <QString>
#include <QStringList>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nodes/node_graph.h"
#include "nodes/stage.h"
#include "render/render_session.h"
#include "scene/scene.h"
#include "app/undo_hub.h"
#include "ui/render_view.h"

class QAction;
class QLabel;
class QTimer;

namespace sol {

class NodeGraphView;
class MaterialNetworkView;
class ParameterPanel;
class SceneGraphPanel;
class LogPanel;
class TimelineBar;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    void newScene();
    void newSceneFromAlembic(const QString& alembicPath, const QString& hdriPath);
    bool openScene(const QString& path);
    bool saveScene(const QString& path);

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onNewScene();
    void onOpenScene();
    void onSaveScene();
    void onSaveSceneAs();
    void onImportAlembic();
    void onImportUsd();
    void onSaveImage();
    void onStartRender();
    void onStopRender();
    void onCookTimeout();
    void onRenderTick();
    void onCameraMoved();
    void onLookThroughCameraNode();
    void onCopyViewToCameraNode();
    void onShowAbout();
    void onShowShortcuts();
    void onTimelineFrameChanged(int frame);
    void onParameterAction(sol::Node* node, const QString& parameterName);
    void maybeSaveStillFrame();
    void onRenderFinished();

private:
    void createActions();
    void createMenus();
    void createToolBar();
    void createTimeline();
    void createDocks();
    void scheduleCook(int delayMilliseconds = 120);
    void cookNow();
    void restartRender();
    void enterIdlePlaceholder();
    void applyTessellationCache(Scene& scene, bool skipTimeDependent = false) const;
    void storeTessellationCache(const Scene& scene);
    void mergeTessellationCache(const Scene& scene);
    CameraData resolveDicingCamera(const Scene& scene) const;
    // Start pressed (Stop not): edits may cook + restart the session.
    bool renderArmed() const;
    void setRenderArmed(bool armed);
    void updateWindowTitle();
    void updateStatusBar();
    bool maybeSaveChanges();
    Node* findCameraNode() const;
    Node* findCameraNodeByName(const QString& name) const;
    QStringList listCameraNodeNames() const;
    void refreshViewportCameraMenu();
    void lookThroughCamera(const QString& cameraName);
    void applyCameraNodeToView(Node* camera);
    void applyLensFromCameraNode(const Node* camera, CameraData& out) const;
    void writeViewToCameraNode(Node* camera);
    void captureOrbitCameraState(OrbitCameraState& out) const;
    void applyOrbitCameraState(const OrbitCameraState& state);
    void refreshAfterUndo();
    void rebuildGraphAfterUndo();
    void selectDisplayNode();
    void startStillFrameRender(Node* renderSettings);

    NodeGraph graph_;
    RenderSession session_;
    StagePtr stage_;
    ScenePtr scene_;
    // Last Render tessellation, keyed by prim path — kept until the next Render.
    std::vector<std::pair<std::string, MeshPtr>> tessCache_;
    // Fingerprint of authored tess inputs used to build tessCache_.
    std::string tessCacheFingerprint_;

    NodeGraphView* networkView_ = nullptr;
    MaterialNetworkView* materialNetworkView_ = nullptr;
    ParameterPanel* parameterPanel_ = nullptr;
    SceneGraphPanel* sceneGraphPanel_ = nullptr;
    LogPanel* logPanel_ = nullptr;
    RenderView* renderView_ = nullptr;
    TimelineBar* timelineBar_ = nullptr;

    QTimer* cookTimer_ = nullptr;
    QTimer* refreshTimer_ = nullptr;
    QAction* renderAction_ = nullptr;
    QAction* stopAction_ = nullptr;
    QAction* selectToolAction_ = nullptr;
    QAction* translateToolAction_ = nullptr;
    QAction* rotateToolAction_ = nullptr;
    QAction* scaleToolAction_ = nullptr;
    QAction* undoAction_ = nullptr;
    QAction* redoAction_ = nullptr;

    UndoHub undoHub_;
    OrbitCameraState cameraUndoStart_;

    std::atomic<bool> framePending_{false};
    bool cameraOverride_ = false;
    bool renderRequested_ = false;
    // Image → Render: wait for a full spp pass, then write tonemapped EXR.
    bool stillFramePending_ = false;
    QString stillFramePath_;
    // Set by onTimelineFrameChanged; cookNow re-dices only timeDependent meshes.
    bool timelineFrameCook_ = false;
    // Empty = free persp; otherwise Scene Network camera node name (look-through).
    QString lookThroughCameraName_;
    // Source LOP node name for viewport F / framing (survives Select-tool picks).
    QString selectedSourceNode_;
};

}  // namespace sol
