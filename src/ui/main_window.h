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
    void applyTessellationCache(Scene& scene) const;
    void storeTessellationCache(const Scene& scene);
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
    void selectDisplayNode();

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

    std::atomic<bool> framePending_{false};
    bool cameraOverride_ = false;
    bool renderRequested_ = false;
    // Empty = free persp; otherwise Scene Network camera node name (look-through).
    QString lookThroughCameraName_;
    // Source LOP node name for viewport F / framing (survives Select-tool picks).
    QString selectedSourceNode_;
    // Throttle IPR restarts while tumbling with motion blur enabled.
    qint64 lastMbNavIprRestartMs_ = 0;
    bool mbNavIprPending_ = false;
};

}  // namespace sol
