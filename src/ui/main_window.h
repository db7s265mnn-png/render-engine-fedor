// Main application window: network editor, render view, parameters, scene
// graph tree and log, wired to the cook and render pipeline.
#pragma once

#include <QMainWindow>
#include <QString>
#include <memory>

#include "nodes/node_graph.h"
#include "render/render_session.h"
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

private:
    void createActions();
    void createMenus();
    void createToolBar();
    void createDocks();
    void scheduleCook(int delayMilliseconds = 120);
    void cookNow();
    void restartRender();
    void updateWindowTitle();
    void updateStatusBar();
    bool maybeSaveChanges();
    Node* findCameraNode() const;
    void selectDisplayNode();

    NodeGraph graph_;
    RenderSession session_;
    StagePtr stage_;
    ScenePtr scene_;

    NodeGraphView* networkView_ = nullptr;
    MaterialNetworkView* materialNetworkView_ = nullptr;
    ParameterPanel* parameterPanel_ = nullptr;
    SceneGraphPanel* sceneGraphPanel_ = nullptr;
    LogPanel* logPanel_ = nullptr;
    RenderView* renderView_ = nullptr;

    QTimer* cookTimer_ = nullptr;
    QTimer* refreshTimer_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QLabel* backendLabel_ = nullptr;

    QAction* renderAction_ = nullptr;
    QAction* stopAction_ = nullptr;
    QAction* iprAction_ = nullptr;
    QAction* selectToolAction_ = nullptr;
    QAction* translateToolAction_ = nullptr;
    QAction* rotateToolAction_ = nullptr;
    QAction* scaleToolAction_ = nullptr;

    std::atomic<bool> framePending_{false};
    bool cameraOverride_ = false;
    bool renderRequested_ = false;
};

}  // namespace sol
