#include "ui/main_window.h"

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QCloseEvent>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTimer>
#include <QToolBar>
#include <QVector3D>

#include "app/default_scene.h"
#include "app/document.h"
#include "core/log.h"
#include "core/math.h"
#include "io/image_io.h"
#include "nodes/node_registry.h"
#include "render/scene_picker.h"
#include "scene/scene.h"
#include "solstice_config.h"
#include "ui/log_panel.h"
#include "ui/material_network_view.h"
#include "ui/node_graph_view.h"
#include "ui/parameter_panel.h"
#include "ui/scene_graph_panel.h"
#include "ui/theme.h"

namespace sol {
namespace {

QImage toQImage(const Image& image) {
    if (image.empty()) return {};
    QImage result(image.width(), image.height(), QImage::Format_RGB888);
    for (int y = 0; y < image.height(); ++y) {
        uchar* line = result.scanLine(y);
        for (int x = 0; x < image.width(); ++x) {
            const Vec3 c = image.rgb(x, y);
            uchar* px = line + size_t(x) * 3;
            px[0] = uchar(clampf(c.x, 0.0f, 1.0f) * 255.0f + 0.5f);
            px[1] = uchar(clampf(c.y, 0.0f, 1.0f) * 255.0f + 0.5f);
            px[2] = uchar(clampf(c.z, 0.0f, 1.0f) * 255.0f + 0.5f);
        }
    }
    return result;
}

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    registerBuiltinNodes();
    setWindowTitle(SOLSTICE_APP_NAME);
    resize(1720, 1000);

    renderView_ = new RenderView(this);
    setCentralWidget(renderView_);

    createActions();
    createDocks();
    createMenus();
    createToolBar();

    statusLabel_ = new QLabel("Ready");
    backendLabel_ = new QLabel("CPU / Embree 4");
    statusBar()->addWidget(statusLabel_, 1);
    statusBar()->addPermanentWidget(backendLabel_);

    cookTimer_ = new QTimer(this);
    cookTimer_->setSingleShot(true);
    connect(cookTimer_, &QTimer::timeout, this, &MainWindow::onCookTimeout);

    refreshTimer_ = new QTimer(this);
    refreshTimer_->setInterval(66);
    connect(refreshTimer_, &QTimer::timeout, this, &MainWindow::onRenderTick);
    refreshTimer_->start();

    connect(&graph_, &NodeGraph::graphChanged, this, [this] {
        updateWindowTitle();
        scheduleCook();
    });
    connect(&graph_, &NodeGraph::displayNodeChanged, this, [this](Node*) { scheduleCook(0); });

    connect(renderView_, &RenderView::cameraMoved, this, &MainWindow::onCameraMoved);

    renderView_->setPickCallback([this](float u, float v, Vec3& hit) -> bool {
        if (!scene_ || scene_->instances.empty()) return false;
        CameraData camera = scene_->camera;
        camera.cameraToWorld = renderView_->camera().toMatrix();
        camera.focusDistance = renderView_->camera().distance;
        return pickSceneSurface(scene_, camera, scene_->settings.resolutionX, scene_->settings.resolutionY, u, v,
                                hit);
    });

    renderView_->setObjectPickCallback([this](float u, float v, QString& sourceNode) -> bool {
        sourceNode.clear();
        if (!scene_ || scene_->instances.empty()) return false;
        CameraData camera = scene_->camera;
        camera.cameraToWorld = renderView_->camera().toMatrix();
        camera.focusDistance = renderView_->camera().distance;
        Vec3 hit;
        int instanceIndex = -1;
        if (!pickSceneSurface(scene_, camera, scene_->settings.resolutionX, scene_->settings.resolutionY, u, v,
                              hit, &instanceIndex))
            return false;
        if (instanceIndex < 0 || instanceIndex >= int(scene_->instances.size())) return false;
        for (const PrimRecord& prim : scene_->prims) {
            if (prim.instanceIndex != instanceIndex || prim.sourceNode.empty()) continue;
            sourceNode = QString::fromStdString(prim.sourceNode);
            return true;
        }
        return false;
    });

    renderView_->setSceneBoundsCallback([this](Bounds3& out) -> bool {
        if (!scene_ || !scene_->bounds().valid()) return false;
        out = scene_->bounds();
        return true;
    });
    renderView_->setSelectionBoundsCallback([this](Bounds3& out) -> bool {
        if (!scene_) return false;
        Node* target = renderView_->transformTarget();
        if (!target) return false;
        const std::string source = target->name().toStdString();
        Bounds3 bounds;
        bool found = false;
        for (const PrimRecord& prim : scene_->prims) {
            if (prim.sourceNode != source) continue;
            if (prim.instanceIndex < 0 || prim.instanceIndex >= int(scene_->instances.size())) continue;
            const InstanceData& inst = scene_->instances[size_t(prim.instanceIndex)];
            if (inst.meshIndex < 0 || inst.meshIndex >= int(scene_->meshes.size())) continue;
            const MeshPtr& mesh = scene_->meshes[size_t(inst.meshIndex)];
            if (!mesh || !mesh->bounds.valid()) continue;
            bounds.extend(transformBounds(inst.xform, mesh->bounds));
            found = true;
        }
        if (!found) {
            // Transformable node without cooked geo — frame its translate origin.
            if (Parameter* t = target->findParameter("translate")) {
                const Vec3 p = t->toVec3();
                bounds.extend(p - Vec3(0.25f));
                bounds.extend(p + Vec3(0.25f));
                found = true;
            }
        }
        if (!found) return false;
        out = bounds;
        return true;
    });

    // The render thread only flags that new samples exist; the UI timer picks
    // the image up so we never touch widgets from a worker thread.
    session_.setUpdateCallback([this] { framePending_.store(true, std::memory_order_relaxed); });
    session_.setFinishedCallback([this] { framePending_.store(true, std::memory_order_relaxed); });
}

MainWindow::~MainWindow() { session_.stop(); }

void MainWindow::createActions() {
    renderAction_ = new QAction("Render", this);
    renderAction_->setShortcut(QKeySequence("F5"));
    connect(renderAction_, &QAction::triggered, this, &MainWindow::onStartRender);

    stopAction_ = new QAction("Stop", this);
    stopAction_->setShortcut(QKeySequence("Esc"));
    connect(stopAction_, &QAction::triggered, this, &MainWindow::onStopRender);

    iprAction_ = new QAction("Interactive Rendering", this);
    iprAction_->setCheckable(true);
    iprAction_->setChecked(true);
    iprAction_->setToolTip("Re-render automatically after every edit");

    auto* transformGroup = new QActionGroup(this);
    transformGroup->setExclusive(true);

    selectToolAction_ = new QAction("Select", this);
    selectToolAction_->setCheckable(true);
    selectToolAction_->setChecked(true);
    selectToolAction_->setShortcut(QKeySequence("Q"));
    selectToolAction_->setToolTip("Select (Q)");
    transformGroup->addAction(selectToolAction_);

    translateToolAction_ = new QAction("T", this);
    translateToolAction_->setCheckable(true);
    translateToolAction_->setChecked(true);
    translateToolAction_->setShortcut(QKeySequence("T"));
    translateToolAction_->setToolTip("Translate (T)");
    transformGroup->addAction(translateToolAction_);

    rotateToolAction_ = new QAction("R", this);
    rotateToolAction_->setCheckable(true);
    rotateToolAction_->setShortcut(QKeySequence("R"));
    rotateToolAction_->setToolTip("Rotate (R)");
    transformGroup->addAction(rotateToolAction_);

    scaleToolAction_ = new QAction("S", this);
    scaleToolAction_->setCheckable(true);
    scaleToolAction_->setShortcut(QKeySequence("S"));
    scaleToolAction_->setToolTip("Scale (S)");
    transformGroup->addAction(scaleToolAction_);

    selectToolAction_->setChecked(false);

    connect(selectToolAction_, &QAction::triggered, this,
            [this] { renderView_->setTransformTool(TransformTool::Select); });
    connect(translateToolAction_, &QAction::triggered, this,
            [this] { renderView_->setTransformTool(TransformTool::Translate); });
    connect(rotateToolAction_, &QAction::triggered, this,
            [this] { renderView_->setTransformTool(TransformTool::Rotate); });
    connect(scaleToolAction_, &QAction::triggered, this,
            [this] { renderView_->setTransformTool(TransformTool::Scale); });
    connect(renderView_, &RenderView::transformToolChanged, this, [this](TransformTool tool) {
        selectToolAction_->setChecked(tool == TransformTool::Select);
        translateToolAction_->setChecked(tool == TransformTool::Translate);
        rotateToolAction_->setChecked(tool == TransformTool::Rotate);
        scaleToolAction_->setChecked(tool == TransformTool::Scale);
    });
}

void MainWindow::createMenus() {
    QMenu* fileMenu = menuBar()->addMenu("&File");
    fileMenu->addAction("&New Scene", QKeySequence::New, this, &MainWindow::onNewScene);
    fileMenu->addAction("&Open Scene...", QKeySequence::Open, this, &MainWindow::onOpenScene);
    fileMenu->addAction("&Save Scene", QKeySequence::Save, this, &MainWindow::onSaveScene);
    fileMenu->addAction("Save Scene &As...", QKeySequence::SaveAs, this, &MainWindow::onSaveSceneAs);
    fileMenu->addSeparator();
    fileMenu->addAction("&Import Alembic...", QKeySequence("Ctrl+I"), this, &MainWindow::onImportAlembic);
    fileMenu->addAction("Import &USD...", this, &MainWindow::onImportUsd);
    fileMenu->addAction("Save &Image...", QKeySequence("Ctrl+E"), this, &MainWindow::onSaveImage);
    fileMenu->addSeparator();
    fileMenu->addAction("E&xit", QKeySequence::Quit, this, &QWidget::close);

    QMenu* editMenu = menuBar()->addMenu("&Edit");
    editMenu->addAction("Delete Selected Nodes", QKeySequence::Delete, this,
                        [this] { networkView_->deleteSelectedNodes(); });
    editMenu->addAction("Set Display Flag", QKeySequence("Ctrl+D"), this,
                        [this] { networkView_->toggleDisplayFlagOnSelection(); });
    editMenu->addAction("Toggle Bypass", QKeySequence("Ctrl+B"), this,
                        [this] { networkView_->toggleBypassOnSelection(); });
    editMenu->addAction("Frame Network", QKeySequence("Ctrl+F"), this, [this] { networkView_->frameAll(); });
    editMenu->addAction("Lay Out Selection", QKeySequence("Ctrl+L"), this,
                        [this] { networkView_->layoutSelectionVertically(); });

    QMenu* renderMenu = menuBar()->addMenu("&Render");
    renderMenu->addAction(renderAction_);
    renderMenu->addAction(stopAction_);
    renderMenu->addAction(iprAction_);
    renderMenu->addSeparator();
    renderMenu->addAction("Look Through Camera Node", QKeySequence("Ctrl+Shift+C"), this,
                          &MainWindow::onLookThroughCameraNode);
    renderMenu->addAction("Copy View To Camera Node", this, &MainWindow::onCopyViewToCameraNode);

    QMenu* helpMenu = menuBar()->addMenu("&Help");
    helpMenu->addAction("Keyboard Shortcuts", this, &MainWindow::onShowShortcuts);
    helpMenu->addAction(QString("About %1").arg(SOLSTICE_APP_NAME), this, &MainWindow::onShowAbout);
}

void MainWindow::createToolBar() {
    QToolBar* toolBar = addToolBar("Render");
    toolBar->setMovable(false);
    // T/R/S live centered on the viewport itself; this bar is render controls.
    toolBar->addAction(renderAction_);
    toolBar->addAction(stopAction_);
    toolBar->addSeparator();
    toolBar->addAction(iprAction_);
}

void MainWindow::createDocks() {
    setDockOptions(QMainWindow::AnimatedDocks | QMainWindow::AllowNestedDocks |
                   QMainWindow::AllowTabbedDocks);
    // Keep the network pane at the bottom of the window, but put the tab bar
    // (Scene Network / Material Network / Log) on TOP of that pane.
    setTabPosition(Qt::BottomDockWidgetArea, QTabWidget::North);

    auto* networkDock = new QDockWidget("Scene Network", this);
    networkDock->setObjectName("networkDock");
    networkView_ = new NodeGraphView(networkDock);
    networkView_->setGraph(&graph_);
    networkDock->setWidget(networkView_);
    addDockWidget(Qt::BottomDockWidgetArea, networkDock);

    auto* materialNetworkDock = new QDockWidget("Material Network", this);
    materialNetworkDock->setObjectName("materialNetworkDock");
    materialNetworkView_ = new MaterialNetworkView(materialNetworkDock);
    materialNetworkDock->setWidget(materialNetworkView_);
    addDockWidget(Qt::BottomDockWidgetArea, materialNetworkDock);

    auto* parameterDock = new QDockWidget("Parameters", this);
    parameterDock->setObjectName("parameterDock");
    parameterPanel_ = new ParameterPanel(parameterDock);
    parameterDock->setWidget(parameterPanel_);
    parameterDock->setMinimumWidth(380);
    addDockWidget(Qt::RightDockWidgetArea, parameterDock);

    auto* sceneDock = new QDockWidget("Scene Graph", this);
    sceneDock->setObjectName("sceneDock");
    sceneGraphPanel_ = new SceneGraphPanel(sceneDock);
    sceneDock->setWidget(sceneGraphPanel_);
    addDockWidget(Qt::LeftDockWidgetArea, sceneDock);

    auto* logDock = new QDockWidget("Log", this);
    logDock->setObjectName("logDock");
    logPanel_ = new LogPanel(logDock);
    logPanel_->installAsLogSink();
    logDock->setWidget(logPanel_);
    addDockWidget(Qt::BottomDockWidgetArea, logDock);

    tabifyDockWidget(networkDock, materialNetworkDock);
    tabifyDockWidget(networkDock, logDock);
    networkDock->raise();

    resizeDocks({networkDock}, {330}, Qt::Vertical);

    materialNetworkView_->setGraph(&graph_);

    auto syncTransformTarget = [this](Node* node) {
        renderView_->setTransformTarget(node && node->findParameter("translate") ? node : nullptr);
    };

    auto selectNodeForEditing = [this, syncTransformTarget](Node* node, bool centerNetwork = false) {
        parameterPanel_->setNode(node);
        syncTransformTarget(node);
        networkView_->selectNode(node, centerNetwork);
        sceneGraphPanel_->selectBySourceNode(node ? node->name() : QString());
    };

    connect(renderView_, &RenderView::objectSelected, this,
            [this, selectNodeForEditing](const QString& sourceNode) {
                if (sourceNode.isEmpty()) {
                    selectNodeForEditing(nullptr, false);
                    statusBar()->showMessage("Selection cleared", 1500);
                    return;
                }
                if (Node* node = graph_.findNode(sourceNode)) {
                    selectNodeForEditing(node, false);
                    statusBar()->showMessage("Selected " + sourceNode, 2000);
                }
            });

    connect(renderView_, &RenderView::focusDistancePicked, this, [this](float distanceMetres) {
        Node* camera = findCameraNode();
        if (!camera) {
            statusBar()->showMessage("No camera node — add Camera in the Scene Network", 4000);
            parameterPanel_->setFocusPickActive(false);
            return;
        }
        camera->setParameterValue("focusdistance", double(distanceMetres));
        if (scene_) {
            scene_->camera.focusDistance = distanceMetres;
            session_.updateSceneData();
            if (iprAction_->isChecked()) session_.start();
        }
        if (parameterPanel_->node() == camera) parameterPanel_->refresh();
        parameterPanel_->setFocusPickActive(false);
        statusBar()->showMessage(QString("Focus distance set to %1 m").arg(double(distanceMetres), 0, 'f', 3),
                                 3000);
        scheduleCook(0);
    });
    connect(renderView_, &RenderView::focusPickChanged, this,
            [this](bool active) { parameterPanel_->setFocusPickActive(active); });
    connect(parameterPanel_, &ParameterPanel::focusPickToggled, this, [this](bool active) {
        renderView_->setFocusPickActive(active);
        if (active) {
            statusBar()->showMessage("Focus Pick: click geometry in the viewport", 4000);
            renderView_->setFocus(Qt::OtherFocusReason);
        }
    });

    connect(networkView_, &NodeGraphView::nodeSelected, this, [this, syncTransformTarget](Node* node) {
        // Scene Network is the selection source — don't let Material Network steal it.
        parameterPanel_->setNode(node);
        syncTransformTarget(node);
        sceneGraphPanel_->selectBySourceNode(node ? node->name() : QString());
    });
    connect(networkView_, &NodeGraphView::statusMessage, this,
            [this](const QString& message) { statusBar()->showMessage(message, 3000); });
    connect(materialNetworkView_, &MaterialNetworkView::selectionChanged, this, [this, syncTransformTarget] {
        // Only push Material Network selection when that view actually has one.
        MaterialXSelection mtlx;
        if (materialNetworkView_->selectedMaterialX(mtlx)) {
            parameterPanel_->setMaterialXSelection(mtlx);
            syncTransformTarget(nullptr);
            return;
        }
        if (!materialNetworkView_->isInsideMaterial()) {
            Node* node = materialNetworkView_->selectedLopNode();
            if (!node) return;  // empty container selection must not clear Parameters
            parameterPanel_->setNode(node);
            syncTransformTarget(node);
        } else {
            // Inside a material with no MaterialX node selected — show the container.
            parameterPanel_->setNode(materialNetworkView_->currentMaterial());
            syncTransformTarget(nullptr);
        }
    });
    connect(materialNetworkView_, &MaterialNetworkView::materialEdited, this, [this](Node*) {
        scheduleCook();
    });
    connect(materialNetworkView_, &MaterialNetworkView::statusMessage, this,
            [this](const QString& message) { statusBar()->showMessage(message, 3000); });
    connect(sceneGraphPanel_, &SceneGraphPanel::itemSelected, this,
            [this, selectNodeForEditing](const QString& path, const QString& sourceNode) {
                Q_UNUSED(path);
                if (sourceNode.isEmpty()) return;
                if (Node* node = graph_.findNode(sourceNode)) selectNodeForEditing(node, false);
            });
    connect(parameterPanel_, &ParameterPanel::parameterEdited, this,
            [this](Node*, const QString&) {
                if (renderView_->isGizmoDragging()) return;
                scheduleCook();
            });
    connect(parameterPanel_, &ParameterPanel::nodeRenamed, this, [this](Node*) { scheduleCook(); });
    connect(parameterPanel_, &ParameterPanel::materialXRenamed, this,
            [this](Node*, const QString&, const QString& newName) {
                if (materialNetworkView_->renameSelectedMaterialX(newName)) {
                    MaterialXSelection mtlx;
                    if (materialNetworkView_->selectedMaterialX(mtlx))
                        parameterPanel_->setMaterialXSelection(mtlx);
                }
            });
    connect(parameterPanel_, &ParameterPanel::materialXInputEdited, this,
            [this](Node*, const QString&, const QString& inputName, const QString& value) {
                materialNetworkView_->setSelectedMaterialXInput(inputName, value);
            });
    // Live gizmo moves update node params quietly — cook / IPR only on release.
    connect(renderView_, &RenderView::transformEdited, this, [this](Node*) {
        // Keep the gizmo responsive; do not rebuild Parameters or restart IPR.
    });
    connect(renderView_, &RenderView::transformFinished, this, [this](Node* node) {
        if (parameterPanel_->node() == node && !parameterPanel_->showingMaterialX())
            parameterPanel_->refresh();
        scheduleCook(0);
    });
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

void MainWindow::newScene() {
    session_.stop();
    buildDefaultGraph(graph_);
    networkView_->setGraph(&graph_);
    parameterPanel_->clearSelection();
    materialNetworkView_->goUp();
    renderView_->setTransformTarget(nullptr);
    cameraOverride_ = false;
    renderView_->clearImage();
    updateWindowTitle();
    cookNow();
    selectDisplayNode();
}

void MainWindow::newSceneFromAlembic(const QString& alembicPath, const QString& hdriPath) {
    session_.stop();
    buildAlembicGraph(graph_, alembicPath, hdriPath);
    networkView_->setGraph(&graph_);
    parameterPanel_->clearSelection();
    materialNetworkView_->goUp();
    renderView_->setTransformTarget(nullptr);
    cameraOverride_ = false;
    renderView_->clearImage();
    updateWindowTitle();
    cookNow();
    selectDisplayNode();
}

bool MainWindow::openScene(const QString& path) {
    session_.stop();
    QString error;
    if (!loadGraphFromFile(graph_, path, error)) {
        QMessageBox::warning(this, "Open scene", error);
        return false;
    }
    networkView_->setGraph(&graph_);
    parameterPanel_->clearSelection();
    materialNetworkView_->goUp();
    renderView_->setTransformTarget(nullptr);
    cameraOverride_ = false;
    updateWindowTitle();
    cookNow();
    selectDisplayNode();
    return true;
}

bool MainWindow::saveScene(const QString& path) {
    QString error;
    if (!saveGraphToFile(graph_, path, error)) {
        QMessageBox::warning(this, "Save scene", error);
        return false;
    }
    graph_.setFilePath(QFileInfo(path).absoluteFilePath());
    graph_.setModified(false);
    updateWindowTitle();
    statusBar()->showMessage("Saved " + path, 4000);
    return true;
}

void MainWindow::onNewScene() {
    if (!maybeSaveChanges()) return;
    newScene();
}

void MainWindow::onOpenScene() {
    if (!maybeSaveChanges()) return;
    const QString path = QFileDialog::getOpenFileName(this, "Open scene", QString(), kSceneFileFilter);
    if (path.isEmpty()) return;
    openScene(path);
}

void MainWindow::onSaveScene() {
    if (graph_.filePath().isEmpty()) {
        onSaveSceneAs();
        return;
    }
    saveScene(graph_.filePath());
}

void MainWindow::onSaveSceneAs() {
    QString path = QFileDialog::getSaveFileName(this, "Save scene", graph_.filePath(), kSceneFileFilter);
    if (path.isEmpty()) return;
    if (QFileInfo(path).suffix().isEmpty()) path += "." + QString(kSceneFileExtension);
    saveScene(path);
}

void MainWindow::onImportAlembic() {
    const QString path = QFileDialog::getOpenFileName(this, "Import Alembic", QString(), "Alembic (*.abc)");
    if (path.isEmpty()) return;
    Node* node = graph_.createNode("alembic");
    if (!node) return;
    node->setParameterValue("file", path);
    node->setPosition(QPointF(0, -420));
    networkView_->selectNode(node);
    parameterPanel_->setNode(node);
    graph_.setDisplayNode(node);
    scheduleCook(0);
}

void MainWindow::onImportUsd() {
    const QString path =
        QFileDialog::getOpenFileName(this, "Import USD", QString(), "USD (*.usd *.usda *.usdc)");
    if (path.isEmpty()) return;
    Node* node = graph_.createNode("usd");
    if (!node) return;
    node->setParameterValue("file", path);
    node->setPosition(QPointF(0, -420));
    networkView_->selectNode(node);
    parameterPanel_->setNode(node);
    graph_.setDisplayNode(node);
    scheduleCook(0);
}

void MainWindow::onSaveImage() {
    const QString path = QFileDialog::getSaveFileName(this, "Save image", "render.png",
                                                      "PNG (*.png);;OpenEXR (*.exr);;Radiance (*.hdr)");
    if (path.isEmpty()) return;
    const Image linear = session_.linearImage();
    if (linear.empty()) {
        QMessageBox::information(this, "Save image", "Nothing has been rendered yet.");
        return;
    }
    std::string error;
    const RenderSettingsData settings = scene_ ? scene_->settings : RenderSettingsData();
    if (!saveImageAuto(path.toStdString(), linear, settings, error)) {
        QMessageBox::warning(this, "Save image", QString::fromStdString(error));
        return;
    }
    statusBar()->showMessage("Wrote " + path, 4000);
}

// ---------------------------------------------------------------------------
// Cook and render
// ---------------------------------------------------------------------------

void MainWindow::scheduleCook(int delayMilliseconds) {
    if (!cookTimer_) return;
    cookTimer_->start(delayMilliseconds);
}

void MainWindow::onCookTimeout() { cookNow(); }

void MainWindow::cookNow() {
    CookContext context;
    if (!graph_.filePath().isEmpty()) context.sceneDirectory = QFileInfo(graph_.filePath()).absolutePath();

    stage_ = graph_.cookDisplay(context);
    QStringList materialContainers;
    for (const NodePtr& node : graph_.nodes()) {
        if (node && node->typeName() == "material") materialContainers << node->name();
    }
    materialContainers.sort(Qt::CaseInsensitive);
    sceneGraphPanel_->setStage(stage_, materialContainers);
    // Do not rebuild Parameters on every cook — that steals focus / selection while editing.

    if (!stage_) return;
    scene_ = stage_->toScene();

    if (cameraOverride_) {
        scene_->camera.cameraToWorld = renderView_->camera().toMatrix();
        scene_->camera.focusDistance = renderView_->camera().distance;
    } else {
        renderView_->camera().setFromMatrix(scene_->camera.cameraToWorld,
                                            scene_->camera.focusDistance > 0.0f
                                                ? scene_->camera.focusDistance
                                                : std::max(1.0f, scene_->bounds().radius() * 2.0f));
    }
    renderView_->setResolution(scene_->settings.resolutionX, scene_->settings.resolutionY);

    session_.setScene(scene_);
    updateStatusBar();

    if (iprAction_->isChecked() || renderRequested_) restartRender();
}

void MainWindow::restartRender() {
    session_.stop();
    session_.framebuffer().clear();
    session_.start();
    renderRequested_ = false;
}

void MainWindow::onStartRender() {
    renderRequested_ = true;
    cookNow();
}

void MainWindow::onStopRender() {
    session_.stop();
    statusBar()->showMessage("Render stopped", 3000);
    updateStatusBar();
}

void MainWindow::onRenderTick() {
    if (!framePending_.exchange(false, std::memory_order_relaxed)) return;
    renderView_->setImage(toQImage(session_.displayImage()));
    updateStatusBar();
}

void MainWindow::onCameraMoved() {
    cameraOverride_ = true;
    if (!scene_) return;
    scene_->camera.cameraToWorld = renderView_->camera().toMatrix();
    scene_->camera.focusDistance = renderView_->camera().distance;
    // The geometry has not changed, so this keeps the acceleration structures.
    session_.updateSceneData();
    if (iprAction_->isChecked()) session_.start();
}

void MainWindow::onLookThroughCameraNode() {
    cameraOverride_ = false;
    cookNow();
    statusBar()->showMessage("Looking through the camera node", 3000);
}

void MainWindow::selectDisplayNode() {
    Node* node = graph_.displayNode();
    if (!node) return;
    networkView_->selectNode(node);
    parameterPanel_->setNode(node);
    renderView_->setTransformTarget(node->findParameter("translate") ? node : nullptr);
}

Node* MainWindow::findCameraNode() const {
    for (const NodePtr& node : graph_.nodes()) {
        if (node->typeName() == "camera") return node.get();
    }
    return nullptr;
}

void MainWindow::onCopyViewToCameraNode() {
    Node* camera = findCameraNode();
    if (!camera) {
        QMessageBox::information(this, "Copy view",
                                 "This network has no camera node. Add one with Tab > Camera.");
        return;
    }
    const ViewCamera& view = renderView_->camera();
    const Vec3 eye = view.eye();
    camera->setParameterValue("uselookat", true);
    camera->setParameterValue("eye", QVariant::fromValue(QVector3D(eye.x, eye.y, eye.z)));
    camera->setParameterValue("target",
                              QVariant::fromValue(QVector3D(view.pivot.x, view.pivot.y, view.pivot.z)));
    camera->setParameterValue("focusdistance", double(view.distance));
    cameraOverride_ = false;
    parameterPanel_->refresh();
    scheduleCook(0);
    statusBar()->showMessage("View copied to " + camera->name(), 4000);
}

// ---------------------------------------------------------------------------
// Chrome
// ---------------------------------------------------------------------------

void MainWindow::updateWindowTitle() {
    const QString file = graph_.filePath().isEmpty() ? QString("untitled") : QFileInfo(graph_.filePath()).fileName();
    setWindowTitle(QString("%1 %2 - %3%4")
                       .arg(SOLSTICE_APP_NAME, SOLSTICE_VERSION, file, graph_.isModified() ? "*" : ""));
}

void MainWindow::updateStatusBar() {
    const RenderProgress progress = session_.progress();
    QString text;
    if (scene_) {
        text = QString("%1 x %2   |   %3 triangles   |   %4 lights")
                   .arg(scene_->settings.resolutionX)
                   .arg(scene_->settings.resolutionY)
                   .arg(scene_->totalTriangles())
                   .arg(scene_->lights.size());
    }
    if (progress.samplesTarget > 0) {
        text += QString("   |   %1 / %2 samples").arg(progress.samplesDone).arg(progress.samplesTarget);
        if (progress.samplesPerSecond > 0.0)
            text += QString(" (%1 spp/s)").arg(progress.samplesPerSecond, 0, 'f', 2);
    }
    if (session_.isRendering()) text += "   |   rendering";
    statusLabel_->setText(text);

    if (!progress.backendName.empty()) backendLabel_->setText(QString::fromStdString(progress.backendName));

    QString overlay = QString("%1 / %2 spp").arg(progress.samplesDone).arg(progress.samplesTarget);
    if (progress.elapsedSeconds > 0.0) overlay += QString("   %1 s").arg(progress.elapsedSeconds, 0, 'f', 1);
    if (!progress.backendName.empty()) overlay += "   " + QString::fromStdString(progress.backendName);
    renderView_->setStatusText(overlay);
}

bool MainWindow::maybeSaveChanges() {
    if (!graph_.isModified()) return true;
    const QMessageBox::StandardButton answer =
        QMessageBox::question(this, SOLSTICE_APP_NAME, "The current scene has unsaved changes. Save it?",
                              QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
    if (answer == QMessageBox::Cancel) return false;
    if (answer == QMessageBox::Save) {
        onSaveScene();
        return !graph_.isModified();
    }
    return true;
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (!maybeSaveChanges()) {
        event->ignore();
        return;
    }
    session_.stop();
    event->accept();
}

void MainWindow::onShowAbout() {
    QMessageBox::about(
        this, QString("About %1").arg(SOLSTICE_APP_NAME),
        QString("<h3>%1 %2</h3>"
                "<p>A node based path tracer in the spirit of Houdini Solaris.</p>"
                "<ul>"
                "<li>Alembic and USD geometry import</li>"
                "<li>Area, distant and HDRI dome lights</li>"
                "<li>Progressive path tracing on the CPU with Intel Embree</li>"
                "<li>GPU path tracing with NVIDIA OptiX%3</li>"
                "</ul>")
            .arg(SOLSTICE_APP_NAME, SOLSTICE_VERSION,
                 optixBackendCompiledIn() ? "" : " (not compiled into this build)"));
}

void MainWindow::onShowShortcuts() {
    QMessageBox::information(this, "Keyboard shortcuts",
                             "Scene Network\n"
                             "  Tab           add node (search)\n"
                             "  D             set display flag\n"
                             "  B             bypass\n"
                             "  F             frame all\n"
                             "  L             lay out selection\n"
                             "  Del           delete selection\n"
                             "  MMB / Alt+LMB / Space+LMB   pan\n"
                             "  Wheel         zoom to cursor\n\n"
                             "Material Network\n"
                             "  Tab           add MaterialX node (search)\n"
                             "  F             frame all\n"
                             "  Del           delete selection\n"
                             "  ↑             return to materials level\n"
                             "  Double-click  dive into material container\n"
                             "  MMB / Alt+LMB / Space+LMB   pan\n"
                             "  Wheel         zoom to cursor\n\n"
                             "Render view (Houdini style)\n"
                             "  F             frame selected object\n"
                             "  H / Home      frame all\n"
                             "  Sel / Q       select — LMB click geometry (Select tool only)\n"
                             "  T / R / S     translate / rotate / scale\n"
                             "  LMB on gizmo  transform (IPR restarts on release)\n"
                             "  Focus Pick    camera Lens → click geo to set DOF focus\n"
                             "  Alt + LMB     tumble (pivot on geometry under cursor)\n"
                             "  RMB           tumble / orbit\n"
                             "  MMB           pan\n"
                             "  Alt + RMB     dolly\n"
                             "  Wheel         dolly\n\n"
                             "Units\n"
                             "  1 scene unit = 1 metre (Houdini MKS)\n"
                             "  angles in degrees, focal length in millimetres\n\n"
                             "General\n"
                             "  F5            render\n"
                             "  Esc           stop\n"
                             "  Ctrl+E        save image");
}

}  // namespace sol
