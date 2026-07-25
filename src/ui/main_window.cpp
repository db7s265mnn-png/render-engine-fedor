#include "ui/main_window.h"

#include <QAction>
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
#include "io/image_io.h"
#include "nodes/node_registry.h"
#include "solstice_config.h"
#include "ui/log_panel.h"
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
    setWindowTitle("Solstice");
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
}

void MainWindow::createMenus() {
    QMenu* fileMenu = menuBar()->addMenu("&File");
    fileMenu->addAction("&New Scene", QKeySequence::New, this, &MainWindow::onNewScene);
    fileMenu->addAction("&Open Scene...", QKeySequence::Open, this, &MainWindow::onOpenScene);
    fileMenu->addAction("&Save Scene", QKeySequence::Save, this, &MainWindow::onSaveScene);
    fileMenu->addAction("Save Scene &As...", QKeySequence::SaveAs, this, &MainWindow::onSaveSceneAs);
    fileMenu->addSeparator();
    fileMenu->addAction("&Import Alembic...", QKeySequence("Ctrl+I"), this, &MainWindow::onImportAlembic);
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
    helpMenu->addAction("About Solstice", this, &MainWindow::onShowAbout);
}

void MainWindow::createToolBar() {
    QToolBar* toolBar = addToolBar("Render");
    toolBar->setMovable(false);
    toolBar->addAction(renderAction_);
    toolBar->addAction(stopAction_);
    toolBar->addSeparator();
    toolBar->addAction(iprAction_);
}

void MainWindow::createDocks() {
    setDockOptions(QMainWindow::AnimatedDocks | QMainWindow::AllowNestedDocks |
                   QMainWindow::AllowTabbedDocks);

    auto* networkDock = new QDockWidget("Network Editor", this);
    networkDock->setObjectName("networkDock");
    networkView_ = new NodeGraphView(networkDock);
    networkView_->setGraph(&graph_);
    networkDock->setWidget(networkView_);
    addDockWidget(Qt::BottomDockWidgetArea, networkDock);

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
    tabifyDockWidget(networkDock, logDock);
    networkDock->raise();

    resizeDocks({networkDock}, {330}, Qt::Vertical);

    connect(networkView_, &NodeGraphView::nodeSelected, this,
            [this](Node* node) { parameterPanel_->setNode(node); });
    connect(networkView_, &NodeGraphView::statusMessage, this,
            [this](const QString& message) { statusBar()->showMessage(message, 3000); });
    connect(parameterPanel_, &ParameterPanel::parameterEdited, this,
            [this](Node*, const QString&) { scheduleCook(); });
    connect(parameterPanel_, &ParameterPanel::nodeRenamed, this, [this](Node*) { scheduleCook(); });
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

void MainWindow::newScene() {
    session_.stop();
    buildDefaultGraph(graph_);
    networkView_->setGraph(&graph_);
    parameterPanel_->setNode(nullptr);
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
    parameterPanel_->setNode(nullptr);
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
    parameterPanel_->setNode(nullptr);
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
    sceneGraphPanel_->setStage(stage_);
    if (parameterPanel_->node()) parameterPanel_->refresh();

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
    setWindowTitle(QString("Solstice %1 - %2%3").arg(SOLSTICE_VERSION, file, graph_.isModified() ? "*" : ""));
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
        QMessageBox::question(this, "Solstice", "The current scene has unsaved changes. Save it?",
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
        this, "About Solstice",
        QString("<h3>Solstice %1</h3>"
                "<p>A node based path tracer in the spirit of Houdini Solaris.</p>"
                "<ul>"
                "<li>Alembic geometry import</li>"
                "<li>Area, distant and HDRI dome lights</li>"
                "<li>Progressive path tracing on the CPU with Intel Embree</li>"
                "<li>GPU path tracing with NVIDIA OptiX%2</li>"
                "</ul>")
            .arg(SOLSTICE_VERSION, optixBackendCompiledIn() ? "" : " (not compiled into this build)"));
}

void MainWindow::onShowShortcuts() {
    QMessageBox::information(this, "Keyboard shortcuts",
                             "Network editor\n"
                             "  Tab           add node\n"
                             "  D             set display flag\n"
                             "  B             bypass\n"
                             "  F             frame all\n"
                             "  L             lay out selection\n"
                             "  Del           delete selection\n"
                             "  MMB / Alt+LMB pan, wheel zoom\n\n"
                             "Render view\n"
                             "  Alt + LMB     orbit\n"
                             "  MMB           pan\n"
                             "  Alt + RMB     dolly\n"
                             "  Wheel         dolly\n\n"
                             "General\n"
                             "  F5            render\n"
                             "  Esc           stop\n"
                             "  Ctrl+E        save image");
}

}  // namespace sol
