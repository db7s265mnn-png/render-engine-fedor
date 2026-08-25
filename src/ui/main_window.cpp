#include "ui/main_window.h"

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QCloseEvent>
#include <QColor>
#include <QDir>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QMenuBar>
#include <QIcon>
#include <QMessageBox>
#include <QMetaObject>
#include <QPixmap>
#include <QMouseEvent>
#include <QPainter>
#include <QSplitter>
#include <QStatusBar>
#include <QShortcut>
#include <QSizePolicy>
#include <QToolButton>
#include <QSignalBlocker>
#include <QTimer>
#include <QVector3D>
#include <QEventLoop>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <functional>
#include <new>
#include <string>

#include "app/default_scene.h"
#include "app/document.h"
#include "core/expr_eval.h"
#include "core/log.h"
#include "core/math.h"
#include "io/image_io.h"
#include "io/ocio_util.h"
#include "io/tx_cache.h"
#include "nodes/node_registry.h"
#include "nodes/node.h"
#include "render/motion_blur.h"
#include "render/render_device.h"
#include "render/render_session.h"
#include "render/scene_picker.h"
#include "scene/types.h"
#include "scene/scene.h"
#include "scene/tessellate.h"
#include "scene/displace.h"
#include "solstice_config.h"
#include "ui/dock_chrome.h"
#include "ui/log_panel.h"
#include "ui/material_network_view.h"
#include "ui/node_graph_view.h"
#include "ui/parameter_panel.h"
#include "ui/scene_graph_panel.h"
#include "ui/theme.h"
#include "ui/timeline_bar.h"

namespace sol {
namespace {

// Floating viewport host. Not a QDockWidget — putting the viewer in the dock
// layout is what collapsed the central pane and blew up Scene Network.
class ViewportFloatWindow : public QWidget {
public:
    explicit ViewportFloatWindow(QWidget* parent = nullptr) : QWidget(parent, Qt::Window) {
        setObjectName(QStringLiteral("viewportFloatWindow"));
        setWindowTitle(QStringLiteral("Viewport"));
        setMinimumSize(320, 240);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
    }

    void setOnClose(std::function<void()> onClose) { onClose_ = std::move(onClose); }

protected:
    void closeEvent(QCloseEvent* event) override {
        event->ignore();
        if (onClose_) onClose_();
    }

private:
    std::function<void()> onClose_;
};

QMessageBox::StandardButton appMessageBox(QWidget* parent, const QString& title, const QString& text,
                                          QMessageBox::StandardButtons buttons = QMessageBox::Ok,
                                          QMessageBox::StandardButton def = QMessageBox::NoButton) {
    QMessageBox box(parent);
    box.setWindowTitle(title);
    box.setText(text);
    box.setStandardButtons(buttons);
    if (def != QMessageBox::NoButton) box.setDefaultButton(def);
    box.setIconPixmap(QPixmap(QStringLiteral(":/icons/app_icon_64.png")));
    if (parent) box.setWindowIcon(parent->windowIcon());
    return static_cast<QMessageBox::StandardButton>(box.exec());
}

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

QString formatElapsedHms(double seconds) {
    if (seconds < 0.0) seconds = 0.0;
    const int total = int(std::lround(seconds));
    const int h = total / 3600;
    const int m = (total % 3600) / 60;
    const int s = total % 60;
    return QStringLiteral("%1h:%2m:%3s")
        .arg(h, 2, 10, QChar('0'))
        .arg(m, 2, 10, QChar('0'))
        .arg(s, 2, 10, QChar('0'));
}

bool isTypingFocus(QWidget* widget) {
    while (widget) {
        if (widget->inherits("QLineEdit") || widget->inherits("QAbstractSpinBox") ||
            widget->inherits("QPlainTextEdit") || widget->inherits("QTextEdit"))
            return true;
        widget = widget->parentWidget();
    }
    return false;
}

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    registerBuiltinNodes();
    setWindowTitle(SOLSTICE_APP_NAME);
    setWindowIcon(QIcon(QStringLiteral(":/icons/app_icon.png")));
    resize(1720, 1000);

    // Viewport stays the central widget (same proportions as before detach).
    // Docks around it can float; the viewer itself floats via a dedicated window.
    createDocks();
    createActions();
    createMenus();
    createToolBar();
    createTimeline();

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
        refreshViewportCameraMenu();
    });
    connect(&graph_, &NodeGraph::displayNodeChanged, this, [this](Node*) { scheduleCook(0); });

    connect(renderView_, &RenderView::cameraMoved, this, &MainWindow::onCameraMoved);
    connect(renderView_, &RenderView::cameraNavStarted, this, [this] {
        if (!renderArmed()) return;
        session_.setInteractivePreview(true);
    });
    connect(renderView_, &RenderView::cameraNavEnded, this, [this] { session_.setInteractivePreview(false); });
    connect(renderView_, &RenderView::viewTransformChanged, this, [this](int) {
        framePending_.store(true, std::memory_order_relaxed);
        onRenderTick();
    });
    connect(renderView_, &RenderView::colorManagementChanged, this, [this](int) {
        framePending_.store(true, std::memory_order_relaxed);
        onRenderTick();
    });
    connect(renderView_, &RenderView::lookThroughCameraChosen, this, [this](const QString& name) {
        lookThroughCamera(name);
    });
    connect(&graph_, &NodeGraph::nodeAdded, this, [this](Node* node) {
        if (node && node->typeName() == QLatin1String("camera")) {
            refreshViewportCameraMenu();
            if (lookThroughCameraName_.isEmpty()) lookThroughCamera(node->name());
        }
    });
    connect(&graph_, &NodeGraph::nodeAboutToBeRemoved, this, [this](Node* node) {
        if (!node) return;
        if (parameterPanel_ && parameterPanel_->node() == node) parameterPanel_->clearSelection();
        if (renderView_ && renderView_->transformTarget() == node) renderView_->setTransformTarget(nullptr);
        if (selectedSourceNode_ == node->name()) selectedSourceNode_.clear();
        if (node->typeName() != QLatin1String("camera")) return;
        if (lookThroughCameraName_ == node->name()) {
            lookThroughCameraName_.clear();
            cameraOverride_ = true;
        }
        QMetaObject::invokeMethod(this, [this] { refreshViewportCameraMenu(); }, Qt::QueuedConnection);
    });

    renderView_->setPickCallback([this](float u, float v, Vec3& hit) -> bool {
        if (!scene_ || scene_->instances.empty()) return false;
        CameraData camera = scene_->camera;
        camera.cameraToWorld = renderView_->camera().toMatrix();
        // Interactive pick path: pinhole + shutter-center geometry (see ScenePickMode).
        return pickSceneSurface(scene_, camera, scene_->settings.resolutionX, scene_->settings.resolutionY, u, v,
                                hit, nullptr, ScenePickMode::Interactive);
    });

    renderView_->setObjectPickCallback([this](float u, float v, QString& sourceNode) -> bool {
        sourceNode.clear();
        if (!scene_ || scene_->instances.empty()) return false;
        CameraData camera = scene_->camera;
        camera.cameraToWorld = renderView_->camera().toMatrix();
        Vec3 hit;
        int instanceIndex = -1;
        if (!pickSceneSurface(scene_, camera, scene_->settings.resolutionX, scene_->settings.resolutionY, u, v,
                              hit, &instanceIndex, ScenePickMode::Interactive))
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
        QString sourceName = selectedSourceNode_;
        if (sourceName.isEmpty() && renderView_->transformTarget())
            sourceName = renderView_->transformTarget()->name();
        if (sourceName.isEmpty()) return false;
        const std::string source = sourceName.toStdString();
        Bounds3 bounds;
        bool found = false;
        for (const PrimRecord& prim : scene_->prims) {
            if (prim.sourceNode != source) continue;
            if (prim.instanceIndex < 0 || prim.instanceIndex >= int(scene_->instances.size())) continue;
            const InstanceData& inst = scene_->instances[size_t(prim.instanceIndex)];
            if (inst.meshIndex < 0 || inst.meshIndex >= int(scene_->meshes.size())) continue;
            const MeshPtr& mesh = scene_->meshes[size_t(inst.meshIndex)];
            if (!mesh || !mesh->bounds.valid()) continue;
            const Mat4& xform = size_t(prim.instanceIndex) < scene_->pickXforms.size()
                                    ? scene_->pickXforms[size_t(prim.instanceIndex)]
                                    : inst.xform;
            bounds.extend(transformBounds(xform, mesh->bounds));
            found = true;
        }
        if (!found) {
            Node* target = graph_.findNode(sourceName);
            if (!target) target = renderView_->transformTarget();
            if (target) {
                if (Parameter* t = target->findParameter("translate")) {
                    const Vec3 p = t->toVec3();
                    bounds.extend(p - Vec3(0.25f));
                    bounds.extend(p + Vec3(0.25f));
                    found = true;
                }
            }
        }
        if (!found) return false;
        out = bounds;
        return true;
    });

    // The render thread only flags that new samples exist; the UI timer picks
    // the image up so we never touch widgets from a worker thread.
    session_.setUpdateCallback([this] { framePending_.store(true, std::memory_order_relaxed); });
    session_.setFinishedCallback([this] {
        framePending_.store(true, std::memory_order_relaxed);
        QMetaObject::invokeMethod(this, [this] { onRenderFinished(); }, Qt::QueuedConnection);
    });
}

MainWindow::~MainWindow() { session_.stop(); }

void MainWindow::createActions() {
    auto* renderControlGroup = new QActionGroup(this);
    renderControlGroup->setExclusive(true);

    renderAction_ = new QAction("Start", this);
    renderAction_->setCheckable(true);
    renderAction_->setChecked(false);
    renderAction_->setShortcut(QKeySequence("F5"));
    renderAction_->setToolTip("Start cook + render (F5). Stays pressed — graph edits auto-restart "
                              "the render. Displacement dicing re-runs only when you press Start "
                              "(changing Dicing Quality / Subdiv / Screen Adaptive does not).");
    renderControlGroup->addAction(renderAction_);
    connect(renderAction_, &QAction::triggered, this, &MainWindow::onStartRender);

    stopAction_ = new QAction("Stop", this);
    stopAction_->setCheckable(true);
    stopAction_->setChecked(true);
    stopAction_->setShortcut(QKeySequence("Esc"));
    stopAction_->setToolTip("Stop render (Esc). Stays pressed — no cook / no render until Start. "
                            "Camera frozen; last frame kept.");
    renderControlGroup->addAction(stopAction_);
    connect(stopAction_, &QAction::triggered, this, &MainWindow::onStopRender);

    auto* transformGroup = new QActionGroup(this);
    transformGroup->setExclusive(true);

    selectToolAction_ = new QAction("Select", this);
    selectToolAction_->setCheckable(true);
    selectToolAction_->setChecked(true);
    selectToolAction_->setToolTip("Select (Q)");
    transformGroup->addAction(selectToolAction_);

    translateToolAction_ = new QAction("T", this);
    translateToolAction_->setCheckable(true);
    translateToolAction_->setChecked(true);
    translateToolAction_->setToolTip("Translate (T)");
    transformGroup->addAction(translateToolAction_);

    rotateToolAction_ = new QAction("R", this);
    rotateToolAction_->setCheckable(true);
    rotateToolAction_->setToolTip("Rotate (R)");
    transformGroup->addAction(rotateToolAction_);

    scaleToolAction_ = new QAction("S", this);
    scaleToolAction_->setCheckable(true);
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
    // Shortcuts live on the Scene Network view so Ctrl+C still copies prim paths
    // in the Scene Graph when that panel has focus.
    editMenu->addAction("Copy Nodes", this, [this] { networkView_->copySelectedNodes(); });
    editMenu->addAction("Paste Nodes", this, [this] { networkView_->pasteNodes(); });
    editMenu->addSeparator();
    editMenu->addAction("Delete Selected Nodes", QKeySequence::Delete, this,
                        [this] { networkView_->deleteSelectedNodes(); });
    editMenu->addAction("Set Display Flag", QKeySequence("Ctrl+D"), this,
                        [this] { networkView_->toggleDisplayFlagOnSelection(); });
    editMenu->addAction("Toggle Bypass", QKeySequence("Ctrl+B"), this,
                        [this] { networkView_->toggleBypassOnSelection(); });
    editMenu->addAction("Frame Network", QKeySequence("Ctrl+F"), this, [this] { networkView_->frameAll(); });
    editMenu->addAction("Lay Out Selection", QKeySequence("Ctrl+L"), this,
                        [this] { networkView_->layoutSelectionVertically(); });
    editMenu->addSeparator();
    editMenu->addAction("Look Through Camera Node", QKeySequence("Ctrl+Shift+C"), this,
                        &MainWindow::onLookThroughCameraNode);
    editMenu->addAction("Copy View To Camera Node", this, &MainWindow::onCopyViewToCameraNode);

    QMenu* helpMenu = menuBar()->addMenu("&Help");
    helpMenu->addAction("Keyboard Shortcuts", this, &MainWindow::onShowShortcuts);
    helpMenu->addAction(QString("About %1").arg(SOLSTICE_APP_NAME), this, &MainWindow::onShowAbout);
}

void MainWindow::createToolBar() {
    // Start / Stop sit on the viewport chrome, styled like Home / Q / T / R / S.
    addAction(renderAction_);
    addAction(stopAction_);
    addAction(selectToolAction_);
    addAction(translateToolAction_);
    addAction(rotateToolAction_);
    addAction(scaleToolAction_);
    renderView_->attachRenderActions(renderAction_, stopAction_);

    auto bindToolKey = [this](Qt::Key key, QAction* action) {
        auto* shortcut = new QShortcut(QKeySequence(key), this);
        shortcut->setContext(Qt::WindowShortcut);
        shortcut->setAutoRepeat(false);
        connect(shortcut, &QShortcut::activated, this, [action] {
            if (!action || isTypingFocus(QApplication::focusWidget())) return;
            action->trigger();
        });
    };
    bindToolKey(Qt::Key_Q, selectToolAction_);
    bindToolKey(Qt::Key_T, translateToolAction_);
    bindToolKey(Qt::Key_R, rotateToolAction_);
    bindToolKey(Qt::Key_S, scaleToolAction_);
}

void MainWindow::createTimeline() {
    // TimelineBar is created under the viewport (central pane).
    if (!timelineBar_) return;
    connect(timelineBar_, &TimelineBar::frameChanged, this, &MainWindow::onTimelineFrameChanged);
    // After scrubbing / stop, cook once more with full Embree quality when needed.
    // Must keep timelineFrameCook_: a plain click is press+release, and scrubFinished
    // used to schedule a second cook without the flag — that full-applied the Start
    // densify cache and froze Alembic poses until the next Start.
    auto qualityCook = [this] {
        if (!graph_.markTimeDependentDirty()) return;
        timelineFrameCook_ = true;
        scheduleCook(0);
    };
    connect(timelineBar_, &TimelineBar::playbackStopped, this, qualityCook);
    connect(timelineBar_, &TimelineBar::scrubFinished, this, qualityCook);
}

void MainWindow::onTimelineFrameChanged(int) {
    // Only time-dependent caches (animated Alembic/USD) need a recook.
    if (!graph_.markTimeDependentDirty()) {
        updateStatusBar();
        return;
    }
    // Scrubbing coalesces via the single-shot cook timer; clicks/play are immediate.
    const int delay = (timelineBar_ && timelineBar_->isScrubbing()) ? 8 : 0;
    timelineFrameCook_ = true;
    scheduleCook(delay);
}

void MainWindow::createDocks() {
    setDockOptions(QMainWindow::AnimatedDocks | QMainWindow::AllowNestedDocks |
                   QMainWindow::AllowTabbedDocks);
    // Keep the network pane at the bottom of the window, but put the tab bar
    // (Scene Network / Material Network / Log) on TOP of that pane.
    setTabPosition(Qt::BottomDockWidgetArea, QTabWidget::North);
    setCorner(Qt::TopLeftCorner, Qt::LeftDockWidgetArea);
    setCorner(Qt::TopRightCorner, Qt::RightDockWidgetArea);
    setCorner(Qt::BottomLeftCorner, Qt::BottomDockWidgetArea);
    setCorner(Qt::BottomRightCorner, Qt::BottomDockWidgetArea);

    auto* viewportPane = new QWidget(this);
    viewportPane->setObjectName(QStringLiteral("viewportPane"));
    auto* viewportLayout = new QVBoxLayout(viewportPane);
    viewportLayout->setContentsMargins(0, 0, 0, 0);
    viewportLayout->setSpacing(0);

    renderView_ = new RenderView(viewportPane);
    viewportLayout->addWidget(renderView_, 1);

    // Compact Houdini-style timeline stays with the viewport when that pane floats.
    timelineBar_ = new TimelineBar(viewportPane);
    viewportLayout->addWidget(timelineBar_, 0);

    // Viewport + timeline stay the central widget so leftover space after the
    // ~340px side docks and ~330px bottom network goes to the viewer — same
    // split as before detach. Floating uses a separate QWidget window, not a
    // competing QDockWidget (that is what inflated Scene Network).
    auto* viewportHost = new QWidget(this);
    viewportHost->setObjectName(QStringLiteral("viewportHost"));
    viewportHost->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    auto* hostLayout = new QVBoxLayout(viewportHost);
    hostLayout->setContentsMargins(0, 0, 0, 0);
    hostLayout->setSpacing(0);

    auto* viewportPlaceholder = new QWidget(viewportHost);
    viewportPlaceholder->setObjectName(QStringLiteral("viewportPlaceholder"));
    viewportPlaceholder->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    viewportPlaceholder->hide();

    auto* viewportFloat = new ViewportFloatWindow(this);
    auto* floatLayout = qobject_cast<QVBoxLayout*>(viewportFloat->layout());

    auto redockViewport = [this, viewportHost, hostLayout, viewportPane, viewportPlaceholder,
                           viewportFloat, floatLayout] {
        if (viewportPane->parentWidget() == viewportHost) {
            viewportFloat->hide();
            return;
        }
        if (floatLayout) floatLayout->removeWidget(viewportPane);
        viewportPlaceholder->hide();
        viewportPane->setParent(viewportHost);
        hostLayout->addWidget(viewportPane, 1);
        viewportPane->show();
        viewportFloat->hide();
        renderView_->setViewportFloating(false);
    };
    auto detachViewport = [this, hostLayout, viewportPane, viewportPlaceholder, viewportFloat,
                           floatLayout] {
        if (viewportPane->parentWidget() == viewportFloat) {
            viewportFloat->show();
            viewportFloat->raise();
            viewportFloat->activateWindow();
            return;
        }
        hostLayout->removeWidget(viewportPane);
        viewportPlaceholder->show();
        if (floatLayout) floatLayout->addWidget(viewportPane, 1);
        viewportPane->show();
        viewportFloat->resize(std::max(640, viewportPane->width()),
                              std::max(400, viewportPane->height()));
        viewportFloat->show();
        viewportFloat->raise();
        viewportFloat->activateWindow();
        renderView_->setViewportFloating(true);
    };
    viewportFloat->setOnClose(redockViewport);
    renderView_->setOnDetach([viewportPane, viewportFloat, detachViewport, redockViewport] {
        if (viewportPane->parentWidget() == viewportFloat) redockViewport();
        else detachViewport();
    });
    hostLayout->addWidget(viewportPane, 1);
    hostLayout->addWidget(viewportPlaceholder, 1);
    setCentralWidget(viewportHost);

    // Framing shortcuts stay scoped to the viewport pane so F/H are not stolen
    // from Scene / Material Network docks.
    auto* frameSelectedShortcut = new QShortcut(QKeySequence(Qt::Key_F), viewportPane);
    frameSelectedShortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(frameSelectedShortcut, &QShortcut::activated, renderView_, &RenderView::frameSelection);
    auto* frameAllShortcut = new QShortcut(QKeySequence(Qt::Key_H), viewportPane);
    frameAllShortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(frameAllShortcut, &QShortcut::activated, renderView_, &RenderView::frameAll);
    auto* homeShortcut = new QShortcut(QKeySequence(Qt::Key_Home), viewportPane);
    homeShortcut->setContext(Qt::WidgetWithChildrenShortcut);
    connect(homeShortcut, &QShortcut::activated, renderView_, &RenderView::frameAll);

    auto* networkDock = new QDockWidget("Scene Network", this);
    networkDock->setObjectName("networkDock");
    networkView_ = new NodeGraphView(networkDock);
    networkView_->setGraph(&graph_);
    networkDock->setWidget(networkView_);
    installDetachableTitleBar(networkDock);

    auto* materialNetworkDock = new QDockWidget("Material Network", this);
    materialNetworkDock->setObjectName("materialNetworkDock");
    materialNetworkView_ = new MaterialNetworkView(materialNetworkDock);
    materialNetworkDock->setWidget(materialNetworkView_);
    installDetachableTitleBar(materialNetworkDock);

    auto* parameterDock = new QDockWidget("Parameters", this);
    parameterDock->setObjectName("parameterDock");
    parameterPanel_ = new ParameterPanel(parameterDock);
    parameterDock->setWidget(parameterPanel_);
    parameterDock->setMinimumWidth(200);
    installDetachableTitleBar(parameterDock);

    auto* sceneDock = new QDockWidget("Scene Graph", this);
    sceneDock->setObjectName("sceneDock");
    sceneGraphPanel_ = new SceneGraphPanel(sceneDock);
    sceneDock->setWidget(sceneGraphPanel_);
    sceneDock->setMinimumWidth(80);
    installDetachableTitleBar(sceneDock);

    auto* logDock = new QDockWidget("Log", this);
    logDock->setObjectName("logDock");
    logPanel_ = new LogPanel(logDock);
    logPanel_->installAsLogSink();
    logDock->setWidget(logPanel_);
    installDetachableTitleBar(logDock);

    addDockWidget(Qt::LeftDockWidgetArea, sceneDock);
    addDockWidget(Qt::RightDockWidgetArea, parameterDock);
    addDockWidget(Qt::BottomDockWidgetArea, networkDock);
    addDockWidget(Qt::BottomDockWidgetArea, materialNetworkDock);
    addDockWidget(Qt::BottomDockWidgetArea, logDock);

    tabifyDockWidget(networkDock, materialNetworkDock);
    tabifyDockWidget(networkDock, logDock);
    networkDock->raise();

    resizeDocks({networkDock}, {330}, Qt::Vertical);
    constexpr int kSideDockWidth = 340;
    resizeDocks({sceneDock, parameterDock}, {kSideDockWidth, kSideDockWidth}, Qt::Horizontal);
    QTimer::singleShot(0, this, [this, sceneDock, parameterDock, networkDock, kSideDockWidth] {
        resizeDocks({sceneDock, parameterDock}, {kSideDockWidth, kSideDockWidth}, Qt::Horizontal);
        resizeDocks({networkDock}, {330}, Qt::Vertical);
    });

    materialNetworkView_->setGraph(&graph_);

    auto syncTransformTarget = [this](Node* node) {
        selectedSourceNode_ = node ? node->name() : QString();
        renderView_->setTransformTarget(node && node->findParameter("translate") ? node : nullptr);
    };

    // True when this tabified dock is the raised (visible) tab.
    auto dockTabRaised = [](QDockWidget* dock) -> bool {
        return dock && dock->isVisible() && !dock->visibleRegion().isEmpty();
    };

    auto syncParamsFromSceneNetwork = [this, syncTransformTarget] {
        Node* node = networkView_->selectedNode();
        if (!node) {
            parameterPanel_->clearSelection();
            syncTransformTarget(nullptr);
            sceneGraphPanel_->selectBySourceNode(QString());
            return;
        }
        parameterPanel_->setNode(node);
        syncTransformTarget(node);
        sceneGraphPanel_->selectBySourceNode(node->name());
    };

    auto syncParamsFromMaterialNetwork = [this, syncTransformTarget] {
        MaterialXSelection mtlx;
        if (materialNetworkView_->selectedMaterialX(mtlx)) {
            parameterPanel_->setMaterialXSelection(mtlx);
            syncTransformTarget(nullptr);
            return;
        }
        if (materialNetworkView_->isInsideMaterial()) {
            Node* host = materialNetworkView_->currentMaterial();
            if (host) {
                parameterPanel_->setNode(host);
                syncTransformTarget(nullptr);
                return;
            }
        } else if (Node* node = materialNetworkView_->selectedLopNode()) {
            parameterPanel_->setNode(node);
            syncTransformTarget(node);
            return;
        }
        parameterPanel_->clearSelection();
        syncTransformTarget(nullptr);
    };

    connect(networkDock, &QDockWidget::visibilityChanged, this,
            [dockTabRaised, networkDock, syncParamsFromSceneNetwork](bool visible) {
                if (visible && dockTabRaised(networkDock)) syncParamsFromSceneNetwork();
            });
    connect(materialNetworkDock, &QDockWidget::visibilityChanged, this,
            [dockTabRaised, materialNetworkDock, syncParamsFromMaterialNetwork](bool visible) {
                if (visible && dockTabRaised(materialNetworkDock)) syncParamsFromMaterialNetwork();
            });

    auto selectNodeForEditing = [this, syncTransformTarget](Node* node, bool centerNetwork = false) {
        parameterPanel_->setNode(node);
        syncTransformTarget(node);
        networkView_->selectNode(node, centerNetwork);
        sceneGraphPanel_->selectBySourceNode(node ? node->name() : QString());
    };

    connect(renderView_, &RenderView::objectSelected, this,
            [this, selectNodeForEditing](const QString& sourceNode) {
                if (sourceNode.isEmpty()) {
                    selectedSourceNode_.clear();
                    selectNodeForEditing(nullptr, false);
                    statusBar()->showMessage("Selection cleared", 1500);
                    return;
                }
                selectedSourceNode_ = sourceNode;
                if (Node* node = graph_.findNode(sourceNode)) {
                    selectNodeForEditing(node, false);
                    statusBar()->showMessage("Selected " + sourceNode, 2000);
                }
            });

    connect(renderView_, &RenderView::focusDistancePicked, this, [this](float distanceMetres) {
        Node* camera = findCameraNodeByName(lookThroughCameraName_);
        if (!camera) camera = findCameraNode();
        if (!camera) {
            statusBar()->showMessage("No camera node — add Camera in the Scene Network", 4000);
            parameterPanel_->setFocusPickActive(false);
            return;
        }
        camera->setParameterValue("focusdistance", double(distanceMetres));
        if (scene_) {
            applyLensFromCameraNode(camera, scene_->camera);
            scene_->camera.focusDistance = distanceMetres;
            session_.updateSceneData();
            if (renderArmed()) session_.start();
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

    connect(networkView_, &NodeGraphView::nodeSelected, this,
            [this, networkDock, dockTabRaised, syncTransformTarget, syncParamsFromSceneNetwork](Node*) {
                // Only drive Parameters while Scene Network is the raised tab.
                if (!dockTabRaised(networkDock)) return;
                syncParamsFromSceneNetwork();
            });
    connect(networkView_, &NodeGraphView::statusMessage, this,
            [this](const QString& message) { statusBar()->showMessage(message, 3000); });
    connect(materialNetworkView_, &MaterialNetworkView::selectionChanged, this,
            [materialNetworkDock, dockTabRaised, syncParamsFromMaterialNetwork] {
                if (!dockTabRaised(materialNetworkDock)) return;
                syncParamsFromMaterialNetwork();
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
                selectedSourceNode_ = sourceNode;
                if (Node* node = graph_.findNode(sourceNode)) selectNodeForEditing(node, false);
            });
    connect(sceneGraphPanel_, &SceneGraphPanel::frameSelectedRequested, this, [this] {
        renderView_->frameSelection();
    });
    connect(sceneGraphPanel_, &SceneGraphPanel::frameAllRequested, this, [this] {
        renderView_->frameAll();
    });
    connect(parameterPanel_, &ParameterPanel::parameterEdited, this,
            [this](Node* node, const QString& parameterName) {
                if (renderView_->isGizmoDragging()) return;
                scheduleCook();
                // Integrator change may show/hide spectral-only MaterialX inputs.
                if (node && node->typeName() == QLatin1String("rendersettings") &&
                    parameterName == QLatin1String("integrator") && parameterPanel_->showingMaterialX()) {
                    MaterialXSelection mtlx;
                    if (materialNetworkView_->selectedMaterialX(mtlx))
                        parameterPanel_->setMaterialXSelection(mtlx);
                }
            });
    connect(parameterPanel_, &ParameterPanel::parameterAction, this, &MainWindow::onParameterAction);
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
                // Defer: rebuilding Parameters inside QCheckBox::toggled / browse clicked crashes.
                if (inputName == QLatin1String("input_per_axis")) {
                    QTimer::singleShot(0, this, [this] {
                        MaterialXSelection mtlx;
                        if (materialNetworkView_->selectedMaterialX(mtlx))
                            parameterPanel_->setMaterialXSelection(mtlx);
                    });
                }
            });
    connect(parameterPanel_, &ParameterPanel::materialXTypeEdited, this,
            [this](Node*, const QString&, const QString& type) {
                if (materialNetworkView_->setSelectedMaterialXType(type)) {
                    MaterialXSelection mtlx;
                    if (materialNetworkView_->selectedMaterialX(mtlx))
                        parameterPanel_->setMaterialXSelection(mtlx);
                }
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
    buildDefaultGraph(graph_);
    networkView_->setGraph(&graph_);
    parameterPanel_->clearSelection();
    materialNetworkView_->goUp();
    renderView_->setTransformTarget(nullptr);
    selectedSourceNode_.clear();
    cameraOverride_ = false;
    lookThroughCameraName_.clear();
    enterIdlePlaceholder();
    updateWindowTitle();
    selectDisplayNode();
    if (Node* cam = findCameraNode()) lookThroughCamera(cam->name());
    else refreshViewportCameraMenu();
    statusBar()->showMessage("Scene ready — press Start to cook and render", 5000);
}

void MainWindow::newSceneFromAlembic(const QString& alembicPath, const QString& hdriPath) {
    buildAlembicGraph(graph_, alembicPath, hdriPath);
    networkView_->setGraph(&graph_);
    parameterPanel_->clearSelection();
    materialNetworkView_->goUp();
    renderView_->setTransformTarget(nullptr);
    selectedSourceNode_.clear();
    cameraOverride_ = false;
    lookThroughCameraName_.clear();
    enterIdlePlaceholder();
    updateWindowTitle();
    selectDisplayNode();
    if (Node* cam = findCameraNode()) lookThroughCamera(cam->name());
    else refreshViewportCameraMenu();
    statusBar()->showMessage("Scene ready — press Start to cook and render", 5000);
}

bool MainWindow::openScene(const QString& path) {
    QString error;
    if (!loadGraphFromFile(graph_, path, error)) {
        appMessageBox(this, "Open scene", error);
        return false;
    }
    networkView_->setGraph(&graph_);
    parameterPanel_->clearSelection();
    materialNetworkView_->goUp();
    renderView_->setTransformTarget(nullptr);
    selectedSourceNode_.clear();
    cameraOverride_ = false;
    lookThroughCameraName_.clear();
    enterIdlePlaceholder();
    updateWindowTitle();
    selectDisplayNode();
    if (Node* cam = findCameraNode()) lookThroughCamera(cam->name());
    else refreshViewportCameraMenu();
    statusBar()->showMessage("Opened — press Start to cook and render", 5000);
    return true;
}

bool MainWindow::saveScene(const QString& path) {
    QString error;
    if (!saveGraphToFile(graph_, path, error)) {
        appMessageBox(this, "Save scene", error);
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
        appMessageBox(this, "Save image", "Nothing has been rendered yet.");
        return;
    }
    std::string error;
    const RenderSettingsData settings = scene_ ? scene_->settings : RenderSettingsData();
    const bool ok = saveImageAuto(path.toStdString(), linear, settings, error);
    if (!ok) {
        appMessageBox(this, "Save image", QString::fromStdString(error));
        return;
    }
    statusBar()->showMessage("Wrote " + path, 4000);
}

void MainWindow::onParameterAction(Node* node, const QString& parameterName) {
    if (!node) return;
    if (node->typeName() == QLatin1String("rendersettings") &&
        parameterName == QLatin1String("render")) {
        startStillFrameRender(node);
    }
}

void MainWindow::startStillFrameRender(Node* renderSettings) {
    if (!renderSettings) return;
    ocioLogStatus(renderSettings->boolValue("ociousenv", true),
                  renderSettings->stringValue("ocioconfig", "").toStdString());

    QString path = renderSettings->stringValue("outputpath", "render.exr").trimmed();
    if (path.isEmpty()) path = QStringLiteral("render.exr");
    if (QFileInfo(path).suffix().isEmpty()) path += QStringLiteral(".exr");
    if (QFileInfo(path).isRelative()) {
        const QString base = graph_.filePath().isEmpty()
                                 ? QDir::currentPath()
                                 : QFileInfo(graph_.filePath()).absolutePath();
        path = QDir(base).filePath(path);
    }
    stillFramePath_ = QFileInfo(path).absoluteFilePath();
    stillFramePending_ = true;

    // Same arming path as toolbar Start: cook current settings, then one full spp pass.
    const bool fromPlaceholder = renderView_ && renderView_->isShowingPlaceholder();
    setRenderArmed(true);
    if (renderView_) {
        renderView_->setNavigationEnabled(true);
        if (fromPlaceholder) {
            renderView_->beginPlaceholderFade(1000);
        } else {
            renderView_->showPlaceholder(false);
        }
    }
    session_.releaseDeviceKeepDisplay();
    tessCache_.clear();
    tessCacheFingerprint_.clear();
    scene_.reset();
    renderRequested_ = true;
    statusBar()->showMessage(QStringLiteral("Still frame: rendering → %1").arg(stillFramePath_));
    cookNow();
}

void MainWindow::maybeSaveStillFrame() {
    if (!stillFramePending_) return;
    if (session_.isRendering()) return;

    const RenderProgress progress = session_.progress();
    if (progress.samplesDone < progress.samplesTarget || progress.samplesTarget <= 0) {
        // Incomplete finish: keep waiting if Start is still armed (IPR restart mid-pass).
        if (!renderArmed()) {
            stillFramePending_ = false;
            statusBar()->showMessage(QStringLiteral("Still frame cancelled"), 4000);
        }
        return;
    }

    stillFramePending_ = false;
    // Still frame: linear working-space beauty at the configured bit depth.
    Image linear = session_.linearImage();
    if (linear.empty()) {
        appMessageBox(this, QStringLiteral("Still frame"),
                      QStringLiteral("Render finished but the buffer is empty."));
        return;
    }
    if (scene_) {
        for (int y = 0; y < linear.height(); ++y)
            for (int x = 0; x < linear.width(); ++x)
                linear.setRgb(x, y, quantizeRgb(linear.rgb(x, y), scene_->settings.bitDepth));
    }

    const QFileInfo info(stillFramePath_);
    if (!info.absolutePath().isEmpty()) QDir().mkpath(info.absolutePath());

    std::string error;
    const int bitDepth = scene_ ? scene_->settings.bitDepth : 16;
    if (!saveImageExr(stillFramePath_.toStdString(), linear, error, bitDepth)) {
        appMessageBox(this, QStringLiteral("Still frame"), QString::fromStdString(error));
        return;
    }
    statusBar()->showMessage(
        QStringLiteral("Still frame written: %1").arg(stillFramePath_), 6000);
}

void MainWindow::onRenderFinished() {
    const RenderProgress progress = session_.progress();
    if (!progress.message.empty()) {
        stillFramePending_ = false;
        setRenderArmed(false);
        renderRequested_ = false;
        if (renderView_) renderView_->setNavigationEnabled(false);
        const QString text = QString::fromStdString(progress.message);
        statusBar()->showMessage(text, 0);
        updateStatusBar();
        const QString title = progress.backendName.find("XPU") != std::string::npos
                                  ? QStringLiteral("XPU (Embree+OptiX)")
                                  : (progress.backendName.find("OptiX") != std::string::npos
                                         ? QStringLiteral("GPU (OptiX)")
                                         : QStringLiteral("Render"));
        appMessageBox(this, title, text);
        return;
    }
    maybeSaveStillFrame();
}

// ---------------------------------------------------------------------------
// Cook and render
// ---------------------------------------------------------------------------

void MainWindow::scheduleCook(int delayMilliseconds) {
    // Safe idle: while Stop is pressed, do not cook the graph at all.
    if (!renderArmed()) {
        if (cookTimer_) cookTimer_->stop();
        return;
    }
    if (!cookTimer_) return;
    cookTimer_->start(delayMilliseconds);
}

void MainWindow::enterIdlePlaceholder() {
    session_.stop();
    session_.discardPreviousRender();
    setRenderArmed(false);
    renderRequested_ = false;
    if (cookTimer_) cookTimer_->stop();
    tessCache_.clear();
    tessCacheFingerprint_.clear();
    scene_.reset();
    stage_.reset();
    if (sceneGraphPanel_) sceneGraphPanel_->setStage(nullptr, {});
    if (renderView_) {
        renderView_->setNavigationEnabled(false);
        renderView_->showPlaceholder(true);
    }
    framePending_.store(false, std::memory_order_relaxed);
    updateStatusBar();
}

void MainWindow::onCookTimeout() { cookNow(); }

void MainWindow::cookNow() {
    CookContext context;
    if (!graph_.filePath().isEmpty()) context.sceneDirectory = QFileInfo(graph_.filePath()).absolutePath();
    if (timelineBar_) {
        context.frame = timelineBar_->currentFrame();
        context.fps = timelineBar_->fps();
        context.time = timelineBar_->timeSeconds();
    }
    setExprFrame(context.frame);

    const bool timelineInteractive = timelineBar_ && timelineBar_->isInteractive();

    // Arm TX conversion for MaterialX / dome image loads during cook (Start + Render).
    // toScene() also arms TX, but textures are decoded earlier in cookDisplay.
    RenderSettingsData txArm{};
    bool txArmed = false;
    for (const NodePtr& node : graph_.nodes()) {
        if (!node || node->typeName() != QLatin1String("rendersettings")) continue;
        txArm.enableTxCache = node->boolValue("enabletxcache", true) ? 1 : 0;
        txArm.ocioUseEnv = node->boolValue("ociousenv", true) ? 1 : 0;
        {
            const std::string dir = node->stringValue("txcachedir", "tx_cache").toStdString();
            const size_t maxLen = sizeof(txArm.txCacheDir) - 1;
            std::strncpy(txArm.txCacheDir, dir.c_str(), maxLen);
            txArm.txCacheDir[maxLen] = '\0';
        }
        {
            const std::string cfg = node->stringValue("ocioconfig", "").toStdString();
            const size_t maxLen = sizeof(txArm.ocioConfigPath) - 1;
            std::strncpy(txArm.ocioConfigPath, cfg.c_str(), maxLen);
            txArm.ocioConfigPath[maxLen] = '\0';
        }
        txArmed = true;
        break;
    }
    if (txArmed) setActiveTxCacheSettings(&txArm);
    struct TxCookGuard {
        ~TxCookGuard() { setActiveTxCacheSettings(nullptr); }
    } txCookGuard;

    stage_ = graph_.cookDisplay(context);
    if (timelineBar_ && context.hasSuggestedRange)
        timelineBar_->suggestTimeRange(context.suggestedStartTime, context.suggestedEndTime);

    if (!timelineInteractive) {
        QStringList materialContainers;
        for (const NodePtr& node : graph_.nodes()) {
            if (node && node->typeName() == "material") materialContainers << node->name();
        }
        materialContainers.sort(Qt::CaseInsensitive);
        sceneGraphPanel_->setStage(stage_, materialContainers);
    }
    // Do not rebuild Parameters on every cook — that steals focus / selection while editing.

    if (!stage_) return;

    // Keep look-through camera if it still exists.
    if (!lookThroughCameraName_.isEmpty() && !findCameraNodeByName(lookThroughCameraName_)) {
        lookThroughCameraName_.clear();
    }

    // Prefer the look-through camera as the stage render camera so toScene()
    // picks its lens (focusDistance / fStop), not another camera or a framed default.
    if (Node* cam = findCameraNodeByName(lookThroughCameraName_)) {
        for (StagePrim& prim : stage_->prims) {
            if (prim.type != PrimType::Camera || prim.sourceNode != cam->name()) continue;
            stage_->renderCameraPath = prim.path;
            // Live node params win (Focus Pick / quiet nav may be ahead of a stale cache).
            prim.camera.focalLength = float(cam->floatValue("focal", prim.camera.focalLength));
            prim.camera.sensorWidth = float(cam->floatValue("aperture", prim.camera.sensorWidth));
            prim.camera.fStop = float(cam->floatValue("fstop", prim.camera.fStop));
            const float focus = float(cam->floatValue("focusdistance", prim.camera.focusDistance));
            if (focus > 0.0f) prim.camera.focusDistance = focus;
            prim.camera.opticalModel = cam->intValue("opticalmodel", prim.camera.opticalModel);
            prim.camera.lensModel = cam->intValue("lensmodel", prim.camera.lensModel);
            prim.camera.opticalWavelengthNm =
                float(cam->floatValue("wavelength", prim.camera.opticalWavelengthNm));
            prim.camera.chromaticAberration = cam->boolValue("chromatic", false) ? 1 : 0;
            break;
        }
    }

    ScenePtr builtScene = stage_->toScene();
    if (!builtScene) return;
    builtScene->fastRebuild = timelineInteractive;

    if (Node* cam = findCameraNodeByName(lookThroughCameraName_)) {
        // Looking through: transform + full thin-lens params from that camera node.
        applyCameraNodeToView(cam);
        applyLensFromCameraNode(cam, builtScene->camera);
        builtScene->camera.cameraToWorld = renderView_->camera().toMatrix();
    } else if (cameraOverride_) {
        // Free persp — keep authored DOF focusDistance; only override the transform.
        builtScene->camera.cameraToWorld = renderView_->camera().toMatrix();
    } else {
        float frameDistance = renderView_->camera().distance;
        if (Node* camNode = findCameraNode()) {
            if (camNode->boolValue("uselookat", true)) {
                const Vec3 eye = camNode->vec3Value("eye", Vec3(6.0f, 4.0f, 9.0f));
                const Vec3 target = camNode->vec3Value("target", Vec3(0.0f, 1.0f, 0.0f));
                frameDistance = std::max(0.05f, length(eye - target));
            } else if (builtScene->camera.focusDistance > 0.0f) {
                frameDistance = builtScene->camera.focusDistance;
            }
        } else if (builtScene->camera.focusDistance > 0.0f) {
            frameDistance = builtScene->camera.focusDistance;
        } else if (builtScene->bounds().valid()) {
            frameDistance = std::max(1.0f, builtScene->bounds().radius() * 2.0f);
        }
        renderView_->camera().setFromMatrix(builtScene->camera.cameraToWorld, frameDistance);
    }

    if (builtScene->settings.motionBlur) {
        // Stop IPR before the multi-key Alembic cooks — they can take a while on
        // large caches, and must not mutate a Scene still visible to the UI/render tick.
        session_.stop();
        const Mat4 interactiveCam = builtScene->camera.cameraToWorld;
        attachMotionBlurKeys(graph_, context, *builtScene);
        builtScene->camera.cameraToWorld = interactiveCam;
        if (cameraOverride_) {
            // Free / tumbled viewport: geometry MB only — camera keys follow the live view.
            // (Otherwise cameraToWorldAtTime ignores camera.cameraToWorld and orbit looks stuck.)
            if (!builtScene->cameraMotionXforms.empty())
                std::fill(builtScene->cameraMotionXforms.begin(), builtScene->cameraMotionXforms.end(),
                          interactiveCam);
        }
        // Looking through without override: keep authored cameraMotionXforms for camera MB.
        // Motion keys force a full rebuild; deformation buffers need HIGH quality.
        builtScene->fastRebuild = false;
    }

    // Publish only after motion keys are fully installed.
    const std::string tessKey = tessellationFingerprint(*builtScene);
    const bool frameCook = timelineFrameCook_;
    timelineFrameCook_ = false;
    const bool displaceOn = builtScene->settings.enableDisplacement != 0;

    bool needTess = renderRequested_ && displaceOn;
    // Timeline frame change + Displacement: never paste Start-frame densify over
    // freshly cooked animated cages (that froze Alembic while Start was armed).
    // Static grounds stay on tessCache_; timeDependent meshes keep new cages and
    // are re-diced below when they carry geometric displacement.
    const bool partialTimedTess = !needTess && frameCook && displaceOn;
    if (!displaceOn) {
        // Cages only — drop any densify so we never apply stale Pref meshes.
        tessCache_.clear();
        tessCacheFingerprint_.clear();
    } else if (!needTess && !partialTimedTess && !tessCache_.empty() && tessKey != tessCacheFingerprint_) {
        // Subdiv / Screen Adaptive / Dicing Quality / frustum / etc. changed.
        // Keep the previous densify on screen — only an explicit Start re-dices
        // (renderRequested_). While Stopped, drop the cache so cages show.
        if (!renderArmed()) {
            tessCache_.clear();
            tessCacheFingerprint_.clear();
        }
    }

    auto runTess = [&](bool onlyTimeDependent) {
        session_.releaseDeviceKeepDisplay();
        if (!onlyTimeDependent) {
            tessCache_.clear();
            tessCacheFingerprint_.clear();
        }
        scene_.reset();
        const CameraData diceCam = [&]() {
            CameraData cam = builtScene->camera;
            if (!stage_ || builtScene->settings.dicingCameraMode != kDicingCameraCustom) return cam;
            if (stage_->dicingCameraPath.isEmpty()) return cam;
            for (const StagePrim& prim : stage_->prims) {
                if (prim.type != PrimType::Camera || prim.path != stage_->dicingCameraPath) continue;
                cam = prim.camera;
                cam.cameraToWorld = prim.xform;
                return cam;
            }
            logWarning("dicing camera path not found — using render camera");
            return cam;
        }();
        try {
            if (!onlyTimeDependent) tessCacheFingerprint_ = tessellationFingerprint(*builtScene);
            auto lastUi = std::chrono::steady_clock::now();
            tessellateSceneForRender(
                *builtScene, diceCam,
                [&](const std::string& msg) {
                    if (!renderView_) return;
                    renderView_->setStatusTextRight(QString::fromStdString(msg));
                    const auto now = std::chrono::steady_clock::now();
                    if (now - lastUi < std::chrono::milliseconds(50)) return;
                    lastUi = now;
                    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
                },
                onlyTimeDependent);
            if (renderView_) renderView_->setStatusTextRight(QString());
            if (onlyTimeDependent) mergeTessellationCache(*builtScene);
            else storeTessellationCache(*builtScene);
        } catch (const std::bad_alloc&) {
            if (renderView_) renderView_->setStatusTextRight(QString());
            if (!onlyTimeDependent) tessCacheFingerprint_.clear();
            logError("Render tessellation ran out of memory — using undisplaced cages");
            appMessageBox(this, QStringLiteral("Render"),
                          QStringLiteral("Tessellation ran out of memory.\n"
                                         "Lower Subdiv Iterations on the mesh and try again."),
                          QMessageBox::Ok);
        } catch (const std::exception& ex) {
            if (renderView_) renderView_->setStatusTextRight(QString());
            if (!onlyTimeDependent) tessCacheFingerprint_.clear();
            logError(std::string("Render tessellation failed: ") + ex.what());
            appMessageBox(this, QStringLiteral("Render"),
                          QStringLiteral("Tessellation failed:\n%1").arg(ex.what()),
                          QMessageBox::Ok);
        }
    };

    if (needTess) {
        runTess(false);
    } else if (partialTimedTess) {
        // Restore densify for static meshes; keep fresh animated cages.
        applyTessellationCache(*builtScene, /*skipTimeDependent=*/true);
        bool rediceTimed = false;
        auto cacheWasTimed = [&](const std::string& path) {
            for (const auto& entry : tessCache_) {
                if (entry.first == path && entry.second && entry.second->timeDependent) return true;
            }
            return false;
        };
        for (const PrimRecord& prim : builtScene->prims) {
            if (prim.instanceIndex < 0 || size_t(prim.instanceIndex) >= builtScene->instances.size())
                continue;
            const InstanceData& inst = builtScene->instances[size_t(prim.instanceIndex)];
            if (inst.meshIndex < 0 || size_t(inst.meshIndex) >= builtScene->meshes.size()) continue;
            const MeshPtr& m = builtScene->meshes[size_t(inst.meshIndex)];
            if (!m) continue;
            if (!m->timeDependent && !cacheWasTimed(prim.path)) continue;
            // Ensure re-dice sees the flag even if the fresh cage omitted it.
            if (!m->timeDependent) m->timeDependent = true;
            if (inst.materialIndex < 0 || size_t(inst.materialIndex) >= builtScene->materials.size())
                continue;
            if (materialHasGeometricDisplacement(
                    builtScene->materials[size_t(inst.materialIndex)])) {
                rediceTimed = true;
            }
        }
        if (rediceTimed) runTess(true);
        // Always refresh tessCache_ for timeDependent meshes (cages or re-diced).
        // Otherwise a later cook without frameCook pastes the Start-frame character.
        mergeTessellationCache(*builtScene);
    } else if (displaceOn) {
        applyTessellationCache(*builtScene, /*skipTimeDependent=*/false);
    }

    scene_ = builtScene;
    renderView_->setResolution(scene_->settings.resolutionX, scene_->settings.resolutionY);

    session_.setScene(scene_);
    updateStatusBar();
    if (!timelineInteractive) refreshViewportCameraMenu();

    if (renderRequested_ || needTess || partialTimedTess) {
        if (renderRequested_ || renderArmed()) setRenderArmed(true);
        restartRender();
    } else if (renderArmed()) {
        // While Start is pressed, graph edits re-cook and restart (former IPR-on behavior).
        restartRender();
    }
}

void MainWindow::applyTessellationCache(Scene& scene, bool skipTimeDependent) const {
    if (tessCache_.empty()) return;
    bool swapped = false;
    for (const PrimRecord& prim : scene.prims) {
        if (prim.instanceIndex < 0 || size_t(prim.instanceIndex) >= scene.instances.size()) continue;
        InstanceData& inst = scene.instances[size_t(prim.instanceIndex)];
        if (inst.meshIndex < 0 || size_t(inst.meshIndex) >= scene.meshes.size()) continue;
        MeshPtr& mesh = scene.meshes[size_t(inst.meshIndex)];
        if (!mesh) continue;
        if (skipTimeDependent && mesh->timeDependent) continue;
        for (const auto& entry : tessCache_) {
            if (entry.first != prim.path || !entry.second) continue;
            // Cached densify from an animated mesh must not overwrite a new cage,
            // even if the fresh cage lost its timeDependent stamp.
            if (skipTimeDependent && entry.second->timeDependent) break;
            mesh = entry.second;
            swapped = true;
            break;
        }
    }
    // Cached meshes already carry Pref; refresh views so SceneView pointers match.
    if (swapped) scene.finalize();
}

void MainWindow::storeTessellationCache(const Scene& scene) {
    tessCache_.clear();
    for (const PrimRecord& prim : scene.prims) {
        if (prim.instanceIndex < 0 || size_t(prim.instanceIndex) >= scene.instances.size()) continue;
        const InstanceData& inst = scene.instances[size_t(prim.instanceIndex)];
        if (inst.meshIndex < 0 || size_t(inst.meshIndex) >= scene.meshes.size()) continue;
        tessCache_.push_back({prim.path, scene.meshes[size_t(inst.meshIndex)]});
    }
}

void MainWindow::mergeTessellationCache(const Scene& scene) {
    for (const PrimRecord& prim : scene.prims) {
        if (prim.instanceIndex < 0 || size_t(prim.instanceIndex) >= scene.instances.size()) continue;
        const InstanceData& inst = scene.instances[size_t(prim.instanceIndex)];
        if (inst.meshIndex < 0 || size_t(inst.meshIndex) >= scene.meshes.size()) continue;
        const MeshPtr& mesh = scene.meshes[size_t(inst.meshIndex)];
        if (!mesh || !mesh->timeDependent) continue;
        bool found = false;
        for (auto& entry : tessCache_) {
            if (entry.first != prim.path) continue;
            entry.second = mesh;
            found = true;
            break;
        }
        if (!found) tessCache_.push_back({prim.path, mesh});
    }
}

void MainWindow::onStartRender() {
    // Log OCIO availability every time rendering starts.
    {
        bool useEnv = true;
        std::string cfg;
        for (const NodePtr& node : graph_.nodes()) {
            if (!node || node->typeName() != QLatin1String("rendersettings")) continue;
            useEnv = node->boolValue("ociousenv", true);
            cfg = node->stringValue("ocioconfig", "").toStdString();
            break;
        }
        ocioLogStatus(useEnv, cfg);
    }

    const bool fromPlaceholder = renderView_ && renderView_->isShowingPlaceholder();
    setRenderArmed(true);
    if (renderView_) {
        renderView_->setNavigationEnabled(true);
        if (fromPlaceholder) {
            renderView_->beginPlaceholderFade(1000);
        } else {
            // Stop→Start: keep last beauty in the view; do not clear image_.
            renderView_->showPlaceholder(false);
        }
    }
    // Free BVH / scene / accum so cook/tess has RAM headroom, but keep the
    // last display hold so the viewport does not flash black during re-dice.
    session_.releaseDeviceKeepDisplay();
    tessCache_.clear();
    tessCacheFingerprint_.clear();
    scene_.reset();
    renderRequested_ = true;
    cookNow();
}

void MainWindow::onStopRender() {
    setRenderArmed(false);
    renderRequested_ = false;
    if (stillFramePending_) {
        stillFramePending_ = false;
        statusBar()->showMessage("Still frame cancelled", 4000);
    }
    if (cookTimer_) cookTimer_->stop();
    session_.stop();
    if (renderView_) {
        renderView_->setNavigationEnabled(false);
        renderView_->showPlaceholder(false);  // keep last beauty frame
    }
    statusBar()->showMessage("Stopped — last frame held; press Start to cook/render again", 4000);
    updateStatusBar();
}

CameraData MainWindow::resolveDicingCamera(const Scene& scene) const {
    CameraData cam = scene.camera;
    if (!stage_ || scene.settings.dicingCameraMode != kDicingCameraCustom) return cam;
    if (stage_->dicingCameraPath.isEmpty()) return cam;
    for (const StagePrim& prim : stage_->prims) {
        if (prim.type != PrimType::Camera || prim.path != stage_->dicingCameraPath) continue;
        cam = prim.camera;
        cam.cameraToWorld = prim.xform;
        return cam;
    }
    return cam;
}

void MainWindow::restartRender() {
    // Drop stale UI blits so we do not resolve a freshly cleared buffer mid-pass.
    framePending_.store(false, std::memory_order_relaxed);
    session_.stop();
    session_.framebuffer().clear();
    session_.start();
    renderRequested_ = false;
}

bool MainWindow::renderArmed() const {
    return renderAction_ && renderAction_->isChecked();
}

void MainWindow::setRenderArmed(bool armed) {
    if (!renderAction_ || !stopAction_) return;
    // Block signals so toggling check state does not re-enter Start/Stop slots.
    const QSignalBlocker blockStart(renderAction_);
    const QSignalBlocker blockStop(stopAction_);
    renderAction_->setChecked(armed);
    stopAction_->setChecked(!armed);
}

void MainWindow::onRenderTick() {
    const bool fade = renderView_ && renderView_->placeholderFadeActive();
    const bool pending = framePending_.exchange(false, std::memory_order_relaxed);
    if (!pending && !fade) return;
    if (!renderView_) return;
    if (renderArmed() || fade) {
        if (scene_) {
            scene_->settings.viewTransform = renderView_->viewTransform();
            scene_->settings.colorManagement = renderView_->colorManagement();
        }
        renderView_->setImage(toQImage(session_.displayImage()));
        updateStatusBar();
    }
}

void MainWindow::onCameraMoved() {
    // Idle / Stop: camera frozen — ignore orbit/pan/dolly.
    if (!renderArmed()) return;
    cameraOverride_ = true;
    if (Node* cam = findCameraNodeByName(lookThroughCameraName_)) {
        // Looking through: navigation authors the camera node (Houdini lock-to-camera).
        writeViewToCameraNode(cam);
    }
    if (!scene_) return;
    const Mat4 camXform = renderView_->camera().toMatrix();
    scene_->camera.cameraToWorld = camXform;
    // Beauty rays sample cameraMotionXforms when MB is on. If those keys stay at the
    // pre-orbit transform, tumble/pan/dolly appear completely broken.
    if (!scene_->cameraMotionXforms.empty()) {
        std::fill(scene_->cameraMotionXforms.begin(), scene_->cameraMotionXforms.end(), camXform);
    }
    // Do not overwrite CameraData.focusDistance with the orbit radius — that broke DOF.
    // Re-assert lens from the look-through node so DOF survives tumble/IPR refresh.
    if (const Node* cam = findCameraNodeByName(lookThroughCameraName_)) {
        applyLensFromCameraNode(cam, scene_->camera);
    }

    // Live IPR during tumble: soft-restart on the render thread (no UI join).
    // Hard updateSceneData()+start() joined every mousemove and made orbit hitch.
    session_.pushInteractiveRestart();
}

void MainWindow::onLookThroughCameraNode() {
    if (Node* cam = findCameraNode()) {
        lookThroughCamera(cam->name());
        return;
    }
    lookThroughCamera(QString());
    statusBar()->showMessage("No camera node in the network", 3000);
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

Node* MainWindow::findCameraNodeByName(const QString& name) const {
    if (name.isEmpty()) return nullptr;
    return graph_.findNode(name);
}

QStringList MainWindow::listCameraNodeNames() const {
    QStringList names;
    for (const NodePtr& node : graph_.nodes()) {
        if (node && node->typeName() == QLatin1String("camera")) names << node->name();
    }
    names.sort(Qt::CaseInsensitive);
    return names;
}

void MainWindow::refreshViewportCameraMenu() {
    if (!renderView_) return;
    // Drop stale look-through if the node vanished.
    if (!lookThroughCameraName_.isEmpty() && !findCameraNodeByName(lookThroughCameraName_)) {
        lookThroughCameraName_.clear();
    }
    renderView_->setCameraMenu(listCameraNodeNames(), lookThroughCameraName_);
}

void MainWindow::lookThroughCamera(const QString& cameraName) {
    if (cameraName.isEmpty()) {
        lookThroughCameraName_.clear();
        cameraOverride_ = true;
        refreshViewportCameraMenu();
        statusBar()->showMessage("Free perspective (persp)", 2500);
        if (scene_) {
            scene_->camera.cameraToWorld = renderView_->camera().toMatrix();
            if (!scene_->cameraMotionXforms.empty()) {
                std::fill(scene_->cameraMotionXforms.begin(), scene_->cameraMotionXforms.end(),
                          scene_->camera.cameraToWorld);
            }
            session_.updateSceneData();
            if (renderArmed()) session_.start();
        }
        return;
    }
    Node* camera = findCameraNodeByName(cameraName);
    if (!camera || camera->typeName() != QLatin1String("camera")) {
        statusBar()->showMessage("Camera not found: " + cameraName, 3000);
        refreshViewportCameraMenu();
        return;
    }
    lookThroughCameraName_ = camera->name();
    cameraOverride_ = false;
    applyCameraNodeToView(camera);
    refreshViewportCameraMenu();
    if (scene_) {
        applyLensFromCameraNode(camera, scene_->camera);
        scene_->camera.cameraToWorld = renderView_->camera().toMatrix();
        if (!scene_->cameraMotionXforms.empty()) {
            std::fill(scene_->cameraMotionXforms.begin(), scene_->cameraMotionXforms.end(),
                      scene_->camera.cameraToWorld);
        }
        session_.updateSceneData();
        if (renderArmed()) session_.start();
    }
    statusBar()->showMessage("Looking through " + camera->name(), 3000);
}

void MainWindow::applyLensFromCameraNode(const Node* camera, CameraData& out) const {
    if (!camera) return;
    out.focalLength = float(camera->floatValue("focal", out.focalLength));
    out.sensorWidth = float(camera->floatValue("aperture", out.sensorWidth));
    out.fStop = float(camera->floatValue("fstop", out.fStop));
    const float focus = float(camera->floatValue("focusdistance", out.focusDistance));
    if (focus > 0.0f) out.focusDistance = focus;
    out.opticalModel = camera->intValue("opticalmodel", out.opticalModel);
    out.lensModel = camera->intValue("lensmodel", out.lensModel);
    out.opticalWavelengthNm = float(camera->floatValue("wavelength", out.opticalWavelengthNm));
    out.chromaticAberration = camera->boolValue("chromatic", false) ? 1 : 0;
}

void MainWindow::applyCameraNodeToView(Node* camera) {
    if (!camera || !renderView_) return;
    Mat4 cameraToWorld;
    float frameDistance = 5.0f;
    if (camera->boolValue("uselookat", true)) {
        const Vec3 eye = camera->vec3Value("eye", Vec3(6.0f, 4.0f, 9.0f));
        const Vec3 target = camera->vec3Value("target", Vec3(0.0f, 1.0f, 0.0f));
        const Vec3 up = camera->vec3Value("up", Vec3(0.0f, 1.0f, 0.0f));
        cameraToWorld = lookAtMatrix(eye, target, up);
        frameDistance = std::max(0.05f, length(eye - target));
    } else {
        cameraToWorld = transformFromParameters(*camera);
        frameDistance = std::max(0.05f, float(camera->floatValue("focusdistance", 5.0)));
    }
    renderView_->camera().setFromMatrix(cameraToWorld, frameDistance);
    renderView_->update();
}

void MainWindow::writeViewToCameraNode(Node* camera) {
    if (!camera || !renderView_) return;
    const ViewCamera& view = renderView_->camera();
    const Vec3 eye = view.eye();
    const Vec3 target = view.pivot;
    // Quiet writes — avoid cook-on-every-mouse-move; scene is updated live for IPR.
    camera->setParameterValue("uselookat", true, false);
    camera->setParameterValue("eye", QVariant::fromValue(QVector3D(eye.x, eye.y, eye.z)), false);
    camera->setParameterValue("target", QVariant::fromValue(QVector3D(target.x, target.y, target.z)),
                              false);
    camera->setParameterValue("translate", QVariant::fromValue(QVector3D(eye.x, eye.y, eye.z)), false);
    if (!graph_.isModified()) {
        graph_.setModified(true);
        updateWindowTitle();
    }
}

void MainWindow::onCopyViewToCameraNode() {
    Node* camera = findCameraNodeByName(lookThroughCameraName_);
    if (!camera) camera = findCameraNode();
    if (!camera) {
        appMessageBox(this, "Copy view", "This network has no camera node. Add one with Tab > Camera.");
        return;
    }
    writeViewToCameraNode(camera);
    // Do not overwrite focusdistance with orbit radius — that is framing, not DOF focus.
    lookThroughCamera(camera->name());
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
                       .arg(SOLSTICE_APP_NAME, SOLSTICE_BUILD_ID, file, graph_.isModified() ? "*" : ""));
}

void MainWindow::updateStatusBar() {
    const RenderProgress progress = session_.progress();
    QString overlay = QString("%1 / %2 spp").arg(progress.samplesDone).arg(progress.samplesTarget);
    if (progress.elapsedSeconds > 0.0) overlay += QStringLiteral("   %1").arg(formatElapsedHms(progress.elapsedSeconds));
    if (scene_) {
        const RenderSettingsData& rs = scene_->settings;
        const char* engine = "Buckets";
        if (rs.samplingEngine == kSamplingEngineProgressive) engine = "Progressive";
        overlay += QString("   OwenSobol · %1").arg(engine);
        if (rs.noiseThreshold > 0.0f) {
            overlay += QString("   noise %1").arg(double(rs.noiseThreshold), 0, 'g', 3);
            if (progress.noisePixelCount > 0) {
                const int pct = int(100.0 * double(progress.noiseSkipCount) /
                                        double(progress.noisePixelCount) +
                                    0.5);
                overlay += QString("  %1% skip").arg(pct);
            }
        } else {
            overlay += QStringLiteral("   noise off");
        }
    }
    if (!progress.message.empty()) {
        overlay += QStringLiteral("   %1").arg(QString::fromStdString(progress.message));
    }
    if (!renderView_) return;
    renderView_->setStatusText(overlay);

    const bool wantGpu = [&] {
        if (scene_) return renderDeviceUsesGpu(scene_->settings.backend);
        for (const NodePtr& node : graph_.nodes()) {
            if (node && node->typeName() == QLatin1String("rendersettings")) {
                const int b = node->intValue("backend", 0);
                return b == 1 || b == 2;
            }
        }
        return false;
    }();
    const bool wantXpu = [&] {
        if (scene_) return renderDeviceIsXpu(scene_->settings.backend);
        for (const NodePtr& node : graph_.nodes()) {
            if (node && node->typeName() == QLatin1String("rendersettings"))
                return node->intValue("backend", 0) == 2;
        }
        return false;
    }();

    QString active = QStringLiteral("Embree");
    const std::string& live = progress.backendName;
    if (!live.empty()) {
        if (live.find("XPU") != std::string::npos) active = QStringLiteral("XPU");
        else if (live.find("OptiX") != std::string::npos) active = QStringLiteral("OptiX");
        else if (live.find("Embree") != std::string::npos) active = QStringLiteral("Embree");
    } else if (wantXpu && optixRuntimeAvailable()) {
        active = QStringLiteral("XPU");
    } else if (wantGpu && !wantXpu && optixRuntimeAvailable()) {
        active = QStringLiteral("OptiX");
    }
    if (active.startsWith(QLatin1String("XPU")) && progress.backendGpuMs > 0.0) {
        active = QStringLiteral("XPU %1 ms").arg(progress.backendGpuMs, 0, 'f', 1);
    } else if (active.startsWith(QLatin1String("OptiX")) && progress.backendGpuMs > 0.0) {
        active = QStringLiteral("OptiX %1 ms").arg(progress.backendGpuMs, 0, 'f', 1);
    }

    QString support;
    QColor supportColor(255, 80, 80);
    if (wantGpu) {
        support = QStringLiteral("OptiX not supported");
        if (!progress.message.empty()) {
            support = QString::fromStdString(progress.message);
            if (support.size() > 48) support = support.left(46) + QStringLiteral("…");
            supportColor = QColor(255, 80, 80);
            if (wantXpu && active.startsWith(QLatin1String("Embree"))) active = QStringLiteral("XPU");
            else if (!wantXpu && active.startsWith(QLatin1String("Embree"))) active = QStringLiteral("OptiX");
        } else if (!optixBackendCompiledIn()) {
            support = QStringLiteral("OptiX not in this build");
        } else if (optixRuntimeProbePending()) {
            support = QStringLiteral("OptiX checking…");
            supportColor = QColor(255, 210, 70);
        } else {
            std::string err;
            if (optixRuntimeAvailable(&err)) {
                support = QStringLiteral("OptiX supported");
                supportColor = QColor(110, 210, 140);
            } else if (!err.empty()) {
                support = QString::fromStdString(err);
                if (support.size() > 42) support = support.left(40) + QStringLiteral("…");
            }
        }
    }
    renderView_->setBackendHud(active, support, supportColor);
}


bool MainWindow::maybeSaveChanges() {
    if (!graph_.isModified()) return true;
    QMessageBox box(this);
    box.setWindowTitle(SOLSTICE_APP_NAME);
    box.setText("The current scene has unsaved changes. Save it?");
    box.setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
    box.setDefaultButton(QMessageBox::Save);
    // Replace the default "?" system glyph with the Bob app icon.
    box.setIconPixmap(QPixmap(QStringLiteral(":/icons/app_icon_64.png")));
    box.setWindowIcon(windowIcon());
    const auto answer = static_cast<QMessageBox::StandardButton>(box.exec());
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
    QMessageBox about(this);
    about.setWindowTitle(QString("About %1").arg(SOLSTICE_APP_NAME));
    about.setWindowIcon(windowIcon());
    about.setIconPixmap(QPixmap(QStringLiteral(":/icons/app_icon_64.png")));
    about.setTextFormat(Qt::RichText);
    about.setText(
        QString("<h3>%1</h3>"
                "<p><b>Version</b> %2<br/>"
                "<b>Build</b> %3<br/>"
                "<b>Executable</b> %4</p>"
                "<p>A node based path tracer in the spirit of Houdini Solaris.</p>"
                "<ul>"
                "<li>Alembic and USD geometry import</li>"
                "<li>Area, distant and HDRI dome lights</li>"
                "<li>Progressive path tracing on the CPU with Intel Embree</li>"
                "<li>GPU path tracing with NVIDIA OptiX%5</li>"
                "</ul>")
            .arg(SOLSTICE_APP_NAME, SOLSTICE_VERSION_FULL, SOLSTICE_BUILD_ID,
                 SOLSTICE_OUTPUT_NAME,
                 optixBackendCompiledIn() ? "" : " (not compiled into this build)"));
    about.exec();
}

void MainWindow::onShowShortcuts() {
    appMessageBox(this, "Keyboard shortcuts",
                             "Scene Network\n"
                             "  Tab           add node (search)\n"
                             "  Ctrl+C / Ctrl+V   copy / paste nodes\n"
                             "  D             set display flag\n"
                             "  B             bypass\n"
                             "  F             frame all\n"
                             "  L             lay out selection\n"
                             "  Del           delete selection\n"
                             "  MMB / Alt+LMB / Space+LMB   pan\n"
                             "  Wheel         zoom to cursor\n\n"
                             "Material Network\n"
                             "  Tab           add MaterialX node (search)\n"
                             "  Ctrl+C / Ctrl+V   copy / paste nodes\n"
                             "  F             frame all\n"
                             "  Del           delete selection\n"
                             "  ↑             return to materials level\n"
                             "  Double-click  dive into material container\n"
                             "  MMB / Alt+LMB / Space+LMB   pan\n"
                             "  Wheel         zoom to cursor\n\n"
                             "Render view (Houdini style)\n"
                             "  F             frame selected object\n"
                             "  H / Home      frame all\n"
                             "  Q             select — LMB click geometry (Select tool only)\n"
                             "  T / R / S     translate / rotate / scale\n"
                             "  LMB on gizmo  transform (restarts render on release while Start)\n"
                             "  Focus Pick    camera Lens → click geo to set DOF focus\n"
                             "  Cam menu      look through a camera (nav edits that camera)\n"
                             "  Home button   frame all (same as H)\n"
                             "  Alt + LMB     tumble (pivot on raw geometry, ignores MB/lens)\n"
                             "  RMB           tumble / orbit\n"
                             "  MMB           pan\n"
                             "  Alt + RMB     dolly\n"
                             "  Wheel         dolly\n\n"
                             "Units\n"
                             "  1 scene unit = 1 metre (Houdini MKS)\n"
                             "  angles in degrees, focal length in millimetres\n\n"
                             "Timeline (under viewport)\n"
                             "  Start / End boxes    fixed range widgets (editable)\n"
                             "  |<<  /  ▶■  /  >>|   under scrubber (centered)\n"
                             "  Scrubber playhead    current frame (double-click to type)\n"
                             "  Frame → time         Alembic & USD sample time\n\n"
                             "Layout\n"
                             "  square on a pane title   detach / redock Scene Graph, Viewport, Parameters\n\n"
                             "General\n"
                             "  F5            Start cook + render (button stays pressed)\n"
                             "  Esc           Stop (no cook until Start; last frame held)\n"
                             "  Ctrl+E        save image");
}

}  // namespace sol
