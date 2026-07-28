// Solaris-style Material Network: material containers at the root, dive into
// MaterialX graphs on double-click, with a back control to return upstairs.
// Selected node parameters are shown in the shared Parameters dock (not here).
#pragma once

#include <QGraphicsView>
#include <QMetaObject>
#include <QPoint>
#include <QString>
#include <QVector>
#include <QWidget>

#include "nodes/node.h"

class QGraphicsPathItem;
class QGraphicsScene;
class QLabel;
class QLineEdit;
class QListWidget;
class QMouseEvent;
class QToolButton;
class QVBoxLayout;

namespace sol {

class MaterialNetworkGraphView;
class MaterialContainerGraphView;
class NodeGraph;

struct MaterialXInputParam {
    QString name;
    QString type;
    QString value;
    QString nodename;
};

struct MaterialXSelection {
    Node* hostMaterial = nullptr;
    QString category;
    QString type;
    QStringList typeVariants;
    QString name;
    QVector<MaterialXInputParam> inputs;
};

// Public Material Network dock widget: container level + MaterialX canvas.
class MaterialNetworkView : public QWidget {
    Q_OBJECT

public:
    explicit MaterialNetworkView(QWidget* parent = nullptr);

    void setGraph(NodeGraph* graph);
    void goUp();
    bool isInsideMaterial() const { return currentMaterial_ != nullptr; }
    Node* currentMaterial() const { return currentMaterial_; }

    // Selected LOP material container (root level), or host material when inside
    // with no MaterialX node selected.
    Node* selectedLopNode() const;
    bool selectedMaterialX(MaterialXSelection& out) const;
    bool renameSelectedMaterialX(const QString& newName);
    bool setSelectedMaterialXInput(const QString& inputName, const QString& value);
    bool setSelectedMaterialXType(const QString& type);

signals:
    void materialEdited(sol::Node* node);
    void statusMessage(const QString& message);
    // Fired when selection should drive the shared Parameters panel.
    void selectionChanged();

private:
    void refreshContainers();
    void diveInto(Node* material);
    void updateChrome();
    void onContainerSelectionChanged();
    void onMaterialXSelectionChanged();
    void onGraphTopologyChanged();

    NodeGraph* graph_ = nullptr;
    Node* currentMaterial_ = nullptr;
    MaterialContainerGraphView* containerView_ = nullptr;
    MaterialNetworkGraphView* graphView_ = nullptr;
    QToolButton* backButton_ = nullptr;
    QLabel* pathLabel_ = nullptr;
    QMetaObject::Connection nodeAddedConnection_;
    QMetaObject::Connection nodeRemovedConnection_;
    QMetaObject::Connection connectionsChangedConnection_;
};

// Root-level view: one container node per material LOP in the scene graph.
class MaterialContainerGraphView : public QGraphicsView {
    Q_OBJECT

public:
    explicit MaterialContainerGraphView(QWidget* parent = nullptr);

    void setMaterials(const QVector<Node*>& materials);
    Node* selectedMaterial() const;

signals:
    void selectionChanged();
    void diveRequested(sol::Node* material);
    void statusMessage(const QString& message);

protected:
    void showEvent(QShowEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void drawBackground(QPainter* painter, const QRectF& rect) override;
    bool focusNextPrevChild(bool next) override { return false; }

private:
    void frameGraph();
    bool shouldBeginPan(const QMouseEvent* event) const;
    void beginPan(const QPoint& viewPosition);
    void updatePan(const QPoint& viewPosition);
    void endPan();

    QGraphicsScene* graphScene_ = nullptr;
    bool pendingFrame_ = false;
    bool panning_ = false;
    bool spacePressed_ = false;
    QPoint lastPanPoint_;
    QGraphicsView::DragMode savedDragMode_ = QGraphicsView::RubberBandDrag;
    QGraphicsView::ViewportAnchor savedAnchor_ = QGraphicsView::AnchorUnderMouse;
};

// Internal left-to-right MaterialX canvas (inside one material container).
class MaterialNetworkGraphView : public QGraphicsView {
    Q_OBJECT

public:
    struct MtlxInput {
        QString name;
        QString type;
        QString value;
        QString nodename;
    };

    struct MtlxNode {
        QString name;
        QString category;
        QString type;
        QPointF layout;
        QVector<MtlxInput> inputs;
    };

    explicit MaterialNetworkGraphView(QWidget* parent = nullptr);

    void setMaterialNode(Node* node);
    QString selectedNodeName() const;
    const MtlxNode* selectedNode() const;
    bool renameNode(const QString& oldName, const QString& newName);
    bool setInputValue(const QString& nodeName, const QString& inputName, const QString& value);
    bool setNodeType(const QString& nodeName, const QString& type);
    void selectNodeByName(const QString& name);
    // Live-update MaterialX wires while a node is dragged.
    void updateWiresLive();

signals:
    void materialEdited(sol::Node* node);
    // Layout-only persist (xpos/ypos) — mark project dirty, do not cook.
    void materialLayoutChanged();
    void statusMessage(const QString& message);
    void selectionChanged();
    void upRequested();

protected:
    void showEvent(QShowEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void drawBackground(QPainter* painter, const QRectF& rect) override;
    void drawForeground(QPainter* painter, const QRectF& rect) override;
    bool focusNextPrevChild(bool next) override { return false; }

private:
    void rebuild();
    void rebuildFromXml(const QString& xml, bool rewriteRepaired);
    QString serializeGraph() const;
    QString defaultDocument() const;
    void ensureMtlxParameter();
    void writeXmlToMaterial(const QString& xml, bool emitEdited);
    void writeModel(bool emitEdited);
    void frameGraph();
    void showAddNodeMenu(const QPoint& viewPosition);
    void onCreateMenuChosen(const QString& category, const QString& type);
    void addNode(const QString& category, const QString& type, QPointF scenePosition);
    void connectNodes(const QString& sourceName, const QString& targetName, int inputIndex);
    void disconnectInput(const QString& targetName, const QString& inputName);
    void deleteSelectedNodes();
    bool openTextureDialogAt(const QPoint& viewPosition);
    void chooseTexture(const QString& nodeName);
    // MaterialX UDIM: convert concrete tile / pattern → unresolved <UDIM> + udimset.
    QString applyUdimFilename(const QString& path);
    void refreshUdimSetFromFilenames();
    void syncNodePositions();
    void persistLayoutQuietly();
    bool shouldBeginPan(const QMouseEvent* event) const;
    void beginPan(const QPoint& viewPosition);
    void updatePan(const QPoint& viewPosition);
    void endPan();
    void beginWire(const QString& sourceName, QPointF sourcePosition);
    void updateWire(QPointF scenePosition);
    void endWire(const QPoint& viewPosition);
    void emitSelectionChanged();
    QString uniqueNodeName(const QString& baseName) const;
    static void ensureInput(QVector<MtlxInput>& inputs, const QString& name, const QString& type,
                            const QString& value = QString());
    static QStringList canonicalInputOrder(const QString& category);
    static void normalizeInputOrder(QVector<MtlxInput>& inputs, const QString& category);
    static QVector<MtlxInput> defaultInputsForCategory(const QString& category, const QString& type = QString());
    static QString defaultTypeForCategory(const QString& category);
    MtlxNode* findModelNode(const QString& name);
    const MtlxNode* findModelNode(const QString& name) const;

    Node* materialNode_ = nullptr;
    QGraphicsScene* graphScene_ = nullptr;
    QMetaObject::Connection materialChangedConnection_;
    QMetaObject::Connection selectionConnection_;
    QVector<MtlxNode> graphNodes_;
    // MaterialX geominfo udimset (tile ids). Empty → discover from disk at cook.
    QVector<int> udimSet_;
    QGraphicsPathItem* previewWire_ = nullptr;
    class MaterialXCreateMenu* createMenu_ = nullptr;
    QPointF pendingCreateScenePos_;
    bool pendingFrame_ = false;
    bool panning_ = false;
    bool spacePressed_ = false;
    bool wiring_ = false;
    bool suppressMaterialSignal_ = false;
    bool mouseMovedSincePress_ = false;
    QPoint lastPanPoint_;
    QPoint lastMousePoint_;
    QPoint mousePressPoint_;
    QGraphicsView::DragMode savedDragMode_ = QGraphicsView::RubberBandDrag;
    QGraphicsView::ViewportAnchor savedAnchor_ = QGraphicsView::AnchorUnderMouse;
    QString wireSourceNode_;
    QPointF wireSourcePosition_;
    QString clickImageNode_;
    QString preservedSelection_;
};

}  // namespace sol
