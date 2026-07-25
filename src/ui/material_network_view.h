// Solaris-style MaterialX graph view backed by a material node's "mtlx" XML.
// Layout: graph canvas on the left, inspector for the selected MaterialX node on the right.
#pragma once

#include <QGraphicsView>
#include <QMetaObject>
#include <QPoint>
#include <QString>
#include <QVector>
#include <QWidget>

#include "nodes/node.h"

class QFormLayout;
class QGraphicsPathItem;
class QGraphicsScene;
class QLabel;
class QLineEdit;
class QMouseEvent;
class QSplitter;
class QVBoxLayout;

namespace sol {

class MaterialNetworkGraphView;

// Public Material Network dock widget: graph + right-side parameter inspector.
class MaterialNetworkView : public QWidget {
    Q_OBJECT

public:
    explicit MaterialNetworkView(QWidget* parent = nullptr);

    void setMaterialNode(Node* node);

signals:
    void materialEdited(sol::Node* node);
    void statusMessage(const QString& message);

private:
    void onGraphSelectionChanged();
    void rebuildInspector();
    void commitRename();
    void commitInputValue(const QString& inputName, const QString& type, const QString& value);

    MaterialNetworkGraphView* graphView_ = nullptr;
    QSplitter* splitter_ = nullptr;
    QWidget* inspector_ = nullptr;
    QVBoxLayout* inspectorLayout_ = nullptr;
    QLabel* inspectorHint_ = nullptr;
    QLabel* categoryLabel_ = nullptr;
    QLineEdit* nameEdit_ = nullptr;
    QWidget* paramsHost_ = nullptr;
    QFormLayout* paramsForm_ = nullptr;
    QString selectedNodeName_;
    bool updatingInspector_ = false;
};

// Internal left-to-right MaterialX canvas.
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
    void selectNodeByName(const QString& name);

signals:
    void materialEdited(sol::Node* node);
    void statusMessage(const QString& message);
    void selectionChanged();

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
    void addNode(const QString& category, const QString& type, QPointF scenePosition);
    void connectNodes(const QString& sourceName, const QString& targetName, int inputIndex);
    void disconnectInput(const QString& targetName, const QString& inputName);
    void deleteSelectedNodes();
    bool openTextureDialogAt(const QPoint& viewPosition);
    void chooseTexture(const QString& nodeName);
    void syncNodePositions();
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
    static QVector<MtlxInput> defaultInputsForCategory(const QString& category, const QString& type = QString());
    static QString defaultTypeForCategory(const QString& category);
    MtlxNode* findModelNode(const QString& name);
    const MtlxNode* findModelNode(const QString& name) const;

    Node* materialNode_ = nullptr;
    QGraphicsScene* graphScene_ = nullptr;
    QMetaObject::Connection materialChangedConnection_;
    QMetaObject::Connection selectionConnection_;
    QVector<MtlxNode> graphNodes_;
    QGraphicsPathItem* previewWire_ = nullptr;
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
