// Houdini style network editor: nodes flow from top to bottom, Tab opens the
// node creation menu, and the display flag picks what the render view shows.
#pragma once

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QHash>
#include <QList>

#include "nodes/node_graph.h"

class QGraphicsPathItem;
class QLineEdit;
class QListWidget;

namespace sol {

class NodeItem;
class ConnectionItem;

class NodeGraphScene : public QGraphicsScene {
    Q_OBJECT

public:
    explicit NodeGraphScene(QObject* parent = nullptr);

    void setGraph(NodeGraph* graph);
    NodeGraph* graph() const { return graph_; }

    void rebuild();
    void updateConnections();
    NodeItem* itemForNode(Node* node) const;

protected:
    void drawBackground(QPainter* painter, const QRectF& rect) override;

private:
    NodeGraph* graph_ = nullptr;
    QHash<Node*, NodeItem*> nodeItems_;
    QList<ConnectionItem*> connections_;
};

// Searchable node creation popup, opened with Tab.
class NodeCreateMenu : public QWidget {
    Q_OBJECT

public:
    explicit NodeCreateMenu(QWidget* parent = nullptr);
    void popupAt(QPoint globalPosition);

signals:
    void nodeTypeChosen(const QString& typeName);

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;

private:
    void populate(const QString& filter);
    void accept();

    QLineEdit* search_ = nullptr;
    QListWidget* list_ = nullptr;
};

class NodeGraphView : public QGraphicsView {
    Q_OBJECT

public:
    explicit NodeGraphView(QWidget* parent = nullptr);

    void setGraph(NodeGraph* graph);
    NodeGraph* graph() const { return graph_; }
    Node* selectedNode() const;
    void frameAll();
    void selectNode(Node* node);

signals:
    void nodeSelected(sol::Node* node);
    void displayNodeRequested(sol::Node* node);
    void statusMessage(const QString& message);

public slots:
    void createNodeOfType(const QString& typeName);
    void deleteSelectedNodes();
    void toggleDisplayFlagOnSelection();
    void toggleBypassOnSelection();
    void layoutSelectionVertically();

protected:
    void showEvent(QShowEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override;
    // Tab must reach keyPressEvent instead of moving the focus.
    bool focusNextPrevChild(bool next) override { return false; }
    void drawForeground(QPainter* painter, const QRectF& rect) override;

private:
    NodeItem* nodeItemAt(QPoint viewPosition) const;
    void finishWireDrag(QPoint viewPosition);
    QPointF snapWireEndpoint(QPoint viewPosition, bool draggingFromOutput);
    void updateDragWire(QPoint viewPosition);
    void beginPan(const QPoint& viewPos);
    void updatePan(const QPoint& viewPos);
    void endPan();
    qreal zoomFactorFromWheel(const QWheelEvent* event) const;

    NodeGraph* graph_ = nullptr;
    NodeGraphScene* graphScene_ = nullptr;
    NodeCreateMenu* createMenu_ = nullptr;

    bool panning_ = false;
    bool pendingFrameAll_ = true;
    QPoint lastPanPoint_;
    QPointF lastScenePosition_;
    QGraphicsView::ViewportAnchor savedAnchor_ = QGraphicsView::AnchorUnderMouse;
    QGraphicsView::DragMode savedDragMode_ = QGraphicsView::RubberBandDrag;

    // Wire dragging state.
    NodeItem* dragSource_ = nullptr;
    NodeItem* dragDestination_ = nullptr;
    NodeItem* snapTarget_ = nullptr;
    int snapInputIndex_ = -1;
    int dragInputIndex_ = -1;
    QGraphicsPathItem* dragWire_ = nullptr;
};

}  // namespace sol
