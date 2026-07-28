#include "ui/node_graph_view.h"

#include <QAction>
#include <QContextMenuEvent>
#include <QGraphicsPathItem>
#include <QKeyEvent>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QToolTip>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

#include "nodes/node_registry.h"
#include "ui/connection_item.h"
#include "ui/node_item.h"
#include "ui/theme.h"

namespace sol {

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

NodeGraphScene::NodeGraphScene(QObject* parent) : QGraphicsScene(parent) {
    setBackgroundBrush(theme::gridDark());
    setSceneRect(-8000, -8000, 16000, 16000);
}

void NodeGraphScene::setGraph(NodeGraph* graph) {
    graph_ = graph;
    rebuild();
}

NodeItem* NodeGraphScene::itemForNode(Node* node) const { return nodeItems_.value(node, nullptr); }

void NodeGraphScene::rebuild() {
    nodeItems_.clear();
    connections_.clear();
    clear();
    if (!graph_) return;

    for (const NodePtr& node : graph_->nodes()) {
        auto* item = new NodeItem(node.get(), this);
        addItem(item);
        nodeItems_.insert(node.get(), item);
    }
    updateConnections();
}

void NodeGraphScene::updateConnections() {
    if (!graph_) return;
    for (ConnectionItem* connection : connections_) {
        removeItem(connection);
        delete connection;
    }
    connections_.clear();

    for (const NodePtr& node : graph_->nodes()) {
        NodeItem* destination = nodeItems_.value(node.get(), nullptr);
        if (!destination) continue;
        for (int i = 0; i < node->inputCount(); ++i) {
            Node* inputNode = node->input(i);
            if (!inputNode) continue;
            NodeItem* source = nodeItems_.value(inputNode, nullptr);
            if (!source) continue;
            auto* connection = new ConnectionItem(source, destination, i);
            addItem(connection);
            connections_.push_back(connection);
        }
    }
}

void NodeGraphScene::refreshAllNodeItems() {
    for (auto it = nodeItems_.cbegin(); it != nodeItems_.cend(); ++it) {
        if (it.value()) it.value()->refresh();
    }
}

void NodeGraphScene::drawBackground(QPainter* painter, const QRectF& rect) {
    painter->fillRect(rect, theme::gridDark());

    const qreal step = 40.0;
    QPen minor(theme::gridLine(), 0.0);
    painter->setPen(minor);
    const qreal left = std::floor(rect.left() / step) * step;
    const qreal top = std::floor(rect.top() / step) * step;
    for (qreal x = left; x < rect.right(); x += step)
        painter->drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()));
    for (qreal y = top; y < rect.bottom(); y += step)
        painter->drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y));

    QPen major(theme::gridLine().lighter(130), 0.0);
    painter->setPen(major);
    const qreal bigStep = step * 5.0;
    const qreal bigLeft = std::floor(rect.left() / bigStep) * bigStep;
    const qreal bigTop = std::floor(rect.top() / bigStep) * bigStep;
    for (qreal x = bigLeft; x < rect.right(); x += bigStep)
        painter->drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()));
    for (qreal y = bigTop; y < rect.bottom(); y += bigStep)
        painter->drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y));
}

// ---------------------------------------------------------------------------
// Node creation popup
// ---------------------------------------------------------------------------

NodeCreateMenu::NodeCreateMenu(QWidget* parent) : QWidget(parent, Qt::Popup) {
    setObjectName("NodeCreateMenu");
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(4);

    search_ = new QLineEdit(this);
    search_->setPlaceholderText("Add node...");
    layout->addWidget(search_);

    list_ = new QListWidget(this);
    list_->setUniformItemSizes(true);
    layout->addWidget(list_);

    setMinimumWidth(300);
    setMinimumHeight(320);

    connect(search_, &QLineEdit::textChanged, this, [this](const QString& text) { populate(text); });
    connect(search_, &QLineEdit::returnPressed, this, &NodeCreateMenu::accept);
    connect(list_, &QListWidget::itemActivated, this, &NodeCreateMenu::accept);
    connect(list_, &QListWidget::itemClicked, this, &NodeCreateMenu::accept);
}

void NodeCreateMenu::populate(const QString& filter) {
    list_->clear();
    QString currentCategory;
    for (const QString& category : NodeRegistry::instance().categories()) {
        for (const NodeTypeInfo& info : NodeRegistry::instance().types()) {
            if (info.category != category) continue;
            if (!filter.isEmpty() && !info.label.contains(filter, Qt::CaseInsensitive) &&
                !info.typeName.contains(filter, Qt::CaseInsensitive))
                continue;
            if (currentCategory != category) {
                currentCategory = category;
                auto* header = new QListWidgetItem(category.toUpper());
                header->setFlags(Qt::NoItemFlags);
                header->setForeground(theme::accent());
                list_->addItem(header);
            }
            auto* item = new QListWidgetItem("   " + info.label);
            item->setData(Qt::UserRole, info.typeName);
            item->setToolTip(info.description);
            list_->addItem(item);
        }
    }
    for (int i = 0; i < list_->count(); ++i) {
        if (list_->item(i)->flags() & Qt::ItemIsSelectable) {
            list_->setCurrentRow(i);
            break;
        }
    }
}

void NodeCreateMenu::popupAt(QPoint globalPosition) {
    search_->clear();
    populate(QString());
    move(globalPosition);
    show();
    search_->setFocus();
}

void NodeCreateMenu::accept() {
    QListWidgetItem* item = list_->currentItem();
    if (item && (item->flags() & Qt::ItemIsSelectable)) {
        const QString typeName = item->data(Qt::UserRole).toString();
        hide();
        if (!typeName.isEmpty()) emit nodeTypeChosen(typeName);
        return;
    }
    hide();
}

void NodeCreateMenu::keyPressEvent(QKeyEvent* event) {
    switch (event->key()) {
        case Qt::Key_Escape: hide(); return;
        case Qt::Key_Down:
        case Qt::Key_Up: {
            const int direction = event->key() == Qt::Key_Down ? 1 : -1;
            int row = list_->currentRow();
            for (int i = 0; i < list_->count(); ++i) {
                row += direction;
                if (row < 0 || row >= list_->count()) break;
                if (list_->item(row)->flags() & Qt::ItemIsSelectable) {
                    list_->setCurrentRow(row);
                    break;
                }
            }
            return;
        }
        default: break;
    }
    QWidget::keyPressEvent(event);
}

void NodeCreateMenu::focusOutEvent(QFocusEvent* event) {
    QWidget::focusOutEvent(event);
    hide();
}

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

NodeGraphView::NodeGraphView(QWidget* parent) : QGraphicsView(parent) {
    graphScene_ = new NodeGraphScene(this);
    setScene(graphScene_);
    setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
    setDragMode(QGraphicsView::RubberBandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    // Full updates avoid antialiased icon/shadow trails while dragging nodes.
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setFocusPolicy(Qt::StrongFocus);
    setMinimumHeight(180);

    createMenu_ = new NodeCreateMenu(this);
    connect(createMenu_, &NodeCreateMenu::nodeTypeChosen, this, &NodeGraphView::createNodeOfType);

    connect(graphScene_, &QGraphicsScene::selectionChanged, this, [this] {
        emit nodeSelected(selectedNode());
    });
}

void NodeGraphView::setGraph(NodeGraph* graph) {
    graph_ = graph;
    graphScene_->setGraph(graph);
    if (!graph) return;
    connect(graph, &NodeGraph::graphChanged, this, [this] { graphScene_->updateConnections(); });
    connect(graph, &NodeGraph::nodeAdded, this, [this](Node*) { graphScene_->rebuild(); });
    connect(graph, &NodeGraph::nodeAboutToBeRemoved, this, [this](Node*) {
        // Rebuilt after removal completes so dangling items never get painted.
        QMetaObject::invokeMethod(this, [this] { graphScene_->rebuild(); }, Qt::QueuedConnection);
    });
    connect(graph, &NodeGraph::displayNodeChanged, this, [this](Node*) { graphScene_->refreshAllNodeItems(); });
    // Framing needs the final widget size, which is only known once the layout
    // has run, so it is deferred to the first show/resize.
    pendingFrameAll_ = true;
    if (isVisible() && width() > 50) {
        pendingFrameAll_ = false;
        frameAll();
    }
}

void NodeGraphView::showEvent(QShowEvent* event) {
    QGraphicsView::showEvent(event);
    if (pendingFrameAll_ && width() > 50) {
        pendingFrameAll_ = false;
        frameAll();
    }
}

void NodeGraphView::resizeEvent(QResizeEvent* event) {
    QGraphicsView::resizeEvent(event);
    if (pendingFrameAll_ && width() > 50) {
        pendingFrameAll_ = false;
        frameAll();
    }
}

Node* NodeGraphView::selectedNode() const {
    const QList<QGraphicsItem*> selected = graphScene_->selectedItems();
    for (QGraphicsItem* item : selected) {
        if (auto* nodeItem = qgraphicsitem_cast<NodeItem*>(item)) return nodeItem->node();
    }
    return nullptr;
}

void NodeGraphView::selectNode(Node* node, bool centerOnSelection) {
    graphScene_->clearSelection();
    if (NodeItem* item = graphScene_->itemForNode(node)) {
        item->setSelected(true);
        if (centerOnSelection) centerOn(item);
    }
}

void NodeGraphView::frameAll() {
    const QRectF bounds = graphScene_->itemsBoundingRect();
    if (bounds.isEmpty()) {
        resetTransform();
        centerOn(0, 0);
        return;
    }
    fitInView(bounds.adjusted(-60, -60, 60, 60), Qt::KeepAspectRatio);
    // Keep the zoom in a range where node labels stay readable.
    const double scale = transform().m11();
    if (scale > 1.0 || scale < 0.5) {
        resetTransform();
        const double clamped = std::clamp(scale, 0.5, 1.0);
        this->scale(clamped, clamped);
        centerOn(bounds.center());
    }
}

void NodeGraphView::createNodeOfType(const QString& typeName) {
    if (!graph_) return;
    Node* node = graph_->createNode(typeName, QString(), lastScenePosition_);
    if (!node) return;

    // Wire the new node under the current selection, like Houdini does.
    Node* selected = selectedNode();
    if (selected && node->inputCount() > 0 && selected != node) {
        graph_->connectNodes(selected, node, 0);
        node->setPosition(selected->position() + QPointF(0, 90));
    }
    graphScene_->rebuild();
    selectNode(node);
    graph_->setDisplayNode(node);
    emit displayNodeRequested(node);
    emit statusMessage("Created " + node->name());
}

void NodeGraphView::deleteSelectedNodes() {
    if (!graph_) return;
    QList<Node*> toRemove;
    for (QGraphicsItem* item : graphScene_->selectedItems()) {
        if (auto* nodeItem = qgraphicsitem_cast<NodeItem*>(item)) toRemove << nodeItem->node();
    }
    for (Node* node : toRemove) graph_->removeNode(node);
    graphScene_->rebuild();
    emit nodeSelected(selectedNode());
}

void NodeGraphView::toggleDisplayFlagOnSelection() {
    Node* node = selectedNode();
    if (!graph_ || !node) return;
    graph_->setDisplayNode(node);
    graphScene_->refreshAllNodeItems();
    emit displayNodeRequested(node);
}

void NodeGraphView::toggleBypassOnSelection() {
    for (QGraphicsItem* item : graphScene_->selectedItems()) {
        if (auto* nodeItem = qgraphicsitem_cast<NodeItem*>(item))
            nodeItem->node()->setBypassed(!nodeItem->node()->isBypassed());
    }
    graphScene_->refreshAllNodeItems();
}

void NodeGraphView::layoutSelectionVertically() {
    QList<NodeItem*> items;
    for (QGraphicsItem* item : graphScene_->selectedItems()) {
        if (auto* nodeItem = qgraphicsitem_cast<NodeItem*>(item)) items << nodeItem;
    }
    if (items.size() < 2) return;
    std::sort(items.begin(), items.end(),
              [](NodeItem* a, NodeItem* b) { return a->pos().y() < b->pos().y(); });
    const QPointF start = items.first()->pos();
    for (int i = 0; i < items.size(); ++i) items[i]->setPos(start.x(), start.y() + i * 90.0);
    graphScene_->updateConnections();
}

NodeItem* NodeGraphView::nodeItemAt(QPoint viewPosition) const {
    const QList<QGraphicsItem*> items = this->items(viewPosition);
    for (QGraphicsItem* item : items) {
        if (auto* nodeItem = qgraphicsitem_cast<NodeItem*>(item)) return nodeItem;
    }
    return nullptr;
}

void NodeGraphView::wheelEvent(QWheelEvent* event) {
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    const qreal factor = zoomFactorFromWheel(event);
    const double newScale = transform().m11() * factor;
    if (newScale < 0.12 || newScale > 4.0) {
        event->accept();
        return;
    }
    QGraphicsView::scale(factor, factor);
    event->accept();
}

bool NodeGraphView::shouldBeginPan(const QMouseEvent* event) const {
    // Houdini network editor style:
    //   MMB drag       — pan
    //   Alt+LMB drag   — pan
    //   Space+LMB drag — pan
    if (event->button() == Qt::MiddleButton) return true;
    if (event->button() == Qt::LeftButton && (event->modifiers() & Qt::AltModifier)) return true;
    if (event->button() == Qt::LeftButton && spaceHeld_) return true;
    return false;
}

void NodeGraphView::beginPan(const QPoint& viewPos) {
    panning_ = true;
    lastPanPoint_ = viewPos;
    savedDragMode_ = dragMode();
    savedAnchor_ = transformationAnchor();
    setDragMode(QGraphicsView::NoDrag);
    // NoAnchor keeps 1:1 hand-drag — AnchorUnderMouse would warp the pan.
    setTransformationAnchor(QGraphicsView::NoAnchor);
    // Grab the viewport (not the view): coords stay viewport-local and match
    // mapToScene / mouse*Event. Grabbing QGraphicsView itself breaks pan.
    viewport()->grabMouse();
    viewport()->setCursor(Qt::ClosedHandCursor);
}

void NodeGraphView::updatePan(const QPoint& viewPos) {
    if (!panning_) return;
    if (viewPos == lastPanPoint_) return;

    // Sticky hand: keep the scene point that was under the cursor glued to it.
    // mapToScene delta is already in scene units, so no manual /scale needed.
    const QPointF delta = mapToScene(viewPos) - mapToScene(lastPanPoint_);
    translate(delta.x(), delta.y());
    lastPanPoint_ = viewPos;
}

void NodeGraphView::endPan() {
    if (!panning_) return;
    panning_ = false;
    if (QWidget::mouseGrabber() == viewport())
        viewport()->releaseMouse();
    setDragMode(savedDragMode_);
    setTransformationAnchor(savedAnchor_);
    viewport()->unsetCursor();
}

qreal NodeGraphView::zoomFactorFromWheel(const QWheelEvent* event) const {
    const QPoint delta = event->angleDelta().y() != 0 ? event->angleDelta() : event->pixelDelta();
    const qreal steps = qreal(delta.y()) / 120.0;
    if (std::abs(steps) < 1e-4) return 1.0;
    return std::pow(1.08, steps);
}

QPointF NodeGraphView::snapWireEndpoint(QPoint viewPosition, bool draggingFromOutput) {
    snapTarget_ = nullptr;
    snapInputIndex_ = -1;
    const QPointF scenePos = mapToScene(viewPosition);
    const qreal snapRadius = NodeItem::kPortHitRadius * 1.15;
    NodeItem* bestNode = nullptr;
    int bestInput = -1;
    qreal bestDist = snapRadius;

    for (QGraphicsItem* item : graphScene_->items()) {
        auto* nodeItem = qgraphicsitem_cast<NodeItem*>(item);
        if (!nodeItem) continue;
        if (draggingFromOutput) {
            if (dragSource_ && nodeItem == dragSource_) continue;
            const QPointF local = nodeItem->mapFromScene(scenePos);
            const int index = nodeItem->nearestInputPort(local, snapRadius);
            if (index < 0) continue;
            const qreal dist = QLineF(nodeItem->inputPortPosition(index), scenePos).length();
            if (dist < bestDist) {
                bestDist = dist;
                bestNode = nodeItem;
                bestInput = index;
            }
        } else if (nodeItem->node()->hasOutputPort() && nodeItem != dragDestination_) {
            const QPointF local = nodeItem->mapFromScene(scenePos);
            if (!nodeItem->outputPortNear(local, snapRadius)) continue;
            const qreal dist = QLineF(nodeItem->outputPortPosition(), scenePos).length();
            if (dist < bestDist) {
                bestDist = dist;
                bestNode = nodeItem;
                bestInput = -1;
            }
        }
    }

    snapTarget_ = bestNode;
    snapInputIndex_ = bestInput;
    if (draggingFromOutput && bestNode && bestInput >= 0) return bestNode->inputPortPosition(bestInput);
    if (!draggingFromOutput && bestNode) return bestNode->outputPortPosition();
    return scenePos;
}

void NodeGraphView::updateDragWire(QPoint viewPosition) {
    if (!dragWire_) return;
    const QPointF snapped = snapWireEndpoint(viewPosition, dragSource_ != nullptr);
    if (dragSource_) {
        dragWire_->setPath(makeWirePath(dragSource_->outputPortPosition(), snapped));
    } else if (dragDestination_) {
        dragWire_->setPath(makeWirePath(snapped, dragDestination_->inputPortPosition(dragInputIndex_)));
    }
    if (snapTarget_) {
        dragWire_->setPen(QPen(theme::wireActive(), 2.4, Qt::SolidLine, Qt::RoundCap));
    } else {
        dragWire_->setPen(QPen(theme::wireActive(), 1.8, Qt::SolidLine, Qt::RoundCap));
    }
}

void NodeGraphView::mousePressEvent(QMouseEvent* event) {
    lastScenePosition_ = mapToScene(event->pos());

    if (shouldBeginPan(event)) {
        beginPan(event->pos());
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton) {
        if (NodeItem* item = nodeItemAt(event->pos())) {
            const QPointF local = item->mapFromScene(mapToScene(event->pos()));
            switch (item->hitTest(local)) {
                case NodeItem::Hit::DisplayFlag:
                    if (graph_) {
                        graph_->setDisplayNode(item->node());
                        graphScene_->refreshAllNodeItems();
                        emit displayNodeRequested(item->node());
                    }
                    return;
                case NodeItem::Hit::BypassFlag:
                    item->node()->setBypassed(!item->node()->isBypassed());
                    graphScene_->refreshAllNodeItems();
                    return;
                case NodeItem::Hit::Output:
                    dragSource_ = item;
                    dragDestination_ = nullptr;
                    dragInputIndex_ = -1;
                    snapTarget_ = nullptr;
                    snapInputIndex_ = -1;
                    dragWire_ = graphScene_->addPath(QPainterPath(), QPen(theme::wireActive(), 1.8));
                    event->accept();
                    return;
                case NodeItem::Hit::Input: {
                    const int index = item->inputPortAt(local);
                    if (index >= 0) {
                        Node* existing = item->node()->input(index);
                        if (existing && graph_) {
                            graph_->disconnectInput(item->node(), index);
                            graphScene_->updateConnections();
                            dragSource_ = graphScene_->itemForNode(existing);
                            dragDestination_ = nullptr;
                            snapTarget_ = nullptr;
                            snapInputIndex_ = -1;
                            dragWire_ = graphScene_->addPath(QPainterPath(), QPen(theme::wireActive(), 1.8));
                            event->accept();
                            return;
                        }
                        dragDestination_ = item;
                        dragInputIndex_ = index;
                        dragSource_ = nullptr;
                        snapTarget_ = nullptr;
                        snapInputIndex_ = -1;
                        dragWire_ = graphScene_->addPath(QPainterPath(), QPen(theme::wireActive(), 1.8));
                        event->accept();
                    }
                    return;
                }
                default: break;
            }
        }
    }

    QGraphicsView::mousePressEvent(event);
}

void NodeGraphView::mouseMoveEvent(QMouseEvent* event) {
    if (panning_) {
        updatePan(event->pos());
        event->accept();
        return;
    }
    if (dragWire_) {
        updateDragWire(event->pos());
        event->accept();
        return;
    }

    // Hover tip on ports: show the input label / "output" as the port data name.
    if (NodeItem* item = nodeItemAt(event->pos())) {
        const QPointF local = item->mapFromScene(mapToScene(event->pos()));
        QString tip;
        if (item->outputPortNear(local)) {
            tip = QStringLiteral("output");
        } else {
            const int index = item->nearestInputPort(local);
            if (index >= 0 && item->node()) tip = item->node()->inputLabel(index);
        }
        if (!tip.isEmpty())
            QToolTip::showText(event->globalPosition().toPoint(), tip, this);
        else
            QToolTip::hideText();
    } else {
        QToolTip::hideText();
    }

    QGraphicsView::mouseMoveEvent(event);
}

void NodeGraphView::finishWireDrag(QPoint viewPosition) {
    if (!graph_) return;

    updateDragWire(viewPosition);

    if (snapTarget_) {
        if (dragSource_ && snapInputIndex_ >= 0) {
            if (graph_->connectNodes(dragSource_->node(), snapTarget_->node(), snapInputIndex_))
                emit statusMessage(dragSource_->node()->name() + " -> " + snapTarget_->node()->name());
        } else if (dragDestination_ && snapTarget_->node()->hasOutputPort()) {
            if (graph_->connectNodes(snapTarget_->node(), dragDestination_->node(), dragInputIndex_))
                emit statusMessage(snapTarget_->node()->name() + " -> " + dragDestination_->node()->name());
        }
        graphScene_->updateConnections();
        return;
    }

    NodeItem* target = nodeItemAt(viewPosition);
    if (target) {
        const QPointF local = target->mapFromScene(mapToScene(viewPosition));
        if (dragSource_ && dragSource_ != target) {
            int index = target->nearestInputPort(local, NodeItem::kPortHitRadius * 1.35);
            if (index < 0 && target->node()->inputCount() > 0) {
                index = 0;
                for (int i = 0; i < target->node()->inputCount(); ++i) {
                    if (!target->node()->input(i)) {
                        index = i;
                        break;
                    }
                }
            }
            if (index >= 0) {
                if (graph_->connectNodes(dragSource_->node(), target->node(), index))
                    emit statusMessage(dragSource_->node()->name() + " -> " + target->node()->name());
            }
        } else if (dragDestination_ && dragDestination_ != target && target->outputPortNear(local)) {
            if (graph_->connectNodes(target->node(), dragDestination_->node(), dragInputIndex_))
                emit statusMessage(target->node()->name() + " -> " + dragDestination_->node()->name());
        }
    }
    graphScene_->updateConnections();
}

void NodeGraphView::mouseReleaseEvent(QMouseEvent* event) {
    if (panning_) {
        endPan();
        event->accept();
        return;
    }
    if (dragWire_) {
        finishWireDrag(event->pos());
        graphScene_->removeItem(dragWire_);
        delete dragWire_;
        dragWire_ = nullptr;
        dragSource_ = nullptr;
        dragDestination_ = nullptr;
        snapTarget_ = nullptr;
        snapInputIndex_ = -1;
        dragInputIndex_ = -1;
        event->accept();
        return;
    }
    QGraphicsView::mouseReleaseEvent(event);
}

void NodeGraphView::keyPressEvent(QKeyEvent* event) {
    switch (event->key()) {
        case Qt::Key_Space:
            spaceHeld_ = true;
            if (!panning_) viewport()->setCursor(Qt::OpenHandCursor);
            event->accept();
            return;
        case Qt::Key_Tab: {
            const QPoint viewPos = mapFromGlobal(QCursor::pos());
            if (viewport()->rect().contains(viewPos)) lastScenePosition_ = mapToScene(viewPos);
            createMenu_->popupAt(QCursor::pos());
            event->accept();
            return;
        }
        case Qt::Key_Delete:
        case Qt::Key_Backspace:
            deleteSelectedNodes();
            return;
        case Qt::Key_D:
            toggleDisplayFlagOnSelection();
            return;
        case Qt::Key_B:
            toggleBypassOnSelection();
            return;
        case Qt::Key_F:
            frameAll();
            return;
        case Qt::Key_L:
            layoutSelectionVertically();
            return;
        default: break;
    }
    QGraphicsView::keyPressEvent(event);
}

void NodeGraphView::keyReleaseEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Space) {
        spaceHeld_ = false;
        if (!panning_) viewport()->unsetCursor();
        event->accept();
        return;
    }
    QGraphicsView::keyReleaseEvent(event);
}

void NodeGraphView::contextMenuEvent(QContextMenuEvent* event) {
    lastScenePosition_ = mapToScene(event->pos());
    QMenu menu(this);

    ConnectionItem* wire = nullptr;
    for (QGraphicsItem* graphicsItem : items(event->pos())) {
        if (auto* connection = qgraphicsitem_cast<ConnectionItem*>(graphicsItem)) {
            wire = connection;
            break;
        }
    }
    if (wire && wire->destination() && graph_) {
        Node* destination = wire->destination()->node();
        const int inputIndex = wire->inputIndex();
        menu.addAction("Disconnect", this, [this, destination, inputIndex] {
            if (!graph_ || !destination) return;
            graph_->disconnectInput(destination, inputIndex);
            graphScene_->updateConnections();
            emit statusMessage("Disconnected " + destination->name());
        });
        menu.addSeparator();
    }

    NodeItem* item = nodeItemAt(event->pos());
    if (item) {
        Node* node = item->node();
        menu.addAction("Set Display Flag", this, [this, node] {
            if (graph_) graph_->setDisplayNode(node);
            graphScene_->refreshAllNodeItems();
            emit displayNodeRequested(node);
        });
        QAction* bypass = menu.addAction("Bypass");
        bypass->setCheckable(true);
        bypass->setChecked(node->isBypassed());
        connect(bypass, &QAction::triggered, this, [this, node](bool checked) {
            node->setBypassed(checked);
            graphScene_->refreshAllNodeItems();
        });
        menu.addAction("Delete", this, [this, node] {
            if (graph_) graph_->removeNode(node);
            graphScene_->rebuild();
        });
        menu.addSeparator();
    }

    QMenu* addMenu = menu.addMenu("Add Node");
    for (const QString& category : NodeRegistry::instance().categories()) {
        QMenu* categoryMenu = addMenu->addMenu(category);
        for (const NodeTypeInfo& info : NodeRegistry::instance().types()) {
            if (info.category != category) continue;
            const QString typeName = info.typeName;
            categoryMenu->addAction(info.label, this, [this, typeName] { createNodeOfType(typeName); });
        }
    }
    menu.addAction("Frame All", this, &NodeGraphView::frameAll);
    menu.exec(event->globalPos());
}

void NodeGraphView::drawForeground(QPainter* painter, const QRectF& rect) {
    QGraphicsView::drawForeground(painter, rect);
    // Hint bar in the corner of the viewport.
    painter->resetTransform();
    QFont font = painter->font();
    font.setPointSizeF(8.0);
    painter->setFont(font);
    painter->setPen(theme::textDim());
    painter->drawText(QRect(8, height() - 22, width() - 16, 18), Qt::AlignLeft,
                      "Tab: add   F: frame   MMB/Alt+LMB/Space+LMB: pan   Wheel: zoom   D: display   B: bypass   "
                      "Del: delete");
}

}  // namespace sol
