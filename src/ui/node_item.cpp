#include "ui/node_item.h"

#include <QFontMetrics>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <algorithm>

#include "nodes/node_registry.h"
#include "ui/node_graph_view.h"
#include "ui/theme.h"

namespace sol {

NodeItem::NodeItem(Node* node, NodeGraphScene* scene) : node_(node), graphScene_(scene) {
    setFlag(ItemIsMovable, true);
    setFlag(ItemIsSelectable, true);
    setFlag(ItemSendsGeometryChanges, true);
    setCacheMode(NoCache);
    setZValue(1.0);
    setPos(node_->position());
    color_ = headerColor();
}

QColor NodeItem::headerColor() const {
    const NodeTypeInfo* info = NodeRegistry::instance().find(node_->typeName());
    return info ? QColor(info->colorHex) : theme::panelLight();
}

QRectF NodeItem::boundingRect() const {
    QRectF bounds = bodyRect().adjusted(-7.0, -7.0, 8.0, 9.0);
    bounds = bounds.united(labelRect().adjusted(0.0, -3.0, 4.0, 3.0));
    for (int i = 0; i < node_->inputCount(); ++i) {
        const QPointF port = inputPortPosition(i) - pos();
        bounds = bounds.united(QRectF(port.x() - kPortHitRadius, port.y() - kPortHitRadius,
                                      kPortHitRadius * 2.0, kPortHitRadius * 2.0));
    }
    if (node_->hasOutputPort()) {
        const QPointF port = outputPortPosition() - pos();
        bounds = bounds.united(QRectF(port.x() - kPortHitRadius, port.y() - kPortHitRadius,
                                      kPortHitRadius * 2.0, kPortHitRadius * 2.0));
    }
    return bounds;
}

QPainterPath NodeItem::shape() const {
    QPainterPath path;
    path.addRoundedRect(bodyRect(), 5.0, 5.0);
    path.addRect(labelRect());
    for (int i = 0; i < node_->inputCount(); ++i) {
        const QPointF port = inputPortPosition(i) - pos();
        path.addEllipse(port, kPortHitRadius, kPortHitRadius);
    }
    if (node_->hasOutputPort()) {
        const QPointF port = outputPortPosition() - pos();
        path.addEllipse(port, kPortHitRadius, kPortHitRadius);
    }
    return path;
}

QRectF NodeItem::bodyRect() const {
    return QRectF(-kWidth * 0.5, -kHeight * 0.5, kWidth, kHeight);
}

QRectF NodeItem::labelRect() const {
    return QRectF(kWidth * 0.5 + 10.0, -20.0, 140.0, 40.0);
}

QRectF NodeItem::displayFlagRect() const {
    constexpr qreal flagSize = 10.0;
    const QRectF body = bodyRect();
    return QRectF(body.right() - flagSize - 5.0, body.top() + 5.0, flagSize, flagSize);
}

QRectF NodeItem::bypassFlagRect() const {
    constexpr qreal flagSize = 10.0;
    const QRectF body = bodyRect();
    return QRectF(body.right() - flagSize - 5.0, body.bottom() - flagSize - 5.0, flagSize, flagSize);
}

QPointF NodeItem::outputPortPosition() const { return pos() + QPointF(0.0, kHeight * 0.5 + 2.0); }

QPointF NodeItem::inputPortPosition(int index) const {
    const int count = std::max(1, node_->inputCount());
    const qreal span = kWidth * 0.68;
    const qreal step = count > 1 ? span / qreal(count - 1) : 0.0;
    const qreal x = count > 1 ? -span * 0.5 + step * index : 0.0;
    return pos() + QPointF(x, -kHeight * 0.5 - 2.0);
}

int NodeItem::inputPortAt(QPointF localPosition) const {
    return nearestInputPort(localPosition, kPortHitRadius);
}

int NodeItem::nearestInputPort(QPointF localPosition, qreal maxDistance) const {
    int best = -1;
    qreal bestDist = maxDistance;
    for (int i = 0; i < node_->inputCount(); ++i) {
        const QPointF port = inputPortPosition(i) - pos();
        const qreal dist = QLineF(port, localPosition).length();
        if (dist <= bestDist) {
            bestDist = dist;
            best = i;
        }
    }
    return best;
}

bool NodeItem::outputPortNear(QPointF localPosition, qreal maxDistance) const {
    if (!node_->hasOutputPort()) return false;
    const QPointF port = outputPortPosition() - pos();
    return QLineF(port, localPosition).length() <= maxDistance;
}

NodeItem::Hit NodeItem::hitTest(QPointF localPosition) const {
    if (displayFlagRect().contains(localPosition)) return Hit::DisplayFlag;
    if (bypassFlagRect().contains(localPosition)) return Hit::BypassFlag;
    if (outputPortNear(localPosition)) return Hit::Output;
    if (nearestInputPort(localPosition) >= 0) return Hit::Input;
    if (bodyRect().contains(localPosition) || labelRect().contains(localPosition)) return Hit::Body;
    return Hit::None;
}

void NodeItem::refresh() {
    color_ = headerColor();
    update();
}

QVariant NodeItem::itemChange(GraphicsItemChange change, const QVariant& value) {
    if (change == ItemPositionHasChanged && node_) {
        node_->setPosition(value.toPointF());
        if (graphScene_) graphScene_->updateConnections();
    }
    return QGraphicsItem::itemChange(change, value);
}

void NodeItem::paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) {
    painter->setRenderHint(QPainter::Antialiasing, true);
    const QRectF body = bodyRect();
    const bool isDisplay = graphScene_ && graphScene_->graph() && graphScene_->graph()->displayNode() == node_;

    // Drop shadow.
    painter->setPen(Qt::NoPen);
    painter->setBrush(QColor(0, 0, 0, 70));
    painter->drawRoundedRect(body.translated(2.0, 3.0), 5.0, 5.0);

    QLinearGradient gradient(body.topLeft(), body.bottomLeft());
    gradient.setColorAt(0.0, theme::panelLight().lighter(108));
    gradient.setColorAt(1.0, theme::panel().darker(112));
    painter->setBrush(gradient);

    QPen border(node_->errorText().isEmpty() ? QColor(20, 21, 24) : theme::error(), 1.4);
    if (isSelected()) border = QPen(theme::selection(), 2.0);
    painter->setPen(border);
    painter->drawRoundedRect(body, 5.0, 5.0);

    QPainterPath bodyClip;
    bodyClip.addRoundedRect(body, 5.0, 5.0);
    painter->save();
    painter->setClipPath(bodyClip);
    painter->setPen(Qt::NoPen);
    painter->setBrush(color_);
    painter->drawRect(QRectF(body.left(), body.top(), body.width(), 11.0));
    painter->restore();

    if (isDisplay) {
        painter->setPen(QPen(theme::displayFlag().lighter(125), 3.0, Qt::SolidLine, Qt::SquareCap, Qt::RoundJoin));
        painter->setBrush(Qt::NoBrush);
        painter->drawRoundedRect(body.adjusted(-1.8, -1.8, 1.8, 1.8), 6.0, 6.0);
    }

    if (node_->isBypassed()) {
        painter->setPen(QPen(QColor(255, 230, 150, 180), 1.6));
        painter->drawLine(body.topLeft() + QPointF(4, 4), body.bottomRight() - QPointF(4, 4));
    }

    // Name and type live outside the tile, matching Houdini's network editor.
    const QRectF label = labelRect();
    QFont nameFont = painter->font();
    nameFont.setPointSizeF(8.5);
    nameFont.setBold(true);
    painter->setFont(nameFont);
    painter->setPen(QColor(245, 246, 248));
    const QRectF nameRect(label.left(), label.top() + 2.0, label.width(), 18.0);
    painter->drawText(nameRect, Qt::AlignLeft | Qt::AlignVCenter,
                      QFontMetrics(nameFont).elidedText(node_->name(), Qt::ElideRight, int(nameRect.width())));

    QFont typeFont = painter->font();
    typeFont.setPointSizeF(7.0);
    typeFont.setBold(false);
    painter->setFont(typeFont);
    painter->setPen(theme::textDim());
    const QRectF typeRect(label.left(), label.top() + 20.0, label.width(), 16.0);
    painter->drawText(typeRect, Qt::AlignLeft | Qt::AlignVCenter,
                      QFontMetrics(typeFont).elidedText(node_->typeName(), Qt::ElideRight, int(typeRect.width())));

    // Flags.
    const QRectF displayRect = displayFlagRect();
    painter->setPen(QPen(QColor(20, 21, 24), 1.0));
    painter->setBrush(isDisplay ? theme::displayFlag() : QColor(60, 63, 68));
    painter->drawRect(displayRect);

    const QRectF bypassRect = bypassFlagRect();
    painter->setBrush(node_->isBypassed() ? theme::accent() : QColor(60, 63, 68));
    painter->drawRect(bypassRect);

    // Ports.
    for (int i = 0; i < node_->inputCount(); ++i) {
        const QPointF port = inputPortPosition(i) - pos();
        painter->setPen(Qt::NoPen);
        painter->setBrush(QColor(120, 170, 255, node_->input(i) ? 55 : 35));
        painter->drawEllipse(port, kPortRadius + 3.0, kPortRadius + 3.0);
        painter->setPen(QPen(QColor(20, 21, 24), 1.0));
        painter->setBrush(node_->input(i) ? theme::wireActive() : QColor(120, 125, 133));
        painter->drawEllipse(port, kPortRadius, kPortRadius);
    }
    if (node_->hasOutputPort()) {
        const QPointF port = outputPortPosition() - pos();
        painter->setPen(Qt::NoPen);
        painter->setBrush(QColor(120, 170, 255, 35));
        painter->drawEllipse(port, kPortRadius + 3.0, kPortRadius + 3.0);
        painter->setPen(QPen(QColor(20, 21, 24), 1.0));
        painter->setBrush(QColor(120, 125, 133));
        painter->drawEllipse(port, kPortRadius, kPortRadius);
    }
}

}  // namespace sol
