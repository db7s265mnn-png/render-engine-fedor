#include "ui/node_item.h"

#include <QFontMetrics>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QPainterPath>

#include "nodes/node_registry.h"
#include "ui/node_graph_view.h"
#include "ui/theme.h"

namespace sol {

NodeItem::NodeItem(Node* node, NodeGraphScene* scene) : node_(node), graphScene_(scene) {
    setFlag(ItemIsMovable, true);
    setFlag(ItemIsSelectable, true);
    setFlag(ItemSendsGeometryChanges, true);
    setCacheMode(DeviceCoordinateCache);
    setZValue(1.0);
    setPos(node_->position());
    color_ = headerColor();
}

QColor NodeItem::headerColor() const {
    const NodeTypeInfo* info = NodeRegistry::instance().find(node_->typeName());
    return info ? QColor(info->colorHex) : theme::panelLight();
}

QRectF NodeItem::boundingRect() const {
    return QRectF(-kWidth * 0.5 - 8.0, -kHeight * 0.5 - 10.0, kWidth + 16.0, kHeight + 20.0);
}

QPainterPath NodeItem::shape() const {
    QPainterPath path;
    path.addRoundedRect(QRectF(-kWidth * 0.5, -kHeight * 0.5, kWidth, kHeight), 4.0, 4.0);
    return path;
}

QPointF NodeItem::outputPortPosition() const { return pos() + QPointF(0.0, kHeight * 0.5 + 2.0); }

QPointF NodeItem::inputPortPosition(int index) const {
    const int count = std::max(1, node_->inputCount());
    const qreal span = kWidth * 0.62;
    const qreal step = count > 1 ? span / qreal(count - 1) : 0.0;
    const qreal x = count > 1 ? -span * 0.5 + step * index : 0.0;
    return pos() + QPointF(x, -kHeight * 0.5 - 2.0);
}

int NodeItem::inputPortAt(QPointF localPosition) const {
    for (int i = 0; i < node_->inputCount(); ++i) {
        const QPointF port = inputPortPosition(i) - pos();
        if (QLineF(port, localPosition).length() < kPortRadius * 2.4) return i;
    }
    return -1;
}

NodeItem::Hit NodeItem::hitTest(QPointF localPosition) const {
    const QRectF displayRect(kWidth * 0.5 - 16.0, -kHeight * 0.5 + 4.0, 12.0, 12.0);
    const QRectF bypassRect(kWidth * 0.5 - 16.0, kHeight * 0.5 - 16.0, 12.0, 12.0);
    if (displayRect.contains(localPosition)) return Hit::DisplayFlag;
    if (bypassRect.contains(localPosition)) return Hit::BypassFlag;
    if (QLineF(outputPortPosition() - pos(), localPosition).length() < kPortRadius * 2.4) return Hit::Output;
    if (inputPortAt(localPosition) >= 0) return Hit::Input;
    if (shape().contains(localPosition)) return Hit::Body;
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
    const QRectF body(-kWidth * 0.5, -kHeight * 0.5, kWidth, kHeight);
    const bool isDisplay = graphScene_ && graphScene_->graph() && graphScene_->graph()->displayNode() == node_;

    // Drop shadow.
    painter->setPen(Qt::NoPen);
    painter->setBrush(QColor(0, 0, 0, 70));
    painter->drawRoundedRect(body.translated(2.0, 3.0), 4.0, 4.0);

    QLinearGradient gradient(body.topLeft(), body.bottomLeft());
    gradient.setColorAt(0.0, color_.lighter(125));
    gradient.setColorAt(0.45, color_);
    gradient.setColorAt(1.0, color_.darker(135));
    painter->setBrush(gradient);

    QPen border(node_->errorText().isEmpty() ? QColor(20, 21, 24) : theme::error(), 1.4);
    if (isSelected()) border = QPen(theme::selection(), 2.0);
    painter->setPen(border);
    painter->drawRoundedRect(body, 4.0, 4.0);

    if (node_->isBypassed()) {
        painter->setPen(QPen(QColor(255, 230, 150, 180), 1.6));
        painter->drawLine(body.topLeft() + QPointF(4, 4), body.bottomRight() - QPointF(4, 4));
    }

    // Name and type.
    QFont nameFont = painter->font();
    nameFont.setPointSizeF(8.5);
    nameFont.setBold(true);
    painter->setFont(nameFont);
    painter->setPen(QColor(245, 246, 248));
    const QRectF textRect = body.adjusted(8.0, 4.0, -20.0, -kHeight * 0.45);
    painter->drawText(textRect, Qt::AlignLeft | Qt::AlignVCenter,
                      QFontMetrics(nameFont).elidedText(node_->name(), Qt::ElideRight, int(textRect.width())));

    QFont typeFont = painter->font();
    typeFont.setPointSizeF(7.0);
    typeFont.setBold(false);
    painter->setFont(typeFont);
    painter->setPen(QColor(225, 228, 233, 190));
    painter->drawText(body.adjusted(8.0, kHeight * 0.42, -20.0, -4.0), Qt::AlignLeft | Qt::AlignVCenter,
                      node_->typeName());

    // Flags.
    const QRectF displayRect(kWidth * 0.5 - 16.0, -kHeight * 0.5 + 4.0, 12.0, 12.0);
    painter->setPen(QPen(QColor(20, 21, 24), 1.0));
    painter->setBrush(isDisplay ? theme::displayFlag() : QColor(60, 63, 68));
    painter->drawRoundedRect(displayRect, 2.0, 2.0);

    const QRectF bypassRect(kWidth * 0.5 - 16.0, kHeight * 0.5 - 16.0, 12.0, 12.0);
    painter->setBrush(node_->isBypassed() ? QColor(240, 200, 90) : QColor(60, 63, 68));
    painter->drawRoundedRect(bypassRect, 2.0, 2.0);

    // Ports.
    painter->setPen(QPen(QColor(20, 21, 24), 1.0));
    for (int i = 0; i < node_->inputCount(); ++i) {
        const QPointF port = inputPortPosition(i) - pos();
        painter->setBrush(node_->input(i) ? theme::wireActive() : QColor(120, 125, 133));
        painter->drawEllipse(port, kPortRadius, kPortRadius * 0.8);
    }
    if (node_->hasOutputPort()) {
        painter->setBrush(QColor(120, 125, 133));
        painter->drawEllipse(outputPortPosition() - pos(), kPortRadius, kPortRadius * 0.8);
    }
}

}  // namespace sol
