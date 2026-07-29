#include "ui/node_item.h"

#include <QFontMetrics>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <algorithm>

#include "nodes/node_registry.h"
#include "ui/node_graph_view.h"
#include "ui/node_icons.h"
#include "ui/theme.h"

namespace sol {
namespace {

QColor bypassFlagColor(bool active) {
    return active ? QColor(236, 196, 48) : QColor(72, 68, 42);
}

QColor displayFlagColor(bool active) {
    return active ? theme::displayFlag() : QColor(42, 58, 72);
}

// Flag strips sit fully inside the border stroke so corners align with the outline.
QPainterPath flagStripPath(const QRectF& body, qreal cornerRadius, bool leftSide) {
    const qreal inset = NodeItem::kBorderWidth * 0.5;
    const qreal x = leftSide ? body.left() + inset : body.right() - NodeItem::kFlagWidth;
    const qreal w = NodeItem::kFlagWidth - inset;
    QPainterPath strip;
    strip.addRect(QRectF(x, body.top() + inset, w, body.height() - 2.0 * inset));
    QPainterPath rounded;
    rounded.addRoundedRect(body.adjusted(inset, inset, -inset, -inset),
                           std::max(0.0, cornerRadius - inset), std::max(0.0, cornerRadius - inset));
    return strip.intersected(rounded);
}

}  // namespace

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
    QRectF bounds = bodyRect().adjusted(-12.0, -12.0, 13.0, 14.0);
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
    QPainterPath path = bodyPath();
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

QRectF NodeItem::iconArea() const {
    const QRectF body = bodyRect();
    return QRectF(body.left() + kFlagWidth + 2.0, body.top() + 3.0,
                  body.width() - 2.0 * kFlagWidth - 4.0, body.height() - 6.0);
}

QPainterPath NodeItem::bodyPath() const {
    QPainterPath path;
    path.addRoundedRect(bodyRect(), kCornerRadius, kCornerRadius);
    return path;
}

QPainterPath NodeItem::bypassFlagPath() const {
    return flagStripPath(bodyRect(), kCornerRadius, true);
}

QPainterPath NodeItem::displayFlagPath() const {
    return flagStripPath(bodyRect(), kCornerRadius, false);
}

QPointF NodeItem::outputPortPosition() const { return pos() + QPointF(0.0, kHeight * 0.5 + 2.0); }

QPointF NodeItem::inputPortPosition(int index) const {
    const int count = std::max(1, node_->inputCount());
    const qreal span = kWidth * 0.58;
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
    if (displayFlagPath().contains(localPosition)) return Hit::DisplayFlag;
    if (bypassFlagPath().contains(localPosition)) return Hit::BypassFlag;
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
    const bool isBypassed = node_->isBypassed();
    const QPainterPath clip = bodyPath();

    // Soft rectangular display glow — layered rounded rects, not an oval.
    if (isDisplay) {
        const QColor flag = theme::displayFlag();
        painter->setPen(Qt::NoPen);
        for (int i = 4; i >= 1; --i) {
            const qreal pad = 1.2 + qreal(i) * 1.55;
            const int alpha = 8 + i * 9;
            painter->setBrush(QColor(flag.red(), flag.green(), flag.blue(), alpha));
            painter->drawRoundedRect(body.adjusted(-pad, -pad, pad, pad), kCornerRadius + pad * 0.25,
                                     kCornerRadius + pad * 0.25);
        }
        QPen softEdge(QColor(flag.red(), flag.green(), flag.blue(), 110), 2.2);
        softEdge.setJoinStyle(Qt::RoundJoin);
        painter->setBrush(Qt::NoBrush);
        painter->setPen(softEdge);
        painter->drawRoundedRect(body.adjusted(-1.2, -1.2, 1.2, 1.2), kCornerRadius + 0.8,
                                 kCornerRadius + 0.8);
    }

    // Drop shadow.
    painter->setPen(Qt::NoPen);
    painter->setBrush(QColor(0, 0, 0, 70));
    painter->drawRoundedRect(body.translated(2.0, 3.0), kCornerRadius, kCornerRadius);

    const NodeTypeInfo* typeInfo = NodeRegistry::instance().find(node_->typeName());
    const NodeIconKind iconKind =
        nodeIconKind(node_->typeName(), typeInfo ? typeInfo->category : QString());
    const QColor bodyColor = nodeBodyColor(iconKind, color_);

    if (iconKind != NodeIconKind::None) {
        QLinearGradient gradient(body.topLeft(), body.bottomLeft());
        gradient.setColorAt(0.0, bodyColor.lighter(112));
        gradient.setColorAt(1.0, bodyColor.darker(118));
        painter->setBrush(gradient);
    } else {
        QLinearGradient gradient(body.topLeft(), body.bottomLeft());
        gradient.setColorAt(0.0, theme::panelLight().lighter(108));
        gradient.setColorAt(1.0, theme::panel().darker(112));
        painter->setBrush(gradient);
    }

    // Body fill first (no stroke) so flags and border can share one outline.
    painter->setPen(Qt::NoPen);
    painter->drawPath(clip);

    painter->save();
    painter->setClipPath(clip);
    painter->setPen(Qt::NoPen);

    if (iconKind == NodeIconKind::None) {
        painter->setBrush(color_);
        painter->drawRect(QRectF(body.left() + kFlagWidth + 1.0, body.top(),
                                 body.width() - 2.0 * kFlagWidth - 2.0, 10.0));
    }

    // Flags inset to the inner edge of the border stroke.
    painter->setBrush(bypassFlagColor(isBypassed));
    painter->drawPath(bypassFlagPath());
    painter->setBrush(displayFlagColor(isDisplay));
    painter->drawPath(displayFlagPath());

    const qreal inset = kBorderWidth * 0.5;
    painter->setPen(QPen(QColor(0, 0, 0, 90), 1.0));
    painter->drawLine(QPointF(body.left() + kFlagWidth, body.top() + inset + 0.5),
                      QPointF(body.left() + kFlagWidth, body.bottom() - inset - 0.5));
    painter->drawLine(QPointF(body.right() - kFlagWidth, body.top() + inset + 0.5),
                      QPointF(body.right() - kFlagWidth, body.bottom() - inset - 0.5));

    if (iconKind != NodeIconKind::None) paintNodeIcon(*painter, iconKind, iconArea());

    if (isBypassed) {
        painter->setPen(Qt::NoPen);
        painter->setBrush(QColor(0, 0, 0, 55));
        painter->drawRect(body.adjusted(kFlagWidth + 1.0, 0.0, -(kFlagWidth + 1.0), 0.0));
    }
    painter->restore();

    // Border drawn last — one clean contour over body + flags.
    const qreal borderW = isSelected() ? 2.0 : kBorderWidth;
    QPen border(node_->errorText().isEmpty() ? QColor(20, 21, 24) : theme::error(), borderW);
    if (isSelected()) border = QPen(theme::selection(), borderW);
    border.setJoinStyle(Qt::RoundJoin);
    painter->setBrush(Qt::NoBrush);
    painter->setPen(border);
    painter->drawPath(clip);

    // Name and type live outside the tile.
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
