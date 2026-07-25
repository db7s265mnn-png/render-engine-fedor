#include "ui/connection_item.h"

#include <QPainterPath>
#include <QPen>
#include <cmath>

#include "ui/node_item.h"
#include "ui/theme.h"

namespace sol {

QPainterPath makeWirePath(QPointF from, QPointF to) {
    QPainterPath path(from);
    const qreal dy = std::max(qreal(28.0), std::abs(to.y() - from.y()) * 0.45);
    path.cubicTo(from + QPointF(0.0, dy), to - QPointF(0.0, dy), to);
    return path;
}

ConnectionItem::ConnectionItem(NodeItem* source, NodeItem* destination, int inputIndex)
    : source_(source), destination_(destination), inputIndex_(inputIndex) {
    setZValue(0.0);
    setPen(QPen(theme::wire(), 1.8, Qt::SolidLine, Qt::RoundCap));
    setFlag(ItemIsSelectable, false);
    updateGeometry();
}

void ConnectionItem::updateGeometry() {
    if (!source_ || !destination_) return;
    setPath(makeWirePath(source_->outputPortPosition(), destination_->inputPortPosition(inputIndex_)));
}

}  // namespace sol
