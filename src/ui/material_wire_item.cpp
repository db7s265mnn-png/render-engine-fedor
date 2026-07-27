#include "ui/material_wire_item.h"

#include <QPainterPathStroker>
#include <QPen>

#include "ui/theme.h"

namespace sol {
namespace {

constexpr qreal kWireHitWidth = 14.0;

QPainterPath strokedShape(const QPainterPath& path) {
    QPainterPathStroker stroker;
    stroker.setWidth(kWireHitWidth);
    stroker.setCapStyle(Qt::RoundCap);
    stroker.setJoinStyle(Qt::RoundJoin);
    return stroker.createStroke(path);
}

}  // namespace

MaterialWireItem::MaterialWireItem(QString sourceNodeName, QString targetNodeName, QString inputName, QColor color,
                                   QGraphicsItem* parent)
    : QGraphicsPathItem(parent),
      sourceNodeName_(std::move(sourceNodeName)),
      targetNodeName_(std::move(targetNodeName)),
      inputName_(std::move(inputName)) {
    setAcceptHoverEvents(true);
    // Not selectable — Scene Network style: wires are for rewire/context menu,
    // not for stealing node clicks / Parameters selection.
    setFlag(QGraphicsItem::ItemIsSelectable, false);
    setAcceptedMouseButtons(Qt::RightButton);
    setZValue(-0.5);
    setWireColor(color.isValid() ? color : theme::wireActive());
}

void MaterialWireItem::setWirePath(const QPainterPath& path) { setPath(path); }

void MaterialWireItem::setWireColor(const QColor& color) {
    setPen(QPen(color.isValid() ? color : theme::wireActive(), 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
}

QPainterPath MaterialWireItem::shape() const { return strokedShape(path()); }

QRectF MaterialWireItem::boundingRect() const {
    return shape().controlPointRect().adjusted(-2.0, -2.0, 2.0, 2.0);
}

}  // namespace sol
