// Graphics item that draws a single node, Houdini style: inputs on top,
// output at the bottom, display and bypass flags on the right hand side.
#pragma once

#include <QGraphicsItem>
#include <QRectF>

#include "nodes/node.h"

namespace sol {

class NodeGraphScene;

class NodeItem : public QGraphicsItem {
public:
    enum { Type = UserType + 1 };
    enum class Hit { None, Body, Output, Input, DisplayFlag, BypassFlag };

    static constexpr qreal kWidth = 74.0;
    static constexpr qreal kHeight = 50.0;
    static constexpr qreal kPortRadius = 5.5;
    static constexpr qreal kPortHitRadius = 18.0;

    NodeItem(Node* node, NodeGraphScene* scene);

    int type() const override { return Type; }
    Node* node() const { return node_; }

    QRectF boundingRect() const override;
    QPainterPath shape() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;

    QPointF outputPortPosition() const;
    QPointF inputPortPosition(int index) const;
    int inputPortAt(QPointF localPosition) const;
    int nearestInputPort(QPointF localPosition, qreal maxDistance = kPortHitRadius) const;
    bool outputPortNear(QPointF localPosition, qreal maxDistance = kPortHitRadius) const;
    Hit hitTest(QPointF localPosition) const;

    void refresh();

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant& value) override;

private:
    QColor headerColor() const;
    QRectF bodyRect() const;
    QRectF labelRect() const;
    QRectF displayFlagRect() const;
    QRectF bypassFlagRect() const;

    Node* node_ = nullptr;
    NodeGraphScene* graphScene_ = nullptr;
    QColor color_;
};

}  // namespace sol
