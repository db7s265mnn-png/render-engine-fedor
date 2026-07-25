// Bezier wire between a node output and a node input.
#pragma once

#include <QGraphicsPathItem>

namespace sol {

class NodeItem;

class ConnectionItem : public QGraphicsPathItem {
public:
    enum { Type = UserType + 2 };

    ConnectionItem(NodeItem* source, NodeItem* destination, int inputIndex);

    int type() const override { return Type; }
    NodeItem* source() const { return source_; }
    NodeItem* destination() const { return destination_; }
    int inputIndex() const { return inputIndex_; }

    void updateGeometry();

private:
    NodeItem* source_ = nullptr;
    NodeItem* destination_ = nullptr;
    int inputIndex_ = 0;
};

// Builds the curve used both for real connections and for the rubber band
// shown while dragging a new one.
QPainterPath makeWirePath(QPointF from, QPointF to);

}  // namespace sol
