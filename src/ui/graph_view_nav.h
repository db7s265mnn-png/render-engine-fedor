// Shared QGraphicsView zoom that stays safe while a middle-mouse pan is live.
// AnchorUnderMouse + a sticky mapToScene pan remaps the last pan delta through
// the new transform and throws the view across the scene.
#pragma once

#include <QGraphicsView>
#include <QPoint>
#include <QWheelEvent>

namespace sol {

inline QPoint graphicsViewWheelPos(const QGraphicsView* view, const QWheelEvent* event) {
    (void)view;
    return event->position().toPoint();
}

inline void zoomGraphicsViewAt(QGraphicsView* view, qreal factor, const QPoint& viewPos,
                               QPoint* panAnchor = nullptr) {
    view->setTransformationAnchor(QGraphicsView::NoAnchor);
    const QPointF before = view->mapToScene(viewPos);
    view->scale(factor, factor);
    const QPointF after = view->mapToScene(viewPos);
    view->translate(before.x() - after.x(), before.y() - after.y());
    if (panAnchor) *panAnchor = viewPos;
}

}  // namespace sol
