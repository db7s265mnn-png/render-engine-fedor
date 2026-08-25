// Graph pan/zoom that cannot throw the view across the scene.
//
// QGraphicsView::AnchorUnderMouse plus a sticky last-widget-point pan is
// unsafe: wheel events and grabbed mouse-moves live in different widgets
// (view vs viewport), and after scale() the leftover pan delta is a huge
// scene jump. All cursor math goes global → viewport, pan glues one scene
// point to the cursor, and zoom is NoAnchor + re-glue.
#pragma once

#include <QGraphicsView>
#include <QPoint>
#include <QPointF>
#include <QScrollBar>
#include <QTransform>
#include <QWheelEvent>
#include <QWidget>
#include <algorithm>
#include <cmath>

namespace sol {

inline QPoint graphicsViewViewportPos(const QGraphicsView* view, const QPointF& globalPos) {
    if (!view || !view->viewport()) return globalPos.toPoint();
    return view->viewport()->mapFromGlobal(globalPos.toPoint());
}

inline void glueGraphicsViewScenePoint(QGraphicsView* view, const QPointF& scenePoint,
                                       const QPoint& viewportPos) {
    if (!view) return;
    view->setTransformationAnchor(QGraphicsView::NoAnchor);
    view->setResizeAnchor(QGraphicsView::NoAnchor);

    const QPointF onView = view->mapFromScene(scenePoint);
    const qreal dx = onView.x() - qreal(viewportPos.x());
    const qreal dy = onView.y() - qreal(viewportPos.y());
    if (qAbs(dx) < 0.25 && qAbs(dy) < 0.25) return;

    auto* hBar = view->horizontalScrollBar();
    auto* vBar = view->verticalScrollBar();
    const int oldH = hBar ? hBar->value() : 0;
    const int oldV = vBar ? vBar->value() : 0;
    if (hBar) hBar->setValue(oldH + int(std::lround(dx)));
    if (vBar) vBar->setValue(oldV + int(std::lround(dy)));
    const qreal remainX = dx - qreal((hBar ? hBar->value() : oldH) - oldH);
    const qreal remainY = dy - qreal((vBar ? vBar->value() : oldV) - oldV);
    if (qAbs(remainX) < 0.5 && qAbs(remainY) < 0.5) return;

    // Scrollbars are AlwaysOff here; the range is often 0 and the whole
    // offset lives in the transform. Finish the leftover in scene units.
    const QPointF now = view->mapToScene(viewportPos);
    const QPointF sceneDelta = now - scenePoint;
    if (qAbs(sceneDelta.x()) < 1e-4 && qAbs(sceneDelta.y()) < 1e-4) return;
    QTransform t = view->transform();
    t.translate(sceneDelta.x(), sceneDelta.y());
    view->setTransform(t);
}

inline void glueGraphicsViewPan(QGraphicsView* view, const QPointF& panScenePoint,
                                const QPointF& globalPos) {
    glueGraphicsViewScenePoint(view, panScenePoint, graphicsViewViewportPos(view, globalPos));
}

inline qreal graphicsViewWheelZoomFactor(const QWheelEvent* event) {
    QPoint delta = event->angleDelta();
    if (delta.y() == 0) delta = event->pixelDelta();
    qreal steps = qreal(delta.y()) / 120.0;
    steps = std::clamp(steps, -3.0, 3.0);
    if (std::abs(steps) < 1e-4) return 1.0;
    return std::pow(1.08, steps);
}

inline bool zoomGraphicsViewAtCursor(QGraphicsView* view, qreal factor, const QPointF& globalPos,
                                     QPointF* panScenePoint, qreal minScale, qreal maxScale) {
    factor = std::clamp(factor, 0.72, 1.40);
    const qreal newScale = view->transform().m11() * factor;
    if (newScale < minScale || newScale > maxScale) return false;

    view->setTransformationAnchor(QGraphicsView::NoAnchor);
    view->setResizeAnchor(QGraphicsView::NoAnchor);
    const QPoint viewportPos = graphicsViewViewportPos(view, globalPos);
    const QPointF scenePoint = view->mapToScene(viewportPos);
    view->scale(factor, factor);
    glueGraphicsViewScenePoint(view, scenePoint, viewportPos);
    if (panScenePoint) *panScenePoint = view->mapToScene(viewportPos);
    return true;
}

}  // namespace sol
