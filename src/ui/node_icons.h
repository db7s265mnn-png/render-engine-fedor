// Node body icons loaded from embedded PNGs (Houdini / MaterialX / ARRI).
#pragma once

#include <QPainter>
#include <QPixmap>
#include <QRectF>
#include <QString>

namespace sol {

enum class NodeIconKind {
    None = 0,
    Geometry,
    Light,
    Camera,
    Merge,
    Material,
    Render,
};

NodeIconKind nodeIconKind(const QString& typeName, const QString& category);
QColor nodeBodyColor(NodeIconKind kind, const QColor& fallback);

// Draw the category icon centered in `area` (already clipped to the node body).
void paintNodeIcon(QPainter& painter, NodeIconKind kind, const QRectF& area);

// Shared pixmap accessor for Material Network / other views.
QPixmap nodeIconPixmap(NodeIconKind kind);

}  // namespace sol
