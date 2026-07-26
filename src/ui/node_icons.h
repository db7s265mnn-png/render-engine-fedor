// Houdini OBJ node icons (geometry / light / camera) loaded from embedded PNGs.
#pragma once

#include <QPainter>
#include <QRectF>
#include <QString>

namespace sol {

enum class NodeIconKind { None = 0, Geometry, Light, Camera };

NodeIconKind nodeIconKind(const QString& typeName, const QString& category);
QColor nodeBodyColor(NodeIconKind kind, const QColor& fallback);

// Draw the category icon centered in `area` (already clipped to the node body).
void paintNodeIcon(QPainter& painter, NodeIconKind kind, const QRectF& area);

}  // namespace sol
