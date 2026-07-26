#include "ui/node_icons.h"

#include <QHash>
#include <QPixmap>

namespace sol {
namespace {

const QPixmap& iconPixmap(NodeIconKind kind) {
    static QHash<int, QPixmap> cache;
    const int key = static_cast<int>(kind);
    if (cache.contains(key)) return cache[key];

    QString path;
    switch (kind) {
        case NodeIconKind::Geometry:
            path = QStringLiteral(":/icons/OBJ_geo.png");
            break;
        case NodeIconKind::Light:
            path = QStringLiteral(":/icons/OBJ_hlight.png");
            break;
        case NodeIconKind::Camera:
            path = QStringLiteral(":/icons/OBJ_camera.png");
            break;
        default:
            cache.insert(key, QPixmap());
            return cache[key];
    }

    QPixmap pm(path);
    cache.insert(key, pm);
    return cache[key];
}

}  // namespace

NodeIconKind nodeIconKind(const QString& typeName, const QString& category) {
    if (typeName == "camera") return NodeIconKind::Camera;
    if (category == "Lighting" || typeName.endsWith("light")) return NodeIconKind::Light;
    if (category == "Geometry" || typeName == "sphere" || typeName == "grid" || typeName == "box" ||
        typeName == "tube" || typeName == "alembic" || typeName == "usd")
        return NodeIconKind::Geometry;
    return NodeIconKind::None;
}

QColor nodeBodyColor(NodeIconKind kind, const QColor& fallback) {
    switch (kind) {
        case NodeIconKind::Geometry:
            return QColor(168, 170, 174);  // Houdini SOP gray
        case NodeIconKind::Light:
            return QColor(214, 122, 42);  // Houdini light orange
        case NodeIconKind::Camera:
            return QColor(58, 118, 178);  // Houdini camera blue
        default:
            return fallback;
    }
}

void paintNodeIcon(QPainter& painter, NodeIconKind kind, const QRectF& area) {
    if (kind == NodeIconKind::None) return;
    const QPixmap& pm = iconPixmap(kind);
    if (pm.isNull()) return;

    painter.save();
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    // Fit the icon inside the body with a little padding, preserving aspect.
    const qreal pad = qMin(area.width(), area.height()) * 0.08;
    const QRectF target = area.adjusted(pad, pad, -pad, -pad);
    const QSizeF scaled = QSizeF(pm.size()).scaled(target.size(), Qt::KeepAspectRatio);
    const QRectF dest(target.center().x() - scaled.width() * 0.5,
                      target.center().y() - scaled.height() * 0.5, scaled.width(), scaled.height());
    painter.drawPixmap(dest, pm, QRectF(pm.rect()));
    painter.restore();
}

}  // namespace sol
