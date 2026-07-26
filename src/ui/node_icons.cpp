#include "ui/node_icons.h"

#include <QHash>
#include <QPixmap>

namespace sol {
namespace {

QString iconResourcePath(NodeIconKind kind) {
    switch (kind) {
        case NodeIconKind::Geometry:
            return QStringLiteral(":/icons/OBJ_geo.png");
        case NodeIconKind::Light:
            return QStringLiteral(":/icons/OBJ_hlight.png");
        case NodeIconKind::Camera:
            return QStringLiteral(":/icons/OBJ_camera.png");
        case NodeIconKind::Merge:
            return QStringLiteral(":/icons/LOP_merge.png");
        case NodeIconKind::Material:
            return QStringLiteral(":/icons/MaterialX.png");
        case NodeIconKind::Render:
            return QStringLiteral(":/icons/ARRI.png");
        default:
            return {};
    }
}

}  // namespace

QPixmap nodeIconPixmap(NodeIconKind kind) {
    static QHash<int, QPixmap> cache;
    const int key = static_cast<int>(kind);
    if (cache.contains(key)) return cache.value(key);

    const QString path = iconResourcePath(kind);
    QPixmap pm;
    if (!path.isEmpty()) pm = QPixmap(path);
    cache.insert(key, pm);
    return pm;
}

NodeIconKind nodeIconKind(const QString& typeName, const QString& category) {
    if (typeName == "camera") return NodeIconKind::Camera;
    if (typeName == "merge") return NodeIconKind::Merge;
    if (typeName == "material" || category == "Material") return NodeIconKind::Material;
    if (typeName == "rendersettings" || category == "Render") return NodeIconKind::Render;
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
        case NodeIconKind::Merge:
            return QColor(150, 152, 156);  // Houdini merge gray
        case NodeIconKind::Material:
            return QColor(48, 50, 56);  // Dark body so MaterialX mark reads cleanly
        case NodeIconKind::Render:
            return QColor(0, 91, 166);  // ARRI corporate blue
        default:
            return fallback;
    }
}

void paintNodeIcon(QPainter& painter, NodeIconKind kind, const QRectF& area) {
    if (kind == NodeIconKind::None) return;
    const QPixmap pm = nodeIconPixmap(kind);
    if (pm.isNull()) return;

    painter.save();
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    // Fit the icon inside the body; keep ~25% smaller than the available area.
    const qreal pad = qMin(area.width(), area.height()) * 0.08;
    const QRectF target = area.adjusted(pad, pad, -pad, -pad);
    const QSizeF scaled = QSizeF(pm.size()).scaled(target.size() * 0.75, Qt::KeepAspectRatio);
    const QRectF dest(target.center().x() - scaled.width() * 0.5,
                      target.center().y() - scaled.height() * 0.5, scaled.width(), scaled.height());
    painter.drawPixmap(dest, pm, QRectF(pm.rect()));
    painter.restore();
}

}  // namespace sol
