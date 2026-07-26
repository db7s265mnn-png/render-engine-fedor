#include "ui/node_icons.h"

#include <QLinearGradient>
#include <QPainterPath>
#include <QRadialGradient>
#include <QtMath>

namespace sol {
namespace {

void drawShadedEllipse(QPainter& painter, const QRectF& r, const QColor& base, const QPointF& highlight) {
    QRadialGradient g(highlight, r.width() * 0.55);
    g.setColorAt(0.0, base.lighter(160));
    g.setColorAt(0.45, base);
    g.setColorAt(1.0, base.darker(145));
    painter.setPen(QPen(base.darker(170), 0.8));
    painter.setBrush(g);
    painter.drawEllipse(r);
}

void paintGeometryIcon(QPainter& painter, const QRectF& area) {
    const QPointF c = area.center();
    const qreal s = qMin(area.width(), area.height());

    // Blue cylinder (back-left), yellow cone (middle), red sphere (front-right).
    // Cylinder body
    {
        const QRectF cyl(c.x() - s * 0.38, c.y() - s * 0.28, s * 0.22, s * 0.48);
        QLinearGradient body(cyl.topLeft(), cyl.topRight());
        body.setColorAt(0.0, QColor(40, 90, 180));
        body.setColorAt(0.35, QColor(90, 160, 255));
        body.setColorAt(0.7, QColor(55, 120, 220));
        body.setColorAt(1.0, QColor(25, 60, 140));
        painter.setPen(QPen(QColor(20, 40, 90), 0.8));
        painter.setBrush(body);
        painter.drawRoundedRect(cyl, 2.0, 2.0);
        // Top ellipse
        const QRectF top(cyl.left() - 0.5, cyl.top() - s * 0.06, cyl.width() + 1.0, s * 0.14);
        QRadialGradient topG(top.center() + QPointF(-s * 0.02, -s * 0.01), top.width() * 0.5);
        topG.setColorAt(0.0, QColor(140, 190, 255));
        topG.setColorAt(1.0, QColor(40, 90, 180));
        painter.setBrush(topG);
        painter.drawEllipse(top);
        // Bottom rim
        const QRectF bot(cyl.left() - 0.5, cyl.bottom() - s * 0.05, cyl.width() + 1.0, s * 0.12);
        painter.setBrush(QColor(30, 70, 150));
        painter.drawEllipse(bot);
    }

    // Yellow cone
    {
        QPainterPath cone;
        const QPointF tip(c.x() + s * 0.02, c.y() - s * 0.30);
        const QPointF bl(c.x() - s * 0.10, c.y() + s * 0.22);
        const QPointF br(c.x() + s * 0.18, c.y() + s * 0.22);
        cone.moveTo(tip);
        cone.lineTo(bl);
        cone.lineTo(br);
        cone.closeSubpath();
        QLinearGradient g(tip, QPointF(c.x() + s * 0.04, c.y() + s * 0.22));
        g.setColorAt(0.0, QColor(255, 230, 90));
        g.setColorAt(0.55, QColor(240, 180, 30));
        g.setColorAt(1.0, QColor(170, 110, 10));
        painter.setPen(QPen(QColor(120, 80, 10), 0.8));
        painter.setBrush(g);
        painter.drawPath(cone);
        // Base ellipse
        const QRectF base(c.x() - s * 0.10, c.y() + s * 0.16, s * 0.28, s * 0.12);
        QRadialGradient bg(base.center(), base.width() * 0.5);
        bg.setColorAt(0.0, QColor(255, 220, 100));
        bg.setColorAt(1.0, QColor(180, 120, 20));
        painter.setBrush(bg);
        painter.drawEllipse(base);
    }

    // Red sphere (front)
    {
        const QRectF ball(c.x() + s * 0.08, c.y() - s * 0.02, s * 0.30, s * 0.30);
        drawShadedEllipse(painter, ball, QColor(210, 55, 50),
                          ball.center() + QPointF(-s * 0.05, -s * 0.05));
    }
}

void paintLightIcon(QPainter& painter, const QRectF& area) {
    const QPointF c = area.center();
    const qreal s = qMin(area.width(), area.height());

    // Soft glow behind the bulb.
    {
        QRadialGradient glow(c + QPointF(0, -s * 0.05), s * 0.42);
        glow.setColorAt(0.0, QColor(255, 250, 210, 90));
        glow.setColorAt(1.0, QColor(255, 250, 210, 0));
        painter.setPen(Qt::NoPen);
        painter.setBrush(glow);
        painter.drawEllipse(c + QPointF(0, -s * 0.05), s * 0.42, s * 0.42);
    }

    // Glass bulb
    {
        const QRectF glass(c.x() - s * 0.18, c.y() - s * 0.34, s * 0.36, s * 0.42);
        QRadialGradient g(glass.center() + QPointF(-s * 0.06, -s * 0.08), glass.width() * 0.55);
        g.setColorAt(0.0, QColor(245, 250, 255));
        g.setColorAt(0.45, QColor(190, 220, 245));
        g.setColorAt(1.0, QColor(120, 160, 200));
        painter.setPen(QPen(QColor(70, 100, 130), 0.9));
        painter.setBrush(g);
        painter.drawEllipse(glass);

        // Highlight
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(255, 255, 255, 140));
        painter.drawEllipse(QRectF(glass.left() + s * 0.06, glass.top() + s * 0.08, s * 0.10, s * 0.16));
    }

    // Filament
    {
        painter.setPen(QPen(QColor(255, 200, 80), 1.3, Qt::SolidLine, Qt::RoundCap));
        const qreal fx = c.x();
        const qreal fy = c.y() - s * 0.06;
        painter.drawLine(QPointF(fx - s * 0.04, fy + s * 0.06), QPointF(fx - s * 0.02, fy - s * 0.02));
        painter.drawLine(QPointF(fx - s * 0.02, fy - s * 0.02), QPointF(fx + s * 0.02, fy - s * 0.02));
        painter.drawLine(QPointF(fx + s * 0.02, fy - s * 0.02), QPointF(fx + s * 0.04, fy + s * 0.06));
    }

    // Neck
    {
        const QRectF neck(c.x() - s * 0.08, c.y() + s * 0.06, s * 0.16, s * 0.08);
        QLinearGradient g(neck.topLeft(), neck.topRight());
        g.setColorAt(0.0, QColor(150, 150, 155));
        g.setColorAt(0.5, QColor(220, 220, 225));
        g.setColorAt(1.0, QColor(120, 120, 125));
        painter.setPen(QPen(QColor(70, 70, 75), 0.7));
        painter.setBrush(g);
        painter.drawRoundedRect(neck, 1.5, 1.5);
    }

    // Screw base
    {
        const QRectF base(c.x() - s * 0.11, c.y() + s * 0.13, s * 0.22, s * 0.18);
        QLinearGradient g(base.topLeft(), base.topRight());
        g.setColorAt(0.0, QColor(110, 110, 115));
        g.setColorAt(0.45, QColor(200, 200, 205));
        g.setColorAt(1.0, QColor(90, 90, 95));
        painter.setPen(QPen(QColor(50, 50, 55), 0.8));
        painter.setBrush(g);
        painter.drawRoundedRect(base, 2.0, 2.0);
        painter.setPen(QPen(QColor(60, 60, 65, 160), 1.0));
        for (int i = 0; i < 3; ++i) {
            const qreal y = base.top() + base.height() * (0.25 + 0.22 * i);
            painter.drawLine(QPointF(base.left() + 2.0, y), QPointF(base.right() - 2.0, y));
        }
        // Tip
        painter.setBrush(QColor(70, 70, 75));
        painter.setPen(Qt::NoPen);
        painter.drawEllipse(QRectF(c.x() - s * 0.04, base.bottom() - s * 0.02, s * 0.08, s * 0.06));
    }
}

void paintCameraIcon(QPainter& painter, const QRectF& area) {
    const QPointF c = area.center();
    const qreal s = qMin(area.width(), area.height());

    auto metalGrad = [](const QRectF& r) {
        QLinearGradient g(r.topLeft(), r.topRight());
        g.setColorAt(0.0, QColor(35, 35, 38));
        g.setColorAt(0.4, QColor(90, 90, 95));
        g.setColorAt(1.0, QColor(25, 25, 28));
        return g;
    };

    // Body
    {
        const QRectF body(c.x() - s * 0.28, c.y() - s * 0.06, s * 0.42, s * 0.28);
        painter.setPen(QPen(QColor(10, 10, 12), 0.9));
        painter.setBrush(metalGrad(body));
        painter.drawRoundedRect(body, 3.0, 3.0);
        // Side detail
        painter.setBrush(QColor(20, 20, 22));
        painter.drawRoundedRect(QRectF(body.left() + 2.0, body.top() + 3.0, s * 0.08, body.height() - 6.0), 1.5,
                                1.5);
    }

    // Film reels
    auto drawReel = [&](const QPointF& center, qreal radius) {
        QRadialGradient g(center + QPointF(-radius * 0.25, -radius * 0.25), radius);
        g.setColorAt(0.0, QColor(80, 80, 85));
        g.setColorAt(0.55, QColor(30, 30, 33));
        g.setColorAt(1.0, QColor(10, 10, 12));
        painter.setPen(QPen(QColor(0, 0, 0), 0.8));
        painter.setBrush(g);
        painter.drawEllipse(center, radius, radius);
        painter.setBrush(QColor(15, 15, 18));
        painter.drawEllipse(center, radius * 0.35, radius * 0.35);
        painter.setPen(QPen(QColor(120, 120, 125), 0.7));
        painter.setBrush(Qt::NoBrush);
        painter.drawEllipse(center, radius * 0.7, radius * 0.7);
    };
    drawReel(QPointF(c.x() - s * 0.14, c.y() - s * 0.16), s * 0.13);
    drawReel(QPointF(c.x() + s * 0.02, c.y() - s * 0.16), s * 0.13);

    // Bridge between reels
    {
        const QRectF bridge(c.x() - s * 0.14, c.y() - s * 0.20, s * 0.16, s * 0.07);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(25, 25, 28));
        painter.drawRoundedRect(bridge, 1.5, 1.5);
    }

    // Lens barrel
    {
        const QRectF barrel(c.x() + s * 0.12, c.y() - s * 0.02, s * 0.22, s * 0.20);
        QLinearGradient g(barrel.topLeft(), barrel.bottomLeft());
        g.setColorAt(0.0, QColor(55, 55, 60));
        g.setColorAt(0.5, QColor(25, 25, 28));
        g.setColorAt(1.0, QColor(45, 45, 50));
        painter.setPen(QPen(QColor(0, 0, 0), 0.8));
        painter.setBrush(g);
        painter.drawRoundedRect(barrel, 2.0, 2.0);

        // Lens glass
        const QRectF glass(barrel.right() - s * 0.09, barrel.center().y() - s * 0.07, s * 0.14, s * 0.14);
        QRadialGradient lg(glass.center() + QPointF(-s * 0.02, -s * 0.02), glass.width() * 0.5);
        lg.setColorAt(0.0, QColor(180, 210, 230));
        lg.setColorAt(0.4, QColor(40, 70, 110));
        lg.setColorAt(0.75, QColor(10, 20, 40));
        lg.setColorAt(1.0, QColor(0, 0, 0));
        painter.setBrush(lg);
        painter.drawEllipse(glass);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(255, 255, 255, 100));
        painter.drawEllipse(QRectF(glass.left() + s * 0.03, glass.top() + s * 0.02, s * 0.04, s * 0.03));
    }
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
    painter.save();
    painter.setRenderHint(QPainter::Antialiasing, true);
    switch (kind) {
        case NodeIconKind::Geometry:
            paintGeometryIcon(painter, area);
            break;
        case NodeIconKind::Light:
            paintLightIcon(painter, area);
            break;
        case NodeIconKind::Camera:
            paintCameraIcon(painter, area);
            break;
        default:
            break;
    }
    painter.restore();
}

}  // namespace sol
