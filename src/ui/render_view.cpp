#include "ui/render_view.h"

#include <QMouseEvent>
#include <QPainter>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

#include "ui/theme.h"

namespace sol {

Vec3 ViewCamera::eye() const {
    const float yawRad = radians(yaw);
    const float pitchRad = radians(pitch);
    const Vec3 offset(std::cos(pitchRad) * std::sin(yawRad), std::sin(pitchRad),
                      std::cos(pitchRad) * std::cos(yawRad));
    return pivot + offset * distance;
}

Mat4 ViewCamera::toMatrix() const { return lookAtMatrix(eye(), pivot, Vec3(0.0f, 1.0f, 0.0f)); }

void ViewCamera::setFromMatrix(const Mat4& cameraToWorld, float focusDistance) {
    const Vec3 position(cameraToWorld.at(0, 3), cameraToWorld.at(1, 3), cameraToWorld.at(2, 3));
    const Vec3 forward = normalize(
        Vec3(-cameraToWorld.at(0, 2), -cameraToWorld.at(1, 2), -cameraToWorld.at(2, 2)));
    distance = std::max(0.05f, focusDistance);
    pivot = position + forward * distance;
    pitch = degrees(std::asin(clampf(-forward.y, -1.0f, 1.0f)));
    yaw = degrees(std::atan2(-forward.x, -forward.z));
}

void ViewCamera::orbit(float deltaYaw, float deltaPitch) {
    yaw += deltaYaw;
    pitch = clampf(pitch + deltaPitch, -89.0f, 89.0f);
}

void ViewCamera::pan(float dx, float dy) {
    const Vec3 forward = normalize(pivot - eye());
    Vec3 right = cross(forward, Vec3(0.0f, 1.0f, 0.0f));
    if (lengthSquared(right) < 1e-8f) right = Vec3(1.0f, 0.0f, 0.0f);
    right = normalize(right);
    const Vec3 up = normalize(cross(right, forward));
    const float scale = distance * 0.0022f;
    pivot += right * (-dx * scale) + up * (dy * scale);
}

void ViewCamera::dolly(float amount) { distance = clampf(distance * std::pow(1.0025f, -amount), 0.02f, 1e6f); }

// ---------------------------------------------------------------------------

RenderView::RenderView(QWidget* parent) : QWidget(parent) {
    setMinimumSize(320, 200);
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, theme::gridDark());
    setPalette(pal);
}

void RenderView::setImage(const QImage& image) {
    image_ = image;
    update();
}

void RenderView::clearImage() {
    image_ = QImage();
    update();
}

void RenderView::setResolution(int width, int height) {
    resolutionX_ = std::max(1, width);
    resolutionY_ = std::max(1, height);
}

void RenderView::setCamera(const ViewCamera& camera) {
    camera_ = camera;
    update();
}

QRect RenderView::imageRect() const {
    const double aspect = double(resolutionX_) / double(std::max(1, resolutionY_));
    int w = width() - 16;
    int h = int(w / aspect);
    if (h > height() - 16) {
        h = height() - 16;
        w = int(h * aspect);
    }
    return QRect((width() - w) / 2, (height() - h) / 2, std::max(1, w), std::max(1, h));
}

void RenderView::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), theme::gridDark());

    const QRect target = imageRect();
    if (!image_.isNull()) {
        painter.setRenderHint(QPainter::SmoothPixmapTransform, true);
        painter.drawImage(target, image_);
    } else {
        painter.setPen(theme::textDim());
        painter.drawText(rect(), Qt::AlignCenter, "Press Render to start the path tracer");
    }
    painter.setPen(QPen(QColor(0, 0, 0, 120), 1));
    painter.drawRect(target.adjusted(0, 0, -1, -1));

    if (!statusText_.isEmpty()) {
        QFont font = painter.font();
        font.setPointSizeF(8.5);
        painter.setFont(font);
        const QRect textRect(target.left(), target.bottom() - 22, target.width(), 20);
        painter.fillRect(textRect, QColor(0, 0, 0, 130));
        painter.setPen(QColor(235, 237, 240));
        painter.drawText(textRect.adjusted(8, 0, -8, 0), Qt::AlignVCenter | Qt::AlignLeft, statusText_);
    }
}

void RenderView::mousePressEvent(QMouseEvent* event) {
    lastMousePosition_ = event->pos();
    if (!navigationEnabled_) return;
    const bool alt = event->modifiers() & Qt::AltModifier;
    if (event->button() == Qt::LeftButton && alt) {
        mode_ = 1;
    } else if (event->button() == Qt::MiddleButton) {
        mode_ = 2;
    } else if (event->button() == Qt::RightButton && alt) {
        mode_ = 3;
    } else {
        mode_ = 0;
    }
    if (mode_ != 0) setCursor(mode_ == 1 ? Qt::SizeAllCursor : Qt::ClosedHandCursor);
}

void RenderView::mouseMoveEvent(QMouseEvent* event) {
    if (mode_ == 0) return;
    const QPoint delta = event->pos() - lastMousePosition_;
    lastMousePosition_ = event->pos();
    switch (mode_) {
        case 1: camera_.orbit(-float(delta.x()) * 0.35f, float(delta.y()) * 0.35f); break;
        case 2: camera_.pan(float(delta.x()), float(delta.y())); break;
        case 3: camera_.dolly(float(delta.x() + delta.y())); break;
        default: break;
    }
    emit cameraMoved();
}

void RenderView::mouseReleaseEvent(QMouseEvent*) {
    mode_ = 0;
    unsetCursor();
}

void RenderView::wheelEvent(QWheelEvent* event) {
    if (!navigationEnabled_) return;
    camera_.dolly(float(event->angleDelta().y()) * 0.6f);
    emit cameraMoved();
}

}  // namespace sol
