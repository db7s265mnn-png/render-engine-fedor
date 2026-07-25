#include "ui/render_view.h"

#include <QDateTime>
#include <QMouseEvent>
#include <QPainter>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

#include "ui/theme.h"

namespace sol {
namespace {

Vec3 rotateAroundAxis(const Vec3& point, const Vec3& center, const Vec3& axis, float degrees) {
    const float a = radians(degrees);
    const float c = std::cos(a);
    const float s = std::sin(a);
    const Vec3 d = point - center;
    const Vec3 n = normalize(axis);
    return center + d * c + cross(n, d) * s + n * dot(n, d) * (1.0f - c);
}

Vec3 rotateAroundY(const Vec3& point, const Vec3& center, float degrees) {
    return rotateAroundAxis(point, center, Vec3(0.0f, 1.0f, 0.0f), degrees);
}

}  // namespace

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
    const Vec3 forward =
        normalize(Vec3(-cameraToWorld.at(0, 2), -cameraToWorld.at(1, 2), -cameraToWorld.at(2, 2)));
    distance = std::max(0.05f, focusDistance);
    pivot = position + forward * distance;
    const Vec3 toEye = normalize(position - pivot);
    pitch = degrees(std::asin(clampf(toEye.y, -1.0f, 1.0f)));
    yaw = degrees(std::atan2(toEye.x, toEye.z));
}

void ViewCamera::orbit(float deltaYaw, float deltaPitch) {
    yaw += deltaYaw;
    pitch = clampf(pitch + deltaPitch, -89.0f, 89.0f);
}

void ViewCamera::orbitAround(const Vec3& center, float deltaYaw, float deltaPitch) {
    // Rotate both the eye and the look-at point around `center` so the framed
    // image does not jump when the tumble pivot is chosen under the cursor.
    Vec3 e = eye();
    Vec3 p = pivot;

    e = rotateAroundY(e, center, deltaYaw);
    p = rotateAroundY(p, center, deltaYaw);

    Vec3 forward = normalize(p - e);
    Vec3 right = cross(forward, Vec3(0.0f, 1.0f, 0.0f));
    if (lengthSquared(right) < 1e-8f) right = Vec3(1.0f, 0.0f, 0.0f);
    right = normalize(right);

    e = rotateAroundAxis(e, center, right, deltaPitch);
    p = rotateAroundAxis(p, center, right, deltaPitch);

    pivot = p;
    distance = std::max(0.05f, length(e - p));
    const Vec3 toEye = (e - p) / distance;
    pitch = degrees(std::asin(clampf(toEye.y, -1.0f, 1.0f)));
    yaw = degrees(std::atan2(toEye.x, toEye.z));
}

void ViewCamera::pan(float dx, float dy) {
    const Vec3 forward = normalize(pivot - eye());
    Vec3 right = cross(forward, Vec3(0.0f, 1.0f, 0.0f));
    if (lengthSquared(right) < 1e-8f) right = Vec3(1.0f, 0.0f, 0.0f);
    right = normalize(right);
    const Vec3 up = normalize(cross(right, forward));
    const float scale = distance * 0.0018f;
    pivot += right * (-dx * scale) + up * (dy * scale);
}

void ViewCamera::dolly(float amount) {
    distance = clampf(distance * std::pow(1.0018f, -amount), 0.02f, 1e6f);
}

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

bool RenderView::pickUnderMouse(const QPoint& pos, Vec3& hitPoint) const {
    if (!pickCallback_) return false;
    const QRect target = imageRect();
    if (!target.contains(pos) || target.width() <= 0 || target.height() <= 0) return false;
    const float u = (float(pos.x() - target.left()) + 0.5f) / float(target.width());
    const float v = (float(pos.y() - target.top()) + 0.5f) / float(target.height());
    return pickCallback_(u, v, hitPoint);
}

bool RenderView::projectWorldToWidget(const Vec3& world, QPointF& out) const {
    const Mat4 worldToCamera = inverse(camera_.toMatrix());
    const Vec3 cam = transformPoint(worldToCamera, world);
    if (cam.z >= -1e-4f) return false;

    const float sensorWidth = 36.0f;
    const float focalLength = 50.0f;
    const float aspect = float(resolutionX_) / float(std::max(1, resolutionY_));
    const float sensorHeight = sensorWidth / aspect;
    const float sx = (cam.x / -cam.z) * focalLength;
    const float sy = (cam.y / -cam.z) * focalLength;
    const float ndcX = sx / sensorWidth + 0.5f;
    const float ndcY = 0.5f - sy / sensorHeight;
    if (ndcX < 0.0f || ndcX > 1.0f || ndcY < 0.0f || ndcY > 1.0f) return false;

    const QRect target = imageRect();
    out = QPointF(target.left() + ndcX * target.width(), target.top() + ndcY * target.height());
    return true;
}

void RenderView::beginNavigation(int mode, const QPoint& pos) {
    mode_ = mode;
    lastMousePosition_ = pos;
    if (mode_ == 1) {
        // Pick the tumble center under the cursor, but do NOT reframe the camera.
        // The view stays exactly where it is; later drags rotate around this point.
        tumbleCenter_ = camera_.pivot;
        Vec3 hit;
        if (pickUnderMouse(pos, hit)) {
            tumbleCenter_ = hit;
            showPivotMarker_ = true;
            pivotMarkerWorld_ = hit;
            pivotMarkerUntilMs_ = QDateTime::currentMSecsSinceEpoch() + 1600;
            update();
        }
        setCursor(Qt::SizeAllCursor);
    } else {
        setCursor(Qt::ClosedHandCursor);
    }
    grabMouse();
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

    if (showPivotMarker_ && QDateTime::currentMSecsSinceEpoch() < pivotMarkerUntilMs_) {
        QPointF screen;
        if (projectWorldToWidget(pivotMarkerWorld_, screen)) {
            painter.setRenderHint(QPainter::Antialiasing, true);
            painter.setPen(QPen(QColor(255, 210, 70), 1.6));
            painter.setBrush(QColor(255, 210, 70, 180));
            painter.drawEllipse(screen, 5.0, 5.0);
            painter.drawLine(screen + QPointF(-10, 0), screen + QPointF(10, 0));
            painter.drawLine(screen + QPointF(0, -10), screen + QPointF(0, 10));
        }
    } else {
        showPivotMarker_ = false;
    }

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
    if (!navigationEnabled_) {
        QWidget::mousePressEvent(event);
        return;
    }

    const bool alt = event->modifiers() & Qt::AltModifier;
    // Orbit / tumble: Alt+LMB (Houdini) or plain RMB.
    if ((event->button() == Qt::LeftButton && alt) ||
        (event->button() == Qt::RightButton && !alt)) {
        beginNavigation(1, event->pos());
        event->accept();
        return;
    }
    if (event->button() == Qt::MiddleButton) {
        beginNavigation(2, event->pos());
        event->accept();
        return;
    }
    if (event->button() == Qt::RightButton && alt) {
        beginNavigation(3, event->pos());
        event->accept();
        return;
    }

    mode_ = 0;
    QWidget::mousePressEvent(event);
}

void RenderView::mouseMoveEvent(QMouseEvent* event) {
    if (mode_ == 0) {
        QWidget::mouseMoveEvent(event);
        return;
    }
    const QPoint delta = event->pos() - lastMousePosition_;
    lastMousePosition_ = event->pos();
    const float precision = (event->modifiers() & Qt::ShiftModifier) ? 0.3f : 1.0f;
    switch (mode_) {
        case 1:
            camera_.orbitAround(tumbleCenter_, -float(delta.x()) * 0.22f * precision,
                                float(delta.y()) * 0.22f * precision);
            break;
        case 2:
            camera_.pan(float(delta.x()) * precision, float(delta.y()) * precision);
            break;
        case 3:
            camera_.dolly(float(delta.x() + delta.y()) * precision);
            break;
        default:
            break;
    }
    emit cameraMoved();
    update();
    event->accept();
}

void RenderView::mouseReleaseEvent(QMouseEvent* event) {
    if (mode_ != 0) {
        mode_ = 0;
        releaseMouse();
        unsetCursor();
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

void RenderView::wheelEvent(QWheelEvent* event) {
    if (!navigationEnabled_) {
        QWidget::wheelEvent(event);
        return;
    }
    const float precision = (event->modifiers() & Qt::ShiftModifier) ? 0.3f : 1.0f;
    camera_.dolly(float(event->angleDelta().y()) * 0.28f * precision);
    emit cameraMoved();
    event->accept();
}

}  // namespace sol
