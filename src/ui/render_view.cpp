#include "ui/render_view.h"

#include <QApplication>
#include <QButtonGroup>
#include <QDateTime>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QResizeEvent>
#include <QToolButton>
#include <QVector3D>
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

float closestRayAxisParam(const Vec3& rayOrigin, const Vec3& rayDir, const Vec3& axisOrigin,
                          const Vec3& axisDir) {
    // Closest points between ray and infinite axis; return param along axis.
    const Vec3 w0 = rayOrigin - axisOrigin;
    const float a = dot(rayDir, rayDir);
    const float b = dot(rayDir, axisDir);
    const float c = dot(axisDir, axisDir);
    const float d = dot(rayDir, w0);
    const float e = dot(axisDir, w0);
    const float denom = a * c - b * b;
    if (std::fabs(denom) < 1e-10f) return e / std::max(1e-8f, c);
    return (a * e - b * d) / denom;
}

float pointSegmentDistance2D(const QPointF& p, const QPointF& a, const QPointF& b, float* tOut = nullptr) {
    const QPointF ab = b - a;
    const float len2 = float(QPointF::dotProduct(ab, ab));
    float t = 0.0f;
    if (len2 > 1e-8f) {
        t = float(QPointF::dotProduct(p - a, ab) / double(len2));
        t = clampf(t, 0.0f, 1.0f);
    }
    if (tOut) *tOut = t;
    const QPointF closest = a + ab * double(t);
    const QPointF d = p - closest;
    return float(QPointF::dotProduct(d, d));
}

QColor axisColor(int axis, bool active) {
    QColor c;
    if (axis == 0) c = QColor(220, 70, 70);
    else if (axis == 1) c = QColor(70, 200, 90);
    else c = QColor(70, 130, 230);
    if (active) c = c.lighter(130);
    return c;
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

    // Centered T / R / S tool strip above the framebuffer (viewport chrome).
    toolStrip_ = new QWidget(this);
    toolStrip_->setObjectName("viewportTransformStrip");
    toolStrip_->setStyleSheet(
        "QWidget#viewportTransformStrip {"
        "  background: rgba(20, 22, 26, 180);"
        "  border: 1px solid rgba(255,255,255,28);"
        "  border-radius: 6px;"
        "}"
        "QToolButton {"
        "  color: #e8eaed;"
        "  background: transparent;"
        "  border: none;"
        "  border-radius: 4px;"
        "  min-width: 28px;"
        "  min-height: 24px;"
        "  font-weight: 700;"
        "  font-size: 12px;"
        "}"
        "QToolButton:checked {"
        "  background: rgba(80, 170, 255, 70);"
        "  color: #ffffff;"
        "}"
        "QToolButton:hover {"
        "  background: rgba(255,255,255,22);"
        "}");
    auto* stripLayout = new QHBoxLayout(toolStrip_);
    stripLayout->setContentsMargins(4, 3, 4, 3);
    stripLayout->setSpacing(2);

    auto* group = new QButtonGroup(toolStrip_);
    group->setExclusive(true);
    auto makeButton = [&](const QString& text, const QString& tip) {
        auto* button = new QToolButton(toolStrip_);
        button->setText(text);
        button->setCheckable(true);
        button->setToolTip(tip);
        button->setAutoRaise(true);
        group->addButton(button);
        stripLayout->addWidget(button);
        return button;
    };
    translateButton_ = makeButton("T", "Translate (T)");
    rotateButton_ = makeButton("R", "Rotate (R)");
    scaleButton_ = makeButton("S", "Scale (S)");
    translateButton_->setChecked(true);

    connect(translateButton_, &QToolButton::clicked, this, [this] {
        setTransformTool(TransformTool::Translate);
    });
    connect(rotateButton_, &QToolButton::clicked, this, [this] {
        setTransformTool(TransformTool::Rotate);
    });
    connect(scaleButton_, &QToolButton::clicked, this, [this] {
        setTransformTool(TransformTool::Scale);
    });
    layoutToolStrip();
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

void RenderView::setTransformTool(TransformTool tool) {
    if (transformTool_ == tool) {
        syncToolButtons();
        return;
    }
    transformTool_ = tool;
    if (mode_ == 4) endGizmoDrag();
    hoverAxis_ = GizmoAxis::None;
    syncToolButtons();
    emit transformToolChanged(transformTool_);
    update();
}

void RenderView::syncToolButtons() {
    if (!translateButton_) return;
    translateButton_->setChecked(transformTool_ == TransformTool::Translate);
    rotateButton_->setChecked(transformTool_ == TransformTool::Rotate);
    scaleButton_->setChecked(transformTool_ == TransformTool::Scale);
}

void RenderView::layoutToolStrip() {
    if (!toolStrip_) return;
    toolStrip_->adjustSize();
    const int x = std::max(8, (width() - toolStrip_->width()) / 2);
    toolStrip_->move(x, 8);
    toolStrip_->raise();
}

void RenderView::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    layoutToolStrip();
}

void RenderView::setTransformTarget(Node* node) {
    if (transformTarget_ == node) return;
    if (mode_ == 4) endGizmoDrag();
    transformTarget_ = node;
    hoverAxis_ = GizmoAxis::None;
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
    if (ndcX < -0.05f || ndcX > 1.05f || ndcY < -0.05f || ndcY > 1.05f) return false;

    const QRect target = imageRect();
    out = QPointF(target.left() + ndcX * target.width(), target.top() + ndcY * target.height());
    return true;
}

bool RenderView::widgetToCameraRay(const QPoint& pos, Vec3& origin, Vec3& direction) const {
    const QRect target = imageRect();
    if (target.width() <= 0 || target.height() <= 0) return false;
    const float u = (float(pos.x() - target.left()) + 0.5f) / float(target.width());
    const float v = (float(pos.y() - target.top()) + 0.5f) / float(target.height());

    const float sensorWidth = 36.0f;
    const float focalLength = 50.0f;
    const float aspect = float(resolutionX_) / float(std::max(1, resolutionY_));
    const float sensorHeight = sensorWidth / aspect;
    const float sx = (u - 0.5f) * sensorWidth;
    const float sy = (0.5f - v) * sensorHeight;
    const Vec3 dirCam = normalize(Vec3(sx, sy, -focalLength));
    const Mat4 cameraToWorld = camera_.toMatrix();
    origin = transformPoint(cameraToWorld, Vec3(0.0f));
    direction = normalize(transformVector(cameraToWorld, dirCam));
    return true;
}

void RenderView::beginNavigation(int mode, const QPoint& pos) {
    mode_ = mode;
    lastMousePosition_ = pos;
    if (mode_ == 1) {
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

bool RenderView::hasTransformTarget() const {
    return transformTarget_ && transformTarget_->findParameter("translate") &&
           transformTool_ != TransformTool::Select;
}

Mat4 RenderView::targetWorldMatrix() const {
    if (!transformTarget_) return Mat4::identity();
    return transformFromParameters(*transformTarget_);
}

Vec3 RenderView::targetOrigin() const {
    const Mat4 m = targetWorldMatrix();
    return Vec3(m.at(0, 3), m.at(1, 3), m.at(2, 3));
}

void RenderView::targetAxes(Vec3& x, Vec3& y, Vec3& z) const {
    const Mat4 m = targetWorldMatrix();
    x = normalize(Vec3(m.at(0, 0), m.at(1, 0), m.at(2, 0)));
    y = normalize(Vec3(m.at(0, 1), m.at(1, 1), m.at(2, 1)));
    z = normalize(Vec3(m.at(0, 2), m.at(1, 2), m.at(2, 2)));
}

float RenderView::gizmoWorldSize() const {
    return std::max(0.15f, camera_.distance * 0.12f);
}

RenderView::GizmoAxis RenderView::hitTestGizmo(const QPoint& pos) const {
    if (!hasTransformTarget()) return GizmoAxis::None;
    QPointF originScreen;
    if (!projectWorldToWidget(targetOrigin(), originScreen)) return GizmoAxis::None;

    const float size = gizmoWorldSize();
    Vec3 ax, ay, az;
    targetAxes(ax, ay, az);
    const Vec3 o = targetOrigin();
    struct Cand {
        GizmoAxis axis;
        float dist2;
    };
    Cand best{GizmoAxis::None, 14.0f * 14.0f};

    auto considerAxis = [&](GizmoAxis axis, const Vec3& dir) {
        QPointF tip;
        if (!projectWorldToWidget(o + dir * size, tip)) return;
        float t = 0.0f;
        const float d2 = pointSegmentDistance2D(QPointF(pos), originScreen, tip, &t);
        if (d2 < best.dist2) best = {axis, d2};
    };

    const QPointF delta = QPointF(pos) - originScreen;
    if (QPointF::dotProduct(delta, delta) < 10.0 * 10.0) return GizmoAxis::Center;

    considerAxis(GizmoAxis::X, ax);
    considerAxis(GizmoAxis::Y, ay);
    considerAxis(GizmoAxis::Z, az);

    if (transformTool_ == TransformTool::Rotate) {
        // Also hit-test a ring around the origin in screen space.
        const float r = float(std::sqrt(QPointF::dotProduct(
            [&]() {
                QPointF tip;
                projectWorldToWidget(o + ax * size, tip);
                return tip - originScreen;
            }(),
            [&]() {
                QPointF tip;
                projectWorldToWidget(o + ax * size, tip);
                return tip - originScreen;
            }())));
        Q_UNUSED(r);
        // Prefer axis tip hits already considered; ring uses same axes.
    }

    return best.axis;
}

bool RenderView::beginGizmoDrag(const QPoint& pos) {
    activeAxis_ = hitTestGizmo(pos);
    if (activeAxis_ == GizmoAxis::None || !hasTransformTarget()) return false;

    dragStartTranslate_ = transformTarget_->vec3Value("translate", Vec3(0.0f));
    dragStartRotate_ = transformTarget_->vec3Value("rotate", Vec3(0.0f));
    dragStartScale_ = transformTarget_->vec3Value("scale", Vec3(1.0f));
    dragOrigin_ = targetOrigin();
    dragStartMouse_ = pos;

    Vec3 ax, ay, az;
    targetAxes(ax, ay, az);
    if (activeAxis_ == GizmoAxis::X) dragAxisDir_ = ax;
    else if (activeAxis_ == GizmoAxis::Y) dragAxisDir_ = ay;
    else if (activeAxis_ == GizmoAxis::Z) dragAxisDir_ = az;
    else dragAxisDir_ = normalize(cross(Vec3(0, 1, 0), normalize(camera_.eye() - dragOrigin_)));

    Vec3 rayO, rayD;
    if (widgetToCameraRay(pos, rayO, rayD))
        dragStartParam_ = closestRayAxisParam(rayO, rayD, dragOrigin_, dragAxisDir_);
    else
        dragStartParam_ = 0.0f;

    mode_ = 4;
    gizmoDidEdit_ = false;
    dragParameterName_.clear();
    lastMousePosition_ = pos;
    setCursor(Qt::ClosedHandCursor);
    grabMouse();
    return true;
}

void RenderView::updateGizmoDrag(const QPoint& pos) {
    if (!transformTarget_ || activeAxis_ == GizmoAxis::None) return;
    const float precision = (QApplication::keyboardModifiers() & Qt::ShiftModifier) ? 0.25f : 1.0f;

    if (transformTool_ == TransformTool::Translate) {
        Vec3 rayO, rayD;
        if (!widgetToCameraRay(pos, rayO, rayD)) return;
        const float param = closestRayAxisParam(rayO, rayD, dragOrigin_, dragAxisDir_);
        const float delta = (param - dragStartParam_) * precision;
        Vec3 t = dragStartTranslate_;
        if (activeAxis_ == GizmoAxis::Center) {
            // Screen-space pan in the camera plane.
            const QPoint d = pos - dragStartMouse_;
            const Vec3 forward = normalize(camera_.pivot - camera_.eye());
            Vec3 right = cross(forward, Vec3(0, 1, 0));
            if (lengthSquared(right) < 1e-8f) right = Vec3(1, 0, 0);
            right = normalize(right);
            const Vec3 up = normalize(cross(right, forward));
            const float scale = camera_.distance * 0.0018f * precision;
            t = dragStartTranslate_ + right * (-float(d.x()) * scale) + up * (float(d.y()) * scale);
        } else {
            // Move along the chosen world/local axis direction.
            t = dragStartTranslate_ + dragAxisDir_ * delta;
        }
        dragParameterName_ = "translate";
        transformTarget_->setParameterValue("translate", QVariant::fromValue(QVector3D(t.x, t.y, t.z)),
                                            false);
        gizmoDidEdit_ = true;
        emit transformEdited(transformTarget_);
        update();
        return;
    }

    if (transformTool_ == TransformTool::Scale) {
        Vec3 rayO, rayD;
        if (!widgetToCameraRay(pos, rayO, rayD)) return;
        const float param = closestRayAxisParam(rayO, rayD, dragOrigin_, dragAxisDir_);
        const float delta = (param - dragStartParam_) * precision;
        const float factor = std::max(0.01f, 1.0f + delta / std::max(0.15f, gizmoWorldSize()));
        Vec3 s = dragStartScale_;
        if (activeAxis_ == GizmoAxis::Center) {
            s = dragStartScale_ * factor;
        } else if (activeAxis_ == GizmoAxis::X) s.x = dragStartScale_.x * factor;
        else if (activeAxis_ == GizmoAxis::Y) s.y = dragStartScale_.y * factor;
        else if (activeAxis_ == GizmoAxis::Z) s.z = dragStartScale_.z * factor;
        dragParameterName_ = "scale";
        transformTarget_->setParameterValue("scale", QVariant::fromValue(QVector3D(s.x, s.y, s.z)), false);
        gizmoDidEdit_ = true;
        emit transformEdited(transformTarget_);
        update();
        return;
    }

    if (transformTool_ == TransformTool::Rotate) {
        const QPoint d = pos - dragStartMouse_;
        const float degrees = float(d.x() + d.y()) * 0.35f * precision;
        Vec3 r = dragStartRotate_;
        if (activeAxis_ == GizmoAxis::X) r.x = dragStartRotate_.x + degrees;
        else if (activeAxis_ == GizmoAxis::Y) r.y = dragStartRotate_.y + degrees;
        else if (activeAxis_ == GizmoAxis::Z) r.z = dragStartRotate_.z + degrees;
        else {
            r.y = dragStartRotate_.y + float(d.x()) * 0.35f * precision;
            r.x = dragStartRotate_.x + float(d.y()) * 0.35f * precision;
        }
        dragParameterName_ = "rotate";
        transformTarget_->setParameterValue("rotate", QVariant::fromValue(QVector3D(r.x, r.y, r.z)), false);
        gizmoDidEdit_ = true;
        emit transformEdited(transformTarget_);
        update();
    }
}

void RenderView::endGizmoDrag() {
    Node* node = transformTarget_;
    const QString param = dragParameterName_;
    const bool edited = gizmoDidEdit_;
    activeAxis_ = GizmoAxis::None;
    gizmoDidEdit_ = false;
    dragParameterName_.clear();
    if (mode_ == 4) {
        mode_ = 0;
        releaseMouse();
        unsetCursor();
    }
    if (edited && node) {
        if (!param.isEmpty()) node->notifyParameterChanged(param);
        emit transformFinished(node);
    }
}

void RenderView::drawGizmo(QPainter& painter) {
    if (!hasTransformTarget()) return;
    QPointF originScreen;
    const Vec3 o = targetOrigin();
    if (!projectWorldToWidget(o, originScreen)) return;

    const float size = gizmoWorldSize();
    Vec3 ax, ay, az;
    targetAxes(ax, ay, az);
    const Vec3 dirs[3] = {ax, ay, az};
    const GizmoAxis axes[3] = {GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z};

    painter.setRenderHint(QPainter::Antialiasing, true);

    // Center handle
    const bool centerActive = (hoverAxis_ == GizmoAxis::Center || activeAxis_ == GizmoAxis::Center);
    painter.setPen(QPen(centerActive ? QColor(255, 220, 90) : QColor(240, 240, 240), 1.2));
    painter.setBrush(centerActive ? QColor(255, 220, 90, 200) : QColor(230, 230, 230, 180));
    painter.drawEllipse(originScreen, 5.5, 5.5);

    for (int i = 0; i < 3; ++i) {
        QPointF tip;
        if (!projectWorldToWidget(o + dirs[i] * size, tip)) continue;
        const bool active = (hoverAxis_ == axes[i] || activeAxis_ == axes[i]);
        const QColor color = axisColor(i, active);
        painter.setPen(QPen(color, active ? 3.0 : 2.2, Qt::SolidLine, Qt::RoundCap));
        painter.drawLine(originScreen, tip);

        if (transformTool_ == TransformTool::Translate) {
            // Arrow tip
            QPainterPath arrow;
            const QPointF dir = tip - originScreen;
            const double len = std::sqrt(QPointF::dotProduct(dir, dir));
            if (len < 1.0) continue;
            const QPointF n = dir / len;
            const QPointF perp(-n.y(), n.x());
            arrow.moveTo(tip);
            arrow.lineTo(tip - n * 12.0 + perp * 5.0);
            arrow.lineTo(tip - n * 12.0 - perp * 5.0);
            arrow.closeSubpath();
            painter.setBrush(color);
            painter.setPen(Qt::NoPen);
            painter.drawPath(arrow);
        } else if (transformTool_ == TransformTool::Scale) {
            painter.setBrush(color);
            painter.setPen(Qt::NoPen);
            painter.drawRect(QRectF(tip.x() - 4.5, tip.y() - 4.5, 9.0, 9.0));
        } else if (transformTool_ == TransformTool::Rotate) {
            painter.setBrush(Qt::NoBrush);
            painter.setPen(QPen(color, active ? 2.4 : 1.6));
            painter.drawEllipse(tip, 6.0, 6.0);
        }
    }
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

    drawGizmo(painter);

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

    if (event->button() == Qt::LeftButton && !alt && hasTransformTarget()) {
        if (beginGizmoDrag(event->pos())) {
            event->accept();
            return;
        }
    }

    mode_ = 0;
    QWidget::mousePressEvent(event);
}

void RenderView::mouseMoveEvent(QMouseEvent* event) {
    if (mode_ == 4) {
        updateGizmoDrag(event->pos());
        event->accept();
        return;
    }
    if (mode_ == 0) {
        if (hasTransformTarget()) {
            const GizmoAxis hit = hitTestGizmo(event->pos());
            if (hit != hoverAxis_) {
                hoverAxis_ = hit;
                update();
            }
            setCursor(hit != GizmoAxis::None ? Qt::SizeAllCursor : Qt::ArrowCursor);
        }
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
    if (mode_ == 4) {
        endGizmoDrag();
        event->accept();
        return;
    }
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
