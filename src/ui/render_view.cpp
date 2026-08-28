#include "ui/render_view.h"

#include <QApplication>
#include <QButtonGroup>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QLayout>
#include <QKeyEvent>
#include <QAbstractButton>
#include <QAction>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QResizeEvent>
#include <QShowEvent>
#include <QSignalBlocker>
#include <QToolButton>
#include <QVector3D>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>

#include "ui/dock_chrome.h"
#include "ui/theme.h"

namespace sol {
namespace {

constexpr int kChromeButtonHeight = 24;

QString chromeToolButtonStyle(int minWidth) {
    return QStringLiteral(
               "QToolButton {"
               "  color: #e8eaed;"
               "  background: #3a3e44;"
               "  border: 1px solid #4a4f57;"
               "  border-radius: 6px;"
               "  min-width: %1px;"
               "  min-height: %2px;"
               "  max-height: %2px;"
               "  font-weight: 600;"
               "  font-size: 11px;"
               "  padding: 0 8px;"
               "}"
               "QToolButton:checked { %3 }"
               "QToolButton:hover { background: #474c54; }"
               "QToolButton:checked:hover { %4 }")
        .arg(minWidth)
        .arg(kChromeButtonHeight)
        .arg(theme::checkedCss())
        .arg(theme::checkedHoverCss());
}

void fitChromeButton(QAbstractButton* button) {
    if (!button) return;
    button->setFixedHeight(kChromeButtonHeight);
    button->setFocusPolicy(Qt::NoFocus);
    if (auto* tool = qobject_cast<QToolButton*>(button)) tool->setAutoRaise(true);
}

QString findPlaceholderAsset() {
    QStringList roots = {
        QDir::currentPath() + "/examples",
        QDir::currentPath() + "/assets",
        QDir::currentPath() + "/../examples",
        QDir::currentPath() + "/../assets",
        QDir::currentPath() + "/../../examples",
    };
    if (QCoreApplication::instance()) {
        const QString appDir = QCoreApplication::applicationDirPath();
        roots.prepend(appDir + "/examples");
        roots.prepend(appDir + "/assets");
        roots.prepend(appDir + "/../examples");
        roots.prepend(appDir + "/../assets");
    }
    for (const QString& root : roots) {
        const QFileInfo info(QDir(root).absoluteFilePath(QStringLiteral("render_placeholder.jpg")));
        if (info.exists() && info.isFile()) return info.absoluteFilePath();
    }
    return {};
}

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

Mat4 rotationMatrixAxisAngle(Vec3 axis, float degrees) {
    const Vec3 n = normalize(axis);
    const float a = radians(degrees);
    const float c = std::cos(a);
    const float s = std::sin(a);
    const float t = 1.0f - c;
    Mat4 r = Mat4::identity();
    r.at(0, 0) = t * n.x * n.x + c;
    r.at(0, 1) = t * n.x * n.y - s * n.z;
    r.at(0, 2) = t * n.x * n.z + s * n.y;
    r.at(1, 0) = t * n.x * n.y + s * n.z;
    r.at(1, 1) = t * n.y * n.y + c;
    r.at(1, 2) = t * n.y * n.z - s * n.x;
    r.at(2, 0) = t * n.x * n.z - s * n.y;
    r.at(2, 1) = t * n.y * n.z + s * n.x;
    r.at(2, 2) = t * n.z * n.z + c;
    return r;
}

// Inverse of composeTRS rotation part R = Rz * Ry * Rx.
Vec3 extractEulerXYZ(const Mat4& m) {
    Vec3 x(m.at(0, 0), m.at(1, 0), m.at(2, 0));
    Vec3 y(m.at(0, 1), m.at(1, 1), m.at(2, 1));
    Vec3 z(m.at(0, 2), m.at(1, 2), m.at(2, 2));
    const float sx = length(x);
    const float sy = length(y);
    const float sz = length(z);
    if (sx > 1e-8f) x = x * (1.0f / sx);
    if (sy > 1e-8f) y = y * (1.0f / sy);
    if (sz > 1e-8f) z = z * (1.0f / sz);
    // Orthonormalize in case of shear from float error.
    z = normalize(cross(x, y));
    y = normalize(cross(z, x));

    // R = Rz*Ry*Rx → sin(ry) = -R20 = -column0.z
    const float syAsin = clampf(-x.z, -1.0f, 1.0f);
    const float ry = degrees(std::asin(syAsin));
    float rx = 0.0f;
    float rz = 0.0f;
    if (std::fabs(syAsin) < 0.9999f) {
        rx = degrees(std::atan2(y.z, z.z));
        rz = degrees(std::atan2(x.y, x.x));
    } else {
        // Gimbal lock: ry ≈ ±90°, fold into rz.
        rx = 0.0f;
        rz = degrees(std::atan2(-y.x, y.y));
    }
    return Vec3(rx, ry, rz);
}

void decomposeTRS(const Mat4& m, Vec3& translate, Vec3& rotateDeg, Vec3& scale) {
    translate = Vec3(m.at(0, 3), m.at(1, 3), m.at(2, 3));
    Vec3 x(m.at(0, 0), m.at(1, 0), m.at(2, 0));
    Vec3 y(m.at(0, 1), m.at(1, 1), m.at(2, 1));
    Vec3 z(m.at(0, 2), m.at(1, 2), m.at(2, 2));
    scale = Vec3(length(x), length(y), length(z));
    rotateDeg = extractEulerXYZ(m);
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

    // Chrome strip above the framebuffer. Detach lives in a 28px column on the
    // RIGHT of this row so it cannot sit at (0,0) over Start.
    chromeRow_ = new QWidget(this);
    chromeRow_->setObjectName("viewportChromeRow");
    chromeRow_->setFixedHeight(theme::chromeBarHeight());
    chromeRow_->setStyleSheet(
        "QWidget#viewportChromeRow {"
        "  background: #2e3136;"
        "  border-bottom: 1px solid #22242a;"
        "}");
    auto* chromeRowLayout = new QHBoxLayout(chromeRow_);
    chromeRowLayout->setContentsMargins(0, 0, 0, 0);
    chromeRowLayout->setSpacing(0);

    chromeBar_ = new QWidget(chromeRow_);
    chromeBar_->setObjectName("viewportChromeBar");
    chromeBar_->setStyleSheet(
        "QWidget#viewportChromeBar { background: transparent; border: none; }");

    detachSlot_ = new QWidget(chromeRow_);
    detachSlot_->setObjectName("viewportDetachSlot");
    detachSlot_->setFixedWidth(28);
    detachSlot_->setStyleSheet(
        "QWidget#viewportDetachSlot { background: transparent; border: none; }");
    auto* detachSlotLayout = new QHBoxLayout(detachSlot_);
    detachSlotLayout->setContentsMargins(0, 0, 6, 0);
    detachSlotLayout->setSpacing(0);
    detachButton_ = new DockDetachButton(detachSlot_);
    detachSlotLayout->addWidget(detachButton_, 0, Qt::AlignVCenter | Qt::AlignRight);
    connect(detachButton_, &QToolButton::clicked, this, [this] {
        if (onDetach_) onDetach_();
    });
    chromeRowLayout->addWidget(chromeBar_, 1);
    chromeRowLayout->addWidget(detachSlot_, 0);
    detachSlot_->hide();
    detachButton_->hide();

    renderControlStrip_ = new QWidget(chromeBar_);
    renderControlStrip_->setObjectName("viewportRenderControls");
    renderControlStrip_->setStyleSheet(
        "QWidget#viewportRenderControls { background: transparent; border: none; }");
    auto* renderLayout = new QHBoxLayout(renderControlStrip_);
    renderLayout->setContentsMargins(8, 4, 4, 4);
    renderLayout->setSpacing(4);
    startButton_ = new QToolButton(renderControlStrip_);
    startButton_->setText(QStringLiteral("Start"));
    startButton_->setCheckable(true);
    startButton_->setStyleSheet(chromeToolButtonStyle(48));
    fitChromeButton(startButton_);
    stopButton_ = new QToolButton(renderControlStrip_);
    stopButton_->setText(QStringLiteral("Stop"));
    stopButton_->setCheckable(true);
    stopButton_->setChecked(true);
    stopButton_->setStyleSheet(chromeToolButtonStyle(48));
    fitChromeButton(stopButton_);
    renderLayout->addWidget(startButton_);
    renderLayout->addWidget(stopButton_);

    toolStrip_ = new QWidget(chromeBar_);
    toolStrip_->setObjectName("viewportTransformStrip");
    toolStrip_->setStyleSheet(
        QStringLiteral(
            "QWidget#viewportTransformStrip {"
            "  background: transparent;"
            "  border: none;"
            "}"
            "QToolButton {"
            "  color: #e8eaed;"
            "  background: #3a3e44;"
            "  border: 1px solid #4a4f57;"
            "  border-radius: 6px;"
            "  min-width: 28px;"
            "  min-height: 24px;"
            "  max-height: 24px;"
            "  font-weight: 700;"
            "  font-size: 11px;"
            "  padding: 0 4px;"
            "}"
            "QToolButton:checked { %1 }"
            "QToolButton:hover {"
            "  background: #474c54;"
            "}")
            .arg(theme::checkedCss()));
    auto* stripLayout = new QHBoxLayout(toolStrip_);
    stripLayout->setContentsMargins(4, 4, 4, 4);
    stripLayout->setSpacing(4);

    cameraMenuButton_ = new QToolButton(toolStrip_);
    cameraMenuButton_->setText("persp");
    cameraMenuButton_->setToolTip("Look through camera (Houdini-style)");
    cameraMenuButton_->setPopupMode(QToolButton::InstantPopup);
    cameraMenuButton_->setAutoRaise(true);
    cameraMenuButton_->setStyleSheet(
        "QToolButton {"
        "  min-width: 72px; max-width: 140px;"
        "  min-height: 24px; max-height: 24px;"
        "  font-size: 11px; font-weight: 600;"
        "  text-align: left; padding-left: 6px; padding-right: 6px;"
        "  background: #3a3e44; border: 1px solid #4a4f57; border-radius: 6px; color: #e8eaed;"
        "}"
        "QToolButton:hover { background: #474c54; }"
        "QToolButton::menu-indicator { width: 10px; }");
    fitChromeButton(cameraMenuButton_);
    stripLayout->addWidget(cameraMenuButton_);
    rebuildCameraMenu();

    homeButton_ = new QToolButton(toolStrip_);
    homeButton_->setText(QStringLiteral("Home"));
    homeButton_->setToolTip(QStringLiteral("Home — frame all geometry (H)"));
    homeButton_->setAutoRaise(true);
    homeButton_->setStyleSheet(
        "QToolButton {"
        "  min-width: 44px; min-height: 24px; max-height: 24px;"
        "  font-size: 11px; font-weight: 600;"
        "  background: #3a3e44; border: 1px solid #4a4f57; border-radius: 6px; color: #e8eaed;"
        "}"
        "QToolButton:hover { background: #474c54; }");
    fitChromeButton(homeButton_);
    stripLayout->addWidget(homeButton_);
    connect(homeButton_, &QToolButton::clicked, this, [this] { frameAll(); });

    auto* camSep = new QWidget(toolStrip_);
    camSep->setFixedWidth(1);
    camSep->setStyleSheet("background: rgba(255,255,255,40);");
    camSep->setMinimumHeight(18);
    stripLayout->addSpacing(4);
    stripLayout->addWidget(camSep);
    stripLayout->addSpacing(4);

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
        fitChromeButton(button);
        return button;
    };
    selectButton_ = makeButton("Q", "Select (Q) — click objects in the viewport");
    translateButton_ = makeButton("T", "Translate (T)");
    rotateButton_ = makeButton("R", "Rotate (R)");
    scaleButton_ = makeButton("S", "Scale (S)");
    translateButton_->setChecked(true);

    auto* sep = new QWidget(toolStrip_);
    sep->setFixedWidth(1);
    sep->setStyleSheet("background: rgba(255,255,255,40);");
    sep->setMinimumHeight(18);
    stripLayout->addSpacing(4);
    stripLayout->addWidget(sep);
    stripLayout->addSpacing(4);

    auto* spaceGroup = new QButtonGroup(toolStrip_);
    spaceGroup->setExclusive(true);
    auto makeSpaceButton = [&](const QString& text, const QString& tip) {
        auto* button = new QToolButton(toolStrip_);
        button->setText(text);
        button->setCheckable(true);
        button->setToolTip(tip);
        button->setAutoRaise(true);
        button->setStyleSheet(
            QStringLiteral(
                "QToolButton {"
                "  min-width: 42px; min-height: 24px; max-height: 24px;"
                "  font-size: 11px; font-weight: 600;"
                "  background: #3a3e44; border: 1px solid #4a4f57; border-radius: 6px; color: #e8eaed;"
                "}"
                "QToolButton:checked { %1 }"
                "QToolButton:hover { background: #474c54; }")
                .arg(theme::checkedCss()));
        spaceGroup->addButton(button);
        stripLayout->addWidget(button);
        fitChromeButton(button);
        return button;
    };
    localSpaceButton_ = makeSpaceButton("Local", "Local transform space");
    worldSpaceButton_ = makeSpaceButton("World", "World transform space");
    localSpaceButton_->setChecked(true);

    stripLayout->addSpacing(8);
    auto comboStyle = QStringLiteral(
        "QComboBox {"
        "  min-height: 24px; max-height: 24px;"
        "  font-size: 11px; font-weight: 600;"
        "  background: #3a3e44; border: 1px solid #4a4f57; border-radius: 6px; color: #e8eaed;"
        "  padding: 0 6px;"
        "}"
        "QComboBox:hover { background: #474c54; }"
        "QComboBox::drop-down { border: none; width: 16px; }");

    colorManagementCombo_ = new QComboBox(toolStrip_);
    colorManagementCombo_->addItem(QStringLiteral("Classic"), 0);
    colorManagementCombo_->addItem(QStringLiteral("ACES"), 1);
    colorManagementCombo_->setCurrentIndex(1);
    colorManagementCombo_->setToolTip(
        QStringLiteral("Classic: linear → sRGB (no OCIO, no tone map; Houdini-style).\n"
                       "ACES: OpenColorIO Display/View from the ACES config."));
    colorManagementCombo_->setStyleSheet(comboStyle + QStringLiteral(" QComboBox { max-width: 90px; }"));
    colorManagementCombo_->setFixedHeight(kChromeButtonHeight);
    stripLayout->addWidget(colorManagementCombo_);
    connect(colorManagementCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                if (index < 0) return;
                setColorManagement(colorManagementCombo_->itemData(index).toInt());
            });

    viewTransformCombo_ = new QComboBox(toolStrip_);
    viewTransformCombo_->addItem(QStringLiteral("sRGB"), 0);
    viewTransformCombo_->addItem(QStringLiteral("Rec.709"), 1);
    viewTransformCombo_->addItem(QStringLiteral("Rec.2020"), 3);
    viewTransformCombo_->addItem(QStringLiteral("Raw"), 2);
    viewTransformCombo_->setCurrentIndex(0);
    viewTransformCombo_->setToolTip(
        QStringLiteral("Monitor view transform.\n"
                       "Working space is set in Render Settings → Film."));
    viewTransformCombo_->setStyleSheet(comboStyle + QStringLiteral(" QComboBox { max-width: 110px; }"));
    viewTransformCombo_->setFixedHeight(kChromeButtonHeight);
    stripLayout->addWidget(viewTransformCombo_);
    connect(viewTransformCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                if (index < 0) return;
                setViewTransform(viewTransformCombo_->itemData(index).toInt());
            });

    connect(selectButton_, &QToolButton::clicked, this, [this] {
        setTransformTool(TransformTool::Select);
    });
    connect(translateButton_, &QToolButton::clicked, this, [this] {
        setTransformTool(TransformTool::Translate);
    });
    connect(rotateButton_, &QToolButton::clicked, this, [this] {
        setTransformTool(TransformTool::Rotate);
    });
    connect(scaleButton_, &QToolButton::clicked, this, [this] {
        setTransformTool(TransformTool::Scale);
    });
    connect(localSpaceButton_, &QToolButton::clicked, this, [this] {
        setTransformSpace(TransformSpace::Local);
    });
    connect(worldSpaceButton_, &QToolButton::clicked, this, [this] {
        setTransformSpace(TransformSpace::World);
    });
    layoutToolStrip();

    const QString phPath = findPlaceholderAsset();
    if (!phPath.isEmpty()) {
        placeholderImage_ = QImage(phPath);
        if (placeholderImage_.isNull())
            placeholderImage_ = QImage();
    }
}

void RenderView::setImage(const QImage& image) {
    if (fadeActive_ && !placeholderImage_.isNull() && !image.isNull()) {
        const qint64 now = QDateTime::currentMSecsSinceEpoch();
        const float t =
            fadeDurationMs_ > 0
                ? std::clamp(float(now - fadeStartMs_) / float(fadeDurationMs_), 0.0f, 1.0f)
                : 1.0f;
        if (t >= 1.0f) {
            fadeActive_ = false;
            image_ = image;
        } else {
            const QSize sz = image.size();
            QImage ph = coverCroppedPlaceholder(sz).convertToFormat(QImage::Format_RGB32);
            QImage beauty = image.convertToFormat(QImage::Format_RGB32);
            if (ph.size() != sz)
                ph = ph.scaled(sz, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
            QImage blended(sz, QImage::Format_RGB32);
            for (int y = 0; y < sz.height(); ++y) {
                const QRgb* a = reinterpret_cast<const QRgb*>(ph.constScanLine(y));
                const QRgb* b = reinterpret_cast<const QRgb*>(beauty.constScanLine(y));
                QRgb* d = reinterpret_cast<QRgb*>(blended.scanLine(y));
                for (int x = 0; x < sz.width(); ++x) {
                    const int ar = qRed(a[x]), ag = qGreen(a[x]), ab = qBlue(a[x]);
                    const int br = qRed(b[x]), bg = qGreen(b[x]), bb = qBlue(b[x]);
                    d[x] = qRgb(int(ar + (br - ar) * t + 0.5f), int(ag + (bg - ag) * t + 0.5f),
                                int(ab + (bb - ab) * t + 0.5f));
                }
            }
            image_ = blended;
        }
    } else {
        fadeActive_ = false;
        image_ = image;
    }
    update();
}

void RenderView::clearImage() {
    image_ = QImage();
    fadeActive_ = false;
    update();
}

void RenderView::showPlaceholder(bool show) {
    showPlaceholder_ = show;
    if (show) {
        fadeActive_ = false;
        image_ = QImage();
    }
    update();
}

void RenderView::beginPlaceholderFade(int durationMs) {
    fadeActive_ = true;
    fadeStartMs_ = QDateTime::currentMSecsSinceEpoch();
    fadeDurationMs_ = std::max(1, durationMs);
    showPlaceholder_ = false;
    update();
}

QImage RenderView::coverCroppedPlaceholder(const QSize& targetSize) const {
    if (placeholderImage_.isNull() || targetSize.width() <= 0 || targetSize.height() <= 0)
        return QImage();
    const double srcAspect =
        double(placeholderImage_.width()) / double(std::max(1, placeholderImage_.height()));
    const double dstAspect = double(targetSize.width()) / double(targetSize.height());
    QRect src(0, 0, placeholderImage_.width(), placeholderImage_.height());
    if (dstAspect > srcAspect) {
        // Framebuffer wider → crop top/bottom.
        const int h = int(std::lround(placeholderImage_.width() / dstAspect));
        const int y = std::max(0, (placeholderImage_.height() - h) / 2);
        src = QRect(0, y, placeholderImage_.width(), std::min(h, placeholderImage_.height() - y));
    } else if (dstAspect < srcAspect) {
        // Framebuffer narrower → crop sides.
        const int w = int(std::lround(placeholderImage_.height() * dstAspect));
        const int x = std::max(0, (placeholderImage_.width() - w) / 2);
        src = QRect(x, 0, std::min(w, placeholderImage_.width() - x), placeholderImage_.height());
    }
    return placeholderImage_.copy(src).scaled(targetSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
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
    if (focusPickActive_) setFocusPickActive(false);
    hoverAxis_ = GizmoAxis::None;
    syncToolButtons();
    emit transformToolChanged(transformTool_);
    update();
}

void RenderView::syncToolButtons() {
    if (!translateButton_) return;
    if (selectButton_) selectButton_->setChecked(transformTool_ == TransformTool::Select);
    translateButton_->setChecked(transformTool_ == TransformTool::Translate);
    rotateButton_->setChecked(transformTool_ == TransformTool::Rotate);
    scaleButton_->setChecked(transformTool_ == TransformTool::Scale);
    if (localSpaceButton_) {
        localSpaceButton_->setChecked(transformSpace_ == TransformSpace::Local);
        worldSpaceButton_->setChecked(transformSpace_ == TransformSpace::World);
    }
}

void RenderView::setTransformSpace(TransformSpace space) {
    if (transformSpace_ == space) {
        syncToolButtons();
        return;
    }
    transformSpace_ = space;
    if (mode_ == 4) endGizmoDrag();
    hoverAxis_ = GizmoAxis::None;
    syncToolButtons();
    emit transformSpaceChanged(transformSpace_);
    update();
}

void RenderView::setViewTransform(int view) {
    if (view != 0 && view != 1 && view != 2 && view != 3) view = 0;
    if (viewTransform_ == view) return;
    viewTransform_ = view;
    if (viewTransformCombo_) {
        const QSignalBlocker block(viewTransformCombo_);
        const int idx = viewTransformCombo_->findData(view);
        if (idx >= 0) viewTransformCombo_->setCurrentIndex(idx);
    }
    emit viewTransformChanged(viewTransform_);
    update();
}

void RenderView::setColorManagement(int mode) {
    mode = std::clamp(mode, 0, 1);
    if (colorManagement_ == mode) return;
    colorManagement_ = mode;
    if (colorManagementCombo_) {
        const QSignalBlocker block(colorManagementCombo_);
        const int idx = colorManagementCombo_->findData(mode);
        if (idx >= 0) colorManagementCombo_->setCurrentIndex(idx);
    }
    emit colorManagementChanged(colorManagement_);
    update();
}

void RenderView::setFocusPickActive(bool active) {
    if (focusPickActive_ == active) return;
    focusPickActive_ = active;
    if (focusPickActive_) {
        if (mode_ == 4) endGizmoDrag();
        setCursor(Qt::CrossCursor);
    } else if (mode_ == 0) {
        unsetCursor();
    }
    emit focusPickChanged(focusPickActive_);
    update();
}

void RenderView::layoutToolStrip() {
    if (!chromeBar_ || !toolStrip_) return;
    const int chromeH = theme::chromeBarHeight();
    // QWidget::isVisible() is false while any ancestor is hidden, so it cannot
    // drive this layout (slot stays hidden forever, or the button sits at 0,0).
    const bool showDetach = bool(onDetach_);
    if (chromeRow_) {
        chromeRow_->setGeometry(0, 0, width(), chromeH);
    } else {
        chromeBar_->setGeometry(0, 0, width(), chromeH);
    }
    if (detachSlot_) {
        detachSlot_->setFixedWidth(showDetach ? 28 : 0);
        detachSlot_->setVisible(showDetach);
    }
    if (chromeRow_) {
        if (QLayout* rowLayout = chromeRow_->layout()) rowLayout->activate();
    }
    if (detachButton_) detachButton_->setVisible(showDetach);
    int left = 0;
    if (renderControlStrip_) {
        renderControlStrip_->adjustSize();
        const int y = std::max(0, (chromeH - renderControlStrip_->height()) / 2);
        renderControlStrip_->move(0, y);
        left = renderControlStrip_->width() + 4;
    }
    toolStrip_->adjustSize();
    const int available = std::max(0, chromeBar_->width());
    const int x = std::max(left, (available - toolStrip_->width()) / 2);
    const int y = std::max(0, (chromeH - toolStrip_->height()) / 2);
    toolStrip_->move(x, y);
    if (chromeRow_)
        chromeRow_->raise();
    else
        chromeBar_->raise();
    if (renderControlStrip_) renderControlStrip_->raise();
    toolStrip_->raise();
}

void RenderView::attachRenderActions(QAction* start, QAction* stop) {
    if (startButton_ && start) {
        startButton_->setDefaultAction(start);
        startButton_->setToolButtonStyle(Qt::ToolButtonTextOnly);
        startButton_->setStyleSheet(chromeToolButtonStyle(48));
        fitChromeButton(startButton_);
    }
    if (stopButton_ && stop) {
        stopButton_->setDefaultAction(stop);
        stopButton_->setToolButtonStyle(Qt::ToolButtonTextOnly);
        stopButton_->setStyleSheet(chromeToolButtonStyle(48));
        fitChromeButton(stopButton_);
    }
    layoutToolStrip();
}

void RenderView::setOnDetach(std::function<void()> onDetach) {
    onDetach_ = std::move(onDetach);
    if (detachButton_) detachButton_->setToolTip(QStringLiteral("Detach"));
    layoutToolStrip();
}

void RenderView::setViewportFloating(bool floating) {
    if (detachButton_) {
        detachButton_->setToolTip(floating ? QStringLiteral("Dock") : QStringLiteral("Detach"));
    }
}

void RenderView::setCameraMenu(const QStringList& cameraNames, const QString& activeName) {
    cameraMenuNames_ = cameraNames;
    activeCameraName_ = activeName;
    if (cameraMenuButton_) {
        const QString label = activeCameraName_.isEmpty() ? QStringLiteral("persp") : activeCameraName_;
        cameraMenuButton_->setText(label);
        cameraMenuButton_->setToolTip(activeCameraName_.isEmpty()
                                          ? QStringLiteral("Free perspective — choose a camera to look through")
                                          : QStringLiteral("Looking through %1").arg(activeCameraName_));
    }
    rebuildCameraMenu();
}

void RenderView::rebuildCameraMenu() {
    if (!cameraMenuButton_) return;
    auto* menu = new QMenu(cameraMenuButton_);
    auto* freeAction = menu->addAction(QStringLiteral("persp (free)"));
    freeAction->setCheckable(true);
    freeAction->setChecked(activeCameraName_.isEmpty());
    connect(freeAction, &QAction::triggered, this, [this] { emit lookThroughCameraChosen(QString()); });

    if (!cameraMenuNames_.isEmpty()) {
        menu->addSeparator();
        for (const QString& name : cameraMenuNames_) {
            auto* action = menu->addAction(name);
            action->setCheckable(true);
            action->setChecked(name == activeCameraName_);
            connect(action, &QAction::triggered, this, [this, name] { emit lookThroughCameraChosen(name); });
        }
    }
    cameraMenuButton_->setMenu(menu);
}

void RenderView::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    layoutToolStrip();
}

void RenderView::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
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
    const int top = theme::chromeBarHeight() + 8;
    const double aspect = double(resolutionX_) / double(std::max(1, resolutionY_));
    int w = width() - 16;
    int h = int(w / aspect);
    if (h > height() - top - 8) {
        h = height() - top - 8;
        w = int(h * aspect);
    }
    return QRect((width() - w) / 2, top + std::max(0, (height() - top - 8 - h) / 2), std::max(1, w),
                 std::max(1, h));
}

bool RenderView::pickUnderMouse(const QPoint& pos, Vec3& hitPoint) const {
    if (!pickCallback_) return false;
    const QRect target = imageRect();
    if (!target.contains(pos) || target.width() <= 0 || target.height() <= 0) return false;
    const float u = (float(pos.x() - target.left()) + 0.5f) / float(target.width());
    const float v = (float(pos.y() - target.top()) + 0.5f) / float(target.height());
    return pickCallback_(u, v, hitPoint);
}

bool RenderView::pickObjectUnderMouse(const QPoint& pos, QString& sourceNode) const {
    sourceNode.clear();
    if (!objectPickCallback_) return false;
    const QRect target = imageRect();
    if (!target.contains(pos) || target.width() <= 0 || target.height() <= 0) return false;
    const float u = (float(pos.x() - target.left()) + 0.5f) / float(target.width());
    const float v = (float(pos.y() - target.top()) + 0.5f) / float(target.height());
    return objectPickCallback_(u, v, sourceNode);
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
    emit cameraNavStarted();
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
    if (transformSpace_ == TransformSpace::World) {
        x = Vec3(1.0f, 0.0f, 0.0f);
        y = Vec3(0.0f, 1.0f, 0.0f);
        z = Vec3(0.0f, 0.0f, 1.0f);
        return;
    }
    const Mat4 m = targetWorldMatrix();
    x = normalize(Vec3(m.at(0, 0), m.at(1, 0), m.at(2, 0)));
    y = normalize(Vec3(m.at(0, 1), m.at(1, 1), m.at(2, 1)));
    z = normalize(Vec3(m.at(0, 2), m.at(1, 2), m.at(2, 2)));
}

float RenderView::gizmoWorldSize() const {
    return std::max(0.15f, camera_.distance * 0.12f);
}

bool RenderView::rayPlaneHit(const Vec3& rayO, const Vec3& rayD, const Vec3& planeO, const Vec3& planeN,
                             Vec3& out) const {
    const float denom = dot(rayD, planeN);
    if (std::fabs(denom) < 1e-8f) return false;
    const float t = dot(planeO - rayO, planeN) / denom;
    if (t < 0.0f) return false;
    out = rayO + rayD * t;
    return true;
}

float RenderView::angleOnAxisPlane(const Vec3& axis, const Vec3& center, const Vec3& point) const {
    const Vec3 a = normalize(axis);
    Vec3 ref = (std::fabs(a.y) < 0.9f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 u = cross(a, ref);
    if (lengthSquared(u) < 1e-10f) u = cross(a, Vec3(0.0f, 0.0f, 1.0f));
    u = normalize(u);
    const Vec3 v = cross(a, u);
    Vec3 d = point - center;
    d = d - a * dot(d, a);
    if (lengthSquared(d) < 1e-12f) return 0.0f;
    return std::atan2(dot(d, v), dot(d, u));
}

bool RenderView::ringAngleAtMouse(const QPoint& pos, const Vec3& axis, float& angleOut) const {
    Vec3 rayO, rayD;
    if (!widgetToCameraRay(pos, rayO, rayD)) return false;
    Vec3 hit;
    if (!rayPlaneHit(rayO, rayD, targetOrigin(), normalize(axis), hit)) return false;
    angleOut = angleOnAxisPlane(axis, targetOrigin(), hit);
    return true;
}

float RenderView::ringScreenDistance(const QPoint& pos, const Vec3& axis, float radius) const {
    const Vec3 o = targetOrigin();
    const Vec3 n = normalize(axis);
    Vec3 ref = (std::fabs(n.y) < 0.9f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 u = cross(n, ref);
    if (lengthSquared(u) < 1e-10f) u = cross(n, Vec3(0.0f, 0.0f, 1.0f));
    u = normalize(u);
    const Vec3 v = cross(n, u);

    constexpr int kSegs = 64;
    float best = 1.0e9f;
    QPointF prev;
    bool havePrev = false;
    for (int i = 0; i <= kSegs; ++i) {
        const float a = kTwoPi * float(i) / float(kSegs);
        const Vec3 p = o + (u * std::cos(a) + v * std::sin(a)) * radius;
        QPointF screen;
        if (!projectWorldToWidget(p, screen)) {
            havePrev = false;
            continue;
        }
        if (havePrev) {
            float t = 0.0f;
            best = std::min(best, pointSegmentDistance2D(QPointF(pos), prev, screen, &t));
        }
        prev = screen;
        havePrev = true;
    }
    return best;
}

RenderView::GizmoAxis RenderView::hitTestGizmo(const QPoint& pos) const {
    if (!hasTransformTarget()) return GizmoAxis::None;
    QPointF originScreen;
    if (!projectWorldToWidget(targetOrigin(), originScreen)) return GizmoAxis::None;

    const float size = gizmoWorldSize();
    Vec3 ax, ay, az;
    targetAxes(ax, ay, az);
    const Vec3 o = targetOrigin();

    if (transformTool_ == TransformTool::Rotate) {
        struct Cand {
            GizmoAxis axis;
            float dist2;
        };
        Cand best{GizmoAxis::None, 12.0f * 12.0f};
        const float dx = ringScreenDistance(pos, ax, size);
        const float dy = ringScreenDistance(pos, ay, size);
        const float dz = ringScreenDistance(pos, az, size);
        if (dx < best.dist2) best = {GizmoAxis::X, dx};
        if (dy < best.dist2) best = {GizmoAxis::Y, dy};
        if (dz < best.dist2) best = {GizmoAxis::Z, dz};
        return best.axis;
    }

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
    return best.axis;
}

bool RenderView::beginGizmoDrag(const QPoint& pos) {
    activeAxis_ = hitTestGizmo(pos);
    if (activeAxis_ == GizmoAxis::None || !hasTransformTarget()) return false;

    dragStartTranslate_ = transformTarget_->vec3Value("translate", Vec3(0.0f));
    dragStartRotate_ = transformTarget_->vec3Value("rotate", Vec3(0.0f));
    dragStartScale_ = transformTarget_->vec3Value("scale", Vec3(1.0f));
    dragStartMatrix_ = transformFromParameters(*transformTarget_);
    dragOrigin_ = targetOrigin();
    dragStartMouse_ = pos;
    dragStartAngle_ = 0.0f;

    Vec3 ax, ay, az;
    targetAxes(ax, ay, az);
    if (activeAxis_ == GizmoAxis::X) dragAxisDir_ = ax;
    else if (activeAxis_ == GizmoAxis::Y) dragAxisDir_ = ay;
    else if (activeAxis_ == GizmoAxis::Z) dragAxisDir_ = az;
    else dragAxisDir_ = normalize(cross(Vec3(0, 1, 0), normalize(camera_.eye() - dragOrigin_)));

    if (transformTool_ == TransformTool::Rotate) {
        if (!ringAngleAtMouse(pos, dragAxisDir_, dragStartAngle_)) return false;
    } else {
        Vec3 rayO, rayD;
        if (widgetToCameraRay(pos, rayO, rayD))
            dragStartParam_ = closestRayAxisParam(rayO, rayD, dragOrigin_, dragAxisDir_);
        else
            dragStartParam_ = 0.0f;
    }

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
        float angle = dragStartAngle_;
        if (!ringAngleAtMouse(pos, dragAxisDir_, angle)) return;
        float deltaDeg = degrees(angle - dragStartAngle_) * precision;
        // Keep continuity across the ±π wrap.
        while (deltaDeg > 180.0f) deltaDeg -= 360.0f;
        while (deltaDeg < -180.0f) deltaDeg += 360.0f;

        const Mat4 deltaRot = rotationMatrixAxisAngle(dragAxisDir_, deltaDeg);
        const Mat4 toOrigin = Mat4::translate(Vec3(-dragOrigin_.x, -dragOrigin_.y, -dragOrigin_.z));
        const Mat4 fromOrigin = Mat4::translate(dragOrigin_);
        const Mat4 result = fromOrigin * deltaRot * toOrigin * dragStartMatrix_;

        Vec3 t, r, s;
        decomposeTRS(result, t, r, s);
        // Keep the authored scale stable while rotating.
        s = dragStartScale_;
        dragParameterName_ = "rotate";
        transformTarget_->setParameterValue("translate", QVariant::fromValue(QVector3D(t.x, t.y, t.z)), false);
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
        if (param == "rotate") {
            node->notifyParameterChanged("translate");
            node->notifyParameterChanged("rotate");
        } else if (!param.isEmpty()) {
            node->notifyParameterChanged(param);
        }
        emit transformFinished(node);
    }
}

void RenderView::drawRotationRings(QPainter& painter, const Vec3& origin, const Vec3& ax, const Vec3& ay,
                                   const Vec3& az, float radius) {
    const Vec3 axes[3] = {ax, ay, az};
    const GizmoAxis ids[3] = {GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z};
    constexpr int kSegs = 64;

    for (int axisIndex = 0; axisIndex < 3; ++axisIndex) {
        const Vec3 n = normalize(axes[axisIndex]);
        Vec3 ref = (std::fabs(n.y) < 0.9f) ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
        Vec3 u = cross(n, ref);
        if (lengthSquared(u) < 1e-10f) u = cross(n, Vec3(0.0f, 0.0f, 1.0f));
        u = normalize(u);
        const Vec3 v = cross(n, u);
        const bool active = (hoverAxis_ == ids[axisIndex] || activeAxis_ == ids[axisIndex]);
        const QColor color = axisColor(axisIndex, active);

        QPainterPath path;
        bool started = false;
        for (int i = 0; i <= kSegs; ++i) {
            const float a = kTwoPi * float(i) / float(kSegs);
            const Vec3 p = origin + (u * std::cos(a) + v * std::sin(a)) * radius;
            // Back-face fade: hide ring segments facing away from the camera.
            const Vec3 view = normalize(camera_.eye() - p);
            if (dot(n, view) < -0.15f && !active) {
                started = false;
                continue;
            }
            QPointF screen;
            if (!projectWorldToWidget(p, screen)) {
                started = false;
                continue;
            }
            if (!started) {
                path.moveTo(screen);
                started = true;
            } else {
                path.lineTo(screen);
            }
        }
        painter.setBrush(Qt::NoBrush);
        painter.setPen(QPen(color, active ? 3.2 : 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        painter.drawPath(path);
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

    if (transformTool_ == TransformTool::Rotate) {
        drawRotationRings(painter, o, ax, ay, az, size);
        painter.setPen(QPen(QColor(240, 240, 240), 1.0));
        painter.setBrush(QColor(230, 230, 230, 160));
        painter.drawEllipse(originScreen, 4.0, 4.0);
        return;
    }

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
        }
    }
}

void RenderView::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), theme::gridDark());

    const QRect target = imageRect();
    // Coarse nav film (1/32 … 1/4) is a splat: nearest upscale, not a blur.
    const bool fatNavPixels =
        !image_.isNull() && image_.width() > 0 && target.width() > 0 &&
        image_.width() * 2 < target.width();
    painter.setRenderHint(QPainter::SmoothPixmapTransform, !fatNavPixels);
    if (!image_.isNull()) {
        painter.drawImage(target, image_);
    } else if (showPlaceholder_ && !placeholderImage_.isNull()) {
        painter.drawImage(target, coverCroppedPlaceholder(target.size()));
    } else {
        painter.setPen(theme::textDim());
        painter.drawText(rect(), Qt::AlignCenter, "Press Start to cook and render");
    }
    painter.setPen(QPen(QColor(0, 0, 0, 120), 1));
    painter.drawRect(target.adjusted(0, 0, -1, -1));

    if (showPivotMarker_ && QDateTime::currentMSecsSinceEpoch() < pivotMarkerUntilMs_) {
        QPointF screen;
        if (projectWorldToWidget(pivotMarkerWorld_, screen)) {
            painter.setRenderHint(QPainter::Antialiasing, true);
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(255, 210, 70));
            // Neat 6×6 yellow orbit-pivot dot (no crosshair).
            painter.drawEllipse(QRectF(screen.x() - 3.0, screen.y() - 3.0, 6.0, 6.0));
        }
    } else {
        showPivotMarker_ = false;
    }

    drawGizmo(painter);

    const bool showBar = hasBackendHud_ || !statusText_.isEmpty() || !statusTextRight_.isEmpty() ||
                         focusPickActive_;
    if (showBar) {
        QFont font = painter.font();
        font.setPointSizeF(8.5);
        painter.setFont(font);
        const QFontMetrics fm(font);
        const QRect textRect(target.left(), target.bottom() - 22, target.width(), 20);
        painter.fillRect(textRect, QColor(0, 0, 0, 130));

        const int margin = 8;
        int rightLimit = textRect.right() - margin;
        if (hasBackendHud_) {
            const QString sep = QStringLiteral("  ·  ");
            const int activeW = fm.horizontalAdvance(backendActive_);
            const int sepW = optixSupportText_.isEmpty() ? 0 : fm.horizontalAdvance(sep);
            const int supportW =
                optixSupportText_.isEmpty() ? 0 : fm.horizontalAdvance(optixSupportText_);
            const int hudW = activeW + sepW + supportW;
            const int hudX = std::max(textRect.left() + margin, rightLimit - hudW);
            painter.setPen(QColor(235, 237, 240));
            painter.drawText(QRect(hudX, textRect.top(), activeW, textRect.height()),
                             Qt::AlignVCenter | Qt::AlignLeft, backendActive_);
            if (!optixSupportText_.isEmpty()) {
                painter.setPen(QColor(160, 164, 170));
                painter.drawText(QRect(hudX + activeW, textRect.top(), sepW, textRect.height()),
                                 Qt::AlignVCenter | Qt::AlignLeft, sep);
                painter.setPen(optixSupportColor_);
                painter.drawText(QRect(hudX + activeW + sepW, textRect.top(), supportW, textRect.height()),
                                 Qt::AlignVCenter | Qt::AlignLeft, optixSupportText_);
            }
            rightLimit = hudX - margin;
        }

        QString leftText = statusText_;
        if (leftText.isEmpty() && focusPickActive_) {
            leftText = QStringLiteral("Focus Pick — click geometry to set DOF focus distance");
        }
        if (!leftText.isEmpty()) {
            const int leftWidth = std::max(0, rightLimit - (textRect.left() + margin));
            const QRect leftRect(textRect.left() + margin, textRect.top(), leftWidth, textRect.height());
            painter.setPen(focusPickActive_ && statusText_.isEmpty() ? QColor(255, 210, 70)
                                                                     : QColor(235, 237, 240));
            painter.drawText(leftRect, Qt::AlignVCenter | Qt::AlignLeft,
                             fm.elidedText(leftText, Qt::ElideRight, leftWidth));
        }
        if (!statusTextRight_.isEmpty()) {
            const int diceLeft = textRect.left() + textRect.width() / 3;
            const int diceWidth = std::max(0, rightLimit - diceLeft);
            if (diceWidth > 24) {
                painter.setPen(QColor(235, 237, 240));
                painter.drawText(QRect(diceLeft, textRect.top(), diceWidth, textRect.height()),
                                 Qt::AlignVCenter | Qt::AlignLeft,
                                 fm.elidedText(statusTextRight_, Qt::ElideRight, diceWidth));
            }
        }
    }
}

void RenderView::mousePressEvent(QMouseEvent* event) {
    if (!navigationEnabled_) {
        QWidget::mousePressEvent(event);
        return;
    }

    // Keep viewport shortcuts (F / H) working after chrome / dock focus changes.
    setFocus(Qt::MouseFocusReason);

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

    if (event->button() == Qt::LeftButton && !alt) {
        // Focus Pick (camera DOF): click geometry → focus plane depth in camera space.
        if (focusPickActive_) {
            Vec3 hit;
            if (pickUnderMouse(event->pos(), hit)) {
                // Optical-axis depth (metres), matching generateCameraRay's focus plane at z=-focusDistance.
                const Mat4 worldToCam = inverse(camera_.toMatrix());
                const Vec3 hitCam = transformPoint(worldToCam, hit);
                const float distance = std::max(1e-4f, -hitCam.z);
                emit focusDistancePicked(distance);
                setFocusPickActive(false);
                event->accept();
                return;
            }
            event->accept();
            return;
        }

        if (hasTransformTarget()) {
            if (beginGizmoDrag(event->pos())) {
                event->accept();
                return;
            }
        }
        // Object selection only in Select tool — TRS must not steal clicks for pick.
        if (transformTool_ == TransformTool::Select) {
            QString sourceNode;
            if (pickObjectUnderMouse(event->pos(), sourceNode)) {
                emit objectSelected(sourceNode);
            } else {
                emit objectSelected(QString());
            }
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
        if (focusPickActive_) {
            setCursor(Qt::CrossCursor);
            QWidget::mouseMoveEvent(event);
            return;
        }
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
            // Invert vertical orbit: Qt Y grows down, so negate pitch.
            camera_.orbitAround(tumbleCenter_, -float(delta.x()) * 0.22f * precision,
                                -float(delta.y()) * 0.22f * precision);
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
        const bool wasNav = mode_ == 1 || mode_ == 2 || mode_ == 3;
        mode_ = 0;
        releaseMouse();
        unsetCursor();
        if (wasNav) emit cameraNavEnded();
        emit cameraMoved();
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

void RenderView::frameBounds(const Bounds3& bounds) {
    Bounds3 b = bounds;
    if (!b.valid()) {
        b.extend(Vec3(-0.5f));
        b.extend(Vec3(0.5f));
    }
    const Vec3 center = b.center();
    const float radius = std::max(0.05f, b.radius());

    // Match the viewport film back (50mm / 36mm) used for picking / projection.
    constexpr float kFocalLength = 50.0f;
    constexpr float kSensorWidth = 36.0f;
    const QRect img = imageRect();
    const float aspect = img.height() > 0 ? float(img.width()) / float(img.height()) : (16.0f / 9.0f);
    const float sensorHeight = kSensorWidth / std::max(0.01f, aspect);
    const float fovX = 2.0f * std::atan(0.5f * kSensorWidth / kFocalLength);
    const float fovY = 2.0f * std::atan(0.5f * sensorHeight / kFocalLength);
    const float fov = std::max(0.05f, std::min(fovX, fovY));
    const float distance = radius / std::sin(fov * 0.5f) * 1.15f;

    camera_.pivot = center;
    camera_.distance = std::max(0.05f, distance);
    // Keep current orbit angles — only reframe distance/pivot like Houdini F.
    emit cameraMoved();
    update();
}

void RenderView::frameSelection() {
    Bounds3 bounds;
    if (selectionBoundsCallback_ && selectionBoundsCallback_(bounds) && bounds.valid()) {
        frameBounds(bounds);
        return;
    }
    frameAll();
}

void RenderView::frameAll() {
    Bounds3 bounds;
    if (sceneBoundsCallback_ && sceneBoundsCallback_(bounds) && bounds.valid()) {
        frameBounds(bounds);
        return;
    }
    Bounds3 fallback;
    fallback.extend(Vec3(-1.0f));
    fallback.extend(Vec3(1.0f));
    frameBounds(fallback);
}

void RenderView::keyPressEvent(QKeyEvent* event) {
    switch (event->key()) {
        case Qt::Key_F:
            frameSelection();
            event->accept();
            return;
        case Qt::Key_H:
        case Qt::Key_Home:
            frameAll();
            event->accept();
            return;
        case Qt::Key_Q:
            setTransformTool(TransformTool::Select);
            event->accept();
            return;
        case Qt::Key_T:
            setTransformTool(TransformTool::Translate);
            event->accept();
            return;
        case Qt::Key_R:
            setTransformTool(TransformTool::Rotate);
            event->accept();
            return;
        case Qt::Key_S:
            setTransformTool(TransformTool::Scale);
            event->accept();
            return;
        default:
            break;
    }
    QWidget::keyPressEvent(event);
}

}  // namespace sol
