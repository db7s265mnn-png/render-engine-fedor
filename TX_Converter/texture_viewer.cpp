#include "texture_viewer.h"

#include <QAbstractSpinBox>
#include <QButtonGroup>
#include <QColor>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QIcon>
#include <QImageReader>
#include <QIntValidator>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QPixmap>
#include <QPointer>
#include <QPushButton>
#include <QResizeEvent>
#include <QSignalBlocker>
#include <QSlider>
#include <QThreadPool>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <string>

#include "core/expr_eval.h"
#include "core/image.h"
#include "core/math.h"
#include "io/image_io.h"
#include "io/ocio_util.h"
#include "io/tx_convert.h"
#include "scene/types.h"
#include "ui/timeline_bar.h"

namespace sol {
namespace {

constexpr int kMaxPreviewEdge = 2048;
constexpr int kFrameBoxWidth = 44;
constexpr int kFrameBoxHeight = 16;
constexpr float kGradePivot = 0.18f;

bool isFloatPreviewPath(const QString& path) {
    const QString ext = QFileInfo(path).suffix().toLower();
    return ext == QLatin1String("exr") || ext == QLatin1String("hdr") ||
           ext == QLatin1String("rgbe") || ext == QLatin1String("pic") ||
           ext == QLatin1String("tx") || ext == QLatin1String("tif") ||
           ext == QLatin1String("tiff");
}

bool prefersAcesCgWorkingSpace(const QString& path) {
    return QFileInfo(path).suffix().toLower() == QLatin1String("tx");
}

QString formatBytes(qint64 bytes) {
    if (bytes < 1024) return QStringLiteral("%1 B").arg(bytes);
    if (bytes < 1024 * 1024) return QStringLiteral("%1 KB").arg(bytes / 1024.0, 0, 'f', 1);
    return QStringLiteral("%1 MB").arg(bytes / (1024.0 * 1024.0), 0, 'f', 1);
}

QString channelModeLabel(ViewerChannelMode mode) {
    switch (mode) {
        case ViewerChannelMode::R: return QStringLiteral("R");
        case ViewerChannelMode::G: return QStringLiteral("G");
        case ViewerChannelMode::B: return QStringLiteral("B");
        case ViewerChannelMode::A: return QStringLiteral("A");
        case ViewerChannelMode::RGBA:
        default: return QStringLiteral("RGBA");
    }
}

bool gradeEquals(const ViewerGrade& a, const ViewerGrade& b) {
    return a.brightness == b.brightness && a.contrast == b.contrast && a.gamma == b.gamma;
}

// Houdini/mplay-style grade: exposure (stops) -> contrast about the 0.18 pivot -> gamma.
// All done in scene-linear before any display / view transform, so HDR values above
// 1.0 are boosted rather than clamped-then-brightened (which would just wash to grey).
Vec3 applyGrade(Vec3 c, const ViewerGrade& grade) {
    c = c * std::exp2(grade.brightness);
    const float pivot = kGradePivot;
    c = (c - Vec3(pivot)) * grade.contrast + Vec3(pivot);
    const float g = std::max(0.01f, grade.gamma);
    const float invG = 1.0f / g;
    c.x = std::pow(std::max(0.0f, c.x), invG);
    c.y = std::pow(std::max(0.0f, c.y), invG);
    c.z = std::pow(std::max(0.0f, c.z), invG);
    return c;
}

// src/out are tightly packed RGBA float (4 components per pixel); alpha survives the resize.
void downscaleLinearRgba(const float* src, int srcW, int srcH, std::vector<float>& outRgba, int& outW,
                         int& outH) {
    int stepX = 1;
    int stepY = 1;
    outW = srcW;
    outH = srcH;
    const int edge = std::max(srcW, srcH);
    if (edge > kMaxPreviewEdge) {
        stepX = std::max(1, (srcW + kMaxPreviewEdge - 1) / kMaxPreviewEdge);
        stepY = std::max(1, (srcH + kMaxPreviewEdge - 1) / kMaxPreviewEdge);
        outW = std::max(1, srcW / stepX);
        outH = std::max(1, srcH / stepY);
    }
    outRgba.resize(size_t(outW) * size_t(outH) * 4);
    for (int y = 0; y < outH; ++y) {
        const int sy = std::min(srcH - 1, y * stepY);
        const float* row = src + size_t(sy) * size_t(srcW) * 4;
        float* dst = outRgba.data() + size_t(y) * size_t(outW) * 4;
        for (int x = 0; x < outW; ++x) {
            const int sx = std::min(srcW - 1, x * stepX);
            const float* px = row + size_t(sx) * 4;
            dst[x * 4 + 0] = px[0];
            dst[x * 4 + 1] = px[1];
            dst[x * 4 + 2] = px[2];
            dst[x * 4 + 3] = px[3];
        }
    }
}

void extractLinearFromFloatImage(const Image& image, std::vector<float>& outRgba, int& outW, int& outH) {
    int srcW = image.width();
    int srcH = image.height();
    const float* src = image.data();
    if (image.mipCount() > 1) {
        for (int l = 0; l < image.mipCount(); ++l) {
            const int mw = image.mipWidth(l);
            const int mh = image.mipHeight(l);
            if (std::max(mw, mh) <= kMaxPreviewEdge || l + 1 == image.mipCount()) {
                srcW = mw;
                srcH = mh;
                src = image.mipData(l);
                break;
            }
        }
    }
    downscaleLinearRgba(src, srcW, srcH, outRgba, outW, outH);
}

QImage bakeDisplayImage(const float* linearRgba, int w, int h, ViewerChannelMode channelMode,
                        int colorManagement, int viewTransform, bool ocioUseEnv,
                        const QString& ocioConfigPath, int workingSpace, const ViewerGrade& grade) {
    QImage out(w, h, QImage::Format_RGB888);
    if (!linearRgba || w <= 0 || h <= 0) return out;

    if (colorManagement == kColorOcio) {
        ocioEnsureConfig(ocioUseEnv, ocioConfigPath.toStdString());
    }
    displayPrepareView(workingSpace, colorManagement, viewTransform);

    for (int y = 0; y < h; ++y) {
        uchar* line = out.scanLine(y);
        const float* row = linearRgba + size_t(y) * size_t(w) * 4;
        for (int x = 0; x < w; ++x) {
            const float* px = row + size_t(x) * 4;

            Vec3 linear;
            if (channelMode == ViewerChannelMode::RGBA) {
                linear = Vec3(px[0], px[1], px[2]);
            } else {
                float v = px[0];
                switch (channelMode) {
                    case ViewerChannelMode::R: v = px[0]; break;
                    case ViewerChannelMode::G: v = px[1]; break;
                    case ViewerChannelMode::B: v = px[2]; break;
                    case ViewerChannelMode::A: v = px[3]; break;
                    default: break;
                }
                linear = Vec3(v, v, v);
            }

            // Grade in linear (scene-referred), then hand off to the OCIO/Classic view transform.
            linear = applyGrade(linear, grade);
            Vec3 display = ocioApplyViewPrepared(linear);

            uchar* dst = line + size_t(x) * 3;
            dst[0] = static_cast<uchar>(clampf(display.x, 0.0f, 1.0f) * 255.0f + 0.5f);
            dst[1] = static_cast<uchar>(clampf(display.y, 0.0f, 1.0f) * 255.0f + 0.5f);
            dst[2] = static_cast<uchar>(clampf(display.z, 0.0f, 1.0f) * 255.0f + 0.5f);
        }
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// FloatPreviewCanvas
// ---------------------------------------------------------------------------

FloatPreviewCanvas::FloatPreviewCanvas(QWidget* parent) : QWidget(parent) {
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
    setMinimumSize(160, 160);
    setCursor(Qt::OpenHandCursor);
    setStyleSheet(QStringLiteral("background: #121416;"));
}

void FloatPreviewCanvas::clear() {
    linearRgba_ = nullptr;
    width_ = height_ = 0;
    contentId_ = 0;
    invalidateDisplayCache();
    placeholder_ = QStringLiteral("No texture");
    update();
}

void FloatPreviewCanvas::setPlaceholder(const QString& text) {
    linearRgba_ = nullptr;
    width_ = height_ = 0;
    contentId_ = 0;
    invalidateDisplayCache();
    placeholder_ = text;
    update();
}

void FloatPreviewCanvas::setLinearImage(const float* rgba, int width, int height, quint64 contentId) {
    linearRgba_ = rgba;
    width_ = width;
    height_ = height;
    contentId_ = contentId;
    invalidateDisplayCache();
    if (fitted_) fitToView();
    else {
        clampPan();
        update();
    }
}

void FloatPreviewCanvas::setDisplayParams(int colorManagement, int viewTransform, bool ocioUseEnv,
                                          const QString& ocioConfigPath, int workingSpace) {
    colorManagement_ = colorManagement;
    viewTransform_ = viewTransform;
    ocioUseEnv_ = ocioUseEnv;
    ocioConfigPath_ = ocioConfigPath;
    workingSpace_ = workingSpace;
    invalidateDisplayCache();
    update();
}

void FloatPreviewCanvas::setChannelMode(ViewerChannelMode mode) {
    if (channelMode_ == mode) return;
    channelMode_ = mode;
    invalidateDisplayCache();
    update();
}

void FloatPreviewCanvas::setGrade(const ViewerGrade& grade) {
    grade_ = grade;
    invalidateDisplayCache();
    update();
}

void FloatPreviewCanvas::invalidateDisplayCache() {
    displayCache_ = {};
    displayCacheId_ = 0;
}

void FloatPreviewCanvas::ensureDisplayCache() {
    if (!linearRgba_ || width_ <= 0 || height_ <= 0) return;
    if (!displayCache_.isNull() && displayCacheId_ == contentId_ &&
        displayCacheColorMgmt_ == colorManagement_ && displayCacheView_ == viewTransform_ &&
        displayCacheChannel_ == int(channelMode_) && displayCacheWorking_ == workingSpace_ &&
        displayCacheOcioEnv_ == ocioUseEnv_ && displayCacheOcioPath_ == ocioConfigPath_ &&
        gradeEquals(displayCacheGrade_, grade_)) {
        return;
    }
    displayCache_ = bakeDisplayImage(linearRgba_, width_, height_, channelMode_, colorManagement_,
                                     viewTransform_, ocioUseEnv_, ocioConfigPath_, workingSpace_, grade_);
    displayCacheId_ = contentId_;
    displayCacheColorMgmt_ = colorManagement_;
    displayCacheView_ = viewTransform_;
    displayCacheChannel_ = int(channelMode_);
    displayCacheWorking_ = workingSpace_;
    displayCacheOcioEnv_ = ocioUseEnv_;
    displayCacheOcioPath_ = ocioConfigPath_;
    displayCacheGrade_ = grade_;
}

void FloatPreviewCanvas::fitToView() {
    fitted_ = true;
    if (width_ <= 0 || height_ <= 0 || this->width() < 2 || this->height() < 2) {
        zoom_ = 1.0;
        pan_ = QPointF(0, 0);
        emit zoomChanged(zoom_);
        update();
        return;
    }
    const double sx = double(this->width()) / double(width_);
    const double sy = double(this->height()) / double(height_);
    zoom_ = std::min(sx, sy);
    pan_ = QPointF(0, 0);
    emit zoomChanged(zoom_);
    update();
}

void FloatPreviewCanvas::resetView() { fitToView(); }

QRectF FloatPreviewCanvas::imageRect() const {
    if (width_ <= 0 || height_ <= 0) return {};
    const double w = double(width_) * zoom_;
    const double h = double(height_) * zoom_;
    const double x = (double(this->width()) - w) * 0.5 + pan_.x();
    const double y = (double(this->height()) - h) * 0.5 + pan_.y();
    return QRectF(x, y, w, h);
}

void FloatPreviewCanvas::clampPan() {
    if (width_ <= 0 || height_ <= 0) {
        pan_ = QPointF(0, 0);
        return;
    }
    const double w = double(width_) * zoom_;
    const double h = double(height_) * zoom_;
    // Allow panning even when the image fits (no zoom) — keep ~½ viewport of slack.
    const double slackX = double(this->width()) * 0.5;
    const double slackY = double(this->height()) * 0.5;
    const double maxX = std::max(slackX, (w - double(this->width())) * 0.5 + 32.0);
    const double maxY = std::max(slackY, (h - double(this->height())) * 0.5 + 32.0);
    pan_.setX(std::clamp(pan_.x(), -maxX, maxX));
    pan_.setY(std::clamp(pan_.y(), -maxY, maxY));
}

void FloatPreviewCanvas::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), QColor(18, 20, 22));

    if (!linearRgba_ || width_ <= 0 || height_ <= 0) {
        p.setPen(QColor(120, 126, 134));
        p.drawText(rect(), Qt::AlignCenter, placeholder_);
        return;
    }

    ensureDisplayCache();
    if (displayCache_.isNull()) {
        p.setPen(QColor(120, 126, 134));
        p.drawText(rect(), Qt::AlignCenter, QStringLiteral("No image"));
        return;
    }

    const QRectF dest = imageRect();
    p.setRenderHint(QPainter::SmoothPixmapTransform, zoom_ < 4.0);
    p.drawImage(dest, displayCache_);
}

void FloatPreviewCanvas::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    if (fitted_) fitToView();
    else {
        clampPan();
        update();
    }
}

void FloatPreviewCanvas::wheelEvent(QWheelEvent* event) {
    if (!linearRgba_) {
        event->ignore();
        return;
    }
    fitted_ = false;
    const QPointF mouse = event->position();
    const QRectF before = imageRect();
    const double factor = std::pow(1.0015, event->angleDelta().y());
    zoom_ = std::clamp(zoom_ * factor, 0.05, 64.0);
    if (before.width() > 1.0 && before.height() > 1.0) {
        const double u = (mouse.x() - before.x()) / before.width();
        const double v = (mouse.y() - before.y()) / before.height();
        const QRectF after = imageRect();
        const QPointF afterPt(after.x() + u * after.width(), after.y() + v * after.height());
        pan_ += mouse - afterPt;
    }
    clampPan();
    emit zoomChanged(zoom_);
    update();
    event->accept();
}

void FloatPreviewCanvas::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton) {
        dragging_ = true;
        lastMouse_ = event->pos();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void FloatPreviewCanvas::mouseMoveEvent(QMouseEvent* event) {
    if (dragging_) {
        fitted_ = false;
        const QPoint delta = event->pos() - lastMouse_;
        lastMouse_ = event->pos();
        pan_ += QPointF(delta);
        clampPan();
        update();
        event->accept();
        return;
    }
    QWidget::mouseMoveEvent(event);
}

void FloatPreviewCanvas::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton) {
        dragging_ = false;
        setCursor(Qt::OpenHandCursor);
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

void FloatPreviewCanvas::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        fitToView();
        event->accept();
        return;
    }
    QWidget::mouseDoubleClickEvent(event);
}

void FloatPreviewCanvas::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_F && !event->modifiers()) {
        fitToView();
        event->accept();
        return;
    }
    QWidget::keyPressEvent(event);
}

// ---------------------------------------------------------------------------
// TextureViewerWidget
// ---------------------------------------------------------------------------

TextureViewerWidget::LoadPayload TextureViewerWidget::decodeFrame(const QString& path, int index) {
    LoadPayload payload;
    payload.index = index;
    payload.path = path;
    const QFileInfo info(path);
    payload.fileBytes = info.size();

    if (!info.exists()) {
        payload.error = QStringLiteral("file not found: %1").arg(path);
        return payload;
    }

    if (!isFloatPreviewPath(path)) {
        QImageReader reader(path);
        reader.setAllocationLimit(0);
        reader.setAutoTransform(true);
        QImage qimage = reader.read();
        if (qimage.isNull()) qimage.load(path);
        if (qimage.isNull()) {
            payload.error = QStringLiteral("unsupported or unreadable image: %1 (%2)")
                                .arg(path, reader.errorString());
            return payload;
        }
        qimage = qimage.convertToFormat(QImage::Format_RGBA8888);
        payload.sourceWidth = qimage.width();
        payload.sourceHeight = qimage.height();

        if (std::max(qimage.width(), qimage.height()) > kMaxPreviewEdge) {
            qimage = qimage
                         .scaled(kMaxPreviewEdge, kMaxPreviewEdge, Qt::KeepAspectRatio,
                                 Qt::FastTransformation)
                         .convertToFormat(QImage::Format_RGBA8888);
        }

        std::vector<float> rgba(size_t(qimage.width()) * size_t(qimage.height()) * 4);
        for (int y = 0; y < qimage.height(); ++y) {
            const uchar* line = qimage.constScanLine(y);
            float* row = rgba.data() + size_t(y) * size_t(qimage.width()) * 4;
            for (int x = 0; x < qimage.width(); ++x) {
                const uchar* px = line + size_t(x) * 4;
                row[x * 4 + 0] = srgbToLinear(px[0] / 255.0f);
                row[x * 4 + 1] = srgbToLinear(px[1] / 255.0f);
                row[x * 4 + 2] = srgbToLinear(px[2] / 255.0f);
                row[x * 4 + 3] = px[3] / 255.0f;
            }
        }
        downscaleLinearRgba(rgba.data(), qimage.width(), qimage.height(), payload.linearRgba,
                            payload.previewW, payload.previewH);
        return payload;
    }

    Image image;
    std::string err;
    if (!loadImage(path.toStdString(), image, err, /*srgbColor=*/false)) {
        if (!loadImage(path.toStdString(), image, err, true)) {
            payload.error = QString::fromStdString(err.empty() ? "load failed" : err);
            return payload;
        }
    }
    if (image.empty()) {
        payload.error = QStringLiteral("empty image");
        return payload;
    }
    payload.sourceWidth = image.width();
    payload.sourceHeight = image.height();
    extractLinearFromFloatImage(image, payload.linearRgba, payload.previewW, payload.previewH);
    return payload;
}

QLineEdit* TextureViewerWidget::makeRangeEdit(const QString& tip) {
    auto* edit = new QLineEdit(this);
    edit->setAlignment(Qt::AlignCenter);
    edit->setFixedSize(kFrameBoxWidth, kFrameBoxHeight);
    edit->setMaxLength(7);
    edit->setValidator(new QIntValidator(-999999, 999999, edit));
    edit->setToolTip(tip);
    edit->setStyleSheet(QStringLiteral(
        "QLineEdit {"
        "  background: #1a1c20;"
        "  color: #d8dae0;"
        "  border: 1px solid #7a7e86;"
        "  border-radius: 2px;"
        "  padding: 0 2px;"
        "  font-size: 11px;"
        "  font-weight: 600;"
        "}"
        "QLineEdit:focus { border: 1px solid #50aaff; }"));
    return edit;
}

QPushButton* TextureViewerWidget::makeGradeLabel(const QString& text, const QString& tip) {
    auto* btn = new QPushButton(text, this);
    btn->setFlat(true);
    btn->setCursor(Qt::PointingHandCursor);
    btn->setToolTip(tip);
    btn->setStyleSheet(QStringLiteral(
        "QPushButton {"
        "  color: #9aa0a6;"
        "  background: transparent;"
        "  border: none;"
        "  padding: 0 2px;"
        "  text-align: left;"
        "}"
        "QPushButton:hover { color: #d8dae0; text-decoration: underline; }"));
    return btn;
}

QDoubleSpinBox* TextureViewerWidget::makeGradeSpin(double minV, double maxV, double step, double value,
                                                   const QString& tip) {
    auto* spin = new QDoubleSpinBox(this);
    spin->setRange(minV, maxV);
    spin->setSingleStep(step);
    spin->setDecimals(2);
    spin->setValue(value);
    spin->setFixedWidth(64);
    spin->setToolTip(tip);
    spin->setButtonSymbols(QAbstractSpinBox::NoButtons);
    spin->setStyleSheet(QStringLiteral(
        "QDoubleSpinBox {"
        "  background: #1a1c20;"
        "  color: #d8dae0;"
        "  border: 1px solid #7a7e86;"
        "  border-radius: 2px;"
        "  padding: 1px 4px;"
        "}"
        "QDoubleSpinBox:focus { border: 1px solid #50aaff; }"));
    return spin;
}

QToolButton* TextureViewerWidget::makeChannelButton(const QString& tip, const QColor& fill, bool checker) {
    auto* btn = new QToolButton(this);
    btn->setCheckable(true);
    btn->setAutoRaise(true);
    btn->setToolTip(tip);
    btn->setFixedSize(28, 20);
    btn->setCursor(Qt::PointingHandCursor);

    QString background = fill.name();
    if (checker) {
        QPixmap pix(12, 12);
        QPainter pp(&pix);
        pp.fillRect(0, 0, 6, 6, QColor(150, 150, 150));
        pp.fillRect(6, 6, 6, 6, QColor(150, 150, 150));
        pp.fillRect(6, 0, 6, 6, QColor(210, 210, 210));
        pp.fillRect(0, 6, 6, 6, QColor(210, 210, 210));
        pp.end();
        btn->setIcon(QIcon(pix));
        btn->setIconSize(QSize(12, 12));
        background = QStringLiteral("#3a3d42");
    }

    btn->setStyleSheet(QStringLiteral(
        "QToolButton {"
        "  background: %1;"
        "  color: #14161a;"
        "  border: 1px solid #22242a;"
        "  border-radius: 2px;"
        "  font-size: 10px;"
        "  font-weight: 700;"
        "}"
        "QToolButton:hover { border: 1px solid #7a7e86; }"
        "QToolButton:checked { border: 2px solid #50aaff; }")
        .arg(background));
    return btn;
}

QSlider* makeGradeSlider(QWidget* parent, int steps = 1000) {
    auto* slider = new QSlider(Qt::Horizontal, parent);
    slider->setRange(0, steps);
    slider->setFixedWidth(110);
    return slider;
}

TextureViewerWidget::TextureViewerWidget(QWidget* parent) : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(4);

    // Toolbar order: content (View) combo, channel buttons, Display (Classic/OCIO + view
    // transform), then Fit + zoom%.
    auto* modeRow = new QHBoxLayout();
    modeRow->setContentsMargins(0, 0, 0, 0);
    modeRow->setSpacing(6);

    modeRow->addWidget(new QLabel(QStringLiteral("View")));
    contentCombo_ = new QComboBox();
    contentCombo_->addItem(QStringLiteral("Source Images"), int(ViewerContentKind::SourceImages));
    contentCombo_->addItem(QStringLiteral("Converted"), int(ViewerContentKind::ConvertedTx));
    contentCombo_->setMinimumWidth(120);
    contentCombo_->setToolTip(QStringLiteral("Preview the source images or the converted output."));
    modeRow->addWidget(contentCombo_);

    channelGroup_ = new QButtonGroup(this);
    channelGroup_->setExclusive(true);
    auto addChannelButton = [&](ViewerChannelMode mode, const QString& tip, const QColor& fill,
                                bool checker = false, bool rgbaIcon = false) {
        QToolButton* btn = makeChannelButton(tip, fill, checker);
        if (rgbaIcon) {
            QPixmap pix(14, 14);
            pix.fill(Qt::transparent);
            QPainter pp(&pix);
            pp.fillRect(0, 0, 7, 7, QColor(220, 70, 70));
            pp.fillRect(7, 0, 7, 7, QColor(70, 120, 220));
            pp.fillRect(0, 7, 7, 7, QColor(70, 190, 90));
            pp.fillRect(7, 7, 7, 7, QColor(180, 180, 180));
            pp.end();
            btn->setIcon(QIcon(pix));
            btn->setIconSize(QSize(14, 14));
        }
        channelGroup_->addButton(btn, int(mode));
        modeRow->addWidget(btn);
        return btn;
    };
    QToolButton* rgbaBtn =
        addChannelButton(ViewerChannelMode::RGBA, QStringLiteral("Show all channels in colour"),
                         QColor(58, 61, 66), false, true);
    addChannelButton(ViewerChannelMode::R, QStringLiteral("Isolate red channel (grey)"),
                     QColor(224, 96, 96));
    addChannelButton(ViewerChannelMode::G, QStringLiteral("Isolate green channel (grey)"),
                     QColor(104, 200, 110));
    addChannelButton(ViewerChannelMode::B, QStringLiteral("Isolate blue channel (grey)"),
                     QColor(102, 146, 232));
    addChannelButton(ViewerChannelMode::A, QStringLiteral("Isolate alpha channel (grey)"),
                     QColor(160, 160, 160), true);
    rgbaBtn->setChecked(true);

    modeRow->addWidget(new QLabel(QStringLiteral("Display")));
    colorMgmtCombo_ = new QComboBox();
    colorMgmtCombo_->addItem(QStringLiteral("Classic"), kColorClassic);
    colorMgmtCombo_->addItem(QStringLiteral("OCIO"), kColorOcio);
    colorMgmtCombo_->setCurrentIndex(1);
    colorMgmtCombo_->setMinimumWidth(90);
    colorMgmtCombo_->setToolTip(
        QStringLiteral("Classic: gamma / linear (no OCIO).\nOCIO: OpenColorIO Display/View."));
    modeRow->addWidget(colorMgmtCombo_);
    viewCombo_ = new QComboBox();
    viewCombo_->addItem(QStringLiteral("sRGB"), kViewSrgbAces);
    viewCombo_->addItem(QStringLiteral("Rec.709"), kViewRec709Aces);
    viewCombo_->addItem(QStringLiteral("Rec.2020"), kViewRec2020);
    viewCombo_->addItem(QStringLiteral("Raw"), kViewRaw);
    viewCombo_->setCurrentIndex(0);
    viewCombo_->setMinimumWidth(100);
    viewCombo_->setToolTip(QStringLiteral("Monitor view transform."));
    modeRow->addWidget(viewCombo_);

    fitBtn_ = new QPushButton(QStringLiteral("Fit"));
    fitBtn_->setFixedWidth(44);
    fitBtn_->setToolTip(QStringLiteral("Fit image to view (double-click canvas)"));
    modeRow->addWidget(fitBtn_);
    zoomLabel_ = new QLabel(QStringLiteral("100%"));
    zoomLabel_->setMinimumWidth(44);
    zoomLabel_->setStyleSheet(QStringLiteral("color: #9aa0a6;"));
    modeRow->addWidget(zoomLabel_);
    modeRow->addStretch(1);
    root->addLayout(modeRow);

    // Grade row: click the label ("Brightness" / "Contrast" / "Gamma") to reset just that param.
    auto* gradeRow = new QHBoxLayout();
    gradeRow->setContentsMargins(0, 0, 0, 0);
    gradeRow->setSpacing(4);
    auto addGrade = [&](QPushButton** labelBtn, const QString& text, QDoubleSpinBox** spin,
                        QSlider** slider, double minV, double maxV, double step, double value,
                        const QString& tip) {
        *labelBtn = makeGradeLabel(text, tip);
        gradeRow->addWidget(*labelBtn);
        *spin = makeGradeSpin(minV, maxV, step, value, tip);
        *slider = makeGradeSlider(this);
        const double t = (value - minV) / std::max(1e-9, maxV - minV);
        (*slider)->setValue(int(std::lround(std::clamp(t, 0.0, 1.0) * 1000.0)));
        gradeRow->addWidget(*slider, 1);
        gradeRow->addWidget(*spin);
    };
    addGrade(&brightnessLabelBtn_, QStringLiteral("Brightness"), &brightnessSpin_, &brightnessSlider_,
             -5.0, 5.0, 0.05, 0.0,
             QStringLiteral("Brightness in stops, applied in linear (click \"Brightness\" to reset to 0)"));
    addGrade(&contrastLabelBtn_, QStringLiteral("Contrast"), &contrastSpin_, &contrastSlider_, 0.0, 3.0,
             0.05, 1.0, QStringLiteral("Contrast about the 0.18 linear pivot (click \"Contrast\" to reset to 1)"));
    addGrade(&gammaLabelBtn_, QStringLiteral("Gamma"), &gammaSpin_, &gammaSlider_, 0.20, 3.0, 0.05, 1.0,
             QStringLiteral("Linear gamma before the view transform (click \"Gamma\" to reset to 1)"));
    root->addLayout(gradeRow);

    canvas_ = new FloatPreviewCanvas(this);
    setFocusProxy(canvas_);
    root->addWidget(canvas_, 1);

    // Solstice-style scrub row: [start] [scrubber] [end] — no transport/play buttons.
    // Numbers shown here are absolute UDIM / $F frame numbers, not tile indices.
    auto* scrubHost = new QWidget(this);
    scrubHost->setObjectName(QStringLiteral("viewerTimeline"));
    scrubHost->setFixedHeight(28);
    scrubHost->setStyleSheet(QStringLiteral(
        "QWidget#viewerTimeline {"
        "  background: #2e3136;"
        "  border-top: 1px solid #22242a;"
        "  border-bottom: 1px solid #22242a;"
        "}"));
    auto* scrubRow = new QHBoxLayout(scrubHost);
    scrubRow->setContentsMargins(4, 4, 4, 4);
    scrubRow->setSpacing(0);
    startEdit_ = makeRangeEdit(QStringLiteral("Start frame"));
    endEdit_ = makeRangeEdit(QStringLiteral("End frame"));
    scrubber_ = new TimelineScrubber(scrubHost);
    scrubRow->addWidget(startEdit_, 0);
    scrubRow->addWidget(scrubber_, 1);
    scrubRow->addWidget(endEdit_, 0);
    root->addWidget(scrubHost);

    infoLabel_ = new QLabel(QStringLiteral("No texture"));
    infoLabel_->setWordWrap(true);
    infoLabel_->setStyleSheet(QStringLiteral("color: #b0b4ba;"));
    root->addWidget(infoLabel_);

    connect(scrubber_, &TimelineScrubber::frameChanged, this, &TextureViewerWidget::setTimelineFrame);
    connect(startEdit_, &QLineEdit::editingFinished, this, &TextureViewerWidget::onStartEdited);
    connect(endEdit_, &QLineEdit::editingFinished, this, &TextureViewerWidget::onEndEdited);
    connect(fitBtn_, &QPushButton::clicked, this, &TextureViewerWidget::fitView);
    connect(canvas_, &FloatPreviewCanvas::zoomChanged, this, [this](double z) {
        zoomLabel_->setText(QStringLiteral("%1%").arg(int(std::lround(z * 100.0))));
    });
    connect(colorMgmtCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        setColorManagement(colorMgmtCombo_->currentData().toInt());
    });
    connect(viewCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        setViewTransform(viewCombo_->currentData().toInt());
    });
    connect(contentCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        contentKind_ = ViewerContentKind(contentCombo_->currentData().toInt());
        refreshFromPipeline();
    });
    connect(channelGroup_, &QButtonGroup::idClicked, this,
            [this](int id) { setChannelMode(ViewerChannelMode(id)); });

    auto wireGrade = [this](QDoubleSpinBox* spin, QSlider* slider, double minV, double maxV) {
        const double span = std::max(1e-9, maxV - minV);
        connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [this, slider, minV, span](double v) {
                    QSignalBlocker b(slider);
                    slider->setValue(int(std::lround(std::clamp((v - minV) / span, 0.0, 1.0) * 1000.0)));
                    applyGradeFromUi();
                });
        connect(slider, &QSlider::valueChanged, this, [this, spin, minV, span](int pos) {
            const double v = minV + span * (double(pos) / 1000.0);
            QSignalBlocker b(spin);
            spin->setValue(v);
            applyGradeFromUi();
        });
    };
    wireGrade(brightnessSpin_, brightnessSlider_, -5.0, 5.0);
    wireGrade(contrastSpin_, contrastSlider_, 0.0, 3.0);
    wireGrade(gammaSpin_, gammaSlider_, 0.20, 3.0);

    // Clicking a grade label resets only that parameter.
    connect(brightnessLabelBtn_, &QPushButton::clicked, this, [this] { brightnessSpin_->setValue(0.0); });
    connect(contrastLabelBtn_, &QPushButton::clicked, this, [this] { contrastSpin_->setValue(1.0); });
    connect(gammaLabelBtn_, &QPushButton::clicked, this, [this] { gammaSpin_->setValue(1.0); });

    rebuildTimeline();
}

void TextureViewerWidget::applyGradeFromUi() {
    grade_.brightness = float(brightnessSpin_->value());
    grade_.contrast = float(contrastSpin_->value());
    grade_.gamma = float(gammaSpin_->value());
    canvas_->setGrade(grade_);
}

void TextureViewerWidget::setChannelMode(ViewerChannelMode mode) {
    channelMode_ = mode;
    canvas_->setChannelMode(mode);
    updateInfoBar();
}

void TextureViewerWidget::setOcioConfig(bool useEnv, const QString& configPath) {
    ocioUseEnv_ = useEnv;
    ocioConfigPath_ = configPath.trimmed();
    canvas_->setDisplayParams(colorManagement_, viewTransform_, ocioUseEnv_, ocioConfigPath_,
                              ocioWorkingSpace_);
}

void TextureViewerWidget::setPipelinePaths(const QString& sourcePath, const QString& outputFolder) {
    pipelineSource_ = sourcePath.trimmed();
    pipelineOutputFolder_ = outputFolder.trimmed();
    refreshFromPipeline();
}

void TextureViewerWidget::setOutputExtension(const QString& ext) {
    const QString trimmed = ext.trimmed();
    const QString next = trimmed.isEmpty() ? QStringLiteral("tx") : trimmed;
    if (outputExt_ == next) return;
    outputExt_ = next;
    if (contentKind_ == ViewerContentKind::ConvertedTx) refreshFromPipeline();
}

void TextureViewerWidget::setContentKind(ViewerContentKind kind) {
    contentKind_ = kind;
    if (contentCombo_) {
        const QSignalBlocker block(contentCombo_);
        const int idx = contentCombo_->findData(int(kind));
        if (idx >= 0) contentCombo_->setCurrentIndex(idx);
    }
    refreshFromPipeline();
}

QString TextureViewerWidget::guessConvertedOutputPath(const QString& sourcePath,
                                                      const QString& outputFolder, const QString& ext) {
    const QString src = sourcePath.trimmed();
    const QString outDir = outputFolder.trimmed();
    const QString extension = ext.trimmed().isEmpty() ? QStringLiteral("tx") : ext.trimmed();
    if (src.isEmpty() || outDir.isEmpty()) return {};

    auto swapExt = [&](QString name) {
        const int dot = name.lastIndexOf(QLatin1Char('.'));
        if (dot > 0) return name.left(dot) + QLatin1Char('.') + extension;
        return name + QLatin1Char('.') + extension;
    };

    QFileInfo info(src);
    QString name = info.fileName();
    if (pathHasUdimToken(name) || src.contains(QStringLiteral("$F"))) {
        name = swapExt(name);
    } else if (pathHasUdimToken(src)) {
        name = swapExt(info.fileName());
    } else {
        QString udimPattern;
        std::vector<int> tiles;
        if (resolveUdimPattern(src, QString(), udimPattern, tiles)) {
            QFileInfo pinfo(udimPattern);
            name = swapExt(pinfo.fileName());
        } else {
            name = info.completeBaseName() + QLatin1Char('.') + extension;
        }
    }
    return QDir(outDir).filePath(name);
}

void TextureViewerWidget::clearView() {
    frames_.clear();
    frameIndex_ = 0;
    loadedCount_ = 0;
    ++loadGeneration_;
    canvas_->clear();
    rangeStart_ = 1;
    rangeEnd_ = 1;
    rebuildTimeline();
    infoLabel_->setText(QStringLiteral("No texture"));
}

void TextureViewerWidget::refreshFromPipeline() {
    QString path;
    if (contentKind_ == ViewerContentKind::SourceImages) {
        path = pipelineSource_;
    } else {
        path = guessConvertedOutputPath(pipelineSource_, pipelineOutputFolder_, outputExt_);
    }
    if (path.isEmpty()) {
        clearView();
        return;
    }
    setSourcePath(path);
}

void TextureViewerWidget::setColorManagement(int mode) {
    mode = (mode == kColorClassic) ? kColorClassic : kColorOcio;
    colorManagement_ = mode;
    if (colorMgmtCombo_) {
        const QSignalBlocker block(colorMgmtCombo_);
        const int idx = colorMgmtCombo_->findData(mode);
        if (idx >= 0) colorMgmtCombo_->setCurrentIndex(idx);
    }
    if (mode == kColorOcio) {
        const OcioStatus st = ocioEnsureConfig(ocioUseEnv_, ocioConfigPath_.toStdString());
        if (!st.configLoaded) {
            emit statusMessage(QStringLiteral("Viewer: %1").arg(QString::fromStdString(st.message)));
        }
    }
    canvas_->setDisplayParams(colorManagement_, viewTransform_, ocioUseEnv_, ocioConfigPath_,
                              ocioWorkingSpace_);
    updateInfoBar();
}

void TextureViewerWidget::setViewTransform(int view) {
    if (view != kViewSrgbAces && view != kViewRec709Aces && view != kViewRec2020 &&
        view != kViewRaw) {
        view = kViewSrgbAces;
    }
    viewTransform_ = view;
    if (viewCombo_) {
        const QSignalBlocker block(viewCombo_);
        const int idx = viewCombo_->findData(view);
        if (idx >= 0) viewCombo_->setCurrentIndex(idx);
    }
    canvas_->setDisplayParams(colorManagement_, viewTransform_, ocioUseEnv_, ocioConfigPath_,
                              ocioWorkingSpace_);
    updateInfoBar();
}

void TextureViewerWidget::setSourcePath(const QString& pathIn) {
    frames_.clear();
    frameIndex_ = 0;
    loadedCount_ = 0;
    ++loadGeneration_;
    canvas_->clear();

    const QString path = pathIn.trimmed();
    if (path.isEmpty()) {
        rangeStart_ = 1;
        rangeEnd_ = 1;
        rebuildTimeline();
        infoLabel_->setText(QStringLiteral("No texture"));
        return;
    }

    QStringList paths;
    QString pattern;
    std::vector<int> tiles;
    if (resolveUdimPattern(path, QString(), pattern, tiles) && !tiles.empty()) {
        for (int udim : tiles) {
            const QString tile = expandUdimToken(pattern, udim);
            if (!QFileInfo::exists(tile)) continue;
            paths.push_back(tile);
        }
    }

    if (paths.isEmpty() && path.contains(QStringLiteral("$F"))) {
        const int scanEnd = std::max(10000, rangeEnd_);
        const auto expanded = txExpandFrameSources(path.toStdString(), 1, scanEnd);
        for (const std::string& p : expanded) paths.push_back(QString::fromStdString(p));
    }

    if (paths.isEmpty()) {
        if (!QFileInfo::exists(path)) {
            // Missing source / .tx → default placeholder, not an error banner.
            rangeStart_ = 1;
            rangeEnd_ = 1;
            rebuildTimeline();
            infoLabel_->setText(QStringLiteral("No texture"));
            return;
        }
        paths.push_back(path);
    }

    frames_.resize(paths.size());
    for (int i = 0; i < paths.size(); ++i) {
        frames_[i].path = paths[i];
        frames_[i].frameNumber = txExtractFrameNumber(paths[i].toStdString());
    }

    ocioWorkingSpace_ = prefersAcesCgWorkingSpace(paths.first()) ? kWorkingSpaceAcesCg
                                                                 : kWorkingSpaceSrgbLinear;
    canvas_->setDisplayParams(colorManagement_, viewTransform_, ocioUseEnv_, ocioConfigPath_,
                              ocioWorkingSpace_);
    canvas_->setChannelMode(channelMode_);
    canvas_->setGrade(grade_);

    int minFrame = frames_.first().frameNumber;
    int maxFrame = frames_.first().frameNumber;
    for (const FrameSlot& f : frames_) {
        minFrame = std::min(minFrame, f.frameNumber);
        maxFrame = std::max(maxFrame, f.frameNumber);
    }
    rangeStart_ = minFrame;
    rangeEnd_ = maxFrame;
    rebuildTimeline();
    canvas_->setPlaceholder(QStringLiteral("Loading sequence…"));
    emit statusMessage(QStringLiteral("Viewer: loading %1 tile(s)…").arg(frames_.size()));
    startPreloadAll();
}

void TextureViewerWidget::startPreloadAll() {
    const quint64 generation = loadGeneration_.load();
    QPointer<TextureViewerWidget> self(this);
    for (int i = 0; i < frames_.size(); ++i) {
        const QString path = frames_[i].path;
        QThreadPool::globalInstance()->start([self, path, i, generation]() {
            LoadPayload payload = decodeFrame(path, i);
            if (!self) return;
            QMetaObject::invokeMethod(
                self,
                [self, generation, payload = std::move(payload)]() mutable {
                    if (!self) return;
                    self->onFrameLoaded(generation, std::move(payload));
                },
                Qt::QueuedConnection);
        });
    }
}

void TextureViewerWidget::onFrameLoaded(quint64 generation, LoadPayload payload) {
    if (generation != loadGeneration_.load()) return;
    if (payload.index < 0 || payload.index >= frames_.size()) return;
    if (frames_[payload.index].path != payload.path) return;

    FrameSlot& slot = frames_[payload.index];
    if (!slot.ready) ++loadedCount_;

    slot.sourceWidth = payload.sourceWidth;
    slot.sourceHeight = payload.sourceHeight;
    slot.fileBytes = payload.fileBytes;
    slot.previewW = payload.previewW;
    slot.previewH = payload.previewH;
    slot.linearRgba = std::move(payload.linearRgba);
    slot.error = payload.error;
    slot.ready = payload.error.isEmpty() && !slot.linearRgba.empty();

    if (payload.index == frameIndex_) showCurrentFrame();

    if (loadedCount_ >= frames_.size()) {
        emit statusMessage(QStringLiteral("Viewer: sequence ready (%1 tile(s))").arg(frames_.size()));
    } else {
        emit statusMessage(
            QStringLiteral("Viewer: loaded %1/%2…").arg(loadedCount_).arg(frames_.size()));
    }
    updateInfoBar();
}

QString TextureViewerWidget::currentPath() const {
    if (frameIndex_ < 0 || frameIndex_ >= frames_.size()) return {};
    return frames_[frameIndex_].path;
}

int TextureViewerWidget::indexForFrameNumber(int frame) const {
    if (frames_.isEmpty()) return 0;
    int bestIdx = 0;
    int bestDist = std::abs(frames_[0].frameNumber - frame);
    for (int i = 1; i < frames_.size(); ++i) {
        const int d = std::abs(frames_[i].frameNumber - frame);
        if (d < bestDist) {
            bestDist = d;
            bestIdx = i;
        }
    }
    return bestIdx;
}

int TextureViewerWidget::frameNumberAt(int index) const {
    if (index < 0 || index >= frames_.size()) return rangeStart_;
    return frames_[index].frameNumber;
}

void TextureViewerWidget::setTimelineFrame(int frame) {
    // Scrubber shows absolute UDIM / $F numbers; snap to the nearest existing frame.
    if (frames_.isEmpty()) return;
    setFrame(indexForFrameNumber(frame));
}

void TextureViewerWidget::setFrame(int index) {
    if (frames_.isEmpty()) return;
    index = std::clamp(index, 0, int(frames_.size()) - 1);
    // Keep scrub inside the editable start/end window (also expressed as frame numbers).
    int lo = indexForFrameNumber(rangeStart_);
    int hi = indexForFrameNumber(rangeEnd_);
    if (lo > hi) std::swap(lo, hi);
    index = std::clamp(index, lo, hi);
    frameIndex_ = index;
    setExprFrame(frameNumberAt(frameIndex_));
    if (!updatingTimeline_) {
        updatingTimeline_ = true;
        scrubber_->setFrame(frameNumberAt(frameIndex_));
        updatingTimeline_ = false;
    }
    showCurrentFrame();
}

void TextureViewerWidget::nextFrame() {
    if (frames_.size() <= 1) return;
    int lo = indexForFrameNumber(rangeStart_);
    int hi = indexForFrameNumber(rangeEnd_);
    if (lo > hi) std::swap(lo, hi);
    const int next = frameIndex_ + 1;
    if (next > hi) setFrame(lo);
    else setFrame(next);
}

void TextureViewerWidget::prevFrame() {
    if (frames_.size() <= 1) return;
    int lo = indexForFrameNumber(rangeStart_);
    int hi = indexForFrameNumber(rangeEnd_);
    if (lo > hi) std::swap(lo, hi);
    const int prev = frameIndex_ - 1;
    if (prev < lo) setFrame(hi);
    else setFrame(prev);
}

void TextureViewerWidget::fitView() { canvas_->fitToView(); }

void TextureViewerWidget::onStartEdited() {
    if (updatingTimeline_ || !startEdit_) return;
    bool ok = false;
    int value = startEdit_->text().trimmed().toInt(&ok);
    if (!ok) {
        updatingTimeline_ = true;
        startEdit_->setText(QString::number(rangeStart_));
        updatingTimeline_ = false;
        return;
    }
    if (!frames_.isEmpty()) {
        int lo = frames_.first().frameNumber;
        int hi = frames_.first().frameNumber;
        for (const FrameSlot& f : frames_) {
            lo = std::min(lo, f.frameNumber);
            hi = std::max(hi, f.frameNumber);
        }
        value = std::clamp(value, lo, hi);
    }
    rangeStart_ = value;
    if (rangeEnd_ < rangeStart_) rangeEnd_ = rangeStart_;
    rebuildTimeline();
    setFrame(frameIndex_);
}

void TextureViewerWidget::onEndEdited() {
    if (updatingTimeline_ || !endEdit_) return;
    bool ok = false;
    int value = endEdit_->text().trimmed().toInt(&ok);
    if (!ok) {
        updatingTimeline_ = true;
        endEdit_->setText(QString::number(rangeEnd_));
        updatingTimeline_ = false;
        return;
    }
    if (!frames_.isEmpty()) {
        int lo = frames_.first().frameNumber;
        int hi = frames_.first().frameNumber;
        for (const FrameSlot& f : frames_) {
            lo = std::min(lo, f.frameNumber);
            hi = std::max(hi, f.frameNumber);
        }
        value = std::clamp(value, lo, hi);
    }
    rangeEnd_ = value;
    if (rangeEnd_ < rangeStart_) rangeStart_ = rangeEnd_;
    rebuildTimeline();
    setFrame(frameIndex_);
}

void TextureViewerWidget::rebuildTimeline() {
    int lo = 1;
    int hi = 1;
    if (!frames_.isEmpty()) {
        lo = hi = frames_.first().frameNumber;
        for (const FrameSlot& f : frames_) {
            lo = std::min(lo, f.frameNumber);
            hi = std::max(hi, f.frameNumber);
        }
    }
    rangeStart_ = std::clamp(rangeStart_, lo, hi);
    rangeEnd_ = std::clamp(rangeEnd_, lo, hi);
    if (rangeEnd_ < rangeStart_) rangeEnd_ = rangeStart_;

    updatingTimeline_ = true;
    if (startEdit_) startEdit_->setText(QString::number(rangeStart_));
    if (endEdit_) endEdit_->setText(QString::number(rangeEnd_));
    scrubber_->setRange(rangeStart_, rangeEnd_);
    scrubber_->setFrame(std::clamp(frameNumberAt(frameIndex_), rangeStart_, rangeEnd_));
    updatingTimeline_ = false;
}

void TextureViewerWidget::updateInfoBar() {
    if (frames_.isEmpty() || frameIndex_ < 0 || frameIndex_ >= frames_.size()) {
        infoLabel_->setText(QStringLiteral("No texture"));
        return;
    }
    const FrameSlot& slot = frames_[frameIndex_];
    const QString modeBit = (colorManagement_ == kColorOcio)
                                ? QStringLiteral("OCIO")
                                : QStringLiteral("Classic");
    QString viewBit = QStringLiteral("sRGB");
    if (viewTransform_ == kViewRec709Aces) viewBit = QStringLiteral("Rec.709");
    else if (viewTransform_ == kViewRec2020) viewBit = QStringLiteral("Rec.2020");
    else if (viewTransform_ == kViewRaw) viewBit = QStringLiteral("Raw");
    const QString channelBit = channelModeLabel(channelMode_);
    if (!slot.error.isEmpty()) {
        infoLabel_->setText(slot.error);
        return;
    }
    if (!slot.ready) {
        infoLabel_->setText(QStringLiteral("Loading %1… (%2/%3)")
                                .arg(QFileInfo(slot.path).fileName())
                                .arg(loadedCount_)
                                .arg(frames_.size()));
        return;
    }
    infoLabel_->setText(QStringLiteral("%1  ·  %2×%3  ·  %4  ·  %5/%6  ·  %7  ·  float  ·  %8/%9")
                            .arg(QFileInfo(slot.path).fileName())
                            .arg(slot.sourceWidth)
                            .arg(slot.sourceHeight)
                            .arg(formatBytes(slot.fileBytes))
                            .arg(modeBit)
                            .arg(viewBit)
                            .arg(channelBit)
                            .arg(loadedCount_)
                            .arg(frames_.size()));
}

void TextureViewerWidget::pushFrameToCanvas() {
    if (frames_.isEmpty() || frameIndex_ < 0 || frameIndex_ >= frames_.size()) {
        canvas_->clear();
        return;
    }
    const FrameSlot& slot = frames_[frameIndex_];
    if (!slot.error.isEmpty()) {
        canvas_->setPlaceholder(slot.error);
        return;
    }
    if (!slot.ready || slot.linearRgba.empty()) {
        canvas_->setPlaceholder(QStringLiteral("Loading…"));
        return;
    }
    const quint64 id = (quint64(loadGeneration_.load()) << 32) ^ quint64(frameIndex_ + 1) ^
                       (quint64(slot.previewW) << 16) ^ quint64(slot.previewH);
    canvas_->setLinearImage(slot.linearRgba.data(), slot.previewW, slot.previewH, id);
}

void TextureViewerWidget::showCurrentFrame() {
    if (frames_.isEmpty()) return;
    frameIndex_ = std::clamp(frameIndex_, 0, int(frames_.size()) - 1);
    pushFrameToCanvas();
    updateInfoBar();
}

}  // namespace sol
