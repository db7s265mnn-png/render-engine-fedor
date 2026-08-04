#include "texture_viewer.h"

#include <QAbstractButton>
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
#include <QShortcut>
#include <QSpinBox>
#include <QThreadPool>
#include <QTimer>
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
    if (grade.brightness != 0.0f) c = c * std::exp2(grade.brightness);
    if (grade.contrast != 1.0f) {
        const float pivot = kGradePivot;
        c = (c - Vec3(pivot)) * grade.contrast + Vec3(pivot);
    }
    if (grade.gamma != 1.0f) {
        const float invG = 1.0f / std::max(0.01f, grade.gamma);
        c.x = std::pow(std::max(0.0f, c.x), invG);
        c.y = std::pow(std::max(0.0f, c.y), invG);
        c.z = std::pow(std::max(0.0f, c.z), invG);
    }
    return c;
}

bool gradeIsIdentity(const ViewerGrade& grade) {
    return grade.brightness == 0.0f && grade.contrast == 1.0f && grade.gamma == 1.0f;
}

// Channel extract into packed RGB (no grade / view). Done once per image+channel.
void buildBaseLinear(const float* linearRgba, int w, int h, ViewerChannelMode channelMode,
                     std::vector<float>& outRgb) {
    outRgb.resize(size_t(w) * size_t(h) * 3);
    if (!linearRgba || w <= 0 || h <= 0) return;

    for (int y = 0; y < h; ++y) {
        const float* row = linearRgba + size_t(y) * size_t(w) * 4;
        float* dst = outRgb.data() + size_t(y) * size_t(w) * 3;
        for (int x = 0; x < w; ++x) {
            const float* px = row + size_t(x) * 4;
            float r = px[0], g = px[1], b = px[2];
            if (channelMode != ViewerChannelMode::RGBA) {
                float v = px[0];
                switch (channelMode) {
                    case ViewerChannelMode::R: v = px[0]; break;
                    case ViewerChannelMode::G: v = px[1]; break;
                    case ViewerChannelMode::B: v = px[2]; break;
                    case ViewerChannelMode::A: v = px[3]; break;
                    default: break;
                }
                r = g = b = v;
            }
            dst[x * 3 + 0] = r;
            dst[x * 3 + 1] = g;
            dst[x * 3 + 2] = b;
        }
    }
}

// Bake from pre-extracted linear RGB (3 floats/px). pixelStep>1 = interactive preview.
QImage bakeDisplayFromBase(const float* baseRgb, int w, int h, int colorManagement, int viewTransform,
                           bool ocioUseEnv, const QString& ocioConfigPath, int workingSpace,
                           const ViewerGrade& grade, int pixelStep) {
    pixelStep = std::max(1, pixelStep);
    const int outW = std::max(1, (w + pixelStep - 1) / pixelStep);
    const int outH = std::max(1, (h + pixelStep - 1) / pixelStep);
    QImage out(outW, outH, QImage::Format_RGB888);
    if (!baseRgb || w <= 0 || h <= 0) return out;

    if (colorManagement == kColorOcio) {
        ocioEnsureConfig(ocioUseEnv, ocioConfigPath.toStdString());
    }
    displayPrepareView(workingSpace, colorManagement, viewTransform);
    const bool skipGrade = gradeIsIdentity(grade);

    for (int oy = 0; oy < outH; ++oy) {
        const int y = std::min(h - 1, oy * pixelStep);
        uchar* line = out.scanLine(oy);
        const float* row = baseRgb + size_t(y) * size_t(w) * 3;
        for (int ox = 0; ox < outW; ++ox) {
            const int x = std::min(w - 1, ox * pixelStep);
            Vec3 linear(row[x * 3 + 0], row[x * 3 + 1], row[x * 3 + 2]);
            if (!skipGrade) linear = applyGrade(linear, grade);
            const Vec3 display = ocioApplyViewPrepared(linear);
            uchar* dst = line + size_t(ox) * 3;
            dst[0] = static_cast<uchar>(clampf(display.x, 0.0f, 1.0f) * 255.0f + 0.5f);
            dst[1] = static_cast<uchar>(clampf(display.y, 0.0f, 1.0f) * 255.0f + 0.5f);
            dst[2] = static_cast<uchar>(clampf(display.z, 0.0f, 1.0f) * 255.0f + 0.5f);
        }
    }
    return out;
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
    invalidateBaseLinear();
    placeholder_ = QStringLiteral("No texture");
    update();
}

void FloatPreviewCanvas::setPlaceholder(const QString& text) {
    linearRgba_ = nullptr;
    width_ = height_ = 0;
    contentId_ = 0;
    invalidateBaseLinear();
    placeholder_ = text;
    update();
}

void FloatPreviewCanvas::setLinearImage(const float* rgba, int width, int height, quint64 contentId) {
    linearRgba_ = rgba;
    width_ = width;
    height_ = height;
    contentId_ = contentId;
    invalidateBaseLinear();
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
    invalidateBaseLinear();
    update();
}

void FloatPreviewCanvas::setGrade(const ViewerGrade& grade, bool interactive) {
    const bool sameGrade = gradeEquals(grade_, grade);
    grade_ = grade;
    gradeInteractive_ = interactive;
    // Keep the last display frame while scrubbing (avoid blank flash); ensureDisplayCache
    // rebuilds when grade / step no longer match the cache keys.
    if (!sameGrade || (!interactive && displayCacheStep_ > 1)) {
        // Mark dirty without wiping the QImage so paint can still show the previous frame.
        displayCacheId_ = 0;
    }
    update();
}

void FloatPreviewCanvas::invalidateDisplayCache() {
    displayCache_ = {};
    displayCacheId_ = 0;
    displayCacheStep_ = 0;
}

void FloatPreviewCanvas::invalidateBaseLinear() {
    baseLinearRgb_.clear();
    baseLinearId_ = 0;
    baseLinearChannel_ = -1;
    invalidateDisplayCache();
}

void FloatPreviewCanvas::ensureBaseLinear() {
    if (!linearRgba_ || width_ <= 0 || height_ <= 0) return;
    if (!baseLinearRgb_.empty() && baseLinearId_ == contentId_ &&
        baseLinearChannel_ == int(channelMode_)) {
        return;
    }
    buildBaseLinear(linearRgba_, width_, height_, channelMode_, baseLinearRgb_);
    baseLinearId_ = contentId_;
    baseLinearChannel_ = int(channelMode_);
}

void FloatPreviewCanvas::ensureDisplayCache() {
    if (!linearRgba_ || width_ <= 0 || height_ <= 0) return;

    const int wantStep = gradeInteractive_ ? 3 : 1;
    // Reuse cache when params match and resolution is at least as good as requested.
    if (!displayCache_.isNull() && displayCacheId_ == contentId_ &&
        displayCacheColorMgmt_ == colorManagement_ && displayCacheView_ == viewTransform_ &&
        displayCacheChannel_ == int(channelMode_) && displayCacheWorking_ == workingSpace_ &&
        displayCacheOcioEnv_ == ocioUseEnv_ && displayCacheOcioPath_ == ocioConfigPath_ &&
        gradeEquals(displayCacheGrade_, grade_) && displayCacheStep_ > 0 &&
        displayCacheStep_ <= wantStep) {
        return;
    }

    ensureBaseLinear();
    displayCache_ = bakeDisplayFromBase(baseLinearRgb_.data(), width_, height_, colorManagement_,
                                        viewTransform_, ocioUseEnv_, ocioConfigPath_, workingSpace_,
                                        grade_, wantStep);
    displayCacheId_ = contentId_;
    displayCacheColorMgmt_ = colorManagement_;
    displayCacheView_ = viewTransform_;
    displayCacheChannel_ = int(channelMode_);
    displayCacheWorking_ = workingSpace_;
    displayCacheOcioEnv_ = ocioUseEnv_;
    displayCacheOcioPath_ = ocioConfigPath_;
    displayCacheGrade_ = grade_;
    displayCacheStep_ = wantStep;
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

void FloatPreviewCanvas::setCamera(double zoom, const QPointF& pan, bool fitted) {
    fitted_ = fitted;
    zoom_ = std::clamp(zoom, 0.05, 64.0);
    pan_ = pan;
    if (fitted_) {
        fitToView();
        return;
    }
    clampPan();
    emit zoomChanged(zoom_);
    update();
}

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
    if ((event->key() == Qt::Key_1 || event->key() == Qt::Key_2) && !event->modifiers()) {
        // Let the parent viewer handle Source/Output hotkeys.
        event->ignore();
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

    // Toolbar: Source / Output (1 / 2), channels, Display, Fit, RAM budget.
    auto* modeRow = new QHBoxLayout();
    modeRow->setContentsMargins(0, 0, 0, 0);
    modeRow->setSpacing(6);

    contentGroup_ = new QButtonGroup(this);
    contentGroup_->setExclusive(true);
    sourceBtn_ = new QToolButton();
    sourceBtn_->setText(QStringLiteral("Source"));
    sourceBtn_->setCheckable(true);
    sourceBtn_->setToolTip(QStringLiteral("Show source images (hotkey 1).\n"
                                          "Grade / Display / channels are remembered per view."));
    sourceBtn_->setChecked(true);
    outputBtn_ = new QToolButton();
    outputBtn_->setText(QStringLiteral("Output"));
    outputBtn_->setCheckable(true);
    outputBtn_->setToolTip(QStringLiteral("Show converted output (hotkey 2).\n"
                                          "Grade / Display / channels are remembered per view."));
    contentGroup_->addButton(sourceBtn_, int(ViewerContentKind::SourceImages));
    contentGroup_->addButton(outputBtn_, int(ViewerContentKind::ConvertedTx));
    modeRow->addWidget(sourceBtn_);
    modeRow->addWidget(outputBtn_);

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

    modeRow->addWidget(new QLabel(QStringLiteral("RAM")));
    memoryGbSpin_ = new QSpinBox();
    memoryGbSpin_->setRange(1, 512);
    memoryGbSpin_->setValue(32);
    memoryGbSpin_->setSuffix(QStringLiteral(" GB"));
    memoryGbSpin_->setToolTip(
        QStringLiteral("Viewer texture buffer budget (decoded float previews).\n"
                       "Also used as the convert parallel memory budget."));
    memoryGbSpin_->setFixedWidth(90);
    modeRow->addWidget(memoryGbSpin_);
    clearBufBtn_ = new QPushButton(QStringLiteral("Clear"));
    clearBufBtn_->setFixedWidth(52);
    clearBufBtn_->setToolTip(QStringLiteral("Unload Source and Converted textures from memory."));
    modeRow->addWidget(clearBufBtn_);
    bufferLabel_ = new QLabel(QStringLiteral("buf 0 B"));
    bufferLabel_->setMinimumWidth(70);
    bufferLabel_->setStyleSheet(QStringLiteral("color: #9aa0a6;"));
    modeRow->addWidget(bufferLabel_);
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
    connect(contentGroup_, &QButtonGroup::idClicked, this, [this](int id) {
        setContentKind(ViewerContentKind(id));
    });
    connect(channelGroup_, &QButtonGroup::idClicked, this,
            [this](int id) { setChannelMode(ViewerChannelMode(id)); });
    connect(clearBufBtn_, &QPushButton::clicked, this, [this] { clearTextureBuffers(); });
    connect(memoryGbSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int gb) {
        setMemoryBudgetBytes(qint64(gb) * 1024LL * 1024LL * 1024LL);
    });
    setMemoryBudgetBytes(qint64(memoryGbSpin_->value()) * 1024LL * 1024LL * 1024LL);
    setFocusPolicy(Qt::StrongFocus);

    auto* hotSource = new QShortcut(QKeySequence(Qt::Key_1), this);
    hotSource->setContext(Qt::WidgetWithChildrenShortcut);
    connect(hotSource, &QShortcut::activated, this,
            [this] { setContentKind(ViewerContentKind::SourceImages); });
    auto* hotOutput = new QShortcut(QKeySequence(Qt::Key_2), this);
    hotOutput->setContext(Qt::WidgetWithChildrenShortcut);
    connect(hotOutput, &QShortcut::activated, this,
            [this] { setContentKind(ViewerContentKind::ConvertedTx); });

    gradeTimer_ = new QTimer(this);
    gradeTimer_->setSingleShot(true);
    connect(gradeTimer_, &QTimer::timeout, this, [this] { applyGradeFromUi(gradeScrubbing_); });

    auto wireGrade = [this](QDoubleSpinBox* spin, QSlider* slider, double minV, double maxV) {
        const double span = std::max(1e-9, maxV - minV);
        connect(spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
                [this, slider, minV, span](double v) {
                    QSignalBlocker b(slider);
                    slider->setValue(int(std::lround(std::clamp((v - minV) / span, 0.0, 1.0) * 1000.0)));
                    // Spin edits (typing / label reset): full bake. While scrubbing, spin is blocked.
                    scheduleGradeApply(false);
                });
        connect(slider, &QSlider::sliderPressed, this, [this] { gradeScrubbing_ = true; });
        connect(slider, &QSlider::sliderReleased, this, [this] {
            gradeScrubbing_ = false;
            if (gradeTimer_) gradeTimer_->stop();
            applyGradeFromUi(false);
        });
        connect(slider, &QSlider::valueChanged, this, [this, spin, minV, span](int pos) {
            const double v = minV + span * (double(pos) / 1000.0);
            QSignalBlocker b(spin);
            spin->setValue(v);
            scheduleGradeApply(true);
        });
    };
    wireGrade(brightnessSpin_, brightnessSlider_, -5.0, 5.0);
    wireGrade(contrastSpin_, contrastSlider_, 0.0, 3.0);
    wireGrade(gammaSpin_, gammaSlider_, 0.20, 3.0);

    // Clicking a grade label resets only that parameter.
    connect(brightnessLabelBtn_, &QPushButton::clicked, this, [this] { brightnessSpin_->setValue(0.0); });
    connect(contrastLabelBtn_, &QPushButton::clicked, this, [this] { contrastSpin_->setValue(1.0); });
    connect(gammaLabelBtn_, &QPushButton::clicked, this, [this] { gammaSpin_->setValue(1.0); });

    captureDisplayState(ViewerContentKind::SourceImages);
    captureDisplayState(ViewerContentKind::ConvertedTx);
    rebuildTimeline();
}

void TextureViewerWidget::scheduleGradeApply(bool interactive) {
    if (interactive) gradeScrubbing_ = true;
    if (!gradeTimer_) {
        applyGradeFromUi(interactive);
        return;
    }
    // Coalesce slider ticks to ~60 Hz interactive previews.
    gradeTimer_->start(interactive ? 16 : 0);
}

void TextureViewerWidget::applyGradeFromUi(bool interactive) {
    if (applyingDisplayState_) return;
    grade_.brightness = float(brightnessSpin_->value());
    grade_.contrast = float(contrastSpin_->value());
    grade_.gamma = float(gammaSpin_->value());
    canvas_->setGrade(grade_, interactive);
    captureDisplayState(contentKind_);
}

void TextureViewerWidget::setChannelMode(ViewerChannelMode mode) {
    channelMode_ = mode;
    canvas_->setChannelMode(mode);
    if (!applyingDisplayState_) captureDisplayState(contentKind_);
    updateInfoBar();
}

void TextureViewerWidget::setOcioConfig(bool useEnv, const QString& configPath) {
    ocioUseEnv_ = useEnv;
    ocioConfigPath_ = configPath.trimmed();
    canvas_->setDisplayParams(colorManagement_, viewTransform_, ocioUseEnv_, ocioConfigPath_,
                              ocioWorkingSpace_);
}

TextureViewerWidget::SequenceCache& TextureViewerWidget::cacheFor(ViewerContentKind kind) {
    return kind == ViewerContentKind::SourceImages ? sourceCache_ : convertedCache_;
}

const TextureViewerWidget::SequenceCache& TextureViewerWidget::cacheFor(ViewerContentKind kind) const {
    return kind == ViewerContentKind::SourceImages ? sourceCache_ : convertedCache_;
}

TextureViewerWidget::SequenceCache& TextureViewerWidget::activeCache() {
    return cacheFor(contentKind_);
}

const TextureViewerWidget::SequenceCache& TextureViewerWidget::activeCache() const {
    return cacheFor(contentKind_);
}

TextureViewerWidget::SequenceCache& TextureViewerWidget::inactiveCache() {
    return cacheFor(contentKind_ == ViewerContentKind::SourceImages
                        ? ViewerContentKind::ConvertedTx
                        : ViewerContentKind::SourceImages);
}

void TextureViewerWidget::rememberSharedFrame() {
    const SequenceCache& cache = activeCache();
    if (!cache.frames.isEmpty() && frameIndex_ >= 0 && frameIndex_ < cache.frames.size()) {
        sharedFrameNumber_ = cache.frames[frameIndex_].frameNumber;
    }
}

void TextureViewerWidget::restoreSharedFrame() {
    const SequenceCache& cache = activeCache();
    if (cache.frames.isEmpty()) {
        frameIndex_ = 0;
        return;
    }
    frameIndex_ = indexForFrameNumber(sharedFrameNumber_);
    setFrame(frameIndex_);
}

qint64 TextureViewerWidget::bufferBytesUsed() const {
    return sourceCache_.bytesUsed() + convertedCache_.bytesUsed();
}

void TextureViewerWidget::setMemoryBudgetBytes(qint64 bytes) {
    memoryBudgetBytes_ = std::max<qint64>(qint64(256) * 1024 * 1024, bytes);
    if (memoryGbSpin_) {
        const int gb = int(std::clamp<qint64>(memoryBudgetBytes_ / (1024LL * 1024LL * 1024LL), 1, 512));
        const QSignalBlocker b(memoryGbSpin_);
        memoryGbSpin_->setValue(gb);
    }
    enforceMemoryBudget(&activeCache());
    updateBufferStatus();
}

void TextureViewerWidget::clearTextureBuffers() {
    rememberSharedFrame();
    sourceCache_.clearPixels();
    convertedCache_.clearPixels();
    // Keep path keys / frame lists so we can reload without resetting timeline.
    sourceCache_.loadedCount = 0;
    convertedCache_.loadedCount = 0;
    ++sourceCache_.generation;
    ++convertedCache_.generation;
    canvas_->clear();
    canvas_->setPlaceholder(QStringLiteral("Buffer cleared — reload on demand"));
    updateInfoBar();
    updateBufferStatus();
    emit statusMessage(QStringLiteral("Viewer: texture buffers cleared"));
    // Kick reload for the active view only.
    if (!activeCache().pathKey.isEmpty() && !activeCache().frames.isEmpty()) {
        startPreload(activeCache());
    }
}

void TextureViewerWidget::reloadConvertedBuffer() {
    rememberSharedFrame();
    convertedCache_.reset();
    if (contentKind_ == ViewerContentKind::ConvertedTx) refreshFromPipeline(true);
}

void TextureViewerWidget::updateBufferStatus() {
    if (!bufferLabel_) return;
    bufferLabel_->setText(QStringLiteral("buf %1").arg(formatBytes(bufferBytesUsed())));
    bufferLabel_->setToolTip(
        QStringLiteral("Source %1 · Converted %2 · Budget %3")
            .arg(formatBytes(sourceCache_.bytesUsed()), formatBytes(convertedCache_.bytesUsed()),
                 formatBytes(memoryBudgetBytes_)));
}

void TextureViewerWidget::enforceMemoryBudget(SequenceCache* preferKeep) {
    if (bufferBytesUsed() <= memoryBudgetBytes_) return;

    auto unloadUntilOk = [&](SequenceCache& cache, int keepIndex) {
        for (int pass = 0; pass < cache.frames.size(); ++pass) {
            if (bufferBytesUsed() <= memoryBudgetBytes_) return;
            int best = -1;
            int bestDist = -1;
            for (int i = 0; i < cache.frames.size(); ++i) {
                if (i == keepIndex) continue;
                if (!cache.frames[i].ready || cache.frames[i].linearRgba.empty()) continue;
                const int dist = std::abs(i - std::max(0, keepIndex));
                if (dist > bestDist) {
                    bestDist = dist;
                    best = i;
                }
            }
            if (best < 0) return;
            cache.frames[best].unloadPixels();
            cache.loadedCount = std::max(0, cache.loadedCount - 1);
        }
    };

    // Free the inactive cache first (only while over budget).
    SequenceCache* other =
        (preferKeep == &sourceCache_) ? &convertedCache_
        : (preferKeep == &convertedCache_) ? &sourceCache_
                                           : &convertedCache_;
    if (other != preferKeep) {
        other->clearPixels();
        other->loadedCount = 0;
    }
    if (bufferBytesUsed() > memoryBudgetBytes_ && preferKeep) {
        const int keep = (preferKeep == &activeCache()) ? frameIndex_ : -1;
        unloadUntilOk(*preferKeep, keep);
    }
}

void TextureViewerWidget::setPipelinePaths(const QString& sourcePath, const QString& outputFolder) {
    const QString src = sourcePath.trimmed();
    const QString out = outputFolder.trimmed();
    const bool sourceChanged = src != pipelineSource_;
    const bool outChanged = out != pipelineOutputFolder_;
    pipelineSource_ = src;
    pipelineOutputFolder_ = out;
    if (sourceChanged) {
        sourceCache_.reset();
    }
    if (sourceChanged || outChanged) {
        convertedCache_.reset();
    }
    refreshFromPipeline(false);
}

void TextureViewerWidget::setOutputExtension(const QString& ext) {
    const QString trimmed = ext.trimmed();
    const QString next = trimmed.isEmpty() ? QStringLiteral("tx") : trimmed;
    if (outputExt_ == next) return;
    outputExt_ = next;
    convertedCache_.reset();
    if (contentKind_ == ViewerContentKind::ConvertedTx) refreshFromPipeline(true);
}

TextureViewerWidget::ViewDisplayState& TextureViewerWidget::displayStateFor(ViewerContentKind kind) {
    return kind == ViewerContentKind::SourceImages ? sourceDisplay_ : outputDisplay_;
}

const TextureViewerWidget::ViewDisplayState& TextureViewerWidget::displayStateFor(
    ViewerContentKind kind) const {
    return kind == ViewerContentKind::SourceImages ? sourceDisplay_ : outputDisplay_;
}

void TextureViewerWidget::captureDisplayState(ViewerContentKind kind) {
    ViewDisplayState& st = displayStateFor(kind);
    st.grade = grade_;
    st.channelMode = channelMode_;
    st.colorManagement = colorManagement_;
    st.viewTransform = viewTransform_;
    if (canvas_) {
        st.zoom = canvas_->zoom();
        st.pan = canvas_->pan();
        st.fitted = canvas_->fitted();
    }
    st.initialized = true;
}

void TextureViewerWidget::applyDisplayState(ViewerContentKind kind) {
    ViewDisplayState& st = displayStateFor(kind);
    applyingDisplayState_ = true;

    grade_ = st.grade;
    channelMode_ = st.channelMode;
    colorManagement_ = st.colorManagement;
    viewTransform_ = st.viewTransform;

    if (brightnessSpin_) {
        const QSignalBlocker b1(brightnessSpin_);
        const QSignalBlocker b2(brightnessSlider_);
        brightnessSpin_->setValue(grade_.brightness);
        const double span = 10.0;
        brightnessSlider_->setValue(
            int(std::lround(std::clamp((grade_.brightness - (-5.0)) / span, 0.0, 1.0) * 1000.0)));
    }
    if (contrastSpin_) {
        const QSignalBlocker b1(contrastSpin_);
        const QSignalBlocker b2(contrastSlider_);
        contrastSpin_->setValue(grade_.contrast);
        contrastSlider_->setValue(
            int(std::lround(std::clamp(grade_.contrast / 3.0, 0.0, 1.0) * 1000.0)));
    }
    if (gammaSpin_) {
        const QSignalBlocker b1(gammaSpin_);
        const QSignalBlocker b2(gammaSlider_);
        gammaSpin_->setValue(grade_.gamma);
        const double span = 3.0 - 0.20;
        gammaSlider_->setValue(
            int(std::lround(std::clamp((grade_.gamma - 0.20) / span, 0.0, 1.0) * 1000.0)));
    }
    if (channelGroup_) {
        if (QAbstractButton* btn = channelGroup_->button(int(channelMode_))) btn->setChecked(true);
    }
    if (colorMgmtCombo_) {
        const QSignalBlocker b(colorMgmtCombo_);
        const int idx = colorMgmtCombo_->findData(colorManagement_);
        if (idx >= 0) colorMgmtCombo_->setCurrentIndex(idx);
    }
    if (viewCombo_) {
        const QSignalBlocker b(viewCombo_);
        const int idx = viewCombo_->findData(viewTransform_);
        if (idx >= 0) viewCombo_->setCurrentIndex(idx);
    }

    canvas_->setChannelMode(channelMode_);
    canvas_->setGrade(grade_, false);
    canvas_->setDisplayParams(colorManagement_, viewTransform_, ocioUseEnv_, ocioConfigPath_,
                              ocioWorkingSpace_);
    if (st.initialized) {
        canvas_->setCamera(st.zoom, st.pan, st.fitted);
    }

    applyingDisplayState_ = false;
    updateInfoBar();
}

void TextureViewerWidget::syncContentButtons() {
    if (!contentGroup_) return;
    const QSignalBlocker b(contentGroup_);
    if (QAbstractButton* btn = contentGroup_->button(int(contentKind_))) btn->setChecked(true);
}

void TextureViewerWidget::keyPressEvent(QKeyEvent* event) {
    if (!event->modifiers()) {
        if (event->key() == Qt::Key_1) {
            setContentKind(ViewerContentKind::SourceImages);
            event->accept();
            return;
        }
        if (event->key() == Qt::Key_2) {
            setContentKind(ViewerContentKind::ConvertedTx);
            event->accept();
            return;
        }
    }
    QWidget::keyPressEvent(event);
}

void TextureViewerWidget::setContentKind(ViewerContentKind kind) {
    if (kind != ViewerContentKind::SourceImages && kind != ViewerContentKind::ConvertedTx) {
        kind = ViewerContentKind::SourceImages;
    }
    if (kind == contentKind_) {
        syncContentButtons();
        return;
    }
    rememberSharedFrame();
    captureDisplayState(contentKind_);
    contentKind_ = kind;
    syncContentButtons();
    refreshFromPipeline(false);
    applyDisplayState(contentKind_);
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

void TextureViewerWidget::clearActiveView() {
    canvas_->clear();
    rebuildTimeline();
    infoLabel_->setText(QStringLiteral("No texture"));
    updateBufferStatus();
}

void TextureViewerWidget::refreshFromPipeline(bool forceReload) {
    QString path;
    if (contentKind_ == ViewerContentKind::SourceImages) {
        path = pipelineSource_;
    } else {
        path = guessConvertedOutputPath(pipelineSource_, pipelineOutputFolder_, outputExt_);
    }
    if (path.isEmpty()) {
        activeCache().reset();
        clearActiveView();
        return;
    }

    SequenceCache& cache = activeCache();
    if (!forceReload && cache.pathKey == path && !cache.frames.isEmpty()) {
        bindActiveCacheToUi();
        restoreSharedFrame();
        showCurrentFrame();
        updateBufferStatus();
        return;
    }

    loadSequenceInto(cache, path);
}

void TextureViewerWidget::bindActiveCacheToUi() {
    SequenceCache& cache = activeCache();
    if (cache.frames.isEmpty()) return;
    ocioWorkingSpace_ = prefersAcesCgWorkingSpace(cache.frames.first().path) ? kWorkingSpaceAcesCg
                                                                             : kWorkingSpaceSrgbLinear;
    // Keep per-view grade / display / channels; only refresh OCIO working space for the file type.
    canvas_->setDisplayParams(colorManagement_, viewTransform_, ocioUseEnv_, ocioConfigPath_,
                              ocioWorkingSpace_);
    canvas_->setChannelMode(channelMode_);
    canvas_->setGrade(grade_);
    rebuildTimeline();
}

void TextureViewerWidget::loadSequenceInto(SequenceCache& cache, const QString& pathIn) {
    rememberSharedFrame();
    cache.reset();
    canvas_->clear();

    const QString path = pathIn.trimmed();
    if (path.isEmpty()) {
        clearActiveView();
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
        const int scanEnd = std::max(10000, cache.rangeEnd);
        const auto expanded = txExpandFrameSources(path.toStdString(), 1, scanEnd);
        for (const std::string& p : expanded) paths.push_back(QString::fromStdString(p));
    }

    if (paths.isEmpty()) {
        if (!QFileInfo::exists(path)) {
            clearActiveView();
            return;
        }
        paths.push_back(path);
    }

    cache.pathKey = path;
    cache.frames.resize(paths.size());
    for (int i = 0; i < paths.size(); ++i) {
        cache.frames[i].path = paths[i];
        cache.frames[i].frameNumber = txExtractFrameNumber(paths[i].toStdString());
    }

    int minFrame = cache.frames.first().frameNumber;
    int maxFrame = cache.frames.first().frameNumber;
    for (const FrameSlot& f : cache.frames) {
        minFrame = std::min(minFrame, f.frameNumber);
        maxFrame = std::max(maxFrame, f.frameNumber);
    }
    cache.rangeStart = minFrame;
    cache.rangeEnd = maxFrame;

    bindActiveCacheToUi();
    canvas_->setPlaceholder(QStringLiteral("Loading sequence…"));
    emit statusMessage(QStringLiteral("Viewer: loading %1 tile(s)…").arg(cache.frames.size()));
    restoreSharedFrame();
    startPreload(cache);
    updateBufferStatus();
}

void TextureViewerWidget::setColorManagement(int mode) {
    mode = (mode == kColorClassic) ? kColorClassic : kColorOcio;
    colorManagement_ = mode;
    if (colorMgmtCombo_ && !applyingDisplayState_) {
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
    if (!applyingDisplayState_) captureDisplayState(contentKind_);
    updateInfoBar();
}

void TextureViewerWidget::setViewTransform(int view) {
    if (view != kViewSrgbAces && view != kViewRec709Aces && view != kViewRec2020 &&
        view != kViewRaw) {
        view = kViewSrgbAces;
    }
    viewTransform_ = view;
    if (viewCombo_ && !applyingDisplayState_) {
        const QSignalBlocker block(viewCombo_);
        const int idx = viewCombo_->findData(view);
        if (idx >= 0) viewCombo_->setCurrentIndex(idx);
    }
    canvas_->setDisplayParams(colorManagement_, viewTransform_, ocioUseEnv_, ocioConfigPath_,
                              ocioWorkingSpace_);
    if (!applyingDisplayState_) captureDisplayState(contentKind_);
    updateInfoBar();
}

void TextureViewerWidget::startPreload(SequenceCache& cache) {
    const ViewerContentKind kind =
        (&cache == &sourceCache_) ? ViewerContentKind::SourceImages : ViewerContentKind::ConvertedTx;
    const quint64 generation = cache.generation;
    QPointer<TextureViewerWidget> self(this);
    for (int i = 0; i < cache.frames.size(); ++i) {
        if (cache.frames[i].ready && !cache.frames[i].linearRgba.empty()) continue;
        const QString path = cache.frames[i].path;
        QThreadPool::globalInstance()->start([self, path, i, generation, kind]() {
            LoadPayload payload = decodeFrame(path, i);
            if (!self) return;
            QMetaObject::invokeMethod(
                self,
                [self, kind, generation, payload = std::move(payload)]() mutable {
                    if (!self) return;
                    self->onFrameLoaded(kind, generation, std::move(payload));
                },
                Qt::QueuedConnection);
        });
    }
}

void TextureViewerWidget::onFrameLoaded(ViewerContentKind kind, quint64 generation,
                                        LoadPayload payload) {
    SequenceCache& cache = cacheFor(kind);
    if (generation != cache.generation) return;
    if (payload.index < 0 || payload.index >= cache.frames.size()) return;
    if (cache.frames[payload.index].path != payload.path) return;

    FrameSlot& slot = cache.frames[payload.index];
    const bool wasReady = slot.ready && !slot.linearRgba.empty();

    slot.sourceWidth = payload.sourceWidth;
    slot.sourceHeight = payload.sourceHeight;
    slot.fileBytes = payload.fileBytes;
    slot.previewW = payload.previewW;
    slot.previewH = payload.previewH;
    slot.linearRgba = std::move(payload.linearRgba);
    slot.error = payload.error;
    slot.ready = payload.error.isEmpty() && !slot.linearRgba.empty();
    if (slot.ready && !wasReady) ++cache.loadedCount;

    enforceMemoryBudget(&cache);

    if (kind == contentKind_ && payload.index == frameIndex_) showCurrentFrame();

    if (kind == contentKind_) {
        if (cache.loadedCount >= cache.frames.size()) {
            emit statusMessage(
                QStringLiteral("Viewer: sequence ready (%1 tile(s))").arg(cache.frames.size()));
        } else {
            emit statusMessage(QStringLiteral("Viewer: loaded %1/%2…")
                                   .arg(cache.loadedCount)
                                   .arg(cache.frames.size()));
        }
        updateInfoBar();
    }
    updateBufferStatus();
}

QString TextureViewerWidget::currentPath() const {
    const SequenceCache& cache = activeCache();
    if (frameIndex_ < 0 || frameIndex_ >= cache.frames.size()) return {};
    return cache.frames[frameIndex_].path;
}

int TextureViewerWidget::frameCount() const { return activeCache().frames.size(); }

int TextureViewerWidget::rangeStart() const { return activeCache().rangeStart; }

int TextureViewerWidget::rangeEnd() const { return activeCache().rangeEnd; }

int TextureViewerWidget::indexForFrameNumber(int frame) const {
    const SequenceCache& cache = activeCache();
    if (cache.frames.isEmpty()) return 0;
    int bestIdx = 0;
    int bestDist = std::abs(cache.frames[0].frameNumber - frame);
    for (int i = 1; i < cache.frames.size(); ++i) {
        const int d = std::abs(cache.frames[i].frameNumber - frame);
        if (d < bestDist) {
            bestDist = d;
            bestIdx = i;
        }
    }
    return bestIdx;
}

int TextureViewerWidget::frameNumberAt(int index) const {
    const SequenceCache& cache = activeCache();
    if (index < 0 || index >= cache.frames.size()) return cache.rangeStart;
    return cache.frames[index].frameNumber;
}

void TextureViewerWidget::setTimelineFrame(int frame) {
    if (activeCache().frames.isEmpty()) return;
    sharedFrameNumber_ = frame;
    setFrame(indexForFrameNumber(frame));
}

void TextureViewerWidget::setFrame(int index) {
    SequenceCache& cache = activeCache();
    if (cache.frames.isEmpty()) return;
    index = std::clamp(index, 0, int(cache.frames.size()) - 1);
    int lo = indexForFrameNumber(cache.rangeStart);
    int hi = indexForFrameNumber(cache.rangeEnd);
    if (lo > hi) std::swap(lo, hi);
    index = std::clamp(index, lo, hi);
    frameIndex_ = index;
    sharedFrameNumber_ = frameNumberAt(frameIndex_);
    setExprFrame(sharedFrameNumber_);
    if (!updatingTimeline_) {
        updatingTimeline_ = true;
        scrubber_->setFrame(sharedFrameNumber_);
        updatingTimeline_ = false;
    }
    // Reload if this tile was evicted from the buffer.
    if (!cache.frames[frameIndex_].ready && cache.frames[frameIndex_].error.isEmpty()) {
        startPreload(cache);
    }
    showCurrentFrame();
}

void TextureViewerWidget::nextFrame() {
    const SequenceCache& cache = activeCache();
    if (cache.frames.size() <= 1) return;
    int lo = indexForFrameNumber(cache.rangeStart);
    int hi = indexForFrameNumber(cache.rangeEnd);
    if (lo > hi) std::swap(lo, hi);
    const int next = frameIndex_ + 1;
    if (next > hi) setFrame(lo);
    else setFrame(next);
}

void TextureViewerWidget::prevFrame() {
    const SequenceCache& cache = activeCache();
    if (cache.frames.size() <= 1) return;
    int lo = indexForFrameNumber(cache.rangeStart);
    int hi = indexForFrameNumber(cache.rangeEnd);
    if (lo > hi) std::swap(lo, hi);
    const int prev = frameIndex_ - 1;
    if (prev < lo) setFrame(hi);
    else setFrame(prev);
}

void TextureViewerWidget::fitView() {
    canvas_->fitToView();
    captureDisplayState(contentKind_);
}

void TextureViewerWidget::onStartEdited() {
    if (updatingTimeline_ || !startEdit_) return;
    SequenceCache& cache = activeCache();
    bool ok = false;
    int value = startEdit_->text().trimmed().toInt(&ok);
    if (!ok) {
        updatingTimeline_ = true;
        startEdit_->setText(QString::number(cache.rangeStart));
        updatingTimeline_ = false;
        return;
    }
    if (!cache.frames.isEmpty()) {
        int lo = cache.frames.first().frameNumber;
        int hi = cache.frames.first().frameNumber;
        for (const FrameSlot& f : cache.frames) {
            lo = std::min(lo, f.frameNumber);
            hi = std::max(hi, f.frameNumber);
        }
        value = std::clamp(value, lo, hi);
    }
    cache.rangeStart = value;
    if (cache.rangeEnd < cache.rangeStart) cache.rangeEnd = cache.rangeStart;
    rebuildTimeline();
    setFrame(frameIndex_);
}

void TextureViewerWidget::onEndEdited() {
    if (updatingTimeline_ || !endEdit_) return;
    SequenceCache& cache = activeCache();
    bool ok = false;
    int value = endEdit_->text().trimmed().toInt(&ok);
    if (!ok) {
        updatingTimeline_ = true;
        endEdit_->setText(QString::number(cache.rangeEnd));
        updatingTimeline_ = false;
        return;
    }
    if (!cache.frames.isEmpty()) {
        int lo = cache.frames.first().frameNumber;
        int hi = cache.frames.first().frameNumber;
        for (const FrameSlot& f : cache.frames) {
            lo = std::min(lo, f.frameNumber);
            hi = std::max(hi, f.frameNumber);
        }
        value = std::clamp(value, lo, hi);
    }
    cache.rangeEnd = value;
    if (cache.rangeEnd < cache.rangeStart) cache.rangeStart = cache.rangeEnd;
    rebuildTimeline();
    setFrame(frameIndex_);
}

void TextureViewerWidget::rebuildTimeline() {
    SequenceCache& cache = activeCache();
    int lo = 1;
    int hi = 1;
    if (!cache.frames.isEmpty()) {
        lo = hi = cache.frames.first().frameNumber;
        for (const FrameSlot& f : cache.frames) {
            lo = std::min(lo, f.frameNumber);
            hi = std::max(hi, f.frameNumber);
        }
    }
    cache.rangeStart = std::clamp(cache.rangeStart, lo, hi);
    cache.rangeEnd = std::clamp(cache.rangeEnd, lo, hi);
    if (cache.rangeEnd < cache.rangeStart) cache.rangeEnd = cache.rangeStart;

    updatingTimeline_ = true;
    if (startEdit_) startEdit_->setText(QString::number(cache.rangeStart));
    if (endEdit_) endEdit_->setText(QString::number(cache.rangeEnd));
    scrubber_->setRange(cache.rangeStart, cache.rangeEnd);
    scrubber_->setFrame(std::clamp(sharedFrameNumber_, cache.rangeStart, cache.rangeEnd));
    updatingTimeline_ = false;
}

void TextureViewerWidget::updateInfoBar() {
    const SequenceCache& cache = activeCache();
    if (cache.frames.isEmpty() || frameIndex_ < 0 || frameIndex_ >= cache.frames.size()) {
        infoLabel_->setText(QStringLiteral("No texture"));
        return;
    }
    const FrameSlot& slot = cache.frames[frameIndex_];
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
                                .arg(cache.loadedCount)
                                .arg(cache.frames.size()));
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
                            .arg(cache.loadedCount)
                            .arg(cache.frames.size()));
}

void TextureViewerWidget::pushFrameToCanvas() {
    const SequenceCache& cache = activeCache();
    if (cache.frames.isEmpty() || frameIndex_ < 0 || frameIndex_ >= cache.frames.size()) {
        canvas_->clear();
        return;
    }
    const FrameSlot& slot = cache.frames[frameIndex_];
    if (!slot.error.isEmpty()) {
        canvas_->setPlaceholder(slot.error);
        return;
    }
    if (!slot.ready || slot.linearRgba.empty()) {
        canvas_->setPlaceholder(QStringLiteral("Loading…"));
        return;
    }
    const quint64 id = (quint64(cache.generation) << 32) ^ quint64(frameIndex_ + 1) ^
                       (quint64(slot.previewW) << 16) ^ quint64(slot.previewH) ^
                       (quint64(contentKind_) << 48);
    canvas_->setLinearImage(slot.linearRgba.data(), slot.previewW, slot.previewH, id);
}

void TextureViewerWidget::showCurrentFrame() {
    const SequenceCache& cache = activeCache();
    if (cache.frames.isEmpty()) return;
    frameIndex_ = std::clamp(frameIndex_, 0, int(cache.frames.size()) - 1);
    pushFrameToCanvas();
    updateInfoBar();
}

}  // namespace sol
