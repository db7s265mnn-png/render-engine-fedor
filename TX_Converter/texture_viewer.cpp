#include "texture_viewer.h"

#include <QComboBox>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QImageReader>
#include <QLabel>
#include <QMetaObject>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QPointer>
#include <QPushButton>
#include <QResizeEvent>
#include <QSignalBlocker>
#include <QThreadPool>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "core/image.h"
#include "core/math.h"
#include "io/image_io.h"
#include "io/ocio_util.h"
#include "scene/types.h"

namespace sol {
namespace {

constexpr int kMaxPreviewEdge = 2048;

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

void downscaleLinearRgb(const float* src, int srcW, int srcH, std::vector<float>& outRgb, int& outW,
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
    outRgb.resize(size_t(outW) * size_t(outH) * 3);
    for (int y = 0; y < outH; ++y) {
        const int sy = std::min(srcH - 1, y * stepY);
        const float* row = src + size_t(sy) * size_t(srcW) * 4;
        float* dst = outRgb.data() + size_t(y) * size_t(outW) * 3;
        for (int x = 0; x < outW; ++x) {
            const int sx = std::min(srcW - 1, x * stepX);
            const float* px = row + size_t(sx) * 4;
            dst[x * 3 + 0] = px[0];
            dst[x * 3 + 1] = px[1];
            dst[x * 3 + 2] = px[2];
        }
    }
}

void extractLinearFromFloatImage(const Image& image, std::vector<float>& outRgb, int& outW, int& outH) {
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
    downscaleLinearRgb(src, srcW, srcH, outRgb, outW, outH);
}

Vec3 classicDisplayRgb(Vec3 linear) {
    linear.x = linear.x / (1.0f + std::max(0.0f, linear.x));
    linear.y = linear.y / (1.0f + std::max(0.0f, linear.y));
    linear.z = linear.z / (1.0f + std::max(0.0f, linear.z));
    return linearToSrgbVec(linear);
}

QImage bakeDisplayImage(const float* linearRgb, int w, int h, ViewerDisplayMode mode, bool ocioUseEnv,
                        const QString& ocioConfigPath, int workingSpace) {
    QImage out(w, h, QImage::Format_RGB888);
    if (!linearRgb || w <= 0 || h <= 0) return out;

    bool ocioOk = false;
    if (mode == ViewerDisplayMode::OcioSrgbAces) {
        ocioEnsureConfig(ocioUseEnv, ocioConfigPath.toStdString());
        ocioOk = ocioPrepareView(workingSpace, kViewSrgbAces);
    }

    for (int y = 0; y < h; ++y) {
        uchar* line = out.scanLine(y);
        const float* row = linearRgb + size_t(y) * size_t(w) * 3;
        for (int x = 0; x < w; ++x) {
            Vec3 linear(row[x * 3 + 0], row[x * 3 + 1], row[x * 3 + 2]);
            Vec3 display =
                (mode == ViewerDisplayMode::OcioSrgbAces && ocioOk) ? ocioApplyViewPrepared(linear)
                                                                    : classicDisplayRgb(linear);
            uchar* px = line + size_t(x) * 3;
            px[0] = static_cast<uchar>(clampf(display.x, 0.0f, 1.0f) * 255.0f + 0.5f);
            px[1] = static_cast<uchar>(clampf(display.y, 0.0f, 1.0f) * 255.0f + 0.5f);
            px[2] = static_cast<uchar>(clampf(display.z, 0.0f, 1.0f) * 255.0f + 0.5f);
        }
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// FrameTimelineWidget
// ---------------------------------------------------------------------------

FrameTimelineWidget::FrameTimelineWidget(QWidget* parent) : QWidget(parent) {
    setMouseTracking(true);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void FrameTimelineWidget::setFrameCount(int count) {
    count_ = std::max(0, count);
    current_ = std::clamp(current_, 0, std::max(0, count_ - 1));
    update();
}

void FrameTimelineWidget::setCurrentFrame(int index) {
    if (count_ <= 0) {
        current_ = 0;
        update();
        return;
    }
    current_ = std::clamp(index, 0, count_ - 1);
    update();
}

QSize FrameTimelineWidget::sizeHint() const { return QSize(200, 22); }
QSize FrameTimelineWidget::minimumSizeHint() const { return QSize(40, 18); }

int FrameTimelineWidget::indexAtX(int x) const {
    if (count_ <= 0) return 0;
    const int left = 1;
    const int right = std::max(left + 1, width() - 1);
    const int span = right - left;
    if (count_ == 1) return 0;
    const float t = float(std::clamp(x, left, right) - left) / float(span);
    return std::clamp(int(std::lround(t * float(count_ - 1))), 0, count_ - 1);
}

void FrameTimelineWidget::seekToX(int x) {
    const int index = indexAtX(x);
    if (index == current_) return;
    current_ = index;
    update();
    emit frameChanged(current_);
}

void FrameTimelineWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && count_ > 0) seekToX(event->pos().x());
}

void FrameTimelineWidget::mouseMoveEvent(QMouseEvent* event) {
    if ((event->buttons() & Qt::LeftButton) && count_ > 0) seekToX(event->pos().x());
}

void FrameTimelineWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, false);

    const QRect r = rect();
    p.fillRect(r, QColor(22, 24, 27));
    p.setPen(QColor(50, 54, 60));
    p.drawRect(r.adjusted(0, 0, -1, -1));

    if (count_ <= 0) return;

    // Edge-to-edge track (1 px inset so the border stays visible).
    const int left = 1;
    const int right = std::max(left + 1, width() - 2);
    const int span = right - left;
    const int trackY = height() / 2;
    p.setPen(QPen(QColor(64, 70, 78), 2));
    p.drawLine(left, trackY, right, trackY);

    for (int i = 0; i < count_; ++i) {
        const float t = (count_ == 1) ? 0.5f : float(i) / float(count_ - 1);
        const int x = left + int(std::lround(t * float(span)));
        const bool active = (i == current_);
        const int tickH = active ? 8 : 5;
        p.setPen(QPen(active ? QColor(220, 180, 90) : QColor(130, 138, 148), active ? 2 : 1));
        p.drawLine(x, trackY - tickH, x, trackY + tickH);
    }

    const float t = (count_ == 1) ? 0.5f : float(current_) / float(count_ - 1);
    const int x = left + int(std::lround(t * float(span)));
    p.setBrush(QColor(232, 176, 64));
    p.setPen(Qt::NoPen);
    const QPoint pts[3] = {QPoint(x, trackY - 10), QPoint(x - 4, trackY - 4), QPoint(x + 4, trackY - 4)};
    p.drawPolygon(pts, 3);
}

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
    linearRgb_ = nullptr;
    width_ = height_ = 0;
    contentId_ = 0;
    invalidateDisplayCache();
    placeholder_ = QStringLiteral("No texture");
    update();
}

void FloatPreviewCanvas::setPlaceholder(const QString& text) {
    linearRgb_ = nullptr;
    width_ = height_ = 0;
    contentId_ = 0;
    invalidateDisplayCache();
    placeholder_ = text;
    update();
}

void FloatPreviewCanvas::setLinearImage(const float* rgb, int width, int height, quint64 contentId) {
    linearRgb_ = rgb;
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

void FloatPreviewCanvas::setDisplayMode(ViewerDisplayMode mode, bool ocioUseEnv,
                                        const QString& ocioConfigPath, int workingSpace) {
    displayMode_ = mode;
    ocioUseEnv_ = ocioUseEnv;
    ocioConfigPath_ = ocioConfigPath;
    workingSpace_ = workingSpace;
    // Invalidate only the 8-bit view cache — float source stays put (no laggy rebake of sequence).
    invalidateDisplayCache();
    update();
}

void FloatPreviewCanvas::invalidateDisplayCache() {
    displayCache_ = {};
    displayCacheId_ = 0;
}

void FloatPreviewCanvas::ensureDisplayCache() {
    if (!linearRgb_ || width_ <= 0 || height_ <= 0) return;
    if (!displayCache_.isNull() && displayCacheId_ == contentId_ &&
        displayCacheMode_ == displayMode_ && displayCacheWorking_ == workingSpace_ &&
        displayCacheOcioEnv_ == ocioUseEnv_ && displayCacheOcioPath_ == ocioConfigPath_) {
        return;
    }
    displayCache_ = bakeDisplayImage(linearRgb_, width_, height_, displayMode_, ocioUseEnv_,
                                     ocioConfigPath_, workingSpace_);
    displayCacheId_ = contentId_;
    displayCacheMode_ = displayMode_;
    displayCacheWorking_ = workingSpace_;
    displayCacheOcioEnv_ = ocioUseEnv_;
    displayCacheOcioPath_ = ocioConfigPath_;
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
    const double maxX = std::max(0.0, (w - double(this->width())) * 0.5 + 32.0);
    const double maxY = std::max(0.0, (h - double(this->height())) * 0.5 + 32.0);
    pan_.setX(std::clamp(pan_.x(), -maxX, maxX));
    pan_.setY(std::clamp(pan_.y(), -maxY, maxY));
}

void FloatPreviewCanvas::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), QColor(18, 20, 22));

    if (!linearRgb_ || width_ <= 0 || height_ <= 0) {
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
    if (!linearRgb_) {
        event->ignore();
        return;
    }
    fitted_ = false;
    const QPointF mouse = event->position();
    const QRectF before = imageRect();
    const double oldZoom = zoom_;
    const double factor = std::pow(1.0015, event->angleDelta().y());
    zoom_ = std::clamp(zoom_ * factor, 0.05, 64.0);

    // Keep the texel under the cursor stable.
    if (before.width() > 1.0 && before.height() > 1.0) {
        const double u = (mouse.x() - before.x()) / before.width();
        const double v = (mouse.y() - before.y()) / before.height();
        const QRectF after = imageRect();
        const QPointF afterPt(after.x() + u * after.width(), after.y() + v * after.height());
        pan_ += mouse - afterPt;
    }
    (void)oldZoom;
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
        downscaleLinearRgb(rgba.data(), qimage.width(), qimage.height(), payload.linearRgb,
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
    extractLinearFromFloatImage(image, payload.linearRgb, payload.previewW, payload.previewH);
    return payload;
}

TextureViewerWidget::TextureViewerWidget(QWidget* parent) : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(4);

    auto* modeRow = new QHBoxLayout();
    modeRow->setContentsMargins(0, 0, 0, 0);
    modeRow->setSpacing(6);
    modeRow->addWidget(new QLabel(QStringLiteral("Display")));
    displayCombo_ = new QComboBox();
    displayCombo_->addItem(QStringLiteral("Classic sRGB"), int(ViewerDisplayMode::ClassicSrgb));
    displayCombo_->addItem(QStringLiteral("OCIO sRGB (ACES)"), int(ViewerDisplayMode::OcioSrgbAces));
    displayCombo_->setMinimumWidth(160);
    modeRow->addWidget(displayCombo_);
    fitBtn_ = new QPushButton(QStringLiteral("Fit"));
    fitBtn_->setFixedWidth(44);
    fitBtn_->setToolTip(QStringLiteral("Fit image to view (double-click canvas)"));
    modeRow->addWidget(fitBtn_);
    zoomLabel_ = new QLabel(QStringLiteral("100%"));
    zoomLabel_->setMinimumWidth(48);
    zoomLabel_->setStyleSheet(QStringLiteral("color: #9aa0a6;"));
    modeRow->addWidget(zoomLabel_);
    modeRow->addStretch(1);
    root->addLayout(modeRow);

    canvas_ = new FloatPreviewCanvas(this);
    root->addWidget(canvas_, 1);

    // Timeline row: edge-to-edge, only prev/next chrome.
    auto* tlRow = new QHBoxLayout();
    tlRow->setContentsMargins(0, 0, 0, 0);
    tlRow->setSpacing(2);
    prevBtn_ = new QPushButton(QStringLiteral("◀"));
    prevBtn_->setFixedSize(22, 22);
    nextBtn_ = new QPushButton(QStringLiteral("▶"));
    nextBtn_->setFixedSize(22, 22);
    timeline_ = new FrameTimelineWidget();
    tlRow->addWidget(prevBtn_);
    tlRow->addWidget(timeline_, 1);
    tlRow->addWidget(nextBtn_);
    root->addLayout(tlRow);

    infoLabel_ = new QLabel(QStringLiteral("Drop a source path or click Preview"));
    infoLabel_->setWordWrap(true);
    infoLabel_->setStyleSheet(QStringLiteral("color: #b0b4ba;"));
    root->addWidget(infoLabel_);

    connect(timeline_, &FrameTimelineWidget::frameChanged, this, &TextureViewerWidget::setFrame);
    connect(prevBtn_, &QPushButton::clicked, this, &TextureViewerWidget::prevFrame);
    connect(nextBtn_, &QPushButton::clicked, this, &TextureViewerWidget::nextFrame);
    connect(fitBtn_, &QPushButton::clicked, this, &TextureViewerWidget::fitView);
    connect(canvas_, &FloatPreviewCanvas::zoomChanged, this, [this](double z) {
        zoomLabel_->setText(QStringLiteral("%1%").arg(int(std::lround(z * 100.0))));
    });
    connect(displayCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        setDisplayMode(ViewerDisplayMode(displayCombo_->currentData().toInt()));
    });
}

void TextureViewerWidget::setOcioConfig(bool useEnv, const QString& configPath) {
    ocioUseEnv_ = useEnv;
    ocioConfigPath_ = configPath.trimmed();
    canvas_->setDisplayMode(displayMode_, ocioUseEnv_, ocioConfigPath_, ocioWorkingSpace_);
}

void TextureViewerWidget::setDisplayMode(ViewerDisplayMode mode) {
    displayMode_ = mode;
    {
        const QSignalBlocker block(displayCombo_);
        const int idx = displayCombo_->findData(int(mode));
        if (idx >= 0) displayCombo_->setCurrentIndex(idx);
    }
    if (mode == ViewerDisplayMode::OcioSrgbAces) {
        const OcioStatus st = ocioEnsureConfig(ocioUseEnv_, ocioConfigPath_.toStdString());
        if (!st.configLoaded) {
            emit statusMessage(QStringLiteral("Viewer: %1").arg(QString::fromStdString(st.message)));
        }
    }
    // Live view transform on float buffer — only current frame's display cache rebuilds.
    canvas_->setDisplayMode(displayMode_, ocioUseEnv_, ocioConfigPath_, ocioWorkingSpace_);
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
        rebuildTimeline();
        infoLabel_->setText(QStringLiteral("No path"));
        emit statusMessage(QStringLiteral("Viewer: empty path"));
        return;
    }

    QStringList paths;
    QList<int> udims;
    QString pattern;
    std::vector<int> tiles;
    if (resolveUdimPattern(path, QString(), pattern, tiles) && !tiles.empty()) {
        for (int udim : tiles) {
            const QString tile = expandUdimToken(pattern, udim);
            if (!QFileInfo::exists(tile)) continue;
            paths.push_back(tile);
            udims.push_back(udim);
        }
    }

    if (paths.isEmpty()) {
        if (!QFileInfo::exists(path)) {
            rebuildTimeline();
            canvas_->setPlaceholder(QStringLiteral("File not found"));
            infoLabel_->setText(path);
            emit statusMessage(QStringLiteral("Viewer: file not found"));
            return;
        }
        paths.push_back(path);
        udims.push_back(0);
    }

    frames_.resize(paths.size());
    for (int i = 0; i < paths.size(); ++i) {
        frames_[i].path = paths[i];
        frames_[i].udim = udims[i];
    }

    ocioWorkingSpace_ = prefersAcesCgWorkingSpace(paths.first()) ? kWorkingSpaceAcesCg
                                                                 : kWorkingSpaceSrgbLinear;
    canvas_->setDisplayMode(displayMode_, ocioUseEnv_, ocioConfigPath_, ocioWorkingSpace_);

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
    slot.linearRgb = std::move(payload.linearRgb);
    slot.error = payload.error;
    slot.ready = payload.error.isEmpty() && !slot.linearRgb.empty();

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

void TextureViewerWidget::setFrame(int index) {
    if (frames_.isEmpty()) return;
    index = std::clamp(index, 0, int(frames_.size()) - 1);
    frameIndex_ = index;
    timeline_->setCurrentFrame(frameIndex_);
    showCurrentFrame();
}

void TextureViewerWidget::nextFrame() {
    if (frames_.size() <= 1) return;
    setFrame((frameIndex_ + 1) % frames_.size());
}

void TextureViewerWidget::prevFrame() {
    if (frames_.size() <= 1) return;
    setFrame((frameIndex_ - 1 + frames_.size()) % frames_.size());
}

void TextureViewerWidget::fitView() { canvas_->fitToView(); }

void TextureViewerWidget::rebuildTimeline() {
    const int n = frames_.size();
    prevBtn_->setEnabled(n > 1);
    nextBtn_->setEnabled(n > 1);
    timeline_->setFrameCount(n);
    timeline_->setCurrentFrame(std::clamp(frameIndex_, 0, std::max(0, n - 1)));
}

void TextureViewerWidget::updateInfoBar() {
    if (frames_.isEmpty() || frameIndex_ < 0 || frameIndex_ >= frames_.size()) {
        infoLabel_->setText(QStringLiteral("No texture"));
        return;
    }
    const FrameSlot& slot = frames_[frameIndex_];
    const QString modeBit = (displayMode_ == ViewerDisplayMode::OcioSrgbAces)
                                ? QStringLiteral("OCIO")
                                : QStringLiteral("sRGB");
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
    infoLabel_->setText(QStringLiteral("%1  ·  %2×%3  ·  %4  ·  %5  ·  float  ·  %6/%7")
                            .arg(QFileInfo(slot.path).fileName())
                            .arg(slot.sourceWidth)
                            .arg(slot.sourceHeight)
                            .arg(formatBytes(slot.fileBytes))
                            .arg(modeBit)
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
    if (!slot.ready || slot.linearRgb.empty()) {
        canvas_->setPlaceholder(QStringLiteral("Loading…"));
        return;
    }
    // contentId uniquely tags this frame buffer so display cache invalidates on scrub.
    const quint64 id = (quint64(loadGeneration_.load()) << 32) ^ quint64(frameIndex_ + 1) ^
                       (quint64(slot.previewW) << 16) ^ quint64(slot.previewH);
    canvas_->setLinearImage(slot.linearRgb.data(), slot.previewW, slot.previewH, id);
}

void TextureViewerWidget::showCurrentFrame() {
    if (frames_.isEmpty()) return;
    frameIndex_ = std::clamp(frameIndex_, 0, int(frames_.size()) - 1);
    timeline_->setCurrentFrame(frameIndex_);
    pushFrameToCanvas();
    updateInfoBar();
}

}  // namespace sol
