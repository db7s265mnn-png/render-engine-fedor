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
#include <QScrollArea>
#include <QSignalBlocker>
#include <QThreadPool>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
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
    const QString ext = QFileInfo(path).suffix().toLower();
    // Converted Arnold/OIIO .tx tiles are ACEScg in this tool.
    return ext == QLatin1String("tx");
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
    // Soft clip for HDR highlights, then classic sRGB encode.
    linear.x = linear.x / (1.0f + std::max(0.0f, linear.x));
    linear.y = linear.y / (1.0f + std::max(0.0f, linear.y));
    linear.z = linear.z / (1.0f + std::max(0.0f, linear.z));
    return linearToSrgbVec(linear);
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

void FrameTimelineWidget::setTickLabels(const QStringList& labels) {
    labels_ = labels;
    update();
}

QSize FrameTimelineWidget::sizeHint() const { return QSize(320, 36); }
QSize FrameTimelineWidget::minimumSizeHint() const { return QSize(120, 28); }

void FrameTimelineWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    update();
}

int FrameTimelineWidget::indexAtX(int x) const {
    if (count_ <= 0) return 0;
    const int left = 8;
    const int right = width() - 8;
    const int span = std::max(1, right - left);
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
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRect r = rect().adjusted(1, 1, -1, -1);
    p.fillRect(r, QColor(26, 28, 31));
    p.setPen(QColor(58, 62, 68));
    p.drawRoundedRect(r, 3, 3);

    if (count_ <= 0) {
        p.setPen(QColor(120, 126, 134));
        p.drawText(r, Qt::AlignCenter, QStringLiteral("no frames"));
        return;
    }

    const int left = 8;
    const int right = width() - 8;
    const int span = std::max(1, right - left);
    const int trackY = height() / 2;
    p.setPen(QPen(QColor(70, 76, 84), 2));
    p.drawLine(left, trackY, right, trackY);

    const bool drawLabels = count_ <= 24 && !labels_.isEmpty();
    for (int i = 0; i < count_; ++i) {
        const float t = (count_ == 1) ? 0.5f : float(i) / float(count_ - 1);
        const int x = left + int(std::lround(t * float(span)));
        const bool active = (i == current_);
        const int tickH = active ? 12 : 7;
        p.setPen(QPen(active ? QColor(220, 180, 90) : QColor(140, 148, 158), active ? 2 : 1));
        p.drawLine(x, trackY - tickH, x, trackY + tickH);

        if (drawLabels && i < labels_.size() && !labels_[i].isEmpty()) {
            p.setPen(active ? QColor(230, 200, 120) : QColor(110, 116, 124));
            QFont f = font();
            f.setPointSize(std::max(7, f.pointSize() - 2));
            p.setFont(f);
            p.drawText(QRect(x - 18, trackY + 8, 36, 12), Qt::AlignHCenter | Qt::AlignTop, labels_[i]);
        }
    }

    // Playhead
    {
        const float t = (count_ == 1) ? 0.5f : float(current_) / float(count_ - 1);
        const int x = left + int(std::lround(t * float(span)));
        p.setBrush(QColor(232, 176, 64));
        p.setPen(Qt::NoPen);
        const QPoint pts[3] = {QPoint(x, trackY - 14), QPoint(x - 5, trackY - 6),
                               QPoint(x + 5, trackY - 6)};
        p.drawPolygon(pts, 3);
    }
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
        // Float TX/EXR: leave linear. For 8-bit TIFF fall back with sRGB decode.
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

void TextureViewerWidget::bakeDisplay(FrameSlot& slot) const {
    slot.display = {};
    if (!slot.ready || slot.linearRgb.empty() || slot.previewW <= 0 || slot.previewH <= 0) return;

    QImage out(slot.previewW, slot.previewH, QImage::Format_RGB888);
    const bool useOcio = displayMode_ == ViewerDisplayMode::OcioSrgbAces;
    bool ocioOk = false;
    if (useOcio) {
        ocioEnsureConfig(ocioUseEnv_, ocioConfigPath_.toStdString());
        ocioOk = ocioPrepareView(ocioWorkingSpace_, kViewSrgbAces);
    }

    for (int y = 0; y < slot.previewH; ++y) {
        uchar* line = out.scanLine(y);
        const float* row = slot.linearRgb.data() + size_t(y) * size_t(slot.previewW) * 3;
        for (int x = 0; x < slot.previewW; ++x) {
            Vec3 linear(row[x * 3 + 0], row[x * 3 + 1], row[x * 3 + 2]);
            Vec3 display;
            if (useOcio && ocioOk) {
                display = ocioApplyViewPrepared(linear);
            } else {
                display = classicDisplayRgb(linear);
            }
            uchar* px = line + size_t(x) * 3;
            px[0] = static_cast<uchar>(clampf(display.x, 0.0f, 1.0f) * 255.0f + 0.5f);
            px[1] = static_cast<uchar>(clampf(display.y, 0.0f, 1.0f) * 255.0f + 0.5f);
            px[2] = static_cast<uchar>(clampf(display.z, 0.0f, 1.0f) * 255.0f + 0.5f);
        }
    }
    slot.display = std::move(out);
}

TextureViewerWidget::TextureViewerWidget(QWidget* parent) : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(6);

    auto* modeRow = new QHBoxLayout();
    modeRow->addWidget(new QLabel(QStringLiteral("Display")));
    displayCombo_ = new QComboBox();
    displayCombo_->addItem(QStringLiteral("Classic sRGB"), int(ViewerDisplayMode::ClassicSrgb));
    displayCombo_->addItem(QStringLiteral("OCIO sRGB (ACES)"), int(ViewerDisplayMode::OcioSrgbAces));
    displayCombo_->setMinimumWidth(180);
    modeRow->addWidget(displayCombo_);
    modeRow->addStretch(1);
    root->addLayout(modeRow);

    scroll_ = new QScrollArea(this);
    scroll_->setWidgetResizable(true);
    scroll_->setAlignment(Qt::AlignCenter);
    scroll_->setMinimumHeight(220);
    scroll_->setStyleSheet(QStringLiteral("QScrollArea { background: #1a1c1f; border: 1px solid #3a3e44; }"));

    imageLabel_ = new QLabel(scroll_);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setMinimumSize(64, 64);
    imageLabel_->setText(QStringLiteral("No texture"));
    imageLabel_->setStyleSheet(QStringLiteral("color: #8a9098; background: #1a1c1f;"));
    scroll_->setWidget(imageLabel_);
    root->addWidget(scroll_, 1);

    infoLabel_ = new QLabel(QStringLiteral("Drop a source path or click Preview"));
    infoLabel_->setWordWrap(true);
    infoLabel_->setStyleSheet(QStringLiteral("color: #b0b4ba;"));
    root->addWidget(infoLabel_);

    auto* tlRow = new QHBoxLayout();
    prevBtn_ = new QPushButton(QStringLiteral("◀"));
    prevBtn_->setFixedWidth(36);
    nextBtn_ = new QPushButton(QStringLiteral("▶"));
    nextBtn_->setFixedWidth(36);
    timeline_ = new FrameTimelineWidget();
    frameLabel_ = new QLabel(QStringLiteral("—"));
    frameLabel_->setMinimumWidth(150);
    tlRow->addWidget(prevBtn_);
    tlRow->addWidget(timeline_, 1);
    tlRow->addWidget(nextBtn_);
    tlRow->addWidget(frameLabel_);
    root->addLayout(tlRow);

    resizeDebounce_ = new QTimer(this);
    resizeDebounce_->setSingleShot(true);
    resizeDebounce_->setInterval(40);
    connect(resizeDebounce_, &QTimer::timeout, this, &TextureViewerWidget::updateImageLabel);

    connect(timeline_, &FrameTimelineWidget::frameChanged, this, &TextureViewerWidget::setFrame);
    connect(prevBtn_, &QPushButton::clicked, this, &TextureViewerWidget::prevFrame);
    connect(nextBtn_, &QPushButton::clicked, this, &TextureViewerWidget::nextFrame);
    connect(displayCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        const ViewerDisplayMode mode =
            ViewerDisplayMode(displayCombo_->currentData().toInt());
        setDisplayMode(mode);
    });
}

void TextureViewerWidget::setOcioConfig(bool useEnv, const QString& configPath) {
    ocioUseEnv_ = useEnv;
    ocioConfigPath_ = configPath.trimmed();
    if (displayMode_ == ViewerDisplayMode::OcioSrgbAces) {
        ocioEnsureConfig(ocioUseEnv_, ocioConfigPath_.toStdString());
        rebakeAllDisplays();
        showCurrentFrame();
        refreshDisplayModeUi();
    }
}

void TextureViewerWidget::setDisplayMode(ViewerDisplayMode mode) {
    if (displayMode_ == mode && displayCombo_->currentData().toInt() == int(mode)) {
        // Still rebake if OCIO just became available — fall through only on change.
    }
    const bool changed = displayMode_ != mode;
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
    if (changed || mode == ViewerDisplayMode::OcioSrgbAces) {
        rebakeAllDisplays();
        showCurrentFrame();
    }
    refreshDisplayModeUi();
}

void TextureViewerWidget::refreshDisplayModeUi() {
    if (displayMode_ != ViewerDisplayMode::OcioSrgbAces) return;
    if (!ocioLibraryAvailable()) {
        emit statusMessage(QStringLiteral("Viewer: OCIO library not linked — using classic sRGB encode"));
    }
}

void TextureViewerWidget::rebakeAllDisplays() {
    // Prepare OCIO once for the batch.
    if (displayMode_ == ViewerDisplayMode::OcioSrgbAces) {
        ocioEnsureConfig(ocioUseEnv_, ocioConfigPath_.toStdString());
        ocioPrepareView(ocioWorkingSpace_, kViewSrgbAces);
    }
    for (FrameSlot& slot : frames_) {
        if (slot.ready) bakeDisplay(slot);
    }
}

void TextureViewerWidget::setSourcePath(const QString& pathIn) {
    frames_.clear();
    frameIndex_ = 0;
    loadedCount_ = 0;
    ++loadGeneration_;

    const QString path = pathIn.trimmed();
    if (path.isEmpty()) {
        rebuildTimeline();
        imageLabel_->setText(QStringLiteral("No texture"));
        imageLabel_->setPixmap({});
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
            imageLabel_->setText(QStringLiteral("File not found"));
            imageLabel_->setPixmap({});
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

    // Working space: ACEScg for .tx sequence, else scene-linear sRGB.
    ocioWorkingSpace_ = prefersAcesCgWorkingSpace(paths.first()) ? kWorkingSpaceAcesCg
                                                                 : kWorkingSpaceSrgbLinear;

    rebuildTimeline();
    imageLabel_->setText(QStringLiteral("Loading sequence…"));
    imageLabel_->setPixmap({});
    emit statusMessage(QStringLiteral("Viewer: loading %1 tile(s)…").arg(frames_.size()));
    startPreloadAll();
}

void TextureViewerWidget::startPreloadAll() {
    const quint64 generation = loadGeneration_.load();
    QPointer<TextureViewerWidget> self(this);
    // Cap concurrency a bit so 8K PNG decode doesn't thrash RAM.
    const int ideal = std::max(1, QThreadPool::globalInstance()->maxThreadCount());
    QThreadPool::globalInstance()->setMaxThreadCount(ideal);

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
    if (slot.ready) bakeDisplay(slot);

    if (payload.index == frameIndex_) showCurrentFrame();

    if (loadedCount_ >= frames_.size()) {
        emit statusMessage(QStringLiteral("Viewer: sequence ready (%1 tile(s))").arg(frames_.size()));
    } else {
        emit statusMessage(QStringLiteral("Viewer: loaded %1/%2…")
                               .arg(loadedCount_)
                               .arg(frames_.size()));
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
    if (index == frameIndex_) {
        showCurrentFrame();
        return;
    }
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

void TextureViewerWidget::fitView() { updateImageLabel(); }

void TextureViewerWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    resizeDebounce_->start();
}

void TextureViewerWidget::rebuildTimeline() {
    const int n = frames_.size();
    prevBtn_->setEnabled(n > 1);
    nextBtn_->setEnabled(n > 1);
    timeline_->setFrameCount(n);
    timeline_->setCurrentFrame(std::clamp(frameIndex_, 0, std::max(0, n - 1)));

    QStringList labels;
    labels.reserve(n);
    for (const FrameSlot& slot : frames_) {
        if (slot.udim > 0) labels << QString::number(slot.udim);
        else labels << QString();
    }
    timeline_->setTickLabels(labels);
}

void TextureViewerWidget::updateInfoBar() {
    if (frames_.isEmpty() || frameIndex_ < 0 || frameIndex_ >= frames_.size()) {
        frameLabel_->setText(QStringLiteral("—"));
        return;
    }
    const FrameSlot& slot = frames_[frameIndex_];
    const QString modeBit = (displayMode_ == ViewerDisplayMode::OcioSrgbAces)
                                ? QStringLiteral("OCIO")
                                : QStringLiteral("sRGB");
    if (slot.udim > 0) {
        frameLabel_->setText(QStringLiteral("UDIM %1 (%2/%3)")
                                 .arg(slot.udim)
                                 .arg(frameIndex_ + 1)
                                 .arg(frames_.size()));
    } else {
        frameLabel_->setText(QStringLiteral("%1/%2").arg(frameIndex_ + 1).arg(frames_.size()));
    }

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
    infoLabel_->setText(QStringLiteral("%1  ·  %2×%3  ·  %4  ·  %5  ·  %6/%7 ready")
                            .arg(QFileInfo(slot.path).fileName())
                            .arg(slot.sourceWidth)
                            .arg(slot.sourceHeight)
                            .arg(formatBytes(slot.fileBytes))
                            .arg(modeBit)
                            .arg(loadedCount_)
                            .arg(frames_.size()));
}

void TextureViewerWidget::showCurrentFrame() {
    if (frames_.isEmpty()) return;
    frameIndex_ = std::clamp(frameIndex_, 0, int(frames_.size()) - 1);
    timeline_->setCurrentFrame(frameIndex_);
    updateInfoBar();

    const FrameSlot& slot = frames_[frameIndex_];
    if (!slot.error.isEmpty()) {
        imageLabel_->setPixmap({});
        imageLabel_->setText(slot.error);
        return;
    }
    if (!slot.ready || slot.display.isNull()) {
        imageLabel_->setPixmap({});
        imageLabel_->setText(QStringLiteral("Loading…"));
        return;
    }
    updateImageLabel();
}

void TextureViewerWidget::updateImageLabel() {
    if (frames_.isEmpty() || frameIndex_ < 0 || frameIndex_ >= frames_.size()) return;
    const FrameSlot& slot = frames_[frameIndex_];
    if (slot.display.isNull()) return;

    const QSize viewport = scroll_->viewport()->size();
    if (viewport.width() < 8 || viewport.height() < 8) {
        imageLabel_->setPixmap(QPixmap::fromImage(slot.display));
        imageLabel_->adjustSize();
        return;
    }
    const QImage scaled =
        slot.display.scaled(viewport, Qt::KeepAspectRatio, Qt::FastTransformation);
    imageLabel_->setPixmap(QPixmap::fromImage(scaled));
    imageLabel_->adjustSize();
}

}  // namespace sol
