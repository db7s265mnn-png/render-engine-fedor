#include "texture_viewer.h"

#include <QFileInfo>
#include <QHBoxLayout>
#include <QImageReader>
#include <QLabel>
#include <QMetaObject>
#include <QPointer>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSlider>
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

namespace sol {
namespace {

constexpr int kMaxPreviewEdge = 2048;
constexpr int kCacheLimit = 12;

float linearToSrgbChannel(float c) {
    c = clampf(c, 0.0f, 1.0f);
    if (c <= 0.0031308f) return 12.92f * c;
    return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

Vec3 previewRgb(Vec3 linear) {
    // Soft clip for HDR / float .tx so bright values stay visible.
    linear.x = linear.x / (1.0f + std::max(0.0f, linear.x));
    linear.y = linear.y / (1.0f + std::max(0.0f, linear.y));
    linear.z = linear.z / (1.0f + std::max(0.0f, linear.z));
    return Vec3(linearToSrgbChannel(linear.x), linearToSrgbChannel(linear.y),
                linearToSrgbChannel(linear.z));
}

bool isFloatPreviewPath(const QString& path) {
    const QString ext = QFileInfo(path).suffix().toLower();
    return ext == QLatin1String("exr") || ext == QLatin1String("hdr") ||
           ext == QLatin1String("rgbe") || ext == QLatin1String("pic") ||
           ext == QLatin1String("tx") || ext == QLatin1String("tif") ||
           ext == QLatin1String("tiff");
}

QString formatBytes(qint64 bytes) {
    if (bytes < 1024) return QStringLiteral("%1 B").arg(bytes);
    if (bytes < 1024 * 1024) return QStringLiteral("%1 KB").arg(bytes / 1024.0, 0, 'f', 1);
    return QStringLiteral("%1 MB").arg(bytes / (1024.0 * 1024.0), 0, 'f', 1);
}

}  // namespace

QImage downscaleForPreview(QImage image) {
    if (image.isNull()) return {};
    const int edge = std::max(image.width(), image.height());
    if (edge <= kMaxPreviewEdge) return image.convertToFormat(QImage::Format_RGB888);
    return image
        .scaled(kMaxPreviewEdge, kMaxPreviewEdge, Qt::KeepAspectRatio, Qt::FastTransformation)
        .convertToFormat(QImage::Format_RGB888);
}

QImage floatImageToPreview(const Image& image) {
    if (image.empty()) return {};

    // Prefer a mip ≤ preview budget when available (.tx / maketx pyramids).
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

    int outW = srcW;
    int outH = srcH;
    int stepX = 1;
    int stepY = 1;
    const int edge = std::max(srcW, srcH);
    if (edge > kMaxPreviewEdge) {
        stepX = std::max(1, (srcW + kMaxPreviewEdge - 1) / kMaxPreviewEdge);
        stepY = std::max(1, (srcH + kMaxPreviewEdge - 1) / kMaxPreviewEdge);
        outW = std::max(1, srcW / stepX);
        outH = std::max(1, srcH / stepY);
    }

    QImage out(outW, outH, QImage::Format_RGB888);
    for (int y = 0; y < outH; ++y) {
        uchar* line = out.scanLine(y);
        const int sy = std::min(srcH - 1, y * stepY);
        const float* row = src + size_t(sy) * size_t(srcW) * 4;
        for (int x = 0; x < outW; ++x) {
            const int sx = std::min(srcW - 1, x * stepX);
            const float* px = row + size_t(sx) * 4;
            const Vec3 c = previewRgb(Vec3(px[0], px[1], px[2]));
            uchar* dst = line + size_t(x) * 3;
            dst[0] = static_cast<uchar>(c.x * 255.0f + 0.5f);
            dst[1] = static_cast<uchar>(c.y * 255.0f + 0.5f);
            dst[2] = static_cast<uchar>(c.z * 255.0f + 0.5f);
        }
    }
    return out;
}

TextureViewerWidget::PreviewResult TextureViewerWidget::loadPreviewImage(const QString& path) {
    PreviewResult result;
    const QFileInfo info(path);
    result.fileBytes = info.size();

    if (!info.exists()) {
        result.error = QStringLiteral("file not found: %1").arg(path);
        return result;
    }

    // LDR (PNG/JPEG/…): decode with Qt and downscale — no float Image round-trip.
    // File size may be ~40 MB; full RGBA decode is large, but we never keep 8K float buffers.
    if (!isFloatPreviewPath(path)) {
        QImageReader reader(path);
        reader.setAllocationLimit(0);
        reader.setAutoTransform(true);
        QImage qimage = reader.read();
        if (qimage.isNull()) {
            qimage.load(path);
        }
        if (qimage.isNull()) {
            result.error = QStringLiteral("unsupported or unreadable image: %1 (%2)")
                               .arg(path, reader.errorString());
            return result;
        }
        result.sourceWidth = qimage.width();
        result.sourceHeight = qimage.height();
        result.image = downscaleForPreview(std::move(qimage));
        return result;
    }

    Image image;
    std::string err;
    if (!loadImage(path.toStdString(), image, err, true)) {
        result.error = QString::fromStdString(err.empty() ? "load failed" : err);
        return result;
    }
    if (image.empty()) {
        result.error = QStringLiteral("empty image");
        return result;
    }
    result.sourceWidth = image.width();
    result.sourceHeight = image.height();
    result.image = floatImageToPreview(image);
    return result;
}

TextureViewerWidget::TextureViewerWidget(QWidget* parent) : QWidget(parent) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(6);

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
    timeline_ = new QSlider(Qt::Horizontal);
    timeline_->setMinimum(0);
    timeline_->setMaximum(0);
    timeline_->setEnabled(false);
    frameLabel_ = new QLabel(QStringLiteral("—"));
    frameLabel_->setMinimumWidth(140);
    tlRow->addWidget(prevBtn_);
    tlRow->addWidget(timeline_, 1);
    tlRow->addWidget(nextBtn_);
    tlRow->addWidget(frameLabel_);
    root->addLayout(tlRow);

    resizeDebounce_ = new QTimer(this);
    resizeDebounce_->setSingleShot(true);
    resizeDebounce_->setInterval(40);
    connect(resizeDebounce_, &QTimer::timeout, this, &TextureViewerWidget::updateImageLabel);

    connect(timeline_, &QSlider::valueChanged, this, &TextureViewerWidget::setFrame);
    connect(prevBtn_, &QPushButton::clicked, this, &TextureViewerWidget::prevFrame);
    connect(nextBtn_, &QPushButton::clicked, this, &TextureViewerWidget::nextFrame);
}

void TextureViewerWidget::setSourcePath(const QString& pathIn) {
    paths_.clear();
    udims_.clear();
    frameIndex_ = 0;
    previewImage_ = {};
    sourceWidth_ = sourceHeight_ = 0;
    fileBytes_ = 0;
    loadedPath_.clear();
    cache_.clear();
    cacheOrder_.clear();
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

    QString pattern;
    std::vector<int> tiles;
    if (resolveUdimPattern(path, QString(), pattern, tiles) && !tiles.empty()) {
        for (int udim : tiles) {
            const QString tile = expandUdimToken(pattern, udim);
            if (!QFileInfo::exists(tile)) continue;
            paths_.push_back(tile);
            udims_.push_back(udim);
        }
    }

    if (paths_.isEmpty()) {
        if (!QFileInfo::exists(path)) {
            rebuildTimeline();
            imageLabel_->setText(QStringLiteral("File not found"));
            imageLabel_->setPixmap({});
            infoLabel_->setText(path);
            emit statusMessage(QStringLiteral("Viewer: file not found"));
            return;
        }
        paths_.push_back(path);
        udims_.push_back(0);
    }

    rebuildTimeline();
    showCurrentFrame(true);
}

QString TextureViewerWidget::currentPath() const {
    if (frameIndex_ < 0 || frameIndex_ >= paths_.size()) return {};
    return paths_[frameIndex_];
}

void TextureViewerWidget::setFrame(int index) {
    if (paths_.isEmpty()) return;
    index = std::clamp(index, 0, int(paths_.size()) - 1);
    if (index == frameIndex_ && !previewImage_.isNull() && loadedPath_ == paths_[index]) {
        updateImageLabel();
        return;
    }
    frameIndex_ = index;
    if (timeline_->value() != frameIndex_) {
        const QSignalBlocker block(timeline_);
        timeline_->setValue(frameIndex_);
    }
    showCurrentFrame(false);
}

void TextureViewerWidget::nextFrame() {
    if (paths_.size() <= 1) return;
    setFrame((frameIndex_ + 1) % paths_.size());
}

void TextureViewerWidget::prevFrame() {
    if (paths_.size() <= 1) return;
    setFrame((frameIndex_ - 1 + paths_.size()) % paths_.size());
}

void TextureViewerWidget::fitView() { updateImageLabel(); }

void TextureViewerWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    resizeDebounce_->start();
}

void TextureViewerWidget::rebuildTimeline() {
    const int n = paths_.size();
    timeline_->setEnabled(n > 1);
    prevBtn_->setEnabled(n > 1);
    nextBtn_->setEnabled(n > 1);
    const QSignalBlocker block(timeline_);
    timeline_->setMinimum(0);
    timeline_->setMaximum(std::max(0, n - 1));
    timeline_->setValue(std::clamp(frameIndex_, 0, std::max(0, n - 1)));
}

void TextureViewerWidget::touchCache(const QString& path, const PreviewResult& result) {
    if (result.image.isNull()) return;
    if (!cache_.contains(path)) {
        cacheOrder_.push_back(path);
    } else {
        cacheOrder_.removeAll(path);
        cacheOrder_.push_back(path);
    }
    cache_.insert(path, result);
    while (cacheOrder_.size() > kCacheLimit) {
        const QString oldest = cacheOrder_.takeFirst();
        cache_.remove(oldest);
    }
}

void TextureViewerWidget::applyPreview(const QString& path, int frameIndex, quint64 generation,
                                       const PreviewResult& result) {
    if (generation != loadGeneration_.load()) return;
    if (frameIndex < 0 || frameIndex >= paths_.size() || paths_[frameIndex] != path) return;

    const int udim = (frameIndex < udims_.size()) ? udims_[frameIndex] : 0;
    loadedPath_ = path;
    previewImage_ = result.image;
    sourceWidth_ = result.sourceWidth;
    sourceHeight_ = result.sourceHeight;
    fileBytes_ = result.fileBytes;

    if (previewImage_.isNull()) {
        imageLabel_->setPixmap({});
        imageLabel_->setText(result.error.isEmpty() ? QStringLiteral("Load failed") : result.error);
        infoLabel_->setText(path);
        emit statusMessage(QStringLiteral("Viewer: %1").arg(result.error));
        frameLabel_->setText(udim > 0 ? QStringLiteral("UDIM %1").arg(udim) : QStringLiteral("—"));
        return;
    }

    touchCache(path, result);

    const QString sizeBit = formatBytes(fileBytes_);
    if (udim > 0) {
        frameLabel_->setText(QStringLiteral("UDIM %1 (%2/%3)")
                                 .arg(udim)
                                 .arg(frameIndex + 1)
                                 .arg(paths_.size()));
        infoLabel_->setText(QStringLiteral("%1  ·  %2×%3  ·  %4  ·  tile %5/%6")
                                .arg(QFileInfo(path).fileName())
                                .arg(sourceWidth_)
                                .arg(sourceHeight_)
                                .arg(sizeBit)
                                .arg(frameIndex + 1)
                                .arg(paths_.size()));
    } else {
        frameLabel_->setText(QStringLiteral("1/1"));
        infoLabel_->setText(QStringLiteral("%1  ·  %2×%3  ·  %4")
                                .arg(QFileInfo(path).fileName())
                                .arg(sourceWidth_)
                                .arg(sourceHeight_)
                                .arg(sizeBit));
    }
    updateImageLabel();
    emit statusMessage(QStringLiteral("Viewer: %1").arg(QFileInfo(path).fileName()));
}

void TextureViewerWidget::showCurrentFrame(bool forceReload) {
    if (paths_.isEmpty()) return;
    frameIndex_ = std::clamp(frameIndex_, 0, int(paths_.size()) - 1);
    const QString path = paths_[frameIndex_];
    const int frameIndex = frameIndex_;

    if (!forceReload && path == loadedPath_ && !previewImage_.isNull()) {
        updateImageLabel();
        return;
    }

    if (!forceReload) {
        const auto it = cache_.constFind(path);
        if (it != cache_.cend() && !it->image.isNull()) {
            applyPreview(path, frameIndex, loadGeneration_.load(), *it);
            return;
        }
    }

    const quint64 generation = ++loadGeneration_;
    imageLabel_->setText(QStringLiteral("Loading…"));
    imageLabel_->setPixmap({});
    emit statusMessage(QStringLiteral("Viewer: loading %1…").arg(QFileInfo(path).fileName()));

    QPointer<TextureViewerWidget> self(this);
    QThreadPool::globalInstance()->start([self, path, frameIndex, generation]() {
        PreviewResult result = loadPreviewImage(path);
        if (!self) return;
        QMetaObject::invokeMethod(
            self,
            [self, path, frameIndex, generation, result = std::move(result)]() mutable {
                if (!self) return;
                self->applyPreview(path, frameIndex, generation, result);
            },
            Qt::QueuedConnection);
    });
}

void TextureViewerWidget::updateImageLabel() {
    if (previewImage_.isNull()) return;
    const QSize viewport = scroll_->viewport()->size();
    if (viewport.width() < 8 || viewport.height() < 8) {
        imageLabel_->setPixmap(QPixmap::fromImage(previewImage_));
        imageLabel_->adjustSize();
        return;
    }
    // previewImage_ is already ≤ 2K; FastTransformation is enough for fit-to-view.
    const QImage scaled =
        previewImage_.scaled(viewport, Qt::KeepAspectRatio, Qt::FastTransformation);
    imageLabel_->setPixmap(QPixmap::fromImage(scaled));
    imageLabel_->adjustSize();
}

}  // namespace sol
