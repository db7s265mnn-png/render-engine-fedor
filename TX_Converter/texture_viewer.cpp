#include "texture_viewer.h"

#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSlider>
#include <QSignalBlocker>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/image.h"
#include "core/math.h"
#include "io/image_io.h"

namespace sol {
namespace {

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

QImage imageToPreviewQImage(const Image& image) {
    if (image.empty()) return {};
    // Level 0 only (mip pyramid lives after level-0 floats in Image).
    const int w = image.width();
    const int h = image.height();
    QImage out(w, h, QImage::Format_RGB888);
    for (int y = 0; y < h; ++y) {
        uchar* line = out.scanLine(y);
        for (int x = 0; x < w; ++x) {
            const Vec3 c = previewRgb(image.rgb(x, y));
            uchar* px = line + size_t(x) * 3;
            px[0] = static_cast<uchar>(c.x * 255.0f + 0.5f);
            px[1] = static_cast<uchar>(c.y * 255.0f + 0.5f);
            px[2] = static_cast<uchar>(c.z * 255.0f + 0.5f);
        }
    }
    return out;
}

}  // namespace

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

    connect(timeline_, &QSlider::valueChanged, this, &TextureViewerWidget::setFrame);
    connect(prevBtn_, &QPushButton::clicked, this, &TextureViewerWidget::prevFrame);
    connect(nextBtn_, &QPushButton::clicked, this, &TextureViewerWidget::nextFrame);
}

void TextureViewerWidget::setSourcePath(const QString& pathIn) {
    paths_.clear();
    udims_.clear();
    frameIndex_ = 0;
    fullImage_ = {};
    loadedPath_.clear();

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
    if (index == frameIndex_ && !fullImage_.isNull() && loadedPath_ == paths_[index]) {
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
    updateImageLabel();
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

void TextureViewerWidget::showCurrentFrame(bool forceReload) {
    if (paths_.isEmpty()) return;
    frameIndex_ = std::clamp(frameIndex_, 0, int(paths_.size()) - 1);
    const QString path = paths_[frameIndex_];
    const int udim = (frameIndex_ < udims_.size()) ? udims_[frameIndex_] : 0;

    if (!forceReload && path == loadedPath_ && !fullImage_.isNull()) {
        updateImageLabel();
    } else {
        QString error;
        emit statusMessage(QStringLiteral("Viewer: loading %1…").arg(QFileInfo(path).fileName()));
        fullImage_ = loadPreviewImage(path, error);
        loadedPath_ = path;
        if (fullImage_.isNull()) {
            imageLabel_->setPixmap({});
            imageLabel_->setText(error.isEmpty() ? QStringLiteral("Load failed") : error);
            infoLabel_->setText(path);
            emit statusMessage(QStringLiteral("Viewer: %1").arg(error));
            frameLabel_->setText(udim > 0 ? QStringLiteral("UDIM %1").arg(udim) : QStringLiteral("—"));
            return;
        }
    }

    if (udim > 0) {
        frameLabel_->setText(QStringLiteral("UDIM %1 (%2/%3)")
                                 .arg(udim)
                                 .arg(frameIndex_ + 1)
                                 .arg(paths_.size()));
        infoLabel_->setText(QStringLiteral("%1  ·  %2×%3  ·  tile %4/%5")
                                .arg(QFileInfo(path).fileName())
                                .arg(fullImage_.width())
                                .arg(fullImage_.height())
                                .arg(frameIndex_ + 1)
                                .arg(paths_.size()));
    } else {
        frameLabel_->setText(QStringLiteral("1/1"));
        infoLabel_->setText(QStringLiteral("%1  ·  %2×%3")
                                .arg(QFileInfo(path).fileName())
                                .arg(fullImage_.width())
                                .arg(fullImage_.height()));
    }
    updateImageLabel();
    emit statusMessage(QStringLiteral("Viewer: %1").arg(QFileInfo(path).fileName()));
}

QImage TextureViewerWidget::loadPreviewImage(const QString& path, QString& error) const {
    Image image;
    std::string err;
    // Colour preview: LDR sRGB→linear on load; float .tx/EXR stay linear. Level 0 for .tx.
    if (!loadImage(path.toStdString(), image, err, true)) {
        error = QString::fromStdString(err.empty() ? "load failed" : err);
        return {};
    }
    if (image.empty()) {
        error = QStringLiteral("empty image");
        return {};
    }
    return imageToPreviewQImage(image);
}

void TextureViewerWidget::updateImageLabel() {
    if (fullImage_.isNull()) return;
    const QSize viewport = scroll_->viewport()->size();
    if (viewport.width() < 8 || viewport.height() < 8) {
        imageLabel_->setPixmap(QPixmap::fromImage(fullImage_));
        imageLabel_->adjustSize();
        return;
    }
    const QImage scaled =
        fullImage_.scaled(viewport, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    imageLabel_->setPixmap(QPixmap::fromImage(scaled));
    imageLabel_->adjustSize();
}

}  // namespace sol
