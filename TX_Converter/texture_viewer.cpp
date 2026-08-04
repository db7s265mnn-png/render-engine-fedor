#include "texture_viewer.h"

#include <QAbstractSpinBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QImageReader>
#include <QIntValidator>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QPointer>
#include <QPushButton>
#include <QResizeEvent>
#include <QSignalBlocker>
#include <QSlider>
#include <QThreadPool>
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
constexpr int kFrameBoxWidth = 40;
constexpr int kFrameBoxHeight = 16;

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

bool gradeEquals(const ViewerGrade& a, const ViewerGrade& b) {
    return a.brightness == b.brightness && a.contrast == b.contrast && a.gamma == b.gamma;
}

Vec3 applyGrade(Vec3 c, const ViewerGrade& grade) {
    // Contrast around mid-grey, then brightness offset, then gamma.
    c.x = (c.x - 0.5f) * grade.contrast + 0.5f + grade.brightness;
    c.y = (c.y - 0.5f) * grade.contrast + 0.5f + grade.brightness;
    c.z = (c.z - 0.5f) * grade.contrast + 0.5f + grade.brightness;
    const float g = std::max(0.01f, grade.gamma);
    const float invG = 1.0f / g;
    c.x = std::pow(std::max(0.0f, c.x), invG);
    c.y = std::pow(std::max(0.0f, c.y), invG);
    c.z = std::pow(std::max(0.0f, c.z), invG);
    return c;
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
                        const QString& ocioConfigPath, int workingSpace, const ViewerGrade& grade) {
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
            display = applyGrade(display, grade);
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
    if (!linearRgb_ || width_ <= 0 || height_ <= 0) return;
    if (!displayCache_.isNull() && displayCacheId_ == contentId_ &&
        displayCacheMode_ == displayMode_ && displayCacheWorking_ == workingSpace_ &&
        displayCacheOcioEnv_ == ocioUseEnv_ && displayCacheOcioPath_ == ocioConfigPath_ &&
        gradeEquals(displayCacheGrade_, grade_)) {
        return;
    }
    displayCache_ = bakeDisplayImage(linearRgb_, width_, height_, displayMode_, ocioUseEnv_,
                                     ocioConfigPath_, workingSpace_, grade_);
    displayCacheId_ = contentId_;
    displayCacheMode_ = displayMode_;
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

QLineEdit* TextureViewerWidget::makeRangeEdit(const QString& tip) {
    auto* edit = new QLineEdit(this);
    edit->setAlignment(Qt::AlignCenter);
    edit->setFixedSize(kFrameBoxWidth, kFrameBoxHeight);
    edit->setMaxLength(6);
    edit->setValidator(new QIntValidator(1, 999999, edit));
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

    auto* modeRow = new QHBoxLayout();
    modeRow->setContentsMargins(0, 0, 0, 0);
    modeRow->setSpacing(6);
    modeRow->addWidget(new QLabel(QStringLiteral("Display")));
    displayCombo_ = new QComboBox();
    displayCombo_->addItem(QStringLiteral("Classic sRGB"), int(ViewerDisplayMode::ClassicSrgb));
    displayCombo_->addItem(QStringLiteral("OCIO sRGB (ACES)"), int(ViewerDisplayMode::OcioSrgbAces));
    displayCombo_->setMinimumWidth(150);
    modeRow->addWidget(displayCombo_);
    modeRow->addWidget(new QLabel(QStringLiteral("View")));
    contentCombo_ = new QComboBox();
    contentCombo_->addItem(QStringLiteral("Source Images"), int(ViewerContentKind::SourceImages));
    contentCombo_->addItem(QStringLiteral("Converted TX"), int(ViewerContentKind::ConvertedTx));
    contentCombo_->setMinimumWidth(130);
    modeRow->addWidget(contentCombo_);
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

    auto* gradeRow = new QHBoxLayout();
    gradeRow->setContentsMargins(0, 0, 0, 0);
    gradeRow->setSpacing(4);
    const QString gradeStyle = QStringLiteral("color: #9aa0a6;");
    auto addGrade = [&](const QString& name, QDoubleSpinBox** spin, QSlider** slider, double minV,
                        double maxV, double step, double value, const QString& tip) {
        auto* lab = new QLabel(name);
        lab->setStyleSheet(gradeStyle);
        gradeRow->addWidget(lab);
        *spin = makeGradeSpin(minV, maxV, step, value, tip);
        *slider = makeGradeSlider(this);
        const double t = (value - minV) / std::max(1e-9, maxV - minV);
        (*slider)->setValue(int(std::lround(std::clamp(t, 0.0, 1.0) * 1000.0)));
        gradeRow->addWidget(*slider, 1);
        gradeRow->addWidget(*spin);
    };
    addGrade(QStringLiteral("Bright"), &brightnessSpin_, &brightnessSlider_, -1.0, 1.0, 0.05, 0.0,
             QStringLiteral("Brightness (−1 … +1)"));
    addGrade(QStringLiteral("Contrast"), &contrastSpin_, &contrastSlider_, 0.0, 3.0, 0.05, 1.0,
             QStringLiteral("Contrast (1 = neutral)"));
    addGrade(QStringLiteral("Gamma"), &gammaSpin_, &gammaSlider_, 0.20, 3.0, 0.05, 1.0,
             QStringLiteral("Display gamma (1 = neutral)"));
    gradeResetBtn_ = new QPushButton(QStringLiteral("Reset"));
    gradeResetBtn_->setFixedWidth(52);
    gradeResetBtn_->setToolTip(QStringLiteral("Reset brightness / contrast / gamma"));
    gradeRow->addWidget(gradeResetBtn_);
    root->addLayout(gradeRow);

    canvas_ = new FloatPreviewCanvas(this);
    root->addWidget(canvas_, 1);

    // Solstice-style scrub row: [start] [scrubber] [end] — no transport/play buttons.
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
    connect(gradeResetBtn_, &QPushButton::clicked, this, &TextureViewerWidget::resetGrade);
    connect(canvas_, &FloatPreviewCanvas::zoomChanged, this, [this](double z) {
        zoomLabel_->setText(QStringLiteral("%1%").arg(int(std::lround(z * 100.0))));
    });
    connect(displayCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        setDisplayMode(ViewerDisplayMode(displayCombo_->currentData().toInt()));
    });
    connect(contentCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
        contentKind_ = ViewerContentKind(contentCombo_->currentData().toInt());
        refreshFromPipeline();
    });
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
    wireGrade(brightnessSpin_, brightnessSlider_, -1.0, 1.0);
    wireGrade(contrastSpin_, contrastSlider_, 0.0, 3.0);
    wireGrade(gammaSpin_, gammaSlider_, 0.20, 3.0);

    rebuildTimeline();
}

void TextureViewerWidget::applyGradeFromUi() {
    grade_.brightness = float(brightnessSpin_->value());
    grade_.contrast = float(contrastSpin_->value());
    grade_.gamma = float(gammaSpin_->value());
    canvas_->setGrade(grade_);
}

void TextureViewerWidget::resetGrade() {
    const QSignalBlocker b0(brightnessSpin_);
    const QSignalBlocker b1(contrastSpin_);
    const QSignalBlocker b2(gammaSpin_);
    const QSignalBlocker s0(brightnessSlider_);
    const QSignalBlocker s1(contrastSlider_);
    const QSignalBlocker s2(gammaSlider_);
    brightnessSpin_->setValue(0.0);
    contrastSpin_->setValue(1.0);
    gammaSpin_->setValue(1.0);
    brightnessSlider_->setValue(500);
    contrastSlider_->setValue(int(std::lround((1.0 / 3.0) * 1000.0)));
    gammaSlider_->setValue(int(std::lround(((1.0 - 0.2) / (3.0 - 0.2)) * 1000.0)));
    applyGradeFromUi();
}

void TextureViewerWidget::setOcioConfig(bool useEnv, const QString& configPath) {
    ocioUseEnv_ = useEnv;
    ocioConfigPath_ = configPath.trimmed();
    canvas_->setDisplayMode(displayMode_, ocioUseEnv_, ocioConfigPath_, ocioWorkingSpace_);
}

void TextureViewerWidget::setPipelinePaths(const QString& sourcePath, const QString& outputFolder) {
    pipelineSource_ = sourcePath.trimmed();
    pipelineOutputFolder_ = outputFolder.trimmed();
    refreshFromPipeline();
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

QString TextureViewerWidget::guessConvertedTxPath(const QString& sourcePath,
                                                 const QString& outputFolder) {
    const QString src = sourcePath.trimmed();
    const QString outDir = outputFolder.trimmed();
    if (src.isEmpty() || outDir.isEmpty()) return {};

    QFileInfo info(src);
    QString name = info.fileName();
    if (pathHasUdimToken(name) || src.contains(QStringLiteral("$F"))) {
        const int dot = name.lastIndexOf(QLatin1Char('.'));
        if (dot > 0) name = name.left(dot) + QStringLiteral(".tx");
        else name += QStringLiteral(".tx");
    } else if (pathHasUdimToken(src)) {
        name = info.fileName();
        const int dot = name.lastIndexOf(QLatin1Char('.'));
        if (dot > 0) name = name.left(dot) + QStringLiteral(".tx");
        else name += QStringLiteral(".tx");
    } else {
        QString udimPattern;
        std::vector<int> tiles;
        if (resolveUdimPattern(src, QString(), udimPattern, tiles)) {
            QFileInfo pinfo(udimPattern);
            name = pinfo.fileName();
            const int dot = name.lastIndexOf(QLatin1Char('.'));
            if (dot > 0) name = name.left(dot) + QStringLiteral(".tx");
            else name += QStringLiteral(".tx");
        } else {
            name = info.completeBaseName() + QStringLiteral(".tx");
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
    rebuildTimeline();
    infoLabel_->setText(QStringLiteral("No texture"));
}

void TextureViewerWidget::refreshFromPipeline() {
    QString path;
    if (contentKind_ == ViewerContentKind::SourceImages) {
        path = pipelineSource_;
    } else {
        path = guessConvertedTxPath(pipelineSource_, pipelineOutputFolder_);
    }
    if (path.isEmpty()) {
        clearView();
        return;
    }
    setSourcePath(path);
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
        infoLabel_->setText(QStringLiteral("No texture"));
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

    if (paths.isEmpty() && path.contains(QStringLiteral("$F"))) {
        const int scanEnd = std::max(rangeEnd_, 10000);
        const auto expanded = txExpandFrameSources(path.toStdString(), 1, scanEnd);
        for (const std::string& p : expanded) {
            paths.push_back(QString::fromStdString(p));
            udims.push_back(0);
        }
    }

    if (paths.isEmpty()) {
        if (!QFileInfo::exists(path)) {
            // Missing source / .tx → default placeholder, not an error banner.
            rebuildTimeline();
            infoLabel_->setText(QStringLiteral("No texture"));
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
    canvas_->setGrade(grade_);

    rangeStart_ = 1;
    rangeEnd_ = std::max(1, int(frames_.size()));
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

void TextureViewerWidget::setTimelineFrame(int frame) {
    // Scrubber is 1-based within [rangeStart_, rangeEnd_].
    const int index = std::clamp(frame, rangeStart_, rangeEnd_) - 1;
    setExprFrame(frame);
    setFrame(index);
}

void TextureViewerWidget::setFrame(int index) {
    if (frames_.isEmpty()) return;
    index = std::clamp(index, 0, int(frames_.size()) - 1);
    // Keep scrub inside the editable start/end window.
    index = std::clamp(index, rangeStart_ - 1, rangeEnd_ - 1);
    frameIndex_ = index;
    if (!updatingTimeline_) {
        updatingTimeline_ = true;
        scrubber_->setFrame(frameIndex_ + 1);
        updatingTimeline_ = false;
    }
    showCurrentFrame();
}

void TextureViewerWidget::nextFrame() {
    if (frames_.size() <= 1) return;
    const int next = frameIndex_ + 1;
    if (next > rangeEnd_ - 1) setFrame(rangeStart_ - 1);
    else setFrame(next);
}

void TextureViewerWidget::prevFrame() {
    if (frames_.size() <= 1) return;
    const int prev = frameIndex_ - 1;
    if (prev < rangeStart_ - 1) setFrame(rangeEnd_ - 1);
    else setFrame(prev);
}

void TextureViewerWidget::fitView() { canvas_->fitToView(); }

void TextureViewerWidget::onStartEdited() {
    if (updatingTimeline_ || !startEdit_) return;
    bool ok = false;
    int value = startEdit_->text().trimmed().toInt(&ok);
    const int maxFrame = std::max(1, int(frames_.size()));
    if (!ok) {
        updatingTimeline_ = true;
        startEdit_->setText(QString::number(rangeStart_));
        updatingTimeline_ = false;
        return;
    }
    value = std::clamp(value, 1, maxFrame);
    rangeStart_ = value;
    if (rangeEnd_ < rangeStart_) rangeEnd_ = rangeStart_;
    rebuildTimeline();
    setFrame(frameIndex_);
}

void TextureViewerWidget::onEndEdited() {
    if (updatingTimeline_ || !endEdit_) return;
    bool ok = false;
    int value = endEdit_->text().trimmed().toInt(&ok);
    const int maxFrame = std::max(1, int(frames_.size()));
    if (!ok) {
        updatingTimeline_ = true;
        endEdit_->setText(QString::number(rangeEnd_));
        updatingTimeline_ = false;
        return;
    }
    value = std::clamp(value, 1, maxFrame);
    rangeEnd_ = value;
    if (rangeEnd_ < rangeStart_) rangeStart_ = rangeEnd_;
    rebuildTimeline();
    setFrame(frameIndex_);
}

void TextureViewerWidget::rebuildTimeline() {
    const int n = std::max(1, int(frames_.size()));
    if (frames_.isEmpty()) {
        rangeStart_ = 1;
        rangeEnd_ = 1;
    } else {
        rangeStart_ = std::clamp(rangeStart_, 1, n);
        rangeEnd_ = std::clamp(rangeEnd_, 1, n);
        if (rangeEnd_ < rangeStart_) rangeEnd_ = rangeStart_;
    }

    updatingTimeline_ = true;
    if (startEdit_) startEdit_->setText(QString::number(rangeStart_));
    if (endEdit_) endEdit_->setText(QString::number(rangeEnd_));
    scrubber_->setRange(rangeStart_, rangeEnd_);
    scrubber_->setFrame(std::clamp(frameIndex_ + 1, rangeStart_, rangeEnd_));
    updatingTimeline_ = false;
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
    const quint64 id = (quint64(loadGeneration_.load()) << 32) ^ quint64(frameIndex_ + 1) ^
                       (quint64(slot.previewW) << 16) ^ quint64(slot.previewH);
    canvas_->setLinearImage(slot.linearRgb.data(), slot.previewW, slot.previewH, id);
}

void TextureViewerWidget::showCurrentFrame() {
    if (frames_.isEmpty()) return;
    frameIndex_ = std::clamp(frameIndex_, 0, int(frames_.size()) - 1);
    pushFrameToCanvas();
    updateInfoBar();
}

}  // namespace sol
