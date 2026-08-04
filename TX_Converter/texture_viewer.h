// Texture preview for TX Converter: right-side float canvas, tick timeline, OCIO/sRGB.
#pragma once

#include <QPointF>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QWidget>

#include <atomic>
#include <vector>

class QLabel;
class QComboBox;
class QPushButton;

namespace sol {

enum class ViewerDisplayMode : int {
    ClassicSrgb = 0,
    OcioSrgbAces = 1,
};

// Horizontal timeline with a tick per frame — no UDIM/frame text.
class FrameTimelineWidget : public QWidget {
    Q_OBJECT
public:
    explicit FrameTimelineWidget(QWidget* parent = nullptr);

    void setFrameCount(int count);
    void setCurrentFrame(int index);
    int currentFrame() const { return current_; }
    int frameCount() const { return count_; }

signals:
    void frameChanged(int index);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

private:
    int indexAtX(int x) const;
    void seekToX(int x);

    int count_ = 0;
    int current_ = 0;
};

// Paints linear float RGB with a live display transform; supports zoom + drag.
class FloatPreviewCanvas : public QWidget {
    Q_OBJECT
public:
    explicit FloatPreviewCanvas(QWidget* parent = nullptr);

    void clear();
    void setPlaceholder(const QString& text);
    void setLinearImage(const float* rgb, int width, int height, quint64 contentId);
    void setDisplayMode(ViewerDisplayMode mode, bool ocioUseEnv, const QString& ocioConfigPath,
                        int workingSpace);
    void fitToView();
    void resetView();
    double zoom() const { return zoom_; }

signals:
    void zoomChanged(double zoom);

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    void invalidateDisplayCache();
    void ensureDisplayCache();
    QRectF imageRect() const;
    void clampPan();

    const float* linearRgb_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    quint64 contentId_ = 0;

    ViewerDisplayMode displayMode_ = ViewerDisplayMode::ClassicSrgb;
    bool ocioUseEnv_ = true;
    QString ocioConfigPath_;
    int workingSpace_ = 1;

    QImage displayCache_;
    quint64 displayCacheId_ = 0;
    ViewerDisplayMode displayCacheMode_ = ViewerDisplayMode::ClassicSrgb;
    int displayCacheWorking_ = -1;
    QString displayCacheOcioPath_;
    bool displayCacheOcioEnv_ = true;

    double zoom_ = 1.0;
    QPointF pan_{0.0, 0.0};  // pan in widget pixels
    bool fitted_ = true;
    bool dragging_ = false;
    QPoint lastMouse_;
    QString placeholder_ = QStringLiteral("No texture");
};

class TextureViewerWidget : public QWidget {
    Q_OBJECT
public:
    explicit TextureViewerWidget(QWidget* parent = nullptr);

    void setSourcePath(const QString& path);
    void setOcioConfig(bool useEnv, const QString& configPath);
    void setDisplayMode(ViewerDisplayMode mode);
    ViewerDisplayMode displayMode() const { return displayMode_; }

    QString currentPath() const;
    int frameCount() const { return frames_.size(); }
    int currentFrame() const { return frameIndex_; }

public slots:
    void setFrame(int index);
    void nextFrame();
    void prevFrame();
    void fitView();

signals:
    void statusMessage(const QString& text);

private:
    struct FrameSlot {
        QString path;
        int udim = 0;
        int sourceWidth = 0;
        int sourceHeight = 0;
        qint64 fileBytes = 0;
        int previewW = 0;
        int previewH = 0;
        std::vector<float> linearRgb;  // linear RGB, source of truth for display
        QString error;
        bool ready = false;
    };

    struct LoadPayload {
        int index = 0;
        QString path;
        int sourceWidth = 0;
        int sourceHeight = 0;
        qint64 fileBytes = 0;
        int previewW = 0;
        int previewH = 0;
        std::vector<float> linearRgb;
        QString error;
    };

    void rebuildTimeline();
    void updateInfoBar();
    void showCurrentFrame();
    void startPreloadAll();
    void onFrameLoaded(quint64 generation, LoadPayload payload);
    void pushFrameToCanvas();

    static LoadPayload decodeFrame(const QString& path, int index);

    FloatPreviewCanvas* canvas_ = nullptr;
    QLabel* infoLabel_ = nullptr;
    QLabel* zoomLabel_ = nullptr;
    QComboBox* displayCombo_ = nullptr;
    FrameTimelineWidget* timeline_ = nullptr;
    QPushButton* prevBtn_ = nullptr;
    QPushButton* nextBtn_ = nullptr;
    QPushButton* fitBtn_ = nullptr;

    QVector<FrameSlot> frames_;
    int frameIndex_ = 0;
    int loadedCount_ = 0;

    ViewerDisplayMode displayMode_ = ViewerDisplayMode::ClassicSrgb;
    bool ocioUseEnv_ = true;
    QString ocioConfigPath_;
    int ocioWorkingSpace_ = 1;

    std::atomic<quint64> loadGeneration_{0};
};

}  // namespace sol
