// Texture preview for TX Converter: UDIM timeline, full-sequence preload, OCIO/sRGB display.
#pragma once

#include <QImage>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QWidget>

#include <atomic>
#include <vector>

class QLabel;
class QComboBox;
class QPushButton;
class QScrollArea;
class QTimer;

namespace sol {

enum class ViewerDisplayMode : int {
    ClassicSrgb = 0,
    OcioSrgbAces = 1,
};

// Horizontal timeline with a tick per frame (UDIM tile / sequence index).
class FrameTimelineWidget : public QWidget {
    Q_OBJECT
public:
    explicit FrameTimelineWidget(QWidget* parent = nullptr);

    void setFrameCount(int count);
    void setCurrentFrame(int index);
    void setTickLabels(const QStringList& labels);  // optional per-frame labels (e.g. UDIM)
    int currentFrame() const { return current_; }
    int frameCount() const { return count_; }

signals:
    void frameChanged(int index);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

private:
    int indexAtX(int x) const;
    void seekToX(int x);

    int count_ = 0;
    int current_ = 0;
    QStringList labels_;
};

class TextureViewerWidget : public QWidget {
    Q_OBJECT
public:
    explicit TextureViewerWidget(QWidget* parent = nullptr);

    // Load a single file or a UDIM sequence; preloads every tile in the background.
    void setSourcePath(const QString& path);

    // OCIO config used when display mode is OCIO (env and/or explicit path).
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

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    struct FrameSlot {
        QString path;
        int udim = 0;
        int sourceWidth = 0;
        int sourceHeight = 0;
        qint64 fileBytes = 0;
        int previewW = 0;
        int previewH = 0;
        // Linear RGB preview (previewW*previewH*3), downscaled.
        std::vector<float> linearRgb;
        QImage display;  // baked for current display mode
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
    void updateImageLabel();
    void showCurrentFrame();
    void startPreloadAll();
    void onFrameLoaded(quint64 generation, LoadPayload payload);
    void bakeDisplay(FrameSlot& slot) const;
    void rebakeAllDisplays();
    void refreshDisplayModeUi();

    static LoadPayload decodeFrame(const QString& path, int index);

    QScrollArea* scroll_ = nullptr;
    QLabel* imageLabel_ = nullptr;
    QLabel* infoLabel_ = nullptr;
    QComboBox* displayCombo_ = nullptr;
    FrameTimelineWidget* timeline_ = nullptr;
    QPushButton* prevBtn_ = nullptr;
    QPushButton* nextBtn_ = nullptr;
    QLabel* frameLabel_ = nullptr;
    QTimer* resizeDebounce_ = nullptr;

    QVector<FrameSlot> frames_;
    int frameIndex_ = 0;
    int loadedCount_ = 0;

    ViewerDisplayMode displayMode_ = ViewerDisplayMode::ClassicSrgb;
    bool ocioUseEnv_ = true;
    QString ocioConfigPath_;
    int ocioWorkingSpace_ = 1;  // default ACEScg; updated from file type when loading

    std::atomic<quint64> loadGeneration_{0};
};

}  // namespace sol
