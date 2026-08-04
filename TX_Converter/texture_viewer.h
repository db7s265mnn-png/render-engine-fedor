// Texture preview for TX Converter: float canvas, Solstice-style scrubber, grade controls.
#pragma once

#include <QPointF>
#include <QString>
#include <QVector>
#include <QWidget>

#include <atomic>
#include <cstdint>
#include <vector>

#include "scene/types.h"

class QLabel;
class QComboBox;
class QPushButton;
class QToolButton;
class QLineEdit;
class QDoubleSpinBox;
class QSlider;
class QSpinBox;
class QButtonGroup;
class QKeyEvent;
class QTimer;

namespace sol {

class TimelineScrubber;

enum class ViewerChannelMode : int {
    RGBA = 0,
    R = 1,
    G = 2,
    B = 3,
    A = 4,
};

struct ViewerGrade {
    float brightness = 0.0f;  // exposure stops (× 2^b), Houdini/mplay-style
    float contrast = 1.0f;    // pivot 0.18 linear
    float gamma = 1.0f;       // linear gamma before view transform
};

// Paints linear float RGBA with a live display transform; supports zoom + drag.
class FloatPreviewCanvas : public QWidget {
    Q_OBJECT
public:
    explicit FloatPreviewCanvas(QWidget* parent = nullptr);

    void clear();
    void setPlaceholder(const QString& text);
    // rgba = tightly packed RGBA float (4 components per pixel).
    void setLinearImage(const float* rgba, int width, int height, quint64 contentId);
    void setDisplayParams(int colorManagement, int viewTransform, bool ocioUseEnv,
                          const QString& ocioConfigPath, int workingSpace);
    void setChannelMode(ViewerChannelMode mode);
    // interactive=true: cheaper preview bake while scrubbing grade sliders.
    void setGrade(const ViewerGrade& grade, bool interactive = false);
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
    void keyPressEvent(QKeyEvent* event) override;

private:
    void invalidateDisplayCache();
    void invalidateBaseLinear();
    void ensureBaseLinear();
    void ensureDisplayCache();
    QRectF imageRect() const;
    void clampPan();

    const float* linearRgba_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    quint64 contentId_ = 0;

    int colorManagement_ = kColorOcio;
    int viewTransform_ = kViewSrgbAces;
    ViewerChannelMode channelMode_ = ViewerChannelMode::RGBA;
    bool ocioUseEnv_ = true;
    QString ocioConfigPath_;
    int workingSpace_ = 1;
    ViewerGrade grade_;
    bool gradeInteractive_ = false;

    // Channel-extracted linear RGB (no grade / view). Rebuilt on image/channel change.
    std::vector<float> baseLinearRgb_;
    quint64 baseLinearId_ = 0;
    int baseLinearChannel_ = -1;

    QImage displayCache_;
    quint64 displayCacheId_ = 0;
    int displayCacheColorMgmt_ = -1;
    int displayCacheView_ = -1;
    int displayCacheChannel_ = -1;
    int displayCacheWorking_ = -1;
    int displayCacheStep_ = 0;
    QString displayCacheOcioPath_;
    bool displayCacheOcioEnv_ = true;
    ViewerGrade displayCacheGrade_;

    double zoom_ = 1.0;
    QPointF pan_{0.0, 0.0};
    bool fitted_ = true;
    bool dragging_ = false;
    QPoint lastMouse_;
    QString placeholder_ = QStringLiteral("No texture");
};

class TextureViewerWidget : public QWidget {
    Q_OBJECT
public:
    enum class ViewerContentKind {
        SourceImages = 0,
        ConvertedTx = 1,
    };

    explicit TextureViewerWidget(QWidget* parent = nullptr);

    void setPipelinePaths(const QString& sourcePath, const QString& outputFolder);
    void setOutputExtension(const QString& ext);  // tx / png / jpg
    void setContentKind(ViewerContentKind kind);
    ViewerContentKind contentKind() const { return contentKind_; }

    static QString guessConvertedOutputPath(const QString& sourcePath, const QString& outputFolder,
                                            const QString& ext);

    void setOcioConfig(bool useEnv, const QString& configPath);
    void setColorManagement(int mode);
    void setViewTransform(int view);
    int colorManagement() const { return colorManagement_; }
    int viewTransform() const { return viewTransform_; }

    // Viewer RAM budget for decoded float previews (default 32 GiB).
    void setMemoryBudgetBytes(qint64 bytes);
    qint64 memoryBudgetBytes() const { return memoryBudgetBytes_; }
    qint64 bufferBytesUsed() const;
    void clearTextureBuffers();
    void reloadConvertedBuffer();

    QString currentPath() const;
    int frameCount() const;
    int currentFrame() const { return frameIndex_; }
    int rangeStart() const;
    int rangeEnd() const;

public slots:
    void setFrame(int index);          // 0-based tile index
    void setTimelineFrame(int frame);  // absolute frame / UDIM number
    void nextFrame();
    void prevFrame();
    void fitView();

signals:
    void statusMessage(const QString& text);

private:
    struct FrameSlot {
        QString path;
        int frameNumber = 0;  // UDIM / $F number from filename
        int sourceWidth = 0;
        int sourceHeight = 0;
        qint64 fileBytes = 0;
        int previewW = 0;
        int previewH = 0;
        std::vector<float> linearRgba;  // RGBA float
        QString error;
        bool ready = false;

        qint64 previewBytes() const {
            return qint64(linearRgba.size()) * qint64(sizeof(float));
        }
        void unloadPixels() {
            linearRgba.clear();
            linearRgba.shrink_to_fit();
            ready = false;
        }
    };

    struct SequenceCache {
        QString pathKey;
        QVector<FrameSlot> frames;
        int loadedCount = 0;
        quint64 generation = 0;
        int rangeStart = 1;
        int rangeEnd = 1;

        qint64 bytesUsed() const {
            qint64 n = 0;
            for (const FrameSlot& f : frames) n += f.previewBytes();
            return n;
        }
        void clearPixels() {
            for (FrameSlot& f : frames) f.unloadPixels();
            loadedCount = 0;
        }
        void reset() {
            frames.clear();
            loadedCount = 0;
            ++generation;
            pathKey.clear();
            rangeStart = 1;
            rangeEnd = 1;
        }
    };

    struct LoadPayload {
        int index = 0;
        QString path;
        int sourceWidth = 0;
        int sourceHeight = 0;
        qint64 fileBytes = 0;
        int previewW = 0;
        int previewH = 0;
        std::vector<float> linearRgba;
        QString error;
    };

    SequenceCache& cacheFor(ViewerContentKind kind);
    const SequenceCache& cacheFor(ViewerContentKind kind) const;
    SequenceCache& activeCache();
    const SequenceCache& activeCache() const;
    SequenceCache& inactiveCache();

    void rememberSharedFrame();
    void restoreSharedFrame();
    void bindActiveCacheToUi();
    void loadSequenceInto(SequenceCache& cache, const QString& path);
    void ensureActiveSequence();
    void clearActiveView();
    void refreshFromPipeline(bool forceReload = false);
    void rebuildTimeline();
    void updateInfoBar();
    void updateBufferStatus();
    void showCurrentFrame();
    void startPreload(SequenceCache& cache);
    void onFrameLoaded(ViewerContentKind kind, quint64 generation, LoadPayload payload);
    void pushFrameToCanvas();
    void enforceMemoryBudget(SequenceCache* preferKeep);
    void applyGradeFromUi(bool interactive = false);
    void scheduleGradeApply(bool interactive);
    void setChannelMode(ViewerChannelMode mode);
    void onStartEdited();
    void onEndEdited();
    int indexForFrameNumber(int frame) const;
    int frameNumberAt(int index) const;
    QLineEdit* makeRangeEdit(const QString& tip);
    QPushButton* makeGradeLabel(const QString& text, const QString& tip);
    QDoubleSpinBox* makeGradeSpin(double minV, double maxV, double step, double value,
                                  const QString& tip);
    QToolButton* makeChannelButton(const QString& tip, const QColor& fill, bool checker = false);

    static LoadPayload decodeFrame(const QString& path, int index);

    FloatPreviewCanvas* canvas_ = nullptr;
    QLabel* infoLabel_ = nullptr;
    QLabel* zoomLabel_ = nullptr;
    QLabel* bufferLabel_ = nullptr;
    QComboBox* contentCombo_ = nullptr;
    QComboBox* colorMgmtCombo_ = nullptr;
    QComboBox* viewCombo_ = nullptr;
    QButtonGroup* channelGroup_ = nullptr;
    QPushButton* fitBtn_ = nullptr;
    QPushButton* clearBufBtn_ = nullptr;
    QSpinBox* memoryGbSpin_ = nullptr;

    QString pipelineSource_;
    QString pipelineOutputFolder_;
    QString outputExt_ = QStringLiteral("tx");
    ViewerContentKind contentKind_ = ViewerContentKind::SourceImages;
    ViewerChannelMode channelMode_ = ViewerChannelMode::RGBA;

    QPushButton* brightnessLabelBtn_ = nullptr;
    QPushButton* contrastLabelBtn_ = nullptr;
    QPushButton* gammaLabelBtn_ = nullptr;
    QDoubleSpinBox* brightnessSpin_ = nullptr;
    QDoubleSpinBox* contrastSpin_ = nullptr;
    QDoubleSpinBox* gammaSpin_ = nullptr;
    QSlider* brightnessSlider_ = nullptr;
    QSlider* contrastSlider_ = nullptr;
    QSlider* gammaSlider_ = nullptr;
    QTimer* gradeTimer_ = nullptr;
    bool gradeScrubbing_ = false;

    QLineEdit* startEdit_ = nullptr;
    QLineEdit* endEdit_ = nullptr;
    TimelineScrubber* scrubber_ = nullptr;

    SequenceCache sourceCache_;
    SequenceCache convertedCache_;
    int frameIndex_ = 0;
    int sharedFrameNumber_ = 1;
    bool updatingTimeline_ = false;
    qint64 memoryBudgetBytes_ = 32LL * 1024 * 1024 * 1024;

    int colorManagement_ = kColorOcio;
    int viewTransform_ = kViewSrgbAces;
    bool ocioUseEnv_ = true;
    QString ocioConfigPath_;
    int ocioWorkingSpace_ = 1;
    ViewerGrade grade_;
};

}  // namespace sol
