// Texture preview for TX Converter: LDR/HDR/TX + UDIM timeline (lazy per-tile load).
#pragma once

#include <QHash>
#include <QImage>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <atomic>

class QLabel;
class QSlider;
class QPushButton;
class QScrollArea;
class QTimer;

namespace sol {

class TextureViewerWidget : public QWidget {
    Q_OBJECT
public:
    explicit TextureViewerWidget(QWidget* parent = nullptr);

    // Load a single file or a UDIM sequence (pattern / concrete tile / <UDIM>).
    // Timeline length = number of existing UDIM tiles (or 1 for a single file).
    void setSourcePath(const QString& path);

    QString currentPath() const;
    int frameCount() const { return paths_.size(); }
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
    struct PreviewResult {
        QImage image;
        int sourceWidth = 0;
        int sourceHeight = 0;
        qint64 fileBytes = 0;
        QString error;
    };

    void rebuildTimeline();
    void showCurrentFrame(bool forceReload);
    void applyPreview(const QString& path, int frameIndex, quint64 generation, const PreviewResult& result);
    void updateImageLabel();
    void touchCache(const QString& path, const PreviewResult& result);

    static PreviewResult loadPreviewImage(const QString& path);

    QScrollArea* scroll_ = nullptr;
    QLabel* imageLabel_ = nullptr;
    QLabel* infoLabel_ = nullptr;
    QSlider* timeline_ = nullptr;
    QPushButton* prevBtn_ = nullptr;
    QPushButton* nextBtn_ = nullptr;
    QLabel* frameLabel_ = nullptr;
    QTimer* resizeDebounce_ = nullptr;

    QStringList paths_;  // concrete files in timeline order
    QList<int> udims_;   // parallel UDIM ids (0 if non-UDIM)
    int frameIndex_ = 0;

    QImage previewImage_;  // current frame, already capped for preview
    int sourceWidth_ = 0;
    int sourceHeight_ = 0;
    qint64 fileBytes_ = 0;
    QString loadedPath_;

    QHash<QString, PreviewResult> cache_;
    QStringList cacheOrder_;
    std::atomic<quint64> loadGeneration_{0};
};

}  // namespace sol
