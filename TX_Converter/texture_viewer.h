// Texture preview for TX Converter: LDR/HDR/TX + UDIM timeline (lazy per-tile load).
#pragma once

#include <QString>
#include <QStringList>
#include <QWidget>

class QLabel;
class QSlider;
class QPushButton;
class QScrollArea;

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
    void rebuildTimeline();
    void showCurrentFrame(bool forceReload);
    QImage loadPreviewImage(const QString& path, QString& error) const;
    void updateImageLabel();

    QScrollArea* scroll_ = nullptr;
    QLabel* imageLabel_ = nullptr;
    QLabel* infoLabel_ = nullptr;
    QSlider* timeline_ = nullptr;
    QPushButton* prevBtn_ = nullptr;
    QPushButton* nextBtn_ = nullptr;
    QLabel* frameLabel_ = nullptr;

    QStringList paths_;       // concrete files in timeline order
    QList<int> udims_;        // parallel UDIM ids (0 if non-UDIM)
    int frameIndex_ = 0;
    QImage fullImage_;        // current frame pixels (display-encoded)
    QString loadedPath_;
};

}  // namespace sol
