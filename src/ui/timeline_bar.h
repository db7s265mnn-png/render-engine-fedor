// Compact Houdini-style playback strip under the viewport.
#pragma once

#include <QWidget>

class QLineEdit;
class QTimer;
class QToolButton;

namespace sol {

// Scrub track with a fixed-width playhead that shows the frame number.
// Double-click the playhead to type a frame.
class TimelineScrubber : public QWidget {
    Q_OBJECT

public:
    explicit TimelineScrubber(QWidget* parent = nullptr);

    void setRange(int startFrame, int endFrame);
    void setFrame(int frame);
    int frame() const { return frame_; }
    bool isScrubbing() const { return dragging_; }

signals:
    void frameChanged(int frame);
    void scrubStarted();
    void scrubFinished();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    QRectF trackRect() const;
    QRectF playheadRect() const;
    int frameAtX(qreal x) const;
    qreal xForFrame(int frame) const;
    void beginFrameEdit();
    void commitFrameEdit();
    void cancelFrameEdit();

    int startFrame_ = 1;
    int endFrame_ = 240;
    int frame_ = 1;
    bool dragging_ = false;
    QLineEdit* editor_ = nullptr;
};

class TimelineBar : public QWidget {
    Q_OBJECT

public:
    explicit TimelineBar(QWidget* parent = nullptr);

    int startFrame() const { return startFrame_; }
    int endFrame() const { return endFrame_; }
    int currentFrame() const { return currentFrame_; }
    double fps() const { return fps_; }
    bool isPlaying() const { return playing_; }
    bool isScrubbing() const;
    // Playing or dragging — cook/render should stay lightweight.
    bool isInteractive() const { return isPlaying() || isScrubbing(); }

    // Houdini-style: time = (frame - 1) / fps.
    double timeSeconds() const;

    void setRange(int startFrame, int endFrame);
    void setFps(double fps);
    void setCurrentFrame(int frame);
    void suggestTimeRange(double startSeconds, double endSeconds, bool force = false);

public slots:
    void play();
    void stopPlayback();  // pause in place (Play/Stop toggle → Stop)
    void goToStart();
    void goToEnd();

signals:
    void frameChanged(int frame);
    void playbackStopped();
    void scrubFinished();

private slots:
    void onPlayStop();
    void onTick();
    void onScrubFrame(int frame);
    void onStartEdited();
    void onEndEdited();

private:
    void syncFromState();
    void clampFrame();
    void updatePlayStopIcon();
    void emitFrame();
    QLineEdit* makeRangeEdit(const QString& tip);

    QLineEdit* startEdit_ = nullptr;
    QLineEdit* endEdit_ = nullptr;
    QToolButton* toStartBtn_ = nullptr;
    QToolButton* playStopBtn_ = nullptr;
    QToolButton* toEndBtn_ = nullptr;
    TimelineScrubber* scrubber_ = nullptr;
    QTimer* timer_ = nullptr;

    int startFrame_ = 1;
    int endFrame_ = 240;
    int currentFrame_ = 1;
    double fps_ = 24.0;
    bool playing_ = false;
    bool rangeTouched_ = false;
    bool updating_ = false;
};

}  // namespace sol
