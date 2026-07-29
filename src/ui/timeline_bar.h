// Bottom playback bar for Alembic / animated USD: range, FPS, scrub, transport.
#pragma once

#include <QWidget>

class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;
class QSpinBox;
class QTimer;

namespace sol {

class TimelineBar : public QWidget {
    Q_OBJECT

public:
    explicit TimelineBar(QWidget* parent = nullptr);

    int startFrame() const;
    int endFrame() const;
    int currentFrame() const;
    double fps() const;
    bool isPlaying() const { return playing_; }

    // Seconds for CookContext (Houdini-style): time = (frame - 1) / fps.
    double timeSeconds() const;

    void setRange(int startFrame, int endFrame);
    void setFps(double fps);
    void setCurrentFrame(int frame);
    // Suggest a range from an archive (seconds → frames). Ignored once the user
    // has scrubbed/edited the range, unless force=true.
    void suggestTimeRange(double startSeconds, double endSeconds, bool force = false);

public slots:
    void play();
    void pause();
    void stop();
    void stepBackward();
    void stepForward();
    void goToStart();
    void goToEnd();

signals:
    void frameChanged(int frame);
    void playbackStopped();

private slots:
    void onPlayPause();
    void onTick();
    void onSliderMoved(int value);
    void onStartEdited(int value);
    void onEndEdited(int value);
    void onFpsEdited(double value);
    void onFrameEdited(int value);

private:
    void syncWidgetsFromState();
    void clampFrame();
    void emitFrame();

    QSpinBox* startSpin_ = nullptr;
    QSpinBox* endSpin_ = nullptr;
    QSpinBox* frameSpin_ = nullptr;
    QDoubleSpinBox* fpsSpin_ = nullptr;
    QSlider* scrub_ = nullptr;
    QPushButton* playButton_ = nullptr;
    QLabel* timeLabel_ = nullptr;
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
