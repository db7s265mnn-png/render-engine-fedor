#include "ui/timeline_bar.h"

#include <QHBoxLayout>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QTimer>
#include <cmath>

#include "ui/theme.h"

namespace sol {
namespace {

QSpinBox* makeFrameSpin(QWidget* parent, int value) {
    auto* spin = new QSpinBox(parent);
    spin->setRange(-1000000, 1000000);
    spin->setValue(value);
    spin->setFixedWidth(72);
    spin->setButtonSymbols(QAbstractSpinBox::UpDownArrows);
    spin->setToolTip("Frame");
    return spin;
}

}  // namespace

TimelineBar::TimelineBar(QWidget* parent) : QWidget(parent) {
    setObjectName("timelineBar");
    setFixedHeight(36);
    setStyleSheet(QStringLiteral(
        "QWidget#timelineBar {"
        "  background: %1;"
        "  border-top: 1px solid %2;"
        "}"
        "QLabel { color: %3; }"
        "QPushButton {"
        "  background: %4; color: %5; border: 1px solid %2;"
        "  border-radius: 3px; padding: 2px 8px; min-width: 28px;"
        "}"
        "QPushButton:hover { background: %6; }"
        "QPushButton:pressed { background: %2; }"
        "QSlider::groove:horizontal {"
        "  height: 6px; background: %2; border-radius: 3px;"
        "}"
        "QSlider::handle:horizontal {"
        "  width: 12px; margin: -5px 0; border-radius: 6px;"
        "  background: %7;"
        "}"
        "QSpinBox, QDoubleSpinBox {"
        "  background: %4; color: %5; border: 1px solid %2; border-radius: 3px;"
        "  padding: 1px 4px;"
        "}")
                      .arg(theme::panel().name(), theme::panelLight().name(), theme::textDim().name(),
                           theme::panelDark().name(), theme::text().name(), theme::panelLight().lighter(110).name(),
                           theme::accent().name()));

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(8, 4, 8, 4);
    layout->setSpacing(6);

    auto* startLabel = new QLabel("Start", this);
    startSpin_ = makeFrameSpin(this, startFrame_);
    startSpin_->setToolTip("Timeline start frame");

    auto* endLabel = new QLabel("End", this);
    endSpin_ = makeFrameSpin(this, endFrame_);
    endSpin_->setToolTip("Timeline end frame");

    fpsSpin_ = new QDoubleSpinBox(this);
    fpsSpin_->setRange(1.0, 240.0);
    fpsSpin_->setDecimals(3);
    fpsSpin_->setValue(fps_);
    fpsSpin_->setSuffix(" fps");
    fpsSpin_->setFixedWidth(88);
    fpsSpin_->setToolTip("Frames per second — converts the current frame to sample time");

    auto* toStart = new QPushButton("|◀", this);
    toStart->setToolTip("Go to start");
    auto* stepBack = new QPushButton("◀", this);
    stepBack->setToolTip("Previous frame");
    playButton_ = new QPushButton("▶", this);
    playButton_->setToolTip("Play / Pause");
    playButton_->setFixedWidth(36);
    auto* stopButton = new QPushButton("■", this);
    stopButton->setToolTip("Stop (return to start)");
    auto* stepFwd = new QPushButton("▶", this);
    stepFwd->setToolTip("Next frame");
    auto* toEnd = new QPushButton("▶|", this);
    toEnd->setToolTip("Go to end");

    scrub_ = new QSlider(Qt::Horizontal, this);
    scrub_->setRange(startFrame_, endFrame_);
    scrub_->setValue(currentFrame_);
    scrub_->setToolTip("Scrub timeline");

    frameSpin_ = makeFrameSpin(this, currentFrame_);
    frameSpin_->setToolTip("Current frame");

    timeLabel_ = new QLabel(this);
    timeLabel_->setMinimumWidth(64);
    timeLabel_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    layout->addWidget(startLabel);
    layout->addWidget(startSpin_);
    layout->addSpacing(4);
    layout->addWidget(toStart);
    layout->addWidget(stepBack);
    layout->addWidget(playButton_);
    layout->addWidget(stopButton);
    layout->addWidget(stepFwd);
    layout->addWidget(toEnd);
    layout->addSpacing(6);
    layout->addWidget(scrub_, 1);
    layout->addWidget(frameSpin_);
    layout->addWidget(timeLabel_);
    layout->addSpacing(8);
    layout->addWidget(endLabel);
    layout->addWidget(endSpin_);
    layout->addWidget(fpsSpin_);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &TimelineBar::onTick);

    connect(playButton_, &QPushButton::clicked, this, &TimelineBar::onPlayPause);
    connect(stopButton, &QPushButton::clicked, this, &TimelineBar::stop);
    connect(toStart, &QPushButton::clicked, this, &TimelineBar::goToStart);
    connect(toEnd, &QPushButton::clicked, this, &TimelineBar::goToEnd);
    connect(stepBack, &QPushButton::clicked, this, &TimelineBar::stepBackward);
    connect(stepFwd, &QPushButton::clicked, this, &TimelineBar::stepForward);
    connect(scrub_, &QSlider::valueChanged, this, &TimelineBar::onSliderMoved);
    connect(startSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this, &TimelineBar::onStartEdited);
    connect(endSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this, &TimelineBar::onEndEdited);
    connect(fpsSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &TimelineBar::onFpsEdited);
    connect(frameSpin_, QOverload<int>::of(&QSpinBox::valueChanged), this, &TimelineBar::onFrameEdited);

    syncWidgetsFromState();
}

int TimelineBar::startFrame() const { return startFrame_; }
int TimelineBar::endFrame() const { return endFrame_; }
int TimelineBar::currentFrame() const { return currentFrame_; }
double TimelineBar::fps() const { return fps_; }

double TimelineBar::timeSeconds() const {
    const double rate = std::max(1e-6, fps_);
    return double(currentFrame_ - 1) / rate;
}

void TimelineBar::setRange(int startFrame, int endFrame) {
    if (endFrame < startFrame) endFrame = startFrame;
    startFrame_ = startFrame;
    endFrame_ = endFrame;
    clampFrame();
    syncWidgetsFromState();
}

void TimelineBar::setFps(double fps) {
    fps_ = std::max(1.0, fps);
    syncWidgetsFromState();
}

void TimelineBar::setCurrentFrame(int frame) {
    currentFrame_ = frame;
    clampFrame();
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::suggestTimeRange(double startSeconds, double endSeconds, bool force) {
    if (rangeTouched_ && !force) return;
    if (!(endSeconds >= startSeconds)) return;
    const double rate = std::max(1e-6, fps_);
    // Map archive seconds onto integer frames with time = (frame - 1) / fps.
    int start = 1 + int(std::floor(startSeconds * rate + 1e-6));
    int end = 1 + int(std::ceil(endSeconds * rate - 1e-6));
    if (start < 1) start = 1;
    if (end < start) end = start;
    startFrame_ = start;
    endFrame_ = end;
    clampFrame();
    syncWidgetsFromState();
}

void TimelineBar::play() {
    if (playing_) return;
    if (currentFrame_ >= endFrame_) currentFrame_ = startFrame_;
    playing_ = true;
    playButton_->setText("❚❚");
    const int intervalMs = int(std::round(1000.0 / std::max(1.0, fps_)));
    timer_->start(std::max(1, intervalMs));
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::pause() {
    if (!playing_) return;
    playing_ = false;
    playButton_->setText("▶");
    timer_->stop();
    emit playbackStopped();
}

void TimelineBar::stop() {
    pause();
    currentFrame_ = startFrame_;
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::stepBackward() {
    pause();
    --currentFrame_;
    clampFrame();
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::stepForward() {
    pause();
    ++currentFrame_;
    clampFrame();
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::goToStart() {
    pause();
    currentFrame_ = startFrame_;
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::goToEnd() {
    pause();
    currentFrame_ = endFrame_;
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::onPlayPause() {
    if (playing_) pause();
    else play();
}

void TimelineBar::onTick() {
    if (!playing_) return;
    ++currentFrame_;
    if (currentFrame_ > endFrame_) {
        currentFrame_ = startFrame_;  // loop
    }
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::onSliderMoved(int value) {
    if (updating_) return;
    rangeTouched_ = true;
    pause();
    currentFrame_ = value;
    clampFrame();
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::onStartEdited(int value) {
    if (updating_) return;
    rangeTouched_ = true;
    startFrame_ = value;
    if (endFrame_ < startFrame_) endFrame_ = startFrame_;
    clampFrame();
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::onEndEdited(int value) {
    if (updating_) return;
    rangeTouched_ = true;
    endFrame_ = value;
    if (endFrame_ < startFrame_) startFrame_ = endFrame_;
    clampFrame();
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::onFpsEdited(double value) {
    if (updating_) return;
    fps_ = std::max(1.0, value);
    if (playing_) {
        timer_->setInterval(std::max(1, int(std::round(1000.0 / fps_))));
    }
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::onFrameEdited(int value) {
    if (updating_) return;
    rangeTouched_ = true;
    pause();
    currentFrame_ = value;
    clampFrame();
    syncWidgetsFromState();
    emitFrame();
}

void TimelineBar::syncWidgetsFromState() {
    updating_ = true;
    startSpin_->setValue(startFrame_);
    endSpin_->setValue(endFrame_);
    frameSpin_->setRange(startFrame_, endFrame_);
    frameSpin_->setValue(currentFrame_);
    scrub_->setRange(startFrame_, endFrame_);
    scrub_->setValue(currentFrame_);
    fpsSpin_->setValue(fps_);
    timeLabel_->setText(QString("%1 s").arg(timeSeconds(), 0, 'f', 3));
    playButton_->setText(playing_ ? QStringLiteral("❚❚") : QStringLiteral("▶"));
    updating_ = false;
}

void TimelineBar::clampFrame() {
    if (endFrame_ < startFrame_) endFrame_ = startFrame_;
    if (currentFrame_ < startFrame_) currentFrame_ = startFrame_;
    if (currentFrame_ > endFrame_) currentFrame_ = endFrame_;
}

void TimelineBar::emitFrame() { emit frameChanged(currentFrame_); }

}  // namespace sol
