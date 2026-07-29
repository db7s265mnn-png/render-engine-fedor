#include "ui/timeline_bar.h"

#include <QPainter>
#include <QPainterPath>
#include <QLineEdit>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QToolButton>
#include <QTimer>
#include <QFontMetrics>
#include <cmath>
#include <algorithm>

#include "ui/theme.h"

namespace sol {
namespace {

QIcon makeHoudiniTransportIcon(const QString& kind, int size = 18) {
    QPixmap pm(size, size);
    pm.fill(Qt::transparent);
    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing, true);
    const QColor ink(220, 222, 226);
    p.setPen(Qt::NoPen);
    p.setBrush(ink);

    const qreal s = size;
    if (kind == QLatin1String("toStart")) {
        // |<<  — vertical bar + double left chevrons (Houdini "jump to start")
        p.drawRect(QRectF(s * 0.14, s * 0.22, s * 0.10, s * 0.56));
        QPainterPath a;
        a.moveTo(s * 0.52, s * 0.22);
        a.lineTo(s * 0.28, s * 0.50);
        a.lineTo(s * 0.52, s * 0.78);
        a.closeSubpath();
        p.drawPath(a);
        QPainterPath b;
        b.moveTo(s * 0.78, s * 0.22);
        b.lineTo(s * 0.54, s * 0.50);
        b.lineTo(s * 0.78, s * 0.78);
        b.closeSubpath();
        p.drawPath(b);
    } else if (kind == QLatin1String("play")) {
        QPainterPath tri;
        tri.moveTo(s * 0.30, s * 0.20);
        tri.lineTo(s * 0.78, s * 0.50);
        tri.lineTo(s * 0.30, s * 0.80);
        tri.closeSubpath();
        p.drawPath(tri);
    } else if (kind == QLatin1String("stop")) {
        p.drawRoundedRect(QRectF(s * 0.28, s * 0.28, s * 0.44, s * 0.44), 1.0, 1.0);
    } else if (kind == QLatin1String("toEnd")) {
        // >>| 
        QPainterPath a;
        a.moveTo(s * 0.22, s * 0.22);
        a.lineTo(s * 0.46, s * 0.50);
        a.lineTo(s * 0.22, s * 0.78);
        a.closeSubpath();
        p.drawPath(a);
        QPainterPath b;
        b.moveTo(s * 0.46, s * 0.22);
        b.lineTo(s * 0.70, s * 0.50);
        b.lineTo(s * 0.46, s * 0.78);
        b.closeSubpath();
        p.drawPath(b);
        p.drawRect(QRectF(s * 0.76, s * 0.22, s * 0.10, s * 0.56));
    }
    return QIcon(pm);
}

QToolButton* makeTransportButton(QWidget* parent, const QString& kind, const QString& tip) {
    auto* btn = new QToolButton(parent);
    btn->setIcon(makeHoudiniTransportIcon(kind));
    btn->setIconSize(QSize(16, 16));
    btn->setToolTip(tip);
    btn->setAutoRaise(false);
    btn->setFixedSize(26, 24);
    btn->setFocusPolicy(Qt::NoFocus);
    return btn;
}

}  // namespace

// ---------------------------------------------------------------------------
// TimelineScrubber
// ---------------------------------------------------------------------------

TimelineScrubber::TimelineScrubber(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(26);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setMouseTracking(true);
    setCursor(Qt::PointingHandCursor);
}

void TimelineScrubber::setRange(int startFrame, int endFrame) {
    if (endFrame < startFrame) endFrame = startFrame;
    startFrame_ = startFrame;
    endFrame_ = endFrame;
    if (frame_ < startFrame_) frame_ = startFrame_;
    if (frame_ > endFrame_) frame_ = endFrame_;
    update();
}

void TimelineScrubber::setFrame(int frame) {
    frame_ = std::clamp(frame, startFrame_, endFrame_);
    update();
}

QRectF TimelineScrubber::trackRect() const {
    return QRectF(36.0, height() * 0.5 - 1.0, width() - 72.0, 2.0);
}

qreal TimelineScrubber::xForFrame(int frame) const {
    const QRectF track = trackRect();
    const int span = std::max(1, endFrame_ - startFrame_);
    const qreal t = double(frame - startFrame_) / double(span);
    return track.left() + t * track.width();
}

int TimelineScrubber::frameAtX(qreal x) const {
    const QRectF track = trackRect();
    if (track.width() < 1.0) return startFrame_;
    const qreal t = std::clamp((x - track.left()) / track.width(), 0.0, 1.0);
    const int span = std::max(1, endFrame_ - startFrame_);
    return startFrame_ + int(std::lround(t * span));
}

QRectF TimelineScrubber::playheadRect() const {
    const qreal x = xForFrame(frame_);
    const QString text = QString::number(frame_);
    QFont font = this->font();
    font.setPointSizeF(9.0);
    font.setBold(true);
    const QFontMetrics fm(font);
    const qreal tw = std::max(28.0, fm.horizontalAdvance(text) + 10.0);
    const qreal th = 16.0;
    return QRectF(x - tw * 0.5, height() * 0.5 - th * 0.5, tw, th);
}

void TimelineScrubber::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRectF track = trackRect();
    // Groove
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(90, 94, 100));
    p.drawRoundedRect(track.adjusted(0, -1, 0, 1), 2.0, 2.0);

    // Tick marks
    const int span = std::max(1, endFrame_ - startFrame_);
    int step = 1;
    if (span > 200) step = 10;
    else if (span > 80) step = 5;
    else if (span > 40) step = 2;
    p.setPen(QPen(QColor(130, 134, 140), 1.0));
    for (int f = startFrame_; f <= endFrame_; f += step) {
        const qreal x = xForFrame(f);
        const bool major = ((f - startFrame_) % (step * 4) == 0) || f == startFrame_ || f == endFrame_;
        const qreal h = major ? 6.0 : 3.5;
        p.drawLine(QPointF(x, track.center().y() - h), QPointF(x, track.center().y() + h));
    }

    // Start / end labels
    QFont labelFont = font();
    labelFont.setPointSizeF(8.5);
    p.setFont(labelFont);
    p.setPen(theme::textDim());
    p.drawText(QRectF(2, 0, 32, height()), Qt::AlignVCenter | Qt::AlignRight, QString::number(startFrame_));
    p.drawText(QRectF(width() - 34, 0, 32, height()), Qt::AlignVCenter | Qt::AlignLeft,
               QString::number(endFrame_));

    // Playhead frame box (Houdini-like black field)
    const QRectF head = playheadRect();
    p.setPen(QPen(QColor(20, 21, 24), 1.0));
    p.setBrush(QColor(12, 13, 15));
    p.drawRoundedRect(head, 2.0, 2.0);
    // Stem through the track
    const qreal cx = head.center().x();
    p.setPen(QPen(QColor(220, 222, 226), 1.2));
    p.drawLine(QPointF(cx, track.center().y() - 7.0), QPointF(cx, track.center().y() + 7.0));

    QFont frameFont = font();
    frameFont.setPointSizeF(9.0);
    frameFont.setBold(true);
    p.setFont(frameFont);
    p.setPen(QColor(235, 237, 240));
    p.drawText(head, Qt::AlignCenter, QString::number(frame_));
}

void TimelineScrubber::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    if (editor_) {
        commitFrameEdit();
        return;
    }
    dragging_ = true;
    emit scrubStarted();
    const int f = frameAtX(event->position().x());
    if (f != frame_) {
        frame_ = f;
        update();
        emit frameChanged(frame_);
    }
    event->accept();
}

void TimelineScrubber::mouseMoveEvent(QMouseEvent* event) {
    if (!dragging_) return;
    const int f = frameAtX(event->position().x());
    if (f != frame_) {
        frame_ = f;
        update();
        emit frameChanged(frame_);
    }
    event->accept();
}

void TimelineScrubber::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton || !dragging_) return;
    dragging_ = false;
    emit scrubFinished();
    event->accept();
}

void TimelineScrubber::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    // Double-click playhead (or nearby) to type a frame.
    if (playheadRect().adjusted(-8, -4, 8, 4).contains(event->position())) {
        beginFrameEdit();
        event->accept();
        return;
    }
    QWidget::mouseDoubleClickEvent(event);
}

void TimelineScrubber::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    if (editor_) {
        const QRectF head = playheadRect();
        editor_->setGeometry(head.toRect().adjusted(-2, -1, 2, 1));
    }
}

void TimelineScrubber::beginFrameEdit() {
    if (editor_) return;
    editor_ = new QLineEdit(this);
    editor_->setAlignment(Qt::AlignCenter);
    editor_->setText(QString::number(frame_));
    editor_->setStyleSheet(
        "QLineEdit { background: #0c0d0f; color: #ebebef; border: 1px solid #50aaff;"
        " border-radius: 2px; padding: 0 4px; font-weight: 700; }");
    const QRectF head = playheadRect();
    editor_->setGeometry(head.toRect().adjusted(-4, -2, 4, 2));
    editor_->selectAll();
    editor_->show();
    editor_->setFocus(Qt::MouseFocusReason);
    connect(editor_, &QLineEdit::editingFinished, this, &TimelineScrubber::commitFrameEdit);
    connect(editor_, &QLineEdit::returnPressed, this, &TimelineScrubber::commitFrameEdit);
}

void TimelineScrubber::commitFrameEdit() {
    if (!editor_) return;
    bool ok = false;
    const int value = editor_->text().trimmed().toInt(&ok);
    editor_->deleteLater();
    editor_ = nullptr;
    if (!ok) {
        update();
        return;
    }
    const int clamped = std::clamp(value, startFrame_, endFrame_);
    if (clamped != frame_) {
        frame_ = clamped;
        emit frameChanged(frame_);
    }
    update();
}

void TimelineScrubber::cancelFrameEdit() {
    if (!editor_) return;
    editor_->deleteLater();
    editor_ = nullptr;
    update();
}

// ---------------------------------------------------------------------------
// TimelineBar
// ---------------------------------------------------------------------------

TimelineBar::TimelineBar(QWidget* parent) : QWidget(parent) {
    setObjectName("timelineBar");
    setFixedHeight(30);
    setStyleSheet(QStringLiteral(
        "QWidget#timelineBar {"
        "  background: #2e3136;"
        "  border-top: 1px solid #22242a;"
        "  border-bottom: 1px solid #22242a;"
        "}"
        "QToolButton {"
        "  background: #3a3e44;"
        "  border: 1px solid #4a4f57;"
        "  border-radius: 3px;"
        "  padding: 0;"
        "}"
        "QToolButton:hover { background: #474c54; }"
        "QToolButton:pressed { background: #2a2d32; }"
        "QToolButton:checked {"
        "  background: rgba(80, 170, 255, 70);"
        "  border-color: #50aaff;"
        "}"));

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(6, 2, 8, 2);
    layout->setSpacing(3);

    toStartBtn_ = makeTransportButton(this, QStringLiteral("toStart"), QStringLiteral("Go to start frame"));
    playStopBtn_ = makeTransportButton(this, QStringLiteral("play"), QStringLiteral("Play / Stop"));
    playStopBtn_->setCheckable(true);
    toEndBtn_ = makeTransportButton(this, QStringLiteral("toEnd"), QStringLiteral("Go to end frame"));

    scrubber_ = new TimelineScrubber(this);

    layout->addWidget(toStartBtn_);
    layout->addWidget(playStopBtn_);
    layout->addWidget(toEndBtn_);
    layout->addSpacing(6);
    layout->addWidget(scrubber_, 1);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &TimelineBar::onTick);
    connect(toStartBtn_, &QToolButton::clicked, this, &TimelineBar::goToStart);
    connect(toEndBtn_, &QToolButton::clicked, this, &TimelineBar::goToEnd);
    connect(playStopBtn_, &QToolButton::clicked, this, &TimelineBar::onPlayStop);
    connect(scrubber_, &TimelineScrubber::frameChanged, this, &TimelineBar::onScrubFrame);
    connect(scrubber_, &TimelineScrubber::scrubStarted, this, [this] {
        if (playing_) stopPlayback();
    });

    syncFromState();
}

double TimelineBar::timeSeconds() const {
    return double(currentFrame_ - 1) / std::max(1e-6, fps_);
}

void TimelineBar::setRange(int startFrame, int endFrame) {
    if (endFrame < startFrame) endFrame = startFrame;
    startFrame_ = startFrame;
    endFrame_ = endFrame;
    clampFrame();
    syncFromState();
}

void TimelineBar::setFps(double fps) {
    fps_ = std::max(1.0, fps);
    if (playing_) timer_->setInterval(std::max(1, int(std::round(1000.0 / fps_))));
}

void TimelineBar::setCurrentFrame(int frame) {
    currentFrame_ = frame;
    clampFrame();
    syncFromState();
    emitFrame();
}

void TimelineBar::suggestTimeRange(double startSeconds, double endSeconds, bool force) {
    if (rangeTouched_ && !force) return;
    if (!(endSeconds >= startSeconds)) return;
    const double rate = std::max(1e-6, fps_);
    int start = 1 + int(std::floor(startSeconds * rate + 1e-6));
    int end = 1 + int(std::ceil(endSeconds * rate - 1e-6));
    if (start < 1) start = 1;
    if (end < start) end = start;
    startFrame_ = start;
    endFrame_ = end;
    clampFrame();
    syncFromState();
}

void TimelineBar::play() {
    if (playing_) return;
    if (currentFrame_ >= endFrame_) currentFrame_ = startFrame_;
    playing_ = true;
    const int intervalMs = int(std::round(1000.0 / std::max(1.0, fps_)));
    timer_->start(std::max(1, intervalMs));
    updatePlayStopIcon();
    syncFromState();
    emitFrame();
}

void TimelineBar::stopPlayback() {
    if (!playing_) return;
    playing_ = false;
    timer_->stop();
    updatePlayStopIcon();
    emit playbackStopped();
}

void TimelineBar::goToStart() {
    stopPlayback();
    currentFrame_ = startFrame_;
    syncFromState();
    emitFrame();
}

void TimelineBar::goToEnd() {
    stopPlayback();
    currentFrame_ = endFrame_;
    syncFromState();
    emitFrame();
}

void TimelineBar::onPlayStop() {
    if (playing_) stopPlayback();
    else play();
}

void TimelineBar::onTick() {
    if (!playing_) return;
    ++currentFrame_;
    if (currentFrame_ > endFrame_) currentFrame_ = startFrame_;
    syncFromState();
    emitFrame();
}

void TimelineBar::onScrubFrame(int frame) {
    if (updating_) return;
    rangeTouched_ = true;
    currentFrame_ = frame;
    clampFrame();
    emitFrame();
}

void TimelineBar::syncFromState() {
    updating_ = true;
    scrubber_->setRange(startFrame_, endFrame_);
    scrubber_->setFrame(currentFrame_);
    updatePlayStopIcon();
    updating_ = false;
}

void TimelineBar::clampFrame() {
    if (endFrame_ < startFrame_) endFrame_ = startFrame_;
    if (currentFrame_ < startFrame_) currentFrame_ = startFrame_;
    if (currentFrame_ > endFrame_) currentFrame_ = endFrame_;
}

void TimelineBar::updatePlayStopIcon() {
    if (!playStopBtn_) return;
    playStopBtn_->setChecked(playing_);
    playStopBtn_->setIcon(makeHoudiniTransportIcon(playing_ ? QStringLiteral("stop")
                                                            : QStringLiteral("play")));
    playStopBtn_->setToolTip(playing_ ? QStringLiteral("Stop") : QStringLiteral("Play"));
}

void TimelineBar::emitFrame() { emit frameChanged(currentFrame_); }

}  // namespace sol
