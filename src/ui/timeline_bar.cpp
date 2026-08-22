#include "ui/timeline_bar.h"

#include <QPainter>
#include <QPainterPath>
#include <QLineEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QMouseEvent>
#include <QToolButton>
#include <QTimer>
#include <QIntValidator>
#include <QFont>
#include <QFontMetrics>
#include <cmath>
#include <algorithm>

#include "ui/theme.h"

namespace sol {
namespace {

// Shared chrome for start / playhead / end frame boxes.
constexpr int kFrameBoxWidth = 40;
constexpr int kFrameBoxHeight = 16;
constexpr int kFrameNumberPixelSize = 11;
constexpr qreal kPlayheadWidth = qreal(kFrameBoxWidth);
constexpr qreal kPlayheadHeight = qreal(kFrameBoxHeight);

QColor frameBoxBg() { return QColor(0x1a, 0x1c, 0x20); }
QColor frameBoxBorder() { return QColor(0x7a, 0x7e, 0x86); }
QColor frameBoxText() { return QColor(0xd8, 0xda, 0xe0); }

QFont frameNumberFont() {
    QFont font;
    font.setPixelSize(kFrameNumberPixelSize);
    font.setBold(true);
    return font;
}

QString frameBoxStyleSheet() {
    return QStringLiteral(
        "QLineEdit {"
        "  background: #1a1c20;"
        "  color: #d8dae0;"
        "  border: 1px solid #7a7e86;"
        "  border-radius: 2px;"
        "  padding: 0px;"
        "  font-size: %1px;"
        "  font-weight: 700;"
        "}"
        "QLineEdit:focus { border: 1px solid #50aaff; }")
        .arg(kFrameNumberPixelSize);
}

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
        // |<<
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
    setFixedHeight(kFrameBoxHeight);
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
    // Inset by half the playhead so the box stays inside the scrubber and docks
    // flush against the start/end widgets at the range ends (no overlap).
    const qreal inset = kPlayheadWidth * 0.5;
    return QRectF(inset, height() * 0.5 - 1.0, std::max(8.0, width() - 2.0 * inset), 2.0);
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
    return QRectF(x - kPlayheadWidth * 0.5, height() * 0.5 - kPlayheadHeight * 0.5, kPlayheadWidth,
                  kPlayheadHeight);
}

void TimelineScrubber::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRectF track = trackRect();
    // Groove with grey border
    p.setPen(QPen(QColor(110, 114, 122), 1.0));
    p.setBrush(QColor(72, 76, 82));
    p.drawRoundedRect(track.adjusted(0, -2, 0, 2), 2.0, 2.0);

    // Tick marks
    const int span = std::max(1, endFrame_ - startFrame_);
    int step = 1;
    if (span > 200) step = 10;
    else if (span > 80) step = 5;
    else if (span > 40) step = 2;
    p.setPen(QPen(QColor(140, 144, 150), 1.0));
    for (int f = startFrame_; f <= endFrame_; f += step) {
        const qreal x = xForFrame(f);
        const bool major = ((f - startFrame_) % (step * 4) == 0) || f == startFrame_ || f == endFrame_;
        const qreal h = major ? 5.5 : 3.0;
        p.drawLine(QPointF(x, track.center().y() - h), QPointF(x, track.center().y() + h));
    }

    // Playhead — same grey chrome as start/end boxes, no center stem.
    const QRectF head = playheadRect();
    p.setPen(QPen(frameBoxBorder(), 1.0));
    p.setBrush(frameBoxBg());
    p.drawRoundedRect(head, 2.0, 2.0);

    p.setFont(frameNumberFont());
    p.setPen(frameBoxText());
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
        editor_->setGeometry(head.toRect());
    }
}

void TimelineScrubber::beginFrameEdit() {
    if (editor_) return;
    editor_ = new QLineEdit(this);
    editor_->setAlignment(Qt::AlignCenter);
    editor_->setText(QString::number(frame_));
    editor_->setValidator(new QIntValidator(-999999, 999999, editor_));
    editor_->setFixedSize(kFrameBoxWidth, kFrameBoxHeight);
    editor_->setFont(frameNumberFont());
    editor_->setTextMargins(0, 0, 0, 0);
    editor_->setStyleSheet(
        "QLineEdit { background: #1a1c20; color: #d8dae0; border: 1px solid #50aaff;"
        " border-radius: 2px; padding: 0px; font-size: 11px; font-weight: 700; }");
    editor_->move(playheadRect().toRect().topLeft());
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

QLineEdit* TimelineBar::makeRangeEdit(const QString& tip) {
    auto* edit = new QLineEdit(this);
    edit->setAlignment(Qt::AlignCenter);
    edit->setFixedSize(kFrameBoxWidth, kFrameBoxHeight);
    edit->setMaxLength(6);
    edit->setValidator(new QIntValidator(-999999, 999999, edit));
    edit->setToolTip(tip);
    edit->setFont(frameNumberFont());
    edit->setTextMargins(0, 0, 0, 0);
    edit->setStyleSheet(frameBoxStyleSheet());
    return edit;
}

TimelineBar::TimelineBar(QWidget* parent) : QWidget(parent) {
    setObjectName("timelineBar");
    setFixedHeight(54);
    setStyleSheet(QStringLiteral(
        "QWidget#timelineBar {"
        "  background: #2e3136;"
        "  border-top: 1px solid #22242a;"
        "  border-bottom: 1px solid #22242a;"
        "}"
        "QToolButton {"
        "  background: #3a3e44;"
        "  border: 1px solid #4a4f57;"
        "  border-radius: 6px;"
        "  padding: 0;"
        "}"
        "QToolButton:hover { background: #474c54; }"
        "QToolButton:pressed { background: #2a2d32; }"
        "QToolButton:checked {"
        "  background: rgba(80, 170, 255, 70);"
        "  border-color: #50aaff;"
        "}"));

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8, 3, 8, 3);
    root->setSpacing(2);

    auto* scrubRow = new QHBoxLayout();
    scrubRow->setContentsMargins(0, 0, 0, 0);
    scrubRow->setSpacing(0);

    startEdit_ = makeRangeEdit(QStringLiteral("Start frame"));
    endEdit_ = makeRangeEdit(QStringLiteral("End frame"));
    scrubber_ = new TimelineScrubber(this);

    scrubRow->addWidget(startEdit_, 0);
    scrubRow->addWidget(scrubber_, 1);
    scrubRow->addWidget(endEdit_, 0);

    auto* buttonRow = new QHBoxLayout();
    buttonRow->setContentsMargins(0, 0, 0, 0);
    buttonRow->setSpacing(4);

    toStartBtn_ = makeTransportButton(this, QStringLiteral("toStart"), QStringLiteral("Go to start frame"));
    playStopBtn_ = makeTransportButton(this, QStringLiteral("play"), QStringLiteral("Play / Stop"));
    playStopBtn_->setCheckable(true);
    toEndBtn_ = makeTransportButton(this, QStringLiteral("toEnd"), QStringLiteral("Go to end frame"));

    buttonRow->addStretch(1);
    buttonRow->addWidget(toStartBtn_);
    buttonRow->addWidget(playStopBtn_);
    buttonRow->addWidget(toEndBtn_);
    buttonRow->addStretch(1);

    root->addLayout(scrubRow);
    root->addLayout(buttonRow);

    timer_ = new QTimer(this);
    connect(timer_, &QTimer::timeout, this, &TimelineBar::onTick);
    connect(toStartBtn_, &QToolButton::clicked, this, &TimelineBar::goToStart);
    connect(toEndBtn_, &QToolButton::clicked, this, &TimelineBar::goToEnd);
    connect(playStopBtn_, &QToolButton::clicked, this, &TimelineBar::onPlayStop);
    connect(scrubber_, &TimelineScrubber::frameChanged, this, &TimelineBar::onScrubFrame);
    connect(scrubber_, &TimelineScrubber::scrubStarted, this, [this] {
        if (playing_) stopPlayback();
    });
    connect(scrubber_, &TimelineScrubber::scrubFinished, this, [this] { emit scrubFinished(); });
    connect(startEdit_, &QLineEdit::editingFinished, this, &TimelineBar::onStartEdited);
    connect(endEdit_, &QLineEdit::editingFinished, this, &TimelineBar::onEndEdited);

    syncFromState();
}

bool TimelineBar::isScrubbing() const { return scrubber_ && scrubber_->isScrubbing(); }

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

void TimelineBar::onStartEdited() {
    if (updating_ || !startEdit_) return;
    bool ok = false;
    int value = startEdit_->text().trimmed().toInt(&ok);
    if (!ok) {
        updating_ = true;
        startEdit_->setText(QString::number(startFrame_));
        updating_ = false;
        return;
    }
    rangeTouched_ = true;
    startFrame_ = value;
    if (endFrame_ < startFrame_) endFrame_ = startFrame_;
    clampFrame();
    syncFromState();
    emitFrame();
}

void TimelineBar::onEndEdited() {
    if (updating_ || !endEdit_) return;
    bool ok = false;
    int value = endEdit_->text().trimmed().toInt(&ok);
    if (!ok) {
        updating_ = true;
        endEdit_->setText(QString::number(endFrame_));
        updating_ = false;
        return;
    }
    rangeTouched_ = true;
    endFrame_ = value;
    if (endFrame_ < startFrame_) startFrame_ = endFrame_;
    clampFrame();
    syncFromState();
    emitFrame();
}

void TimelineBar::syncFromState() {
    updating_ = true;
    if (startEdit_) startEdit_->setText(QString::number(startFrame_));
    if (endEdit_) endEdit_->setText(QString::number(endFrame_));
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
