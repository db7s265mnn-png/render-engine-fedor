// Shared dock chrome: a plain square to float / redock panes without changing
// QMainWindow's default split (viewport stays the central widget).
#pragma once

#include <functional>

#include <QColor>
#include <QDockWidget>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QShowEvent>
#include <QSizePolicy>
#include <QToolButton>

#include "ui/theme.h"

namespace sol {

class DockDetachButton : public QToolButton {
public:
    explicit DockDetachButton(QWidget* parent = nullptr) : QToolButton(parent) {
        setObjectName(QStringLiteral("dockDetachButton"));
        setFocusPolicy(Qt::NoFocus);
        setCursor(Qt::ArrowCursor);
        setFixedSize(18, 18);
        setAutoRaise(true);
        setAttribute(Qt::WA_Hover, true);
        setToolTip(QStringLiteral("Detach"));
        setStyleSheet(QStringLiteral(
            "QToolButton#dockDetachButton {"
            "  background: transparent;"
            "  border: none;"
            "  padding: 0px;"
            "}"));
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing, false);
        constexpr int kSide = 10;
        const QRect square((width() - kSide) / 2, (height() - kSide) / 2, kSide, kSide);
        painter.fillRect(square, underMouse() ? QColor(0x7a, 0x7e, 0x86) : QColor(0x5c, 0x60, 0x66));
        painter.setPen(QColor(0x2a, 0x2d, 0x32));
        painter.drawRect(square.adjusted(0, 0, -1, -1));
    }

    void mousePressEvent(QMouseEvent* event) override {
        QToolButton::mousePressEvent(event);
        event->accept();
    }
};

class DockTitleBar : public QWidget {
public:
    DockTitleBar(const QString& title, QWidget* parent, std::function<void()> onDetach,
                 QDockWidget* dock = nullptr)
        : QWidget(parent), onDetach_(std::move(onDetach)), dock_(dock) {
        setObjectName("dockTitleBar");
        setStyleSheet(
            "QWidget#dockTitleBar {"
            "  background: #2e3136;"
            "  border-bottom: 1px solid #22242a;"
            "}"
            "QLabel {"
            "  color: #dcdee2;"
            "  font-weight: 700;"
            "  background: transparent;"
            "  border: none;"
            "}");
        auto* layout = new QHBoxLayout(this);
        // Right margin reserved for the overlay detach square so title text
        // never runs under it.
        layout->setContentsMargins(10, 0, 18 + 12, 0);
        layout->setSpacing(6);
        auto* label = new QLabel(title, this);
        // Let the dock shrink below the title string; the label elides.
        label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
        label->setMinimumWidth(0);
        layout->addWidget(label, 1);
        detach_ = new DockDetachButton(this);
        detach_->setToolTip(QStringLiteral("Detach"));
        if (dock_) {
            connect(dock_, &QDockWidget::windowTitleChanged, label, &QLabel::setText);
            connect(dock_, &QDockWidget::topLevelChanged, this, [this](bool floating) {
                if (detach_)
                    detach_->setToolTip(floating ? QStringLiteral("Dock") : QStringLiteral("Detach"));
            });
        }
        connect(detach_, &QToolButton::clicked, this, [this] {
            if (onDetach_) onDetach_();
        });
        layoutDetachButton();
    }

    void setOnDetach(std::function<void()> onDetach) { onDetach_ = std::move(onDetach); }

    QSize sizeHint() const override { return {120, theme::chromeBarHeight()}; }
    QSize minimumSizeHint() const override { return {56, theme::chromeBarHeight()}; }

protected:
    void mousePressEvent(QMouseEvent* event) override { event->ignore(); }
    void mouseReleaseEvent(QMouseEvent* event) override { event->ignore(); }
    void mouseMoveEvent(QMouseEvent* event) override { event->ignore(); }
    void mouseDoubleClickEvent(QMouseEvent* event) override { event->ignore(); }
    void resizeEvent(QResizeEvent* event) override {
        QWidget::resizeEvent(event);
        layoutDetachButton();
    }
    void showEvent(QShowEvent* event) override {
        QWidget::showEvent(event);
        layoutDetachButton();
    }

private:
    void layoutDetachButton() {
        if (!detach_) return;
        constexpr int kMargin = 6;
        const int y = qMax(0, (height() - detach_->height()) / 2);
        detach_->move(width() - detach_->width() - kMargin, y);
        detach_->raise();
    }

    std::function<void()> onDetach_;
    QDockWidget* dock_ = nullptr;
    DockDetachButton* detach_ = nullptr;
};

void installDetachableTitleBar(QDockWidget* dock, std::function<void()> onDetach = {});

}  // namespace sol
