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
        layout->setContentsMargins(10, 0, 8, 0);
        layout->setSpacing(6);
        auto* label = new QLabel(title, this);
        // Let the dock shrink below the title string; the label elides.
        label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
        label->setMinimumWidth(0);
        layout->addWidget(label, 1);
        auto* detach = new DockDetachButton(this);
        detach->setToolTip(QStringLiteral("Detach"));
        layout->addWidget(detach, 0, Qt::AlignVCenter);
        if (dock_) {
            connect(dock_, &QDockWidget::windowTitleChanged, label, &QLabel::setText);
            connect(dock_, &QDockWidget::topLevelChanged, this, [detach](bool floating) {
                detach->setToolTip(floating ? QStringLiteral("Dock") : QStringLiteral("Detach"));
            });
        }
        connect(detach, &QToolButton::clicked, this, [this] {
            if (onDetach_) onDetach_();
        });
    }

    void setOnDetach(std::function<void()> onDetach) { onDetach_ = std::move(onDetach); }

    QSize sizeHint() const override { return {120, theme::chromeBarHeight()}; }
    QSize minimumSizeHint() const override { return {56, theme::chromeBarHeight()}; }

protected:
    void mousePressEvent(QMouseEvent* event) override { event->ignore(); }
    void mouseReleaseEvent(QMouseEvent* event) override { event->ignore(); }
    void mouseMoveEvent(QMouseEvent* event) override { event->ignore(); }
    void mouseDoubleClickEvent(QMouseEvent* event) override { event->ignore(); }

private:
    std::function<void()> onDetach_;
    QDockWidget* dock_ = nullptr;
};

void installDetachableTitleBar(QDockWidget* dock, std::function<void()> onDetach = {});

}  // namespace sol
