#include "ui/dock_chrome.h"

#include <QAbstractButton>
#include <QApplication>
#include <QChildEvent>
#include <QEvent>
#include <QTabBar>
#include <QTimer>

namespace sol {
namespace {

bool isNativeDockWindowButton(QObject* object) {
    auto* button = qobject_cast<QAbstractButton*>(object);
    if (!button) return false;
    const QString name = button->objectName();
    return name == QLatin1String("qt_dockwidget_floatbutton")
        || name == QLatin1String("qt_dockwidget_closebutton");
}

void hideNativeDockWindowButtons(QWidget* root) {
    if (!root) return;
    for (QAbstractButton* button : root->findChildren<QAbstractButton*>()) {
        if (!isNativeDockWindowButton(button)) continue;
        button->hide();
        button->setEnabled(false);
        button->setFixedSize(0, 0);
    }
}

class NativeDockButtonFilter : public QObject {
public:
    using QObject::QObject;
    bool eventFilter(QObject* watched, QEvent* event) override {
        if (event->type() == QEvent::ChildAdded) {
            QObject* child = static_cast<QChildEvent*>(event)->child();
            if (isNativeDockWindowButton(child)) {
                auto* button = static_cast<QAbstractButton*>(child);
                button->hide();
                button->setEnabled(false);
                button->setFixedSize(0, 0);
            } else if (qobject_cast<QTabBar*>(child) || qobject_cast<QDockWidget*>(child)) {
                child->installEventFilter(this);
                if (auto* widget = qobject_cast<QWidget*>(child))
                    hideNativeDockWindowButtons(widget);
            }
        } else if (event->type() == QEvent::Show) {
            if (auto* widget = qobject_cast<QWidget*>(watched))
                hideNativeDockWindowButtons(widget);
        }
        return QObject::eventFilter(watched, event);
    }
};

void ensureNativeDockButtonFilter(QWidget* widget) {
    static NativeDockButtonFilter* filter = nullptr;
    if (!filter && qApp) {
        filter = new NativeDockButtonFilter(qApp);
        qApp->installEventFilter(filter);
    }
    if (!widget) return;
    hideNativeDockWindowButtons(widget);
    if (filter) widget->installEventFilter(filter);
    if (QWidget* window = widget->window()) {
        if (filter) window->installEventFilter(filter);
        hideNativeDockWindowButtons(window);
        for (QTabBar* bar : window->findChildren<QTabBar*>()) {
            if (filter) bar->installEventFilter(filter);
            hideNativeDockWindowButtons(bar);
        }
    }
}

}  // namespace

void installDetachableTitleBar(QDockWidget* dock, std::function<void()> onDetach) {
    if (!dock) return;
    dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
    if (!onDetach) {
        onDetach = [dock] { dock->setFloating(!dock->isFloating()); };
    }
    dock->setTitleBarWidget(new DockTitleBar(dock->windowTitle(), dock, onDetach, dock));
    // Closing a floating window must redock it — these panes have no close button
    // and must not vanish from the layout.
    QObject::connect(dock, &QDockWidget::visibilityChanged, dock, [dock, onDetach](bool visible) {
        if (visible || !dock->isFloating()) return;
        QTimer::singleShot(0, dock, [dock, onDetach] {
            if (dock->isVisible()) return;
            if (onDetach) onDetach();
            if (dock->isFloating()) {
                dock->setFloating(false);
                dock->show();
            }
        });
    });
    ensureNativeDockButtonFilter(dock);
}

}  // namespace sol
