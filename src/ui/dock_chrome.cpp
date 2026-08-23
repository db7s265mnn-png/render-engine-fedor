#include "ui/dock_chrome.h"

#include <QTimer>

namespace sol {

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
}

}  // namespace sol
