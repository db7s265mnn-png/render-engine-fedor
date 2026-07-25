#include "ui/theme.h"

#include <QPalette>
#include <QStyleFactory>

namespace sol {

void applyDarkTheme(QApplication& application) {
    application.setStyle(QStyleFactory::create("Fusion"));

    QPalette palette;
    palette.setColor(QPalette::Window, theme::background());
    palette.setColor(QPalette::WindowText, theme::text());
    palette.setColor(QPalette::Base, theme::gridDark());
    palette.setColor(QPalette::AlternateBase, theme::panel());
    palette.setColor(QPalette::ToolTipBase, theme::panelLight());
    palette.setColor(QPalette::ToolTipText, theme::text());
    palette.setColor(QPalette::Text, theme::text());
    palette.setColor(QPalette::Button, theme::panel());
    palette.setColor(QPalette::ButtonText, theme::text());
    palette.setColor(QPalette::BrightText, Qt::red);
    palette.setColor(QPalette::Link, theme::accent());
    palette.setColor(QPalette::Highlight, theme::accent());
    palette.setColor(QPalette::HighlightedText, QColor(20, 20, 20));
    palette.setColor(QPalette::Disabled, QPalette::Text, theme::textDim());
    palette.setColor(QPalette::Disabled, QPalette::ButtonText, theme::textDim());
    application.setPalette(palette);

    application.setStyleSheet(R"(
        QToolTip { color: #dcdee2; background-color: #3a3e44; border: 1px solid #22242a; }
        QDockWidget { titlebar-close-icon: none; font-weight: bold; }
        QDockWidget::title { background: #2e3136; padding: 5px; }
        QGroupBox { border: 1px solid #3a3e44; border-radius: 3px; margin-top: 14px; padding-top: 6px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #ffa82e; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTreeWidget {
            background: #22242a; border: 1px solid #3a3e44; border-radius: 3px; padding: 2px 4px;
        }
        QPushButton {
            background: #3a3e44; border: 1px solid #4a4f57; border-radius: 3px; padding: 4px 10px;
        }
        QPushButton:hover { background: #474c54; }
        QPushButton:pressed { background: #2b2e33; }
        QSlider::groove:horizontal { height: 4px; background: #22242a; border-radius: 2px; }
        QSlider::handle:horizontal {
            background: #8a8f98; width: 10px; margin: -4px 0; border-radius: 5px;
        }
        QSlider::handle:horizontal:hover { background: #ffa82e; }
        QStatusBar { background: #2e3136; }
        QHeaderView::section { background: #2e3136; padding: 3px; border: none; }
    )");
}

}  // namespace sol
