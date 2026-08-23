#include "ui/theme.h"

#include <QIcon>
#include <QLocale>
#include <QPainter>
#include <QPalette>
#include <QPixmap>
#include <QProxyStyle>
#include <QStyleFactory>
#include <QStyleOption>

namespace sol {
namespace {

// Dock / splitter resize grips: three larger square dots in place of Fusion's
// tiny speckles, keeping the same overall hit area.
class SolsticeStyle : public QProxyStyle {
public:
    explicit SolsticeStyle(QStyle* base) : QProxyStyle(base) {}

    int pixelMetric(PixelMetric metric, const QStyleOption* option = nullptr,
                    const QWidget* widget = nullptr) const override {
        if (metric == PM_DockWidgetSeparatorExtent || metric == PM_SplitterWidth) return 8;
        return QProxyStyle::pixelMetric(metric, option, widget);
    }

    void drawPrimitive(PrimitiveElement element, const QStyleOption* option, QPainter* painter,
                       const QWidget* widget = nullptr) const override {
        if (element == PE_IndicatorDockWidgetResizeHandle) {
            drawGrip(option, painter);
            return;
        }
        QProxyStyle::drawPrimitive(element, option, painter, widget);
    }

    void drawControl(ControlElement element, const QStyleOption* option, QPainter* painter,
                     const QWidget* widget = nullptr) const override {
        if (element == CE_Splitter) {
            drawGrip(option, painter);
            return;
        }
        QProxyStyle::drawControl(element, option, painter, widget);
    }

    QIcon standardIcon(StandardPixmap icon, const QStyleOption* option = nullptr,
                       const QWidget* widget = nullptr) const override {
        if (icon == SP_TitleBarNormalButton) return detachSquareIcon();
        return QProxyStyle::standardIcon(icon, option, widget);
    }

private:
    static QIcon detachSquareIcon() {
        QPixmap pixmap(16, 16);
        pixmap.fill(Qt::transparent);
        QPainter painter(&pixmap);
        painter.setRenderHint(QPainter::Antialiasing, false);
        painter.fillRect(3, 3, 10, 10, QColor(0x5c, 0x60, 0x66));
        painter.setPen(QColor(0x2a, 0x2d, 0x32));
        painter.drawRect(3, 3, 9, 9);
        painter.end();
        return QIcon(pixmap);
    }
    static void drawGrip(const QStyleOption* option, QPainter* painter) {
        if (!option || !painter) return;
        const QRect r = option->rect;
        painter->fillRect(r, QColor(0x2e, 0x31, 0x36));

        const bool horizontal = option->state & State_Horizontal;
        constexpr int kDot = 3;
        constexpr int kGap = 4;
        constexpr int kCount = 3;
        const int span = kCount * kDot + (kCount - 1) * kGap;
        const QColor fill = (option->state & State_MouseOver) ? QColor(0xb0, 0xb4, 0xbc)
                                                              : QColor(0x8a, 0x8e, 0x96);

        painter->save();
        painter->setRenderHint(QPainter::Antialiasing, false);
        painter->setPen(Qt::NoPen);
        painter->setBrush(fill);

        if (horizontal) {
            // Separator runs left–right: three squares in a horizontal cluster.
            const int x0 = r.center().x() - span / 2;
            const int y0 = r.center().y() - kDot / 2;
            for (int i = 0; i < kCount; ++i) {
                painter->drawRect(x0 + i * (kDot + kGap), y0, kDot, kDot);
            }
        } else {
            // Separator runs top–bottom: three squares stacked.
            const int x0 = r.center().x() - kDot / 2;
            const int y0 = r.center().y() - span / 2;
            for (int i = 0; i < kCount; ++i) {
                painter->drawRect(x0, y0 + i * (kDot + kGap), kDot, kDot);
            }
        }
        painter->restore();
    }
};

}  // namespace

void applyDarkTheme(QApplication& application) {
    auto* style = new SolsticeStyle(QStyleFactory::create("Fusion"));
    application.setStyle(style);
    // Force '.' as the decimal separator in spin boxes and number formatting.
    QLocale::setDefault(QLocale::c());

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
        QDockWidget::title {
            background: #2e3136;
            padding: 0 10px;
            min-height: 34px;
            max-height: 34px;
        }
        QMainWindow::separator {
            background: #2e3136;
            width: 8px;
            height: 8px;
        }
        QSplitter::handle {
            background: #2e3136;
        }
        QGroupBox { border: 1px solid #3a3e44; border-radius: 2px; margin-top: 14px; padding-top: 6px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #ffa82e; }
        QTabWidget::pane {
            border: 1px solid #555960;
            background: #3a3e44;
            padding: 2px;
        }
        QTabBar::tab {
            background: #2a2e33;
            color: #d0d4da;
            padding: 6px 12px;
            margin-right: 2px;
            border: 1px solid #555960;
            min-height: 22px;
            min-width: 4em;
        }
        QTabBar::tab:selected {
            background: rgba(255, 190, 90, 90);
            color: #ffffff;
            border: 1px solid #ffbe5a;
        }
        QTabBar::tab:hover:!selected {
            color: #e8eaed;
            background: #32363c;
        }
        QTabBar::scroller { width: 24px; }
        QLineEdit, QComboBox, QPlainTextEdit, QTreeWidget {
            background: #22242a; border: 1px solid #3a3e44; border-radius: 2px; padding: 2px 4px;
        }
        /* Padding on QSpinBox breaks the embedded line-edit hit-test (can't place caret). */
        QSpinBox, QDoubleSpinBox {
            background: #22242a; border: 1px solid #3a3e44; border-radius: 2px; padding: 0px;
        }
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            width: 0px; border: none;
        }
        QPushButton {
            background: #3a3e44; border: 1px solid #4a4f57; border-radius: 6px; padding: 4px 10px;
        }
        QPushButton:hover { background: #474c54; }
        QPushButton:pressed { background: #2b2e33; }
        QSlider::groove:horizontal { height: 4px; background: #22242a; border-radius: 2px; }
        QSlider::handle:horizontal {
            background: #8a8f98; width: 10px; margin: -4px 0; border-radius: 5px;
        }
        QSlider::handle:horizontal:hover { background: #ffbe5a; }
        QStatusBar { background: #2e3136; }
        QHeaderView::section { background: #2e3136; padding: 3px; border: none; }
    )");
}

}  // namespace sol
