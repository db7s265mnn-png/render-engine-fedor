// Dark application theme shared by every widget.
#pragma once

#include <QApplication>
#include <QColor>
#include <QString>

namespace sol {

void applyDarkTheme(QApplication& application);

namespace theme {
inline QColor background() { return QColor(37, 39, 43); }
inline QColor panel() { return QColor(46, 49, 54); }
inline QColor panelLight() { return QColor(58, 62, 68); }
inline QColor gridDark() { return QColor(32, 34, 38); }
inline QColor gridLine() { return QColor(48, 51, 56); }
inline QColor text() { return QColor(220, 222, 226); }
inline QColor textDim() { return QColor(150, 154, 160); }
inline QColor accent() { return QColor(255, 168, 46); }
inline QColor selection() { return QColor(255, 200, 90); }
inline QColor wire() { return QColor(150, 155, 165); }
inline QColor wireActive() { return QColor(255, 190, 90); }
inline QColor displayFlag() { return QColor(80, 170, 255); }
inline QColor error() { return QColor(220, 90, 80); }
}  // namespace theme

}  // namespace sol
