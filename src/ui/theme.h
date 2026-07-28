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

// Shared height for viewport chrome + side-dock title bars.
inline constexpr int chromeBarHeight() { return 34; }

// Houdini / Arnold-style MaterialX port colours by data type.
inline QColor colorForMaterialXType(const QString& type) {
    const QString t = type.trimmed().toLower();
    if (t == "float") return QColor(92, 196, 92);
    if (t == "integer" || t == "int") return QColor(58, 158, 110);
    if (t == "boolean" || t == "bool") return QColor(220, 140, 55);
    if (t == "color3" || t == "color") return QColor(230, 200, 70);
    if (t == "color4") return QColor(210, 175, 55);
    if (t == "vector2") return QColor(90, 155, 230);
    if (t == "vector3" || t == "vector") return QColor(70, 135, 220);
    if (t == "vector4") return QColor(55, 115, 205);
    if (t.startsWith("matrix")) return QColor(55, 175, 165);
    if (t == "filename") return QColor(200, 90, 170);
    if (t == "string" || t == "token") return QColor(170, 120, 185);
    if (t == "surfaceshader" || t == "bsdf" || t == "edf" || t == "vdf") return QColor(230, 140, 55);
    if (t == "displacementshader") return QColor(180, 110, 70);
    if (t == "volumeshader") return QColor(120, 160, 200);
    if (t == "lightshader") return QColor(230, 200, 100);
    if (t == "material") return QColor(160, 100, 210);
    if (t.endsWith("shader")) return QColor(230, 140, 55);
    return QColor(140, 145, 155);
}
}  // namespace theme

}  // namespace sol
