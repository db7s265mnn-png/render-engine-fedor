// Node parameters. The parameter panel builds its widgets straight from this
// description, and the scene serialiser reads and writes it verbatim.
#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>
#include <QVariant>
#include <QVector3D>

#include "core/math.h"

namespace sol {

enum class ParamType {
    Float,
    Int,
    Bool,
    Vec3,
    Color,
    String,
    FilePath,
    Menu,
    Label,
    Button,
};

class Parameter {
public:
    QString name;
    QString label;
    QString tooltip;
    QString group;        // parameter folder shown in the UI
    // Declarative show/hide vs sibling params, e.g. "integrator==4",
    // "caustics==1&&causticsengine==2", "integrator==0||integrator==1".
    // Empty = always visible. "false" / "0" = never.
    QString visibleWhen;
    ParamType type = ParamType::Float;
    QVariant value;
    QVariant defaultValue;
    // Houdini-style expression (math + $F/$F#). When non-empty, value is the last
    // evaluated cache and floatValue/intValue re-evaluate from expression.
    QString expression;
    double minValue = 0.0;
    double maxValue = 1.0;
    bool hasRange = false;
    bool locked = false;
    QStringList menuItems;
    QString fileFilter;
    bool fileSaveMode = false;  // FilePath: save dialog instead of open
    bool fileDirectoryMode = false;  // FilePath: choose a directory

    Parameter() = default;

    static Parameter makeFloat(const QString& name, const QString& label, double value, double minValue = 0.0,
                               double maxValue = 1.0, bool hasRange = true);
    static Parameter makeInt(const QString& name, const QString& label, int value, int minValue = 0,
                             int maxValue = 100, bool hasRange = true);
    static Parameter makeBool(const QString& name, const QString& label, bool value);
    static Parameter makeVec3(const QString& name, const QString& label, Vec3 value);
    static Parameter makeColor(const QString& name, const QString& label, Vec3 value);
    static Parameter makeString(const QString& name, const QString& label, const QString& value);
    static Parameter makeFile(const QString& name, const QString& label, const QString& value,
                              const QString& filter);
    static Parameter makeMenu(const QString& name, const QString& label, const QStringList& items, int index);
    static Parameter makeLabel(const QString& name, const QString& text);
    static Parameter makeButton(const QString& name, const QString& label);

    Parameter& withGroup(const QString& groupName);
    Parameter& withTooltip(const QString& text);
    Parameter& withVisibleWhen(const QString& expression);
    Parameter& withFileSaveMode(bool enabled = true);
    Parameter& withDirectoryMode(bool enabled = true);

    double toDouble() const { return value.toDouble(); }
    int toInt() const { return value.toInt(); }
    bool toBool() const { return value.toBool(); }
    QString toString() const { return value.toString(); }
    Vec3 toVec3() const;
    void setVec3(Vec3 v);

    bool hasExpression() const { return !expression.trimmed().isEmpty(); }
    // Evaluate expression (or return literal). Uses sol::exprFrame() when frame < 0.
    double evaluatedNumber(int frame = -1) const;
    QString evaluatedString(int frame = -1) const;

    QJsonObject toJson() const;
    void fromJson(const QJsonObject& json);
};

QString paramTypeName(ParamType type);
ParamType paramTypeFromName(const QString& name);

// Evaluate a Parameter::visibleWhen expression against a node’s current values.
bool evaluateVisibleWhen(const QString& expression, const class Node& node);

// Empty-group tab title. Scene-graph sources (geo / file / lights / camera / vdb)
// use "Base"; everyone else keeps "Parameters".
inline QString defaultParameterFolderTitle(const QString& nodeType) {
    static const QStringList kBaseTypes{
        QStringLiteral("alembic"),
        QStringLiteral("usd"),
        QStringLiteral("sphere"),
        QStringLiteral("grid"),
        QStringLiteral("box"),
        QStringLiteral("tube"),
        QStringLiteral("camera"),
        QStringLiteral("domelight"),
        QStringLiteral("distantlight"),
        QStringLiteral("rectlight"),
        QStringLiteral("disklight"),
        QStringLiteral("spherelight"),
        QStringLiteral("physicalskylight"),
        QStringLiteral("vdbfrompolygons"),
        QStringLiteral("vdbfile"),
        QStringLiteral("sdftopolygons_vdb"),
        QStringLiteral("sdftopolygons_dcsdd"),
    };
    return kBaseTypes.contains(nodeType) ? QStringLiteral("Base") : QStringLiteral("Parameters");
}

}  // namespace sol
