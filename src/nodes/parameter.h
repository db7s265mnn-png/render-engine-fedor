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
};

class Parameter {
public:
    QString name;
    QString label;
    QString tooltip;
    QString group;        // parameter folder shown in the UI
    ParamType type = ParamType::Float;
    QVariant value;
    QVariant defaultValue;
    double minValue = 0.0;
    double maxValue = 1.0;
    bool hasRange = false;
    bool locked = false;
    QStringList menuItems;
    QString fileFilter;

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

    Parameter& withGroup(const QString& groupName);
    Parameter& withTooltip(const QString& text);

    double toDouble() const { return value.toDouble(); }
    int toInt() const { return value.toInt(); }
    bool toBool() const { return value.toBool(); }
    QString toString() const { return value.toString(); }
    Vec3 toVec3() const;
    void setVec3(Vec3 v);

    QJsonObject toJson() const;
    void fromJson(const QJsonObject& json);
};

QString paramTypeName(ParamType type);
ParamType paramTypeFromName(const QString& name);

}  // namespace sol
