#include "nodes/parameter.h"

#include <QJsonArray>

namespace sol {

namespace {
QVariant vecToVariant(Vec3 v) { return QVariant::fromValue(QVector3D(v.x, v.y, v.z)); }
}  // namespace

Parameter Parameter::makeFloat(const QString& name, const QString& label, double value, double minValue,
                               double maxValue, bool hasRange) {
    Parameter p;
    p.name = name;
    p.label = label;
    p.type = ParamType::Float;
    p.value = value;
    p.defaultValue = value;
    p.minValue = minValue;
    p.maxValue = maxValue;
    p.hasRange = hasRange;
    return p;
}

Parameter Parameter::makeInt(const QString& name, const QString& label, int value, int minValue, int maxValue,
                             bool hasRange) {
    Parameter p;
    p.name = name;
    p.label = label;
    p.type = ParamType::Int;
    p.value = value;
    p.defaultValue = value;
    p.minValue = minValue;
    p.maxValue = maxValue;
    p.hasRange = hasRange;
    return p;
}

Parameter Parameter::makeBool(const QString& name, const QString& label, bool value) {
    Parameter p;
    p.name = name;
    p.label = label;
    p.type = ParamType::Bool;
    p.value = value;
    p.defaultValue = value;
    return p;
}

Parameter Parameter::makeVec3(const QString& name, const QString& label, Vec3 value) {
    Parameter p;
    p.name = name;
    p.label = label;
    p.type = ParamType::Vec3;
    p.value = vecToVariant(value);
    p.defaultValue = p.value;
    return p;
}

Parameter Parameter::makeColor(const QString& name, const QString& label, Vec3 value) {
    Parameter p;
    p.name = name;
    p.label = label;
    p.type = ParamType::Color;
    p.value = vecToVariant(value);
    p.defaultValue = p.value;
    return p;
}

Parameter Parameter::makeString(const QString& name, const QString& label, const QString& value) {
    Parameter p;
    p.name = name;
    p.label = label;
    p.type = ParamType::String;
    p.value = value;
    p.defaultValue = value;
    return p;
}

Parameter Parameter::makeFile(const QString& name, const QString& label, const QString& value,
                              const QString& filter) {
    Parameter p;
    p.name = name;
    p.label = label;
    p.type = ParamType::FilePath;
    p.value = value;
    p.defaultValue = value;
    p.fileFilter = filter;
    return p;
}

Parameter Parameter::makeMenu(const QString& name, const QString& label, const QStringList& items, int index) {
    Parameter p;
    p.name = name;
    p.label = label;
    p.type = ParamType::Menu;
    p.menuItems = items;
    p.value = index;
    p.defaultValue = index;
    return p;
}

Parameter Parameter::makeLabel(const QString& name, const QString& text) {
    Parameter p;
    p.name = name;
    p.label = text;
    p.type = ParamType::Label;
    p.value = text;
    p.defaultValue = text;
    return p;
}

Parameter& Parameter::withGroup(const QString& groupName) {
    group = groupName;
    return *this;
}

Parameter& Parameter::withTooltip(const QString& text) {
    tooltip = text;
    return *this;
}

Vec3 Parameter::toVec3() const {
    const QVector3D v = value.value<QVector3D>();
    return Vec3(v.x(), v.y(), v.z());
}

void Parameter::setVec3(Vec3 v) { value = vecToVariant(v); }

QString paramTypeName(ParamType type) {
    switch (type) {
        case ParamType::Float: return "float";
        case ParamType::Int: return "int";
        case ParamType::Bool: return "bool";
        case ParamType::Vec3: return "vec3";
        case ParamType::Color: return "color";
        case ParamType::String: return "string";
        case ParamType::FilePath: return "file";
        case ParamType::Menu: return "menu";
        case ParamType::Label: return "label";
    }
    return "float";
}

ParamType paramTypeFromName(const QString& name) {
    if (name == "int") return ParamType::Int;
    if (name == "bool") return ParamType::Bool;
    if (name == "vec3") return ParamType::Vec3;
    if (name == "color") return ParamType::Color;
    if (name == "string") return ParamType::String;
    if (name == "file") return ParamType::FilePath;
    if (name == "menu") return ParamType::Menu;
    if (name == "label") return ParamType::Label;
    return ParamType::Float;
}

QJsonObject Parameter::toJson() const {
    QJsonObject json;
    json["name"] = name;
    json["type"] = paramTypeName(type);
    switch (type) {
        case ParamType::Vec3:
        case ParamType::Color: {
            const Vec3 v = toVec3();
            QJsonArray array{v.x, v.y, v.z};
            json["value"] = array;
            break;
        }
        case ParamType::Bool: json["value"] = value.toBool(); break;
        case ParamType::Int:
        case ParamType::Menu: json["value"] = value.toInt(); break;
        case ParamType::String:
        case ParamType::FilePath:
        case ParamType::Label: json["value"] = value.toString(); break;
        default: json["value"] = value.toDouble(); break;
    }
    return json;
}

void Parameter::fromJson(const QJsonObject& json) {
    const QJsonValue jsonValue = json.value("value");
    switch (type) {
        case ParamType::Vec3:
        case ParamType::Color: {
            const QJsonArray array = jsonValue.toArray();
            if (array.size() == 3)
                setVec3(Vec3(float(array[0].toDouble()), float(array[1].toDouble()), float(array[2].toDouble())));
            break;
        }
        case ParamType::Bool: value = jsonValue.toBool(); break;
        case ParamType::Int:
        case ParamType::Menu: value = jsonValue.toInt(); break;
        case ParamType::String:
        case ParamType::FilePath:
        case ParamType::Label: value = jsonValue.toString(); break;
        default: value = jsonValue.toDouble(); break;
    }
}

}  // namespace sol
