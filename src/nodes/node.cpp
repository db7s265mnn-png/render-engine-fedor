#include "nodes/node.h"

#include <algorithm>
#include <cmath>

#include "core/expr_eval.h"
#include "core/log.h"
#include "core/units.h"

namespace sol {

void CookContext::reportError(const Node* node, const QString& message) {
    const QString text = (node ? node->name() + ": " : QString()) + message;
    errors << text;
    logError(text.toStdString());
}

void CookContext::reportWarning(const Node* node, const QString& message) {
    const QString text = (node ? node->name() + ": " : QString()) + message;
    warnings << text;
    logWarning(text.toStdString());
}

Node::Node(QString typeName, QString name) : typeName_(std::move(typeName)), name_(std::move(name)) {
    setInputLabels({"Input"});
}

Node::~Node() = default;

void Node::setName(const QString& name) {
    if (name_ == name || name.isEmpty()) return;
    name_ = name;
    emit nameChanged(this);
}

void Node::setBypassed(bool bypassed) {
    if (bypassed_ == bypassed) return;
    bypassed_ = bypassed;
    dirty_ = true;
    emit bypassChanged(this);
}

Parameter* Node::findParameter(const QString& name) {
    for (Parameter& parameter : parameters_) {
        if (parameter.name == name) return &parameter;
    }
    return nullptr;
}

const Parameter* Node::findParameter(const QString& name) const {
    for (const Parameter& parameter : parameters_) {
        if (parameter.name == name) return &parameter;
    }
    return nullptr;
}

void Node::addParameter(Parameter parameter) {
    // Duplicate names would silently shadow each other in every lookup.
    if (findParameter(parameter.name)) {
        logError("Node " + typeName_.toStdString() + " declares the parameter '" +
                 parameter.name.toStdString() + "' twice");
        return;
    }
    parameters_.push_back(std::move(parameter));
}

void Node::setParameterValue(const QString& name, const QVariant& value, bool notify) {
    Parameter* parameter = findParameter(name);
    if (!parameter) return;
    if (parameter->value == value && parameter->expression.isEmpty()) return;
    parameter->value = value;
    parameter->expression.clear();
    dirty_ = true;
    if (notify) emit parameterChanged(this, name);
}

void Node::setParameterExpression(const QString& name, const QString& expr, bool notify) {
    Parameter* parameter = findParameter(name);
    if (!parameter) return;
    const QString trimmed = expr.trimmed();
    if (trimmed.isEmpty() || !looksLikeExpression(trimmed)) {
        // Treat as literal assignment when possible.
        if (parameter->type == ParamType::Float || parameter->type == ParamType::Int) {
            bool ok = false;
            const double v = trimmed.toDouble(&ok);
            if (ok) {
                setParameterValue(name,
                                  parameter->type == ParamType::Int ? QVariant(int(std::lround(v)))
                                                                    : QVariant(v),
                                  notify);
                return;
            }
        }
        if (parameter->type == ParamType::String || parameter->type == ParamType::FilePath) {
            setParameterValue(name, trimmed, notify);
            return;
        }
    }
    if (parameter->expression == trimmed) return;
    parameter->expression = trimmed;
    // Cache evaluated numeric value.
    if (parameter->type == ParamType::Float || parameter->type == ParamType::Int) {
        double out = 0.0;
        if (evalExpression(trimmed, exprFrame(), out)) {
            parameter->value =
                parameter->type == ParamType::Int ? QVariant(int(std::lround(out))) : QVariant(out);
        }
    } else if (parameter->type == ParamType::String || parameter->type == ParamType::FilePath) {
        parameter->value = trimmed;  // store expression text as value too for display
    }
    dirty_ = true;
    if (notify) emit parameterChanged(this, name);
}

void Node::notifyParameterChanged(const QString& name) {
    dirty_ = true;
    emit parameterChanged(this, name);
}

double Node::floatValue(const QString& name, double fallback) const {
    const Parameter* parameter = findParameter(name);
    if (!parameter) return fallback;
    return parameter->evaluatedNumber();
}

int Node::intValue(const QString& name, int fallback) const {
    const Parameter* parameter = findParameter(name);
    if (!parameter) return fallback;
    return int(std::lround(parameter->evaluatedNumber()));
}

bool Node::boolValue(const QString& name, bool fallback) const {
    const Parameter* parameter = findParameter(name);
    return parameter ? parameter->toBool() : fallback;
}

QString Node::stringValue(const QString& name, const QString& fallback) const {
    const Parameter* parameter = findParameter(name);
    if (!parameter) return fallback;
    if (parameter->type == ParamType::FilePath || parameter->type == ParamType::String)
        return parameter->evaluatedString();
    return parameter->toString();
}

Vec3 Node::vec3Value(const QString& name, Vec3 fallback) const {
    const Parameter* parameter = findParameter(name);
    return parameter ? parameter->toVec3() : fallback;
}

QString Node::inputLabel(int index) const {
    return index >= 0 && index < inputLabels_.size() ? inputLabels_[index] : QString("Input");
}

Node* Node::input(int index) const {
    return index >= 0 && index < int(inputs_.size()) ? inputs_[size_t(index)] : nullptr;
}

void Node::setInput(int index, Node* node) {
    if (index < 0 || index >= int(inputs_.size())) return;
    if (inputs_[size_t(index)] == node) return;
    inputs_[size_t(index)] = node;
    dirty_ = true;
}

void Node::detachFrom(Node* node) {
    for (Node*& input : inputs_) {
        if (input == node) {
            input = nullptr;
            dirty_ = true;
        }
    }
}

void Node::setInputLabels(QStringList labels) {
    inputLabels_ = std::move(labels);
    inputs_.assign(size_t(inputLabels_.size()), nullptr);
}

// ---------------------------------------------------------------------------

void addTransformParameters(Node& node, Vec3 translate, Vec3 rotate, Vec3 scale) {
    node.addParameter(Parameter::makeVec3("translate", "Translate", translate)
                          .withGroup("Transform")
                          .withTooltip(units::lengthTooltip()));
    node.addParameter(Parameter::makeVec3("rotate", "Rotate", rotate)
                          .withGroup("Transform")
                          .withTooltip("Euler rotation in degrees (Houdini-compatible XYZ)"));
    node.addParameter(Parameter::makeVec3("scale", "Scale", scale).withGroup("Transform"));
    node.addParameter(Parameter::makeFloat("uniformscale", "Uniform Scale", 1.0, 0.001, 100.0, false)
                          .withGroup("Transform"));
}

Mat4 transformFromParameters(const Node& node) {
    const Vec3 translate = node.vec3Value("translate", Vec3(0.0f));
    const Vec3 rotate = node.vec3Value("rotate", Vec3(0.0f));
    Vec3 scale = node.vec3Value("scale", Vec3(1.0f));
    const float uniform = float(node.floatValue("uniformscale", 1.0));
    scale = scale * uniform;
    return composeTRS(translate, rotate, scale);
}

void addTessellationParameters(Node& node) {
    node.addParameter(Parameter::makeMenu("subdivtype", "Subdiv Type",
                                          {"None", "Catclark", "Linear"}, 1)
                          .withGroup("Subdivision")
                          .withTooltip("None: displace cage only.\n"
                                       "Catclark: OpenSubdiv Catmull-Clark (triangle cages "
                                       "fall back to Linear).\n"
                                       "Linear: mid-edge triangle splits."));
    node.addParameter(Parameter::makeInt("subdiviterations", "Subdiv Iterations", 3, 0, 100)
                          .withGroup("Subdivision")
                          .withTooltip(
                              "Uniform densify passes when Screen Adaptive is off. "
                              "Ignored when Screen Adaptive is on (density from Dicing Quality). "
                              "With Frustum Cull on, only the in-frame patch densifies. "
                              "Changing subdiv settings does not restart the render — press Start "
                              "to re-dice."));
    node.addParameter(Parameter::makeFloat("dicingquality", "Dicing Quality", 1.0, 0.01, 16.0, false)
                          .withGroup("Subdivision")
                          .withTooltip(
                              "Screen Adaptive only: target edge ≈ 1/Quality pixels "
                              "(Karma/Mantra/PRMan). Higher = denser. "
                              "1 ≈ 1 µpoly/px; 0.01 ≈ 100 px edge (much coarser); "
                              "2 ≈ 0.5 px (denser). Ignored when Screen Adaptive is off. "
                              "Changing this does not restart the render — press Start to re-dice."));
    node.addParameter(Parameter::makeFloat("boundspadding", "Bounds Padding", 0.0, 0.0, 100.0, false)
                          .withGroup("Subdivision")
                          .withTooltip("Extra AABB padding after displacement (scene units)."));
}

void applyTessellationParameters(const Node& node, StagePrim& prim) {
    prim.subdivType = node.intValue("subdivtype", kSubdivCatclark);
    prim.subdivIterations = std::clamp(node.intValue("subdiviterations", 3), 0, 100);
    prim.dicingQuality = float(node.floatValue("dicingquality", 1.0));
    prim.boundsPadding = float(node.floatValue("boundspadding", 0.0));
}

void addMediumParameters(Node& node) {
    node.addParameter(Parameter::makeMenu(
                          "mediumtype", "Medium Type",
                          {"None", "Homogeneous", "OpenVDB"}, 0)
                          .withGroup("Medium")
                          .withTooltip("None: no participating medium on this prim.\n"
                                       "Homogeneous: uniform fog/smoke described by sigma_a/sigma_s.\n"
                                       "OpenVDB: heterogeneous volume from a VDB file (path below)."));
    node.addParameter(Parameter::makeColor("medium_sigma_a", "Absorption (σa)", Vec3(0.0f))
                          .withGroup("Medium")
                          .withVisibleWhen("mediumtype!=0")
                          .withTooltip("Absorption coefficient per scene unit (RGB)"));
    node.addParameter(Parameter::makeColor("medium_sigma_s", "Scattering (σs)", Vec3(0.0f))
                          .withGroup("Medium")
                          .withVisibleWhen("mediumtype!=0")
                          .withTooltip("Scattering coefficient per scene unit (RGB)"));
    node.addParameter(Parameter::makeFloat("medium_g", "Phase (g)", 0.0, -1.0, 1.0, false)
                          .withGroup("Medium")
                          .withVisibleWhen("mediumtype!=0")
                          .withTooltip("Henyey-Greenstein asymmetry: 0 = isotropic, "
                                       ">0 = forward scatter, <0 = back scatter"));
    node.addParameter(Parameter::makeFloat("medium_density", "Density", 1.0, 0.0, 1000.0, false)
                          .withGroup("Medium")
                          .withVisibleWhen("mediumtype!=0")
                          .withTooltip("Global density scale multiplied with sigma_a and sigma_s"));
    node.addParameter(Parameter::makeFile("medium_vdbfile", "VDB File", "",
                                          "OpenVDB (*.vdb)")
                          .withGroup("Medium")
                          .withVisibleWhen("mediumtype==2")
                          .withTooltip("Path to the OpenVDB file (.vdb). "
                                       "The volume integrator will load and sample this file."));
}

void applyMediumParameters(const Node& node, StagePrim& prim) {
    const int type = node.intValue("mediumtype", 0);
    prim.medium.type = type;
    if (type == 0) {
        prim.mediumAssigned = false;
        return;
    }
    prim.mediumAssigned = true;
    prim.medium.sigmaA = node.vec3Value("medium_sigma_a", Vec3(0.0f));
    prim.medium.sigmaS = node.vec3Value("medium_sigma_s", Vec3(0.0f));
    prim.medium.g = float(node.floatValue("medium_g", 0.0));
    prim.medium.density = float(node.floatValue("medium_density", 1.0));
    prim.vdbPath = (type == 2) ? node.stringValue("medium_vdbfile") : QString();
    prim.medium.volumeIndex = -1;  // resolved later in Stage::toScene
}

}  // namespace sol
