// Base class for every node in the network.
#pragma once

#include <QJsonObject>
#include <QObject>
#include <QPointF>
#include <QString>
#include <QStringList>
#include <memory>
#include <vector>

#include "nodes/parameter.h"
#include "nodes/stage.h"

namespace sol {

class Node;

struct CookContext {
    double time = 0.0;
    int frame = 1;
    QString sceneDirectory;  // used to resolve relative file paths
    QStringList warnings;
    QStringList errors;

    void reportError(const Node* node, const QString& message);
    void reportWarning(const Node* node, const QString& message);
};

class Node : public QObject {
    Q_OBJECT

public:
    Node(QString typeName, QString name);
    ~Node() override;

    const QString& typeName() const { return typeName_; }
    const QString& name() const { return name_; }
    void setName(const QString& name);

    QPointF position() const { return position_; }
    void setPosition(QPointF position) { position_ = position; }

    bool isBypassed() const { return bypassed_; }
    void setBypassed(bool bypassed);

    // Parameters ------------------------------------------------------------
    const std::vector<Parameter>& parameters() const { return parameters_; }
    std::vector<Parameter>& parameters() { return parameters_; }
    Parameter* findParameter(const QString& name);
    const Parameter* findParameter(const QString& name) const;
    void addParameter(Parameter parameter);
    // notify=false updates the value / dirty flag without emitting parameterChanged
    // (used while dragging viewport gizmos so cook/IPR wait for mouse release).
    void setParameterValue(const QString& name, const QVariant& value, bool notify = true);
    void notifyParameterChanged(const QString& name);

    double floatValue(const QString& name, double fallback = 0.0) const;
    int intValue(const QString& name, int fallback = 0) const;
    bool boolValue(const QString& name, bool fallback = false) const;
    QString stringValue(const QString& name, const QString& fallback = QString()) const;
    Vec3 vec3Value(const QString& name, Vec3 fallback = Vec3(0.0f)) const;

    // Inputs ----------------------------------------------------------------
    int inputCount() const { return int(inputLabels_.size()); }
    QString inputLabel(int index) const;
    Node* input(int index) const;
    const std::vector<Node*>& inputs() const { return inputs_; }
    void setInput(int index, Node* node);
    void detachFrom(Node* node);
    bool hasOutputPort() const { return hasOutput_; }

    // Cooking ---------------------------------------------------------------
    // When true the graph pre-fills the output stage with a copy of input 0.
    virtual bool copiesFirstInput() const { return true; }
    virtual void cook(CookContext& context, const std::vector<StagePtr>& inputStages, Stage& stage) = 0;

    bool isDirty() const { return dirty_; }
    void setDirty(bool dirty) { dirty_ = dirty; }
    StagePtr cachedStage() const { return cache_; }
    void setCachedStage(StagePtr stage) { cache_ = std::move(stage); }

    const QString& errorText() const { return errorText_; }
    void setErrorText(const QString& text) { errorText_ = text; }

    virtual QJsonObject extraStateToJson() const { return {}; }
    virtual void extraStateFromJson(const QJsonObject&) {}

signals:
    void parameterChanged(sol::Node* node, const QString& parameterName);
    void nameChanged(sol::Node* node);
    void bypassChanged(sol::Node* node);

protected:
    void setInputLabels(QStringList labels);
    void setHasOutput(bool hasOutput) { hasOutput_ = hasOutput; }

private:
    QString typeName_;
    QString name_;
    QPointF position_;
    bool bypassed_ = false;
    bool dirty_ = true;
    bool hasOutput_ = true;
    QString errorText_;
    std::vector<Parameter> parameters_;
    QStringList inputLabels_;
    std::vector<Node*> inputs_;
    StagePtr cache_;
};

using NodePtr = std::unique_ptr<Node>;

// Shared parameter blocks reused by several node types.
void addTransformParameters(Node& node, Vec3 translate = Vec3(0.0f), Vec3 rotate = Vec3(0.0f),
                            Vec3 scale = Vec3(1.0f));
Mat4 transformFromParameters(const Node& node);

}  // namespace sol
