#include "app/undo_hub.h"

#include <QUndoCommand>
#include <cmath>

#include "nodes/node.h"
#include "nodes/node_graph.h"

namespace sol {
namespace {

class ParameterCommand : public QUndoCommand {
public:
    ParameterCommand(UndoHub& hub, QString nodeName, QString param, QVariant oldValue, QVariant newValue,
                     QString text)
        : hub_(hub),
          nodeName_(std::move(nodeName)),
          param_(std::move(param)),
          old_(std::move(oldValue)),
          new_(std::move(newValue)) {
        if (text.isEmpty()) text = QStringLiteral("Edit %1").arg(param_);
        setText(text);
    }

    void undo() override { apply(old_); }
    void redo() override {
        if (virgin_) {
            virgin_ = false;
            return;
        }
        apply(new_);
    }

private:
    void apply(const QVariant& value) {
        NodeGraph* graph = hub_.graph();
        if (!graph) return;
        UndoHub::IgnoreGuard guard(hub_);
        if (Node* node = graph->findNode(nodeName_)) node->setParameterValue(param_, value);
        hub_.refreshUi();
    }

    UndoHub& hub_;
    QString nodeName_;
    QString param_;
    QVariant old_;
    QVariant new_;
    bool virgin_ = true;
};

class GraphSnapshotCommand : public QUndoCommand {
public:
    GraphSnapshotCommand(UndoHub& hub, QJsonObject before, QJsonObject after, QString text)
        : hub_(hub), before_(std::move(before)), after_(std::move(after)) {
        setText(text);
    }

    void undo() override { apply(before_); }
    void redo() override {
        if (virgin_) {
            virgin_ = false;
            return;
        }
        apply(after_);
    }

private:
    void apply(const QJsonObject& json) {
        NodeGraph* graph = hub_.graph();
        if (!graph) return;
        UndoHub::IgnoreGuard guard(hub_);
        QString error;
        graph->fromJson(json, error);
        hub_.rebuildGraph();
        hub_.refreshUi();
    }

    UndoHub& hub_;
    QJsonObject before_;
    QJsonObject after_;
    bool virgin_ = true;
};

class NodePositionsCommand : public QUndoCommand {
public:
    NodePositionsCommand(UndoHub& hub, QHash<QString, QPointF> oldPos, QHash<QString, QPointF> newPos)
        : hub_(hub), old_(std::move(oldPos)), new_(std::move(newPos)) {
        setText(QStringLiteral("Move nodes"));
    }

    void undo() override { apply(old_); }
    void redo() override {
        if (virgin_) {
            virgin_ = false;
            return;
        }
        apply(new_);
    }

private:
    void apply(const QHash<QString, QPointF>& pos) {
        NodeGraph* graph = hub_.graph();
        if (!graph) return;
        UndoHub::IgnoreGuard guard(hub_);
        for (auto it = pos.constBegin(); it != pos.constEnd(); ++it) {
            if (Node* node = graph->findNode(it.key())) node->setPosition(it.value());
        }
        hub_.rebuildGraph();
        hub_.refreshUi();
    }

    UndoHub& hub_;
    QHash<QString, QPointF> old_;
    QHash<QString, QPointF> new_;
    bool virgin_ = true;
};

class CameraCommand : public QUndoCommand {
public:
    CameraCommand(UndoHub& hub, OrbitCameraState before, OrbitCameraState after)
        : hub_(hub), before_(std::move(before)), after_(std::move(after)) {
        setText(QStringLiteral("Camera"));
    }

    void undo() override { apply(before_); }
    void redo() override {
        if (virgin_) {
            virgin_ = false;
            return;
        }
        apply(after_);
    }

private:
    void apply(const OrbitCameraState& state) {
        NodeGraph* graph = hub_.graph();
        UndoHub::IgnoreGuard guard(hub_);
        if (graph && state.hasNode) {
            if (Node* node = graph->findNode(state.nodeName)) {
                node->setParameterValue(QStringLiteral("uselookat"), state.uselookat, false);
                if (state.eye.isValid())
                    node->setParameterValue(QStringLiteral("eye"), state.eye, false);
                if (state.target.isValid())
                    node->setParameterValue(QStringLiteral("target"), state.target, false);
                if (state.translate.isValid())
                    node->setParameterValue(QStringLiteral("translate"), state.translate, false);
                node->notifyParameterChanged(QStringLiteral("eye"));
            }
        }
        hub_.applyCamera(state);
        hub_.refreshUi();
    }

    UndoHub& hub_;
    OrbitCameraState before_;
    OrbitCameraState after_;
    bool virgin_ = true;
};

}  // namespace

UndoHub::UndoHub(QObject* parent) : QObject(parent) {
    stack_.setUndoLimit(100);
}

void UndoHub::pushParameter(const QString& nodeName, const QString& param, const QVariant& oldValue,
                            const QVariant& newValue, const QString& text) {
    if (ignore_ > 0 || !graph_ || nodeName.isEmpty() || param.isEmpty()) return;
    if (oldValue == newValue) return;
    stack_.push(new ParameterCommand(*this, nodeName, param, oldValue, newValue, text));
}

void UndoHub::beginParameter(const QString& nodeName, const QString& param, const QVariant& oldValue) {
    if (ignore_ > 0) return;
    dragNode_ = nodeName;
    dragParam_ = param;
    dragOld_ = oldValue;
    dragging_ = true;
}

void UndoHub::endParameter(const QString& nodeName, const QString& param, const QVariant& newValue) {
    if (!dragging_) return;
    dragging_ = false;
    if (nodeName != dragNode_ || param != dragParam_) return;
    pushParameter(dragNode_, dragParam_, dragOld_, newValue);
    dragNode_.clear();
    dragParam_.clear();
    dragOld_ = QVariant();
}

void UndoHub::pushGraphSnapshot(const QJsonObject& before, const QJsonObject& after, const QString& text) {
    if (ignore_ > 0 || !graph_) return;
    if (before == after) return;
    stack_.push(new GraphSnapshotCommand(*this, before, after, text));
}

void UndoHub::pushNodePositions(const QHash<QString, QPointF>& oldPos,
                                const QHash<QString, QPointF>& newPos) {
    if (ignore_ > 0 || !graph_) return;
    if (oldPos == newPos) return;
    bool any = false;
    for (auto it = newPos.constBegin(); it != newPos.constEnd(); ++it) {
        if (oldPos.value(it.key()) != it.value()) {
            any = true;
            break;
        }
    }
    if (!any) return;
    stack_.push(new NodePositionsCommand(*this, oldPos, newPos));
}

void UndoHub::pushCamera(const OrbitCameraState& before, const OrbitCameraState& after) {
    if (ignore_ > 0) return;
    if (orbitCameraStateEqual(before, after)) return;
    stack_.push(new CameraCommand(*this, before, after));
}

void UndoHub::beginMacro(const QString& text) {
    if (ignore_ > 0) return;
    stack_.beginMacro(text);
}

void UndoHub::endMacro() {
    if (ignore_ > 0) return;
    stack_.endMacro();
}

void UndoHub::clear() {
    stack_.clear();
    dragging_ = false;
}

void UndoHub::undo() {
    if (ignore_ > 0) return;
    stack_.undo();
}

void UndoHub::redo() {
    if (ignore_ > 0) return;
    stack_.redo();
}

}  // namespace sol
