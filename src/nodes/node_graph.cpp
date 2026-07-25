#include "nodes/node_graph.h"

#include <QJsonArray>
#include <QRegularExpression>
#include <algorithm>

#include "core/log.h"
#include "nodes/node_registry.h"

namespace sol {

NodeGraph::NodeGraph(QObject* parent) : QObject(parent) {}

NodeGraph::~NodeGraph() = default;

QString NodeGraph::uniqueNodeName(const QString& base) const {
    QString stem = base;
    static const QRegularExpression trailingDigits("\\d+$");
    stem.remove(trailingDigits);
    if (stem.isEmpty()) stem = "node";
    for (int i = 1; i < 100000; ++i) {
        const QString candidate = stem + QString::number(i);
        if (!findNode(candidate)) return candidate;
    }
    return base;
}

Node* NodeGraph::createNode(const QString& typeName, const QString& name, QPointF position) {
    NodePtr node = NodeRegistry::instance().create(typeName, name.isEmpty() ? uniqueNodeName(typeName) : name);
    if (!node) {
        logError("Unknown node type: " + typeName.toStdString());
        return nullptr;
    }
    if (findNode(node->name())) node->setName(uniqueNodeName(node->name()));
    node->setPosition(position);
    Node* raw = node.get();
    nodes_.push_back(std::move(node));
    hookNode(raw);
    if (!displayNode_) setDisplayNode(raw);
    setModified(true);
    emit nodeAdded(raw);
    emit graphChanged();
    return raw;
}

void NodeGraph::hookNode(Node* node) {
    connect(node, &Node::parameterChanged, this, &NodeGraph::onParameterChanged);
    connect(node, &Node::nameChanged, this, &NodeGraph::onNodeNameChanged);
    connect(node, &Node::bypassChanged, this, &NodeGraph::onNodeBypassChanged);
}

void NodeGraph::removeNode(Node* node) {
    if (!node) return;
    emit nodeAboutToBeRemoved(node);
    for (NodePtr& other : nodes_) {
        if (other.get() != node) other->detachFrom(node);
    }
    const bool wasDisplay = displayNode_ == node;
    nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
                                [node](const NodePtr& candidate) { return candidate.get() == node; }),
                 nodes_.end());
    if (wasDisplay) {
        displayNode_ = nodes_.empty() ? nullptr : nodes_.back().get();
        emit displayNodeChanged(displayNode_);
    }
    markAllDirty();
    setModified(true);
    emit connectionsChanged();
    emit graphChanged();
}

void NodeGraph::clear() {
    displayNode_ = nullptr;
    nodes_.clear();
    filePath_.clear();
    setModified(false);
    emit displayNodeChanged(nullptr);
    emit graphChanged();
}

Node* NodeGraph::findNode(const QString& name) const {
    for (const NodePtr& node : nodes_) {
        if (node->name() == name) return node.get();
    }
    return nullptr;
}

std::vector<Node*> NodeGraph::outputsOf(const Node* node) const {
    std::vector<Node*> result;
    for (const NodePtr& candidate : nodes_) {
        for (Node* input : candidate->inputs()) {
            if (input == node) {
                result.push_back(candidate.get());
                break;
            }
        }
    }
    return result;
}

bool NodeGraph::connectNodes(Node* source, Node* destination, int inputIndex) {
    if (!source || !destination || source == destination) return false;
    if (inputIndex < 0 || inputIndex >= destination->inputCount()) return false;

    // Reject cycles: walk upstream from `source` looking for `destination`.
    std::vector<const Node*> stack{source};
    while (!stack.empty()) {
        const Node* current = stack.back();
        stack.pop_back();
        if (current == destination) {
            logWarning("Connection refused: it would create a cycle");
            return false;
        }
        for (Node* input : current->inputs()) {
            if (input) stack.push_back(input);
        }
    }

    destination->setInput(inputIndex, source);
    markDirty(destination);
    setModified(true);
    emit connectionsChanged();
    emit graphChanged();
    return true;
}

void NodeGraph::disconnectInput(Node* destination, int inputIndex) {
    if (!destination) return;
    destination->setInput(inputIndex, nullptr);
    markDirty(destination);
    setModified(true);
    emit connectionsChanged();
    emit graphChanged();
}

void NodeGraph::setDisplayNode(Node* node) {
    if (displayNode_ == node) return;
    displayNode_ = node;
    emit displayNodeChanged(node);
}

void NodeGraph::markDirty(Node* node) {
    if (!node) return;
    if (node->isDirty() && !node->cachedStage()) {
        // Still propagate: downstream nodes may hold caches.
    }
    node->setDirty(true);
    node->setCachedStage(nullptr);
    emit nodeDirtied(node);
    for (Node* output : outputsOf(node)) markDirty(output);
}

void NodeGraph::markAllDirty() {
    for (const NodePtr& node : nodes_) {
        node->setDirty(true);
        node->setCachedStage(nullptr);
    }
}

void NodeGraph::onParameterChanged(Node* node, const QString& /*parameterName*/) {
    markDirty(node);
    setModified(true);
    emit graphChanged();
}

void NodeGraph::onNodeNameChanged(Node* node) {
    markDirty(node);
    setModified(true);
    emit graphChanged();
}

void NodeGraph::onNodeBypassChanged(Node* node) {
    markDirty(node);
    setModified(true);
    emit graphChanged();
}

void NodeGraph::setModified(bool modified) {
    if (modified_ == modified) return;
    modified_ = modified;
    emit modifiedChanged(modified_);
}

StagePtr NodeGraph::cook(Node* node, CookContext& context) {
    std::vector<const Node*> visiting;
    return cookInternal(node, context, visiting);
}

StagePtr NodeGraph::cookDisplay(CookContext& context) { return cook(displayNode_, context); }

StagePtr NodeGraph::cookInternal(Node* node, CookContext& context, std::vector<const Node*>& visiting) {
    if (!node) return std::make_shared<Stage>();
    if (!node->isDirty() && node->cachedStage()) return node->cachedStage();
    if (std::find(visiting.begin(), visiting.end(), node) != visiting.end()) {
        context.reportError(node, "cycle detected while cooking");
        return std::make_shared<Stage>();
    }
    visiting.push_back(node);

    std::vector<StagePtr> inputStages;
    inputStages.reserve(size_t(node->inputCount()));
    for (int i = 0; i < node->inputCount(); ++i) {
        Node* input = node->input(i);
        inputStages.push_back(input ? cookInternal(input, context, visiting) : nullptr);
    }

    auto stage = std::make_shared<Stage>();
    if (!inputStages.empty() && inputStages[0]) *stage = *inputStages[0];

    node->setErrorText(QString());
    if (!node->isBypassed()) {
        if (!node->copiesFirstInput()) *stage = Stage();
        const int errorsBefore = context.errors.size();
        node->cook(context, inputStages, *stage);
        if (context.errors.size() > errorsBefore) node->setErrorText(context.errors.last());
    }

    node->setCachedStage(stage);
    node->setDirty(false);
    visiting.pop_back();
    return stage;
}

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------

QJsonObject NodeGraph::toJson() const {
    QJsonObject root;
    root["application"] = "Solstice";
    root["version"] = 1;

    QJsonArray nodesArray;
    for (const NodePtr& node : nodes_) {
        QJsonObject nodeJson;
        nodeJson["type"] = node->typeName();
        nodeJson["name"] = node->name();
        nodeJson["x"] = node->position().x();
        nodeJson["y"] = node->position().y();
        nodeJson["bypassed"] = node->isBypassed();

        QJsonArray parametersArray;
        for (const Parameter& parameter : node->parameters()) parametersArray.append(parameter.toJson());
        nodeJson["parameters"] = parametersArray;

        QJsonArray inputsArray;
        for (Node* input : node->inputs()) inputsArray.append(input ? input->name() : QString());
        nodeJson["inputs"] = inputsArray;

        const QJsonObject extra = node->extraStateToJson();
        if (!extra.isEmpty()) nodeJson["state"] = extra;

        nodesArray.append(nodeJson);
    }
    root["nodes"] = nodesArray;
    if (displayNode_) root["displayNode"] = displayNode_->name();
    return root;
}

bool NodeGraph::fromJson(const QJsonObject& json, QString& error) {
    clear();
    const QJsonArray nodesArray = json.value("nodes").toArray();
    for (const QJsonValue& value : nodesArray) {
        const QJsonObject nodeJson = value.toObject();
        const QString type = nodeJson.value("type").toString();
        const QString name = nodeJson.value("name").toString();
        NodePtr node = NodeRegistry::instance().create(type, name);
        if (!node) {
            error = "unknown node type: " + type;
            logWarning(error.toStdString());
            continue;
        }
        node->setPosition(QPointF(nodeJson.value("x").toDouble(), nodeJson.value("y").toDouble()));
        node->setBypassed(nodeJson.value("bypassed").toBool());

        const QJsonArray parametersArray = nodeJson.value("parameters").toArray();
        for (const QJsonValue& parameterValue : parametersArray) {
            const QJsonObject parameterJson = parameterValue.toObject();
            Parameter* parameter = node->findParameter(parameterJson.value("name").toString());
            if (parameter) parameter->fromJson(parameterJson);
        }
        node->extraStateFromJson(nodeJson.value("state").toObject());

        Node* raw = node.get();
        nodes_.push_back(std::move(node));
        hookNode(raw);
    }

    // Second pass: connections, now that all nodes exist.
    int index = 0;
    for (const QJsonValue& value : nodesArray) {
        if (index >= int(nodes_.size())) break;
        const QJsonObject nodeJson = value.toObject();
        Node* node = findNode(nodeJson.value("name").toString());
        if (!node) {
            ++index;
            continue;
        }
        const QJsonArray inputsArray = nodeJson.value("inputs").toArray();
        for (int i = 0; i < inputsArray.size() && i < node->inputCount(); ++i) {
            const QString inputName = inputsArray[i].toString();
            if (inputName.isEmpty()) continue;
            Node* source = findNode(inputName);
            if (source) node->setInput(i, source);
        }
        ++index;
    }

    const QString displayName = json.value("displayNode").toString();
    Node* display = findNode(displayName);
    if (!display && !nodes_.empty()) display = nodes_.back().get();
    displayNode_ = display;

    markAllDirty();
    setModified(false);
    emit displayNodeChanged(displayNode_);
    emit connectionsChanged();
    emit graphChanged();
    return true;
}

}  // namespace sol
