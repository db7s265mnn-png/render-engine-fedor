// The node network: ownership, connections, cooking and serialisation.
#pragma once

#include <QJsonObject>
#include <QList>
#include <QObject>
#include <QString>
#include <memory>
#include <vector>

#include "nodes/node.h"

namespace sol {

class NodeGraph : public QObject {
    Q_OBJECT

public:
    explicit NodeGraph(QObject* parent = nullptr);
    ~NodeGraph() override;

    Node* createNode(const QString& typeName, const QString& name = QString(), QPointF position = QPointF());
    void removeNode(Node* node);
    void clear();

    const std::vector<NodePtr>& nodes() const { return nodes_; }
    Node* findNode(const QString& name) const;
    bool contains(const Node* node) const;
    QString uniqueNodeName(const QString& base) const;
    std::vector<Node*> outputsOf(const Node* node) const;

    bool connectNodes(Node* source, Node* destination, int inputIndex);
    void disconnectInput(Node* destination, int inputIndex);

    Node* displayNode() const { return displayNode_; }
    void setDisplayNode(Node* node);

    // Cooks the display node (or an explicit node) and returns the flattened
    // stage. Cook results are cached per node and invalidated on edits.
    StagePtr cook(Node* node, CookContext& context);
    StagePtr cookDisplay(CookContext& context);

    void markDirty(Node* node);
    void markAllDirty();
    // Dirties nodes whose cook depends on timeline time (and their outputs).
    // Returns true when at least one node was dirtied.
    bool markTimeDependentDirty();

    QJsonObject toJson() const;
    bool fromJson(const QJsonObject& json, QString& error);

    // Clipboard subset: selected nodes + wires between them (external inputs dropped).
    QJsonObject nodesToClipboardJson(const QList<Node*>& nodes) const;
    // Pastes clipboard JSON near pasteOrigin (centers the selection on that point).
    // Returns newly created nodes; names are uniquified and remapped.
    QList<Node*> pasteNodesFromClipboardJson(const QJsonObject& json, QPointF pasteOrigin, QString& error);

    const QString& filePath() const { return filePath_; }
    void setFilePath(const QString& path) { filePath_ = path; }

    bool isModified() const { return modified_; }
    void setModified(bool modified);

signals:
    void nodeAdded(sol::Node* node);
    void nodeAboutToBeRemoved(sol::Node* node);
    void connectionsChanged();
    void displayNodeChanged(sol::Node* node);
    void graphChanged();
    void nodeDirtied(sol::Node* node);
    void modifiedChanged(bool modified);

private slots:
    void onParameterChanged(sol::Node* node, const QString& parameterName);
    void onNodeNameChanged(sol::Node* node);
    void onNodeBypassChanged(sol::Node* node);

private:
    StagePtr cookInternal(Node* node, CookContext& context, std::vector<const Node*>& visiting);
    void hookNode(Node* node);

    std::vector<NodePtr> nodes_;
    Node* displayNode_ = nullptr;
    QString filePath_;
    bool modified_ = false;
};

}  // namespace sol
