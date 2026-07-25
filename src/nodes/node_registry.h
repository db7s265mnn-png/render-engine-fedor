// Registry of node types. The Tab menu in the network editor is built from it.
#pragma once

#include <QList>
#include <QString>
#include <functional>

#include "nodes/node.h"

namespace sol {

struct NodeTypeInfo {
    QString typeName;
    QString label;
    QString category;     // Geometry, Lighting, Camera, Material, Utility, Render
    QString description;
    QString colorHex = "#4b5f78";
    std::function<NodePtr(const QString& name)> factory;
};

class NodeRegistry {
public:
    static NodeRegistry& instance();

    void registerType(NodeTypeInfo info);
    NodePtr create(const QString& typeName, const QString& name) const;
    const NodeTypeInfo* find(const QString& typeName) const;
    const QList<NodeTypeInfo>& types() const { return types_; }
    QStringList categories() const;

private:
    QList<NodeTypeInfo> types_;
};

// Registers every built in node type. Safe to call multiple times.
void registerBuiltinNodes();

}  // namespace sol
