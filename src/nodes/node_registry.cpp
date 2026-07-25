#include "nodes/node_registry.h"

namespace sol {

NodeRegistry& NodeRegistry::instance() {
    static NodeRegistry registry;
    return registry;
}

void NodeRegistry::registerType(NodeTypeInfo info) {
    for (NodeTypeInfo& existing : types_) {
        if (existing.typeName == info.typeName) {
            existing = std::move(info);
            return;
        }
    }
    types_.push_back(std::move(info));
}

NodePtr NodeRegistry::create(const QString& typeName, const QString& name) const {
    const NodeTypeInfo* info = find(typeName);
    if (!info || !info->factory) return nullptr;
    return info->factory(name.isEmpty() ? typeName + "1" : name);
}

const NodeTypeInfo* NodeRegistry::find(const QString& typeName) const {
    for (const NodeTypeInfo& info : types_) {
        if (info.typeName == typeName) return &info;
    }
    return nullptr;
}

QStringList NodeRegistry::categories() const {
    QStringList result;
    for (const NodeTypeInfo& info : types_) {
        if (!result.contains(info.category)) result << info.category;
    }
    return result;
}

}  // namespace sol
