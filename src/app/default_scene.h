// Builds the network a new document starts with.
#pragma once

#include <QString>

#include "nodes/node_graph.h"

namespace sol {

// Ground plane + sphere + material + dome light + sun + camera + render
// settings, wired into a merge and a final render settings node.
void buildDefaultGraph(NodeGraph& graph);

// A network that imports an Alembic file and lights it with an HDRI, used by
// the CLI when only geometry and an environment are given.
void buildAlembicGraph(NodeGraph& graph, const QString& alembicPath, const QString& hdriPath);

}  // namespace sol
