// Reading and writing .solstice scene files (JSON node networks).
#pragma once

#include <QString>

#include "nodes/node_graph.h"

namespace sol {

constexpr const char* kSceneFileExtension = "solstice";
constexpr const char* kSceneFileFilter = "Bob_Render scene (*.solstice);;All files (*)";

bool saveGraphToFile(const NodeGraph& graph, const QString& path, QString& error);
bool loadGraphFromFile(NodeGraph& graph, const QString& path, QString& error);

}  // namespace sol
