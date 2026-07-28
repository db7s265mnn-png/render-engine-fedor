// Reading and writing .bobsc scene files (JSON node networks).
// Legacy .solstice files still load.
#pragma once

#include <QString>

#include "nodes/node_graph.h"

namespace sol {

constexpr const char* kSceneFileExtension = "bobsc";
constexpr const char* kSceneFileFilter =
    "Bob_Render scene (*.bobsc *.solstice);;Bob_Render scene (*.bobsc);;All files (*)";

bool saveGraphToFile(const NodeGraph& graph, const QString& path, QString& error);
bool loadGraphFromFile(NodeGraph& graph, const QString& path, QString& error);

}  // namespace sol
