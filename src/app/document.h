// Reading and writing .scene files (JSON node networks).
#pragma once

#include <QString>

#include "nodes/node_graph.h"

namespace sol {

constexpr const char* kSceneFileExtension = "scene";
constexpr const char* kSceneFileFilter =
    "Grendizer_Render scene (*.scene);;Legacy Bob_Render scene (*.bobsc);;All files (*)";

bool saveGraphToFile(const NodeGraph& graph, const QString& path, QString& error);
bool loadGraphFromFile(NodeGraph& graph, const QString& path, QString& error);

}  // namespace sol
