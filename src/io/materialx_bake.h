// CPU bake of MaterialX procedural / texture subgraphs into UV textures.
// Prefer shade-time compilation (materialx_compile + render/procedural.h); this
// bake path remains for tooling / fallbacks only.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <QString>

#include "core/image.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_MATERIALX
#  include <MaterialXCore/Node.h>
#endif

namespace sol {

#if SOLSTICE_HAVE_MATERIALX

// Returns true when the node (or a reachable subgraph) can be baked to a UV map.
bool materialXNodeIsBakable(MaterialX::NodePtr node);

// Evaluate `root` over UV space into an RGBA32F image. Handles noise*, fractal*,
// cellnoise*, worleynoise*, constant, math (multiply/mix/add/...), image,
// tiledimage, and triplanarprojection (blended from filex/y/z).
std::shared_ptr<Image> bakeMaterialXNodeToTexture(MaterialX::NodePtr root, const QString& searchDirectory,
                                                   const std::vector<int>& udimSet, int resolution,
                                                   std::string& error);

#endif

}  // namespace sol
