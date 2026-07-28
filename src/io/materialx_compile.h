// Compile MaterialX procedural subgraphs into shade-time ProceduralNode bytecode.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <QString>

#include "core/image.h"
#include "scene/types.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_MATERIALX
#  include <MaterialXCore/Node.h>
#endif

namespace sol {

#if SOLSTICE_HAVE_MATERIALX

bool materialXNodeIsProcedural(MaterialX::NodePtr node);

// Compile `root` into `outNodes`. Image leaves append to `outImages`; their
// texture indices (stored on kProcImage/kProcTriplanar) are local to outImages.
// Returns the root index into outNodes, or -1 on failure.
int compileMaterialXNode(MaterialX::NodePtr root, const QString& searchDirectory,
                         const std::vector<int>& udimSet, std::vector<ProceduralNode>& outNodes,
                         std::vector<std::shared_ptr<Image>>& outImages, std::string& error);

#endif

}  // namespace sol
