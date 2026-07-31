// Render-time displacement tessellation: subdiv (none / catclark / linear),
// optional frustum cull, optional screen-space dicing (Karma-like).
#pragma once

#include "scene/displace.h"
#include "scene/scene.h"
#include "scene/types.h"

namespace sol {

// Apply tessellation + displacement to every mesh that needs it.
// Mutates `scene.meshes` in place. Call once at Render / headless start.
void tessellateSceneForRender(Scene& scene, const CameraData& dicingCamera);

}  // namespace sol
