// Render-time displacement tessellation: subdiv (none / catclark / linear),
// optional frustum cull, optional screen-space dicing (Karma-like).
#pragma once

#include <functional>
#include <string>

#include "scene/displace.h"
#include "scene/scene.h"
#include "scene/types.h"

namespace sol {

// Optional UI/headless progress: short status string (e.g. "Dicing 2/5 · pass 3 · 1.2M tris").
using TessProgressFn = std::function<void(const std::string& message)>;

// Apply tessellation + displacement to every mesh that needs it.
// Mutates `scene.meshes` in place. Call once at Render / headless start.
// Uses CPU threads from Render Settings, reserving 2 cores for UI/OS.
void tessellateSceneForRender(Scene& scene, const CameraData& dicingCamera,
                              const TessProgressFn& progress = {});

// Authored tessellation inputs (Render Settings + per-mesh subdiv attrs +
// displacement bindings). When this changes, the tess cache is stale.
std::string tessellationFingerprint(const Scene& scene);

// Quantized dicing-camera pose / lens. Screen Adaptive densify depends on this;
// orbit/dolly without re-tess leaves coarse cages in close-up.
std::string dicingCameraFingerprint(const CameraData& cam);

}  // namespace sol
