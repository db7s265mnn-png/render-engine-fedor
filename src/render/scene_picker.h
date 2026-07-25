// Synchronous ray picking against a cooked scene (used by the viewport).
#pragma once

#include "core/math.h"
#include "scene/scene.h"

namespace sol {

// Casts a camera ray through normalized device coordinates (0..1) and returns
// the first surface hit in world space. Returns false when nothing is hit.
bool pickSceneSurface(const ScenePtr& scene, const Mat4& cameraToWorld, float aspectRatio, float u, float v,
                      Vec3& hitPoint);

}  // namespace sol
