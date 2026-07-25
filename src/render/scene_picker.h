// Synchronous ray picking against a cooked scene (used by the viewport).
#pragma once

#include "core/math.h"
#include "scene/scene.h"

namespace sol {

// Casts a camera ray through normalized image coordinates (u,v in 0..1) using
// the supplied camera and returns the first surface hit in world space.
bool pickSceneSurface(const ScenePtr& scene, const CameraData& camera, int resolutionX, int resolutionY,
                      float u, float v, Vec3& hitPoint);

}  // namespace sol
