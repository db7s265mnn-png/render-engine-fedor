// Synchronous ray picking against a cooked scene (used by the viewport).
#pragma once

#include "core/math.h"
#include "scene/scene.h"

namespace sol {

// Interactive picks intentionally ignore beauty-only effects (motion blur keys,
// polynomial optics, DOF) so tumble / object select / focus-pick stay stable.
enum class ScenePickMode {
    Interactive = 0,
    Beauty = 1,
};

// Casts a camera ray through normalized image coordinates (u,v in 0..1) using
// the supplied camera and returns the first surface hit in world space.
// When instanceIndexOut is non-null, writes the Scene::instances index of the hit.
bool pickSceneSurface(const ScenePtr& scene, const CameraData& camera, int resolutionX, int resolutionY,
                      float u, float v, Vec3& hitPoint, int* instanceIndexOut = nullptr,
                      ScenePickMode mode = ScenePickMode::Interactive);

}  // namespace sol
