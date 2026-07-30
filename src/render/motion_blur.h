// Arnold-style motion blur: multi-key cooks across the shutter interval.
#pragma once

#include <cmath>

#include "nodes/node_graph.h"
#include "scene/scene.h"

namespace sol {

// Shutter open/close in seconds for a centered shutter of `shutterLength` frames.
inline void shutterIntervalSeconds(double centerTime, double fps, float shutterLength, double& openTime,
                                   double& closeTime) {
    const double half = 0.5 * double(shutterLength) / std::max(1e-6, fps);
    openTime = centerTime - half;
    closeTime = centerTime + half;
}

// After `scene` was built at the shutter-center cook, sample the graph at
// `motionKeys` times across the shutter and attach transform + deformation keys.
// No-op when motion blur is disabled. CPU / Embree consumes the keys.
void attachMotionBlurKeys(NodeGraph& graph, const CookContext& centerContext, Scene& scene);

}  // namespace sol
