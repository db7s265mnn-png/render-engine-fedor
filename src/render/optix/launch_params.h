// Parameters handed to every OptiX launch. Shared verbatim between the host
// code and the device programs.
#pragma once

#include "render/optix/path_state.h"
#include "scene/types.h"

namespace sol {

// Ray types in the shader binding table.
enum OptixRayType : int { kRayTypeRadiance = 0, kRayTypeShadow = 1, kRayTypeCount = 2 };

struct LaunchParams {
    SceneView scene;

    Vec4* accumBuffer = nullptr;  // width * height, rgb + sample weight
    GpuPath* paths = nullptr;
    GpuHit* hits = nullptr;
    GpuShadow* shadows = nullptr;
    int width = 0;
    int height = 0;

    int sampleIndex = 0;
    unsigned int frameSeed = 0;
    int pixelSampler = 0;  // PixelSampler enum

    unsigned long long traversable = 0;  // OptixTraversableHandle
};

}  // namespace sol
