// Parameters handed to every OptiX launch. Shared verbatim between the host
// code and the device programs.
#pragma once

#include "render/optix/path_state.h"
#include "scene/types.h"

namespace sol {

// Ray types in the shader binding table.
enum OptixRayType : int { kRayTypeRadiance = 0, kRayTypeShadow = 1, kRayTypeCount = 2 };

// Dense brick baked from an OpenVDB grid so volume PT can run on the GPU
// without pulling OpenVDB into the OptiX modules (Cycles uses NanoVDB).
struct GpuVolumeGrid {
    const float* density = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int kind = 1;  // 0 = SDF, 1 = fog
    Vec3 bmin{0.0f};
    Vec3 bmax{0.0f};
    float majorant = 1.0f;
};

struct LaunchParams {
    SceneView scene;
    const GpuVolumeGrid* volumes = nullptr;
    int volumeCount = 0;

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
