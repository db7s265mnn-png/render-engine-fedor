// Parameters handed to every OptiX launch. Shared verbatim between the host
// code and the device programs.
#pragma once

#include "render/optix/path_state.h"
#include "scene/types.h"

namespace sol {

// Ray types in the shader binding table.
enum OptixRayType : int { kRayTypeRadiance = 0, kRayTypeShadow = 1, kRayTypeCount = 2 };

// Dense occupancy + Embree majorant bricks so volume PT can run the same
// residual-ratio tracker as CPU (OpenVDB itself stays off the OptiX modules).
struct GpuVolumeGrid {
    const float* density = nullptr;
    const float* majMin = nullptr;
    const float* majMax = nullptr;
    const unsigned char* bricks = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int kind = 1;  // 0 = SDF, 1 = fog
    int nearest = 0;
    int majNx = 0;
    int majNy = 0;
    int majNz = 0;
    int brNx = 0;
    int brNy = 0;
    int brNz = 0;
    int brickSize = 4;
    Vec3 bmin{0.0f};
    Vec3 bmax{0.0f};
    Vec3 majOrigin{0.0f};
    float majorant = 1.0f;
    float majCell = 0.0f;
    float voxelSize = 0.0f;
};

struct alignas(16) LaunchParams {
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
    float manualTestMult = 0.0f;

    int xpuSplit = 0;
    int xpuTileSize = 32;
    int xpuGpuParity = 0;

    unsigned long long traversable = 0;  // OptixTraversableHandle
};

}  // namespace sol
