// Parameters handed to every OptiX launch. Shared verbatim between the host
// code and the device programs.
#pragma once

#include "render/camera_proj.h"
#include "render/optix/path_state.h"
#include "render/photon_aim.h"
#include "render/spectrum_device.h"
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
    float* lumSq = nullptr;       // Σ L(sample)² for the variance oracle
    const unsigned char* skipMask = nullptr;  // 1 = pixel already quiet
    GpuPath* paths = nullptr;
    GpuHit* hits = nullptr;
    GpuShadow* shadows = nullptr;
    GpuMneeJob* mneeJobs = nullptr;

    // Compacted work queues (pixel indices). Host launches 1D on live count.
    int* qIntersect = nullptr;
    int* qIntersectNext = nullptr;
    int* qVolume = nullptr;
    int* qSurface = nullptr;
    int* qBackground = nullptr;
    int* qShadow = nullptr;
    unsigned int* workCounts = nullptr;  // kSlotCount atomics
    int* workItems = nullptr;            // queue this optixLaunch reads
    int workCount = 0;                   // 1D launch width (upper bound)
    int workSlot = -1;                   // >=0: live count is workCounts[slot] on device
    int compactLaunch = 0;               // unused (Iray pool = full W×H, no host count)
    int batchSamples = 1;                // spp folded in this wavefront (GPU regen)
    // 1 = path_tail only drains kQueueExitToDiffuse (MNEE leftover PT must not
    // grow extra bounces). 0 = normal megakernel tail plus ETD.
    int pathTailExitOnly = 0;

    int width = 0;
    int height = 0;
    int pixelOffsetX = 0;
    int pixelOffsetY = 0;

    int sampleIndex = 0;
    unsigned int frameSeed = 0;

    GpuSpectralTables spec{};

    CameraProj camProj{};
    // 1 / (light paths this wavefront). CPU BDPT divides the splat plane by W×H.
    // 0 disables GPU light-trace splats. Iray aiming uses 1/(W×H) (no extra K).
    float splatInvLightPaths = 0.0f;
    int mcmcMutations = 0;  // leftover cone-mutation slots; always 0 (Iray aiming)

    // Iray photon aiming: caster bounding spheres. Mix 1 = aimed only.
    const GpuPhotonCluster* photonClusters = nullptr;
    int photonClusterCount = 0;
    float photonAimMix = 0.0f;

    unsigned long long traversable = 0;  // OptixTraversableHandle
};

}  // namespace sol
