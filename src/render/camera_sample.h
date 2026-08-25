// Camera AA / DoF samples shared by Embree and OptiX.
// PBRT4 Owen-scrambled Sobol: pixel jitter dims 0–1, lens dims 2–3.
// Embree path RNG consumes those dimensions from one shared stream instead.
#pragma once

#include "render/sobol.h"

namespace sol {

SR_INL SR_HD void sampleCameraPixelLens(int x, int y, int sampleIndex, float& jx, float& jy,
                                        float& lensU, float& lensV) {
    pixelSample(x, y, sampleIndex, jx, jy);
    lensSample(x, y, sampleIndex, lensU, lensV);
}

}  // namespace sol
