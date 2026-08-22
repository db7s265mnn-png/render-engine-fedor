// Camera AA / DoF pixel+lens samples shared by Embree and OptiX so XPU tiles
// (and GPU-only) use the same generator as CPU.
#pragma once

#include "core/rng.h"
#include "render/blue_noise.h"
#include "render/rsequence.h"
#include "render/sobol.h"
#include "scene/types.h"

namespace sol {

SR_INL SR_HD void sampleCameraPixelLens(int pixelSampler, int x, int y, int sampleIndex, int width,
                                        uint32_t frameSeed, float manualTestMult, float& jx, float& jy,
                                        float& lensU, float& lensV) {
    jx = 0.5f;
    jy = 0.5f;
    lensU = 0.5f;
    lensV = 0.5f;
    if (pixelSampler == kPixelSamplerBlueNoise) {
        blueNoisePixelJitter(x, y, sampleIndex, jx, jy);
        blueNoiseLensSample(x, y, sampleIndex, lensU, lensV);
        return;
    }
    if (pixelSampler == kPixelSamplerXorshift) {
        Rng xrng = makePixelRngXorshift32(x, y, sampleIndex, frameSeed, 0xCA7E11u);
        jx = xrng.nextFloat();
        jy = xrng.nextFloat();
        lensU = xrng.nextFloat();
        lensV = xrng.nextFloat();
        return;
    }
    if (pixelSampler == kPixelSamplerGenPnt2D) {
        r2PixelJitter(x, y, sampleIndex, width, kR2IndexSpp, jx, jy);
        r2LensSample(x, y, sampleIndex, width, kR2IndexSpp, lensU, lensV);
        return;
    }
    if (pixelSampler == kPixelSamplerManualTest) {
        Rng xrng = makePixelRngXorshift32(x, y, sampleIndex, frameSeed, 0x7E57u);
        const float ux = xrng.nextFloat() * 2.0f - 1.0f;
        const float uy = xrng.nextFloat() * 2.0f - 1.0f;
        jx = srMin(0.999999f, srMax(0.0f, 0.5f + ux * manualTestMult));
        jy = srMin(0.999999f, srMax(0.0f, 0.5f + uy * manualTestMult));
        lensU = 0.5f;
        lensV = 0.5f;
        return;
    }
    pixelSample(x, y, sampleIndex, jx, jy);
    lensSample(x, y, sampleIndex, lensU, lensV);
}

}  // namespace sol
