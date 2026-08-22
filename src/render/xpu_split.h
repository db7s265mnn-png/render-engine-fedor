// XPU work split: even samples on GPU, odd samples on CPU (full frame each).
#pragma once

#include "scene/types.h"

namespace sol {

SR_INL SR_HD bool xpuGpuOwnsSample(int sampleIndex) { return (sampleIndex & 1) == 0; }

}  // namespace sol
