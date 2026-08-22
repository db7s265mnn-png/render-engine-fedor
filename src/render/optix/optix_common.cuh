// Shared OptiX launch-parameter blob. Declared in every device TU in the
// pipeline (OptiX requires the same pipelineLaunchParamsVariableName).
#pragma once

#include <optix.h>

#include "render/optix/launch_params.h"

extern "C" {
__constant__ __align__(16) unsigned char solsticeLaunchParams[sizeof(sol::LaunchParams)];
}

__device__ inline const sol::LaunchParams& launchParams() {
    return *reinterpret_cast<const sol::LaunchParams*>(solsticeLaunchParams);
}
