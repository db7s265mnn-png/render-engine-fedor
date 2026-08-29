// Integrator vs render-device rules.
// BDPT is CPU-only. GPU and XPU hide it in the UI and remember the CPU choice.
#pragma once

#include "scene/types.h"

namespace sol {

// Called when Render Device changes. Updates the remembered CPU/GPU integrator
// indices (never stores BDPT on the GPU side) and returns the integrator that
// should be active on `newBackend`.
inline int switchIntegratorForBackend(int oldBackend, int newBackend, int currentIntegrator, int& cpuMem,
                                      int& gpuMem) {
    if (oldBackend == newBackend) {
        if (renderDeviceUsesGpu(newBackend) && currentIntegrator == kIntegratorBdpt)
            return kIntegratorPathTracer;
        return currentIntegrator;
    }

    const bool oldGpu = renderDeviceUsesGpu(oldBackend);
    const bool newGpu = renderDeviceUsesGpu(newBackend);
    if (!oldGpu)
        cpuMem = currentIntegrator;
    else
        gpuMem = (currentIntegrator == kIntegratorBdpt) ? kIntegratorPathTracer : currentIntegrator;

    if (newGpu) {
        if (currentIntegrator == kIntegratorBdpt) {
            if (gpuMem == kIntegratorBdpt) return kIntegratorPathTracer;
            return gpuMem;
        }
        return currentIntegrator;
    }
    return cpuMem;
}

inline int clampIntegratorForBackend(int backend, int integrator) {
    if (renderDeviceUsesGpu(backend) && integrator == kIntegratorBdpt) return kIntegratorPathTracer;
    return integrator;
}

}  // namespace sol
