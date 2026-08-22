// Fallback used when the project is configured without CUDA/OptiX support.
#include "core/log.h"
#include "render/render_device.h"
#include "solstice_config.h"

#if !SOLSTICE_HAVE_OPTIX

namespace sol {

bool optixBackendCompiledIn() { return false; }

bool optixRuntimeAvailable(std::string* error) {
    if (error) *error = "not compiled into this build";
    return false;
}

RenderDevicePtr createOptixDevice() {
    logWarning("This build has no OptiX backend. Configure with -DSOLSTICE_ENABLE_OPTIX=ON and a CUDA toolkit "
               "plus the OptiX SDK to enable GPU rendering.");
    return nullptr;
}

}  // namespace sol

#endif
