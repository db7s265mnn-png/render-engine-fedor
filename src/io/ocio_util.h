// OpenColorIO helpers: config discovery, status logging, Nuke-style display views.
#pragma once

#include <string>

#include "core/math.h"
#include "scene/types.h"

namespace sol {

struct OcioStatus {
    bool libraryAvailable = false;  // built with OpenColorIO
    bool configLoaded = false;
    bool fromEnvironment = false;
    std::string configPath;
    std::string message;  // human-readable one-liner for the log
};

// Resolve config from env (if useEnv) and/or settingsPath. Loads/caches processors.
OcioStatus ocioEnsureConfig(bool useEnv, const std::string& settingsPath);

// Log current OCIO status (library + config). Call at app start and Start/Render.
void ocioLogStatus(bool useEnv, const std::string& settingsPath);

// Apply Nuke-style viewer process to a linear working-space RGB pixel.
// viewTransform: kViewSrgbAces / kViewRec709Aces / kViewRaw.
// Returns false if Raw or OCIO unavailable (caller may leave linear / use fallback).
bool ocioApplyView(Vec3 linearWorking, int workingSpace, int viewTransform, Vec3& outDisplay);

// Prepare OCIO for many pixels (call once per resolve). Then use ocioApplyViewPrepared.
bool ocioPrepareView(int workingSpace, int viewTransform);
Vec3 ocioApplyViewPrepared(Vec3 linearWorking);  // no lock; requires prepare

// True when this build was linked with OpenColorIO.
bool ocioLibraryAvailable();

}  // namespace sol
