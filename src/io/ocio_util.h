// OpenColorIO helpers: config discovery, status logging, mplay-style display views.
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

// Classic (non-OCIO) monitor transform: Raw = linear clamp; else tone + transfer.
// When workingSpace is ACEScg, converts to linear Rec.709/sRGB primaries first.
Vec3 classicApplyView(Vec3 linearWorking, int workingSpace, int viewTransform);

// Apply display view: Classic or OCIO based on colorManagement.
// viewTransform: kViewSrgbAces / kViewRec709Aces / kViewRec2020 / kViewRaw.
// Returns false if OCIO was requested but unavailable (out still filled via Classic fallback).
bool ocioApplyView(Vec3 linearWorking, int workingSpace, int viewTransform, Vec3& outDisplay);

// Prepare display for many pixels (call once per resolve). Then use ocioApplyViewPrepared.
// colorManagement selects Classic vs OCIO path for the prepared state.
bool ocioPrepareView(int workingSpace, int viewTransform);
bool displayPrepareView(int workingSpace, int colorManagement, int viewTransform);
Vec3 ocioApplyViewPrepared(Vec3 linearWorking);  // no lock; requires prepare

// True when this build was linked with OpenColorIO.
bool ocioLibraryAvailable();

}  // namespace sol
