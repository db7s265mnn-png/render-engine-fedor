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

// Classic (non-OCIO) monitor transform (Houdini-style without OCIO):
// Raw = linear clamp; else working→display primaries + sRGB/Rec.2020 OETF.
// No Reinhard / ACES RRT — textures and beauty keep linear response until the curve.
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

// Convert scene-referred RGB from `inputColorSpace` into ACEScg.
// Uses the loaded OCIO config when available; otherwise a classic ACES 1.2
// matrix / sRGB EOTF fallback for the curated Utility / ACES names.
Vec3 ocioConvertToAcescg(Vec3 rgb, const std::string& inputColorSpace);

#if defined(_WIN32)
// After optixInit: add bin/ocio to the DLL search path (zlib1 + OpenColorIO).
void ocioBindWindowsRuntimeDlls();
#else
inline void ocioBindWindowsRuntimeDlls() {}
#endif

}  // namespace sol
