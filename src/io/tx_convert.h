// Shared maketx / oiiotool TX conversion core (Arnold-style).
// Used by automatic cook-time conversion and by the TX_Converter app.
#pragma once

#include <string>
#include <vector>

namespace sol {

// Curated ACES / Utility input spaces (default UI). Output is always ACEScg.
std::vector<std::string> txCuratedColorSpaces();

// Parse colour space names from an OCIO config file (advanced list).
// Returns curated list if the file is missing / unreadable.
std::vector<std::string> txColorSpacesFromConfig(const std::string& ocioConfigPath);

// Resolve OCIO config path: prefer env OCIO when useEnv is true, else settingsPath.
std::string txResolveOcioConfig(bool useEnv, const std::string& settingsPath);

// True when input space should skip --colorconvert (already ACEScg / Raw / empty).
bool txSkipColorConvert(const std::string& inputColorSpace);

// Expand a path that may contain <UDIM> / %(UDIM)d into concrete tile paths that exist.
// Non-UDIM paths return a single-element list if the file exists (or the path as-is).
std::vector<std::string> txExpandUdimSources(const std::string& sourcePathOrPattern);

// Expand $F / $F2..$F8 in a pattern across [frameStart, frameEnd] (inclusive).
// Returns existing concrete files. If pattern has no $F token, returns empty.
std::vector<std::string> txExpandFrameSources(const std::string& sourcePathOrPattern, int frameStart,
                                              int frameEnd);

// Pick an output .tx path in `outputDir` named like the source basename.
// If that name is taken by a different source, use name_copy_1.tx, name_copy_2.tx, …
// `sourcePath` is recorded in a sibling `.txsrc` sidecar so re-runs stay stable.
std::string txAllocateOutputPath(const std::string& sourcePath, const std::string& outputDir);

struct TxConvertRequest {
    std::string sourcePath;       // concrete file (already expanded)
    std::string outputPath;       // destination .tx
    std::string inputColorSpace;  // e.g. "Utility - sRGB - Texture"; empty/ACEScg/Raw → no convert
    std::string ocioConfigPath;   // optional --colorconfig
    bool updateOnly = true;       // maketx -u
};

struct TxConvertResult {
    bool ok = false;
    std::string outputPath;
    std::string error;
};

// Convert one texture to tiled mipmapped .tx (ACEScg when colorconvert applies).
TxConvertResult txConvertOne(const TxConvertRequest& req);

// Convert source (possibly UDIM / $F pattern) into `outputDir`. Names from sources.
// frameStart/frameEnd used when the path contains $F / $F#.
// Returns false if any tile failed (partial successes still written).
bool txConvertPattern(const std::string& sourcePathOrPattern, const std::string& outputDir,
                      const std::string& inputColorSpace, const std::string& ocioConfigPath,
                      std::vector<TxConvertResult>& results, std::string& error,
                      int frameStart = 1, int frameEnd = 1);

}  // namespace sol
