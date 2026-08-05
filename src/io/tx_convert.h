// Shared maketx / oiiotool conversion core (Arnold-style TX + PNG/JPG reformat).
// Used by automatic cook-time conversion and by the TX_Converter app.
#pragma once

#include <functional>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace sol {

enum class TxOutputFormat : int {
    Tx = 0,
    Png = 1,
    Jpg = 2,
    Exr = 3,
    Original = 4,  // same extension as source; rules follow resolved format
    Tiff = 5,      // .tif / .tiff — settings like EXR (16/32-bit)
};

enum class TxChannelMode : int {
    RGBA = 0,
    RGB = 1,
    R = 2,
    G = 3,
    B = 4,
    A = 5,
};

// Output pixel storage. Original = probe source (depth + uint/half/float kind).
enum class TxPixelType : int {
    Original = 0,
    UInt8 = 1,
    UInt16 = 2,
    Half = 3,   // 16-bit float
    Float = 4,  // 32-bit float
};

// Curated ACES / Utility input spaces (default UI). Output TX is always ACEScg.
std::vector<std::string> txCuratedColorSpaces();

// Parse colour space names from an OCIO config file (advanced list).
// Returns curated list if the file is missing / unreadable.
std::vector<std::string> txColorSpacesFromConfig(const std::string& ocioConfigPath);

// Resolve OCIO config path: prefer env OCIO when useEnv is true, else settingsPath.
std::string txResolveOcioConfig(bool useEnv, const std::string& settingsPath);

// True when input space should skip --colorconvert (already ACEScg / Raw / empty).
bool txSkipColorConvert(const std::string& inputColorSpace);

// Probe on-disk pixel type (never returns Original). Falls back to Float if unknown.
TxPixelType txProbePixelType(const std::string& path);

// Resolve Original → probed type; clamp to what `format` can store (e.g. PNG → uint only).
TxPixelType txResolvePixelType(TxPixelType selected, const std::string& sourcePath,
                               TxOutputFormat format);

// oiiotool/maketx -d token: "uint8" / "uint16" / "half" / "float".
std::string txPixelTypeOiiArg(TxPixelType type);

// UI / info-bar label: "8-bit uint", "16-bit half", "16-bit uint", "32-bit float".
std::string txPixelTypeDisplayLabel(TxPixelType type);

// Expand a path that may contain <UDIM> / %(UDIM)d into concrete tile paths that exist.
// Non-UDIM paths return a single-element list if the file exists (or the path as-is).
// When frameStart/frameEnd > 0, keep only tiles whose trailing number is in range.
std::vector<std::string> txExpandUdimSources(const std::string& sourcePathOrPattern,
                                            int frameStart = 0, int frameEnd = 0);

// Expand $F in a pattern across [frameStart, frameEnd] (inclusive).
// Tries unpadded and zero-padded variants so foo.$F.exr finds foo.0001.exr.
// Returns existing concrete files. If pattern has no $F token, returns empty.
std::vector<std::string> txExpandFrameSources(const std::string& sourcePathOrPattern, int frameStart,
                                              int frameEnd);

// File-name frame / UDIM number (trailing digits before extension), or 0.
int txExtractFrameNumber(const std::string& path);

// Map a path/extension to a concrete output format (Original never returned).
TxOutputFormat txFormatFromPath(const std::string& pathOrExt);

// Resolve Original → format from sourcePath; otherwise return `selected`.
TxOutputFormat txResolveFormat(TxOutputFormat selected, const std::string& sourcePath);

// Extension for the format. For Original, uses sourcePath suffix (fallback "exr").
std::string txOutputExtension(TxOutputFormat format, const std::string& sourcePath = {});

// Pick an output path in `outputDir` named like the source basename + extension.
// If that name is taken by a different source, use name_copy_1.ext, …
// For .tx, a sibling `.txsrc` sidecar keeps re-runs stable.
std::string txAllocateOutputPath(const std::string& sourcePath, const std::string& outputDir,
                                 TxOutputFormat format = TxOutputFormat::Tx);

struct TxConvertRequest {
    std::string sourcePath;       // concrete file (already expanded)
    std::string outputPath;       // destination path
    std::string inputColorSpace;  // TX only; empty/ACEScg/Raw → no convert
    std::string ocioConfigPath;   // optional --colorconfig (TX only)
    bool updateOnly = false;      // true = skip when output newer than source; Convert sets false
    TxOutputFormat format = TxOutputFormat::Tx;
    TxPixelType pixelType = TxPixelType::Original;  // resolved before write
    int longSide = 0;             // 0 = original; else max edge, aspect kept
    TxChannelMode channels = TxChannelMode::RGBA;
};

struct TxConvertOptions {
    std::string inputColorSpace;
    std::string ocioConfigPath;
    TxOutputFormat format = TxOutputFormat::Tx;
    TxPixelType pixelType = TxPixelType::Original;
    int longSide = 0;
    TxChannelMode channels = TxChannelMode::RGBA;
    int frameStart = 1;
    int frameEnd = 1;
    bool updateOnly = false;  // Convert always rewrites; true only for internal incremental paths
    // Parallel convert: 0 = auto from memoryBudgetBytes. Caps concurrent maketx/oiiotool jobs.
    int maxParallelJobs = 0;
    // Soft memory budget for concurrency estimate (default 32 GiB).
    std::int64_t memoryBudgetBytes = 32LL * 1024 * 1024 * 1024;
};

struct TxConvertResult {
    bool ok = false;
    std::string outputPath;
    std::string error;
};

// Progress: completed / total (after each file). May be called from worker threads — keep thread-safe.
using TxConvertProgressFn = std::function<void(int completed, int total, const std::string& sourcePath)>;

// Shared atomics for UI polling during parallel convert (optional).
struct TxConvertProgressState {
    std::atomic<int> completed{0};
    std::atomic<int> total{0};
};

// Convert one texture (TX via maketx; PNG/JPG via oiiotool; TX reformat = oiiotool then maketx).
TxConvertResult txConvertOne(const TxConvertRequest& req);

// Convert source (possibly UDIM / $F pattern) into `outputDir`.
bool txConvertPattern(const std::string& sourcePathOrPattern, const std::string& outputDir,
                      const TxConvertOptions& options, std::vector<TxConvertResult>& results,
                      std::string& error, const TxConvertProgressFn& progress = {});

// Backward-compatible TX-only helper (cook-time cache).
bool txConvertPattern(const std::string& sourcePathOrPattern, const std::string& outputDir,
                      const std::string& inputColorSpace, const std::string& ocioConfigPath,
                      std::vector<TxConvertResult>& results, std::string& error,
                      int frameStart = 1, int frameEnd = 1);

}  // namespace sol
