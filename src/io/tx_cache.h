// Texture TX cache: convert source textures to .tx mipmap TIFF files at cook
// time using maketx (preferred) or oiiotool as a fallback. The cache is keyed
// by source path + mtime + file size so unchanged files are skipped.
#pragma once

#include <string>

#include "scene/types.h"

namespace sol {

// Return the .tx path that would correspond to `sourcePath` in `cacheDir`.
// Does NOT perform any conversion; useful for querying the expected output path.
std::string resolveTxCachedPath(const std::string& sourcePath, const std::string& cacheDir);

// Ensure `sourcePath` has a fresh .tx counterpart in the cache.
//   settings.enableTxCache must be non-zero for any conversion to happen.
//   On success, `outPath` is set to the (possibly new) .tx file path and true is returned.
//   On failure or when the source is already .tx, `outPath` is set to `sourcePath`
//   and an informational / warning message may be written to `error`.
//   Conversion is synchronous (blocks until maketx/oiiotool finishes).
//   Thread-safe with respect to independent source paths; callers should not
//   convert the same path concurrently.
bool ensureTxTexture(const std::string& sourcePath, const RenderSettingsData& settings,
                     std::string& outPath, std::string& error);

// Set / clear the process-wide active TX cache settings used by loadImage and
// loadImageOrUdim. Pass nullptr to disable automatic conversion.
// Typically called at the start of Stage::toScene (or from MainWindow before
// graph cook) so that all subsequent image loads go through the cache.
void setActiveTxCacheSettings(const RenderSettingsData* settings);

// Return true when automatic TX conversion is currently active (settings pointer
// is set and enableTxCache != 0).
bool txCacheActive();

// Convenience: if the process-wide TX cache is active, convert sourcePath and
// return the resulting .tx path; otherwise return sourcePath unchanged.
// Used internally by loadImage / loadImageOrUdim.
std::string txCacheResolve(const std::string& sourcePath);

}  // namespace sol
