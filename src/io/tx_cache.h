// Texture TX cache: convert source textures to .tx mipmap TIFF files using the
// shared tx_convert core (maketx / oiiotool + optional OCIO → ACEScg).
#pragma once

#include <string>

#include "scene/types.h"

namespace sol {

// Default input colour space for automatic conversions (MaterialX may override).
void setTxDefaultInputColorSpace(const std::string& colorSpace);
std::string txDefaultInputColorSpace();

std::string resolveTxCachedPath(const std::string& sourcePath, const std::string& cacheDir);

bool ensureTxTexture(const std::string& sourcePath, const RenderSettingsData& settings,
                     std::string& outPath, std::string& error);

bool ensureTxTexture(const std::string& sourcePath, const RenderSettingsData& settings,
                     const std::string& inputColorSpace, std::string& outPath, std::string& error);

void setActiveTxCacheSettings(const RenderSettingsData* settings);
bool txCacheActive();
std::string txCacheResolve(const std::string& sourcePath);
// Same as txCacheResolve, but uses `inputColorSpace` instead of the sticky MaterialX default.
std::string txCacheResolve(const std::string& sourcePath, const std::string& inputColorSpace);

}  // namespace sol
