// Image loading and saving. Radiance .hdr is handled natively, OpenEXR through
// the optional OpenEXR dependency, Arnold/OIIO .tx (and .tif) mip pyramids through
// libtiff, and every other LDR format through Qt.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <QString>

#include "core/image.h"
#include "scene/types.h"

namespace sol {

// Loads an image as float RGBA. When srgbColor is true (default), LDR colour maps
// are sRGB-decoded to linear. Pass false for data maps (normal / bump / masks).
bool loadImage(const std::string& path, Image& out, std::string& error, bool srgbColor = true);

// Houdini/MaterialX UDIM helpers (<UDIM> / %(UDIM)d, Mari index 1001+U+V*10).
bool pathHasUdimToken(const QString& path);
QString expandUdimToken(const QString& pattern, int udim);
// If `path` is already a MaterialX UDIM pattern, or a concrete tile like name.1001.exr,
// returns the unresolved pattern (with <UDIM>) and discovers existing tiles on disk.
// Matches MaterialX authoring: file keeps <UDIM>; tiles come from udimset / filesystem.
bool resolveUdimPattern(const QString& path, const QString& searchDirectory, QString& outPattern,
                        std::vector<int>& outTiles);
std::vector<int> discoverUdimTiles(const QString& pattern, const QString& searchDirectory,
                                   const std::vector<int>& explicitUdims = {});
// Loads a single image, or bakes a UDIM atlas (MaterialX hwNormalizeUdim-style).
// Optional explicitUdims come from MaterialX geominfo udimset; empty → discover on disk.
std::shared_ptr<Image> loadImageOrUdim(const QString& path, const QString& searchDirectory, std::string& error,
                                       const std::vector<int>& explicitUdims = {}, bool srgbColor = true);

bool saveImagePng(const std::string& path, const Image& displayImage, std::string& error);
bool saveImageHdr(const std::string& path, const Image& linearImage, std::string& error);
bool saveImageExr(const std::string& path, const Image& linearImage, std::string& error, int bitDepth = 16);
// Beauty RGB + optional spectral bin layers (Y channel per bin as mono).
bool saveImageExrSpectral(const std::string& path, const Image& linearImage, int width, int height,
                          int bins, const std::vector<float>& binAccum, int sampleCount,
                          std::string& error);

// Dispatches on the file extension (.png/.jpg/.exr/.hdr). `linear` must hold
// scene referred values; tone mapping is applied for LDR targets only.
bool saveImageAuto(const std::string& path, const Image& linear, const RenderSettingsData& settings,
                   std::string& error);

bool imageFormatIsHdr(const std::string& path);

}  // namespace sol
