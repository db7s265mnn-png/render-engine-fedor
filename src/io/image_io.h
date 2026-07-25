// Image loading and saving. Radiance .hdr is handled natively, OpenEXR through
// the optional OpenEXR dependency and every LDR format through Qt.
#pragma once

#include <memory>
#include <string>

#include <QString>

#include "core/image.h"
#include "scene/types.h"

namespace sol {

// Loads an image and converts it to linear float RGBA. LDR files are assumed to
// be sRGB encoded and are linearised.
bool loadImage(const std::string& path, Image& out, std::string& error);

// Houdini-style UDIM helpers (<UDIM> or %(UDIM)d).
bool pathHasUdimToken(const QString& path);
QString expandUdimToken(const QString& pattern, int udim);
// If path looks like foo.1001.exr and sibling tiles exist, returns foo.<UDIM>.exr.
QString tokenizeUdimPathIfSequence(const QString& path);
// Loads a single image, or a UDIM tile set when the path contains <UDIM>.
std::shared_ptr<Image> loadImageOrUdim(const QString& path, const QString& searchDirectory, std::string& error);

bool saveImagePng(const std::string& path, const Image& displayImage, std::string& error);
bool saveImageHdr(const std::string& path, const Image& linearImage, std::string& error);
bool saveImageExr(const std::string& path, const Image& linearImage, std::string& error);

// Dispatches on the file extension (.png/.jpg/.exr/.hdr). `linear` must hold
// scene referred values; tone mapping is applied for LDR targets only.
bool saveImageAuto(const std::string& path, const Image& linear, const RenderSettingsData& settings,
                   std::string& error);

bool imageFormatIsHdr(const std::string& path);

}  // namespace sol
