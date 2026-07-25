#include "io/image_io.h"

#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QRegularExpression>
#include <QString>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "core/log.h"
#include "render/framebuffer.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPENEXR
#  include <ImfChannelList.h>
#  include <ImfInputFile.h>
#  include <ImfOutputFile.h>
#  include <ImfRgbaFile.h>
#  include <ImfFrameBuffer.h>
#endif

namespace sol {
namespace {

std::string toLowerExtension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return {};
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return ext;
}

// ---------------------------------------------------------------------------
// Radiance .hdr (RGBE)
// ---------------------------------------------------------------------------
Vec3 rgbeToLinear(const unsigned char rgbe[4]) {
    if (rgbe[3] == 0) return Vec3(0.0f);
    const float f = std::ldexp(1.0f, int(rgbe[3]) - (128 + 8));
    return Vec3(float(rgbe[0]) * f, float(rgbe[1]) * f, float(rgbe[2]) * f);
}

void linearToRgbe(Vec3 c, unsigned char rgbe[4]) {
    const float v = std::max({c.x, c.y, c.z, 0.0f});
    if (v < 1e-32f) {
        rgbe[0] = rgbe[1] = rgbe[2] = rgbe[3] = 0;
        return;
    }
    int e = 0;
    const float m = std::frexp(v, &e) * 256.0f / v;
    rgbe[0] = static_cast<unsigned char>(std::clamp(c.x * m, 0.0f, 255.0f));
    rgbe[1] = static_cast<unsigned char>(std::clamp(c.y * m, 0.0f, 255.0f));
    rgbe[2] = static_cast<unsigned char>(std::clamp(c.z * m, 0.0f, 255.0f));
    rgbe[3] = static_cast<unsigned char>(e + 128);
}

bool readHdrScanline(std::FILE* file, std::vector<unsigned char>& scanline, int width) {
    unsigned char header[4];
    if (std::fread(header, 1, 4, file) != 4) return false;
    const bool rle = header[0] == 2 && header[1] == 2 && ((int(header[2]) << 8) | header[3]) == width && width >= 8 &&
                     width < 32768;
    if (!rle) {
        // Flat scanline; the four bytes already read belong to the first pixel.
        scanline[0] = header[0];
        scanline[1] = header[1];
        scanline[2] = header[2];
        scanline[3] = header[3];
        if (width > 1 && std::fread(scanline.data() + 4, 1, size_t(width - 1) * 4, file) != size_t(width - 1) * 4)
            return false;
        return true;
    }
    for (int channel = 0; channel < 4; ++channel) {
        int x = 0;
        while (x < width) {
            unsigned char count = 0;
            if (std::fread(&count, 1, 1, file) != 1) return false;
            if (count > 128) {
                unsigned char value = 0;
                if (std::fread(&value, 1, 1, file) != 1) return false;
                const int run = int(count) - 128;
                if (x + run > width) return false;
                for (int i = 0; i < run; ++i) scanline[size_t(x + i) * 4 + channel] = value;
                x += run;
            } else {
                const int run = int(count);
                if (run == 0 || x + run > width) return false;
                for (int i = 0; i < run; ++i) {
                    unsigned char value = 0;
                    if (std::fread(&value, 1, 1, file) != 1) return false;
                    scanline[size_t(x + i) * 4 + channel] = value;
                }
                x += run;
            }
        }
    }
    return true;
}

bool loadHdr(const std::string& path, Image& out, std::string& error) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) {
        error = "cannot open " + path;
        return false;
    }
    char line[512];
    bool isRadiance = false;
    float exposure = 1.0f;
    while (std::fgets(line, sizeof(line), file)) {
        std::string text(line);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        if (text.rfind("#?", 0) == 0) {
            isRadiance = true;
            continue;
        }
        if (text.empty()) break;  // end of header
        if (text.rfind("EXPOSURE=", 0) == 0) exposure = std::strtof(text.c_str() + 9, nullptr);
    }
    if (!isRadiance) {
        std::fclose(file);
        error = path + " is not a Radiance HDR file";
        return false;
    }
    if (!std::fgets(line, sizeof(line), file)) {
        std::fclose(file);
        error = "truncated HDR header";
        return false;
    }
    int width = 0, height = 0;
    if (std::sscanf(line, "-Y %d +X %d", &height, &width) != 2) {
        std::fclose(file);
        error = "unsupported HDR resolution line";
        return false;
    }
    if (width <= 0 || height <= 0) {
        std::fclose(file);
        error = "invalid HDR resolution";
        return false;
    }

    out.resize(width, height);
    std::vector<unsigned char> scanline(size_t(width) * 4);
    const float invExposure = exposure > 0.0f ? 1.0f / exposure : 1.0f;
    for (int y = 0; y < height; ++y) {
        if (!readHdrScanline(file, scanline, width)) {
            std::fclose(file);
            error = "truncated HDR scanline data";
            return false;
        }
        for (int x = 0; x < width; ++x) {
            unsigned char rgbe[4] = {scanline[size_t(x) * 4 + 0], scanline[size_t(x) * 4 + 1],
                                     scanline[size_t(x) * 4 + 2], scanline[size_t(x) * 4 + 3]};
            out.setRgb(x, y, rgbeToLinear(rgbe) * invExposure, 1.0f);
        }
    }
    std::fclose(file);
    return true;
}

bool writeHdr(const std::string& path, const Image& image, std::string& error) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) {
        error = "cannot write " + path;
        return false;
    }
    std::fprintf(file, "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n", image.height(), image.width());
    std::vector<unsigned char> row(size_t(image.width()) * 4);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) linearToRgbe(image.rgb(x, y), &row[size_t(x) * 4]);
        if (std::fwrite(row.data(), 1, row.size(), file) != row.size()) {
            std::fclose(file);
            error = "failed writing HDR data";
            return false;
        }
    }
    std::fclose(file);
    return true;
}

#if SOLSTICE_HAVE_OPENEXR
bool loadExr(const std::string& path, Image& out, std::string& error) {
    try {
        Imf::RgbaInputFile file(path.c_str());
        const Imath::Box2i dw = file.dataWindow();
        const int width = dw.max.x - dw.min.x + 1;
        const int height = dw.max.y - dw.min.y + 1;
        std::vector<Imf::Rgba> pixels(size_t(width) * size_t(height));
        file.setFrameBuffer(pixels.data() - dw.min.x - size_t(dw.min.y) * width, 1, size_t(width));
        file.readPixels(dw.min.y, dw.max.y);
        out.resize(width, height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const Imf::Rgba& p = pixels[size_t(y) * size_t(width) + size_t(x)];
                // Clamp negatives: some EXRs store below-black values that turn
                // the display path / env sampling black or NaN after tone map.
                const Vec3 c(std::max(0.0f, float(p.r)), std::max(0.0f, float(p.g)),
                             std::max(0.0f, float(p.b)));
                out.setRgb(x, y, c, std::max(0.0f, float(p.a)));
            }
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("OpenEXR: ") + e.what();
        return false;
    }
}

bool writeExr(const std::string& path, const Image& image, std::string& error) {
    try {
        std::vector<Imf::Rgba> pixels(size_t(image.width()) * size_t(image.height()));
        for (int y = 0; y < image.height(); ++y) {
            for (int x = 0; x < image.width(); ++x) {
                const Vec3 c = image.rgb(x, y);
                pixels[size_t(y) * size_t(image.width()) + size_t(x)] = Imf::Rgba(c.x, c.y, c.z, 1.0f);
            }
        }
        Imf::RgbaOutputFile file(path.c_str(), image.width(), image.height(), Imf::WRITE_RGBA);
        file.setFrameBuffer(pixels.data(), 1, size_t(image.width()));
        file.writePixels(image.height());
        return true;
    } catch (const std::exception& e) {
        error = std::string("OpenEXR: ") + e.what();
        return false;
    }
}
#endif

bool loadLdr(const std::string& path, Image& out, std::string& error) {
    QImage qimage;
    if (!qimage.load(QString::fromStdString(path))) {
        error = "unsupported or unreadable image: " + path;
        return false;
    }
    qimage = qimage.convertToFormat(QImage::Format_RGBA8888);
    out.resize(qimage.width(), qimage.height());
    for (int y = 0; y < qimage.height(); ++y) {
        const uchar* line = qimage.constScanLine(y);
        for (int x = 0; x < qimage.width(); ++x) {
            const uchar* px = line + size_t(x) * 4;
            out.setRgb(x, y,
                       Vec3(srgbToLinear(px[0] / 255.0f), srgbToLinear(px[1] / 255.0f),
                            srgbToLinear(px[2] / 255.0f)),
                       px[3] / 255.0f);
        }
    }
    return true;
}

}  // namespace

bool imageFormatIsHdr(const std::string& path) {
    const std::string ext = toLowerExtension(path);
    return ext == "hdr" || ext == "exr" || ext == "rgbe" || ext == "pic";
}

bool loadImage(const std::string& path, Image& out, std::string& error) {
    const std::string ext = toLowerExtension(path);
    if (ext == "hdr" || ext == "rgbe" || ext == "pic") return loadHdr(path, out, error);
    if (ext == "exr") {
#if SOLSTICE_HAVE_OPENEXR
        return loadExr(path, out, error);
#else
        error = "this build has no OpenEXR support, use .hdr instead";
        return false;
#endif
    }
    return loadLdr(path, out, error);
}

bool pathHasUdimToken(const QString& path) {
    return path.contains(QLatin1String("<UDIM>"), Qt::CaseInsensitive) ||
           path.contains(QLatin1String("%(UDIM)d"));
}

QString expandUdimToken(const QString& pattern, int udim) {
    QString result = pattern;
    const QString number = QString::number(udim);
    result.replace(QStringLiteral("<UDIM>"), number, Qt::CaseInsensitive);
    result.replace(QStringLiteral("%(UDIM)d"), number);
    return result;
}

namespace {

QString decodeXmlEntities(QString path) {
    // Decode entities if a Qt writer previously escaped the MaterialX token.
    path.replace(QLatin1String("&lt;"), QLatin1String("<"));
    path.replace(QLatin1String("&gt;"), QLatin1String(">"));
    return path.trimmed();
}

QString makeAbsoluteTexturePath(QString path, const QString& searchDirectory) {
    path = decodeXmlEntities(path);
    const QFileInfo info(path);
    if (!info.isAbsolute() && !searchDirectory.isEmpty()) return QDir(searchDirectory).absoluteFilePath(path);
    return path;
}

// Concrete Mari/MaterialX tile name: ...[._]1xxx.ext  (1001..1999).
bool concreteUdimFilename(const QString& fileName, QString& prefix, int& udim, QString& suffixWithDot) {
    static const QRegularExpression re(QStringLiteral(R"((.*)([._])(1\d{3})(\.[^.]+)$)"));
    const QRegularExpressionMatch match = re.match(fileName);
    if (!match.hasMatch()) return false;
    udim = match.captured(3).toInt();
    if (udim < 1001 || udim >= 2000) return false;
    prefix = match.captured(1) + match.captured(2);
    suffixWithDot = match.captured(4);
    return true;
}

}  // namespace

std::vector<int> discoverUdimTiles(const QString& patternIn, const QString& searchDirectory,
                                   const std::vector<int>& explicitUdims) {
    const QString pattern = makeAbsoluteTexturePath(patternIn, searchDirectory);
    std::vector<int> candidates = explicitUdims;
    if (candidates.empty()) {
        candidates.reserve(100);
        for (int udim = 1001; udim <= 1100; ++udim) candidates.push_back(udim);
    }
    std::vector<int> found;
    found.reserve(candidates.size());
    for (int udim : candidates) {
        if (udim < 1001 || udim >= 2000) continue;
        if (QFileInfo::exists(expandUdimToken(pattern, udim))) found.push_back(udim);
    }
    return found;
}

bool resolveUdimPattern(const QString& pathIn, const QString& searchDirectory, QString& outPattern,
                        std::vector<int>& outTiles) {
    outPattern.clear();
    outTiles.clear();
    QString path = makeAbsoluteTexturePath(pathIn, searchDirectory);
    if (path.isEmpty()) return false;

    if (pathHasUdimToken(path)) {
        outPattern = path;
        outTiles = discoverUdimTiles(path, QString(), {});
        return true;
    }

    const QFileInfo info(path);
    QString prefix;
    QString suffix;
    int udim = 0;
    if (!concreteUdimFilename(info.fileName(), prefix, udim, suffix)) return false;

    // MaterialX authoring form: keep <UDIM> unresolved in the filename.
    outPattern = info.dir().filePath(prefix + QStringLiteral("<UDIM>") + suffix);
    outTiles = discoverUdimTiles(outPattern, QString(), {});
    if (outTiles.empty()) outTiles.push_back(udim);
    return true;
}

std::shared_ptr<Image> loadImageOrUdim(const QString& pathIn, const QString& searchDirectory, std::string& error,
                                       const std::vector<int>& explicitUdims) {
    QString path = makeAbsoluteTexturePath(pathIn, searchDirectory);

    // MaterialX: unresolved <UDIM> in file + tile list from udimset / disk.
    // Also accept a concrete tile path (name.1001.exr) and promote it to a UDIM set.
    QString pattern;
    std::vector<int> discovered;
    const bool isUdim = resolveUdimPattern(path, QString(), pattern, discovered);
    if (isUdim) {
        struct Tile {
            int udim = 0;
            std::shared_ptr<Image> image;
        };
        std::vector<int> udimList = explicitUdims;
        if (udimList.empty()) udimList = discovered;
        if (udimList.empty()) udimList = discoverUdimTiles(pattern, QString(), {});

        std::vector<Tile> tiles;
        tiles.reserve(udimList.size());
        for (int udim : udimList) {
            if (udim < 1001 || udim >= 2000) continue;
            const QString tilePath = expandUdimToken(pattern, udim);
            if (!QFileInfo::exists(tilePath)) continue;
            auto tile = std::make_shared<Image>();
            std::string loadError;
            if (!loadImage(tilePath.toStdString(), *tile, loadError)) {
                logWarning("UDIM tile failed (" + tilePath.toStdString() + "): " + loadError);
                continue;
            }
            tiles.push_back({udim, std::move(tile)});
        }
        if (tiles.empty()) {
            error = "no UDIM tiles found for: " + pattern.toStdString();
            return nullptr;
        }

        // MaterialX Mesh::splitByUdims / Mari: UDIM = 1001 + U + V*10, U,V = floor(uv).
        int maxU = 0;
        int maxV = 0;
        int tileW = tiles.front().image->width();
        int tileH = tiles.front().image->height();
        for (const Tile& tile : tiles) {
            const int u = (tile.udim - 1001) % 10;
            const int v = (tile.udim - 1001) / 10;
            maxU = std::max(maxU, u);
            maxV = std::max(maxV, v);
            tileW = std::max(tileW, tile.image->width());
            tileH = std::max(tileH, tile.image->height());
        }
        const int gridU = maxU + 1;
        const int gridV = maxV + 1;

        // Bake tiles into a UV-space atlas (Mari: UDIM = 1001 + U + V*10).
        // Each tile is V-flipped so OpenGL/MaterialX/Alembic UVs (V=0 at bottom)
        // sample the file correctly when using floor/fract tiling.
        auto atlas = std::make_shared<Image>(tileW * gridU, tileH * gridV, Vec4(0.0f, 0.0f, 0.0f, 0.0f));
        atlas->setUdimGrid(gridU, gridV);
        for (const Tile& tile : tiles) {
            const int u = (tile.udim - 1001) % 10;
            const int v = (tile.udim - 1001) / 10;
            const int dstX0 = u * tileW;
            const int dstY0 = v * tileH;
            const Image& src = *tile.image;
            for (int y = 0; y < tileH; ++y) {
                const int sy = std::min(y, src.height() - 1);
                const int dy = dstY0 + (tileH - 1 - y);  // V-flip within tile
                for (int x = 0; x < tileW; ++x) {
                    const int sx = std::min(x, src.width() - 1);
                    atlas->at(dstX0 + x, dy) = src.at(sx, sy);
                }
            }
        }
        std::string ids;
        for (size_t i = 0; i < tiles.size(); ++i) {
            if (i) ids += ",";
            ids += std::to_string(tiles[i].udim);
        }
        logInfo("UDIM: loaded " + std::to_string(tiles.size()) + " tile(s) [" + ids + "] → atlas " +
                std::to_string(gridU) + "x" + std::to_string(gridV) + " from " + pattern.toStdString());
        return atlas;
    }

    if (!QFileInfo::exists(path)) {
        error = "texture not found: " + path.toStdString();
        return nullptr;
    }
    auto image = std::make_shared<Image>();
    if (!loadImage(path.toStdString(), *image, error)) return nullptr;
    return image;
}

bool saveImagePng(const std::string& path, const Image& displayImage, std::string& error) {
    if (displayImage.empty()) {
        error = "empty image";
        return false;
    }
    QImage qimage(displayImage.width(), displayImage.height(), QImage::Format_RGBA8888);
    for (int y = 0; y < displayImage.height(); ++y) {
        uchar* line = qimage.scanLine(y);
        for (int x = 0; x < displayImage.width(); ++x) {
            const Vec3 c = displayImage.rgb(x, y);
            uchar* px = line + size_t(x) * 4;
            px[0] = static_cast<uchar>(std::lround(clampf(c.x, 0.0f, 1.0f) * 255.0f));
            px[1] = static_cast<uchar>(std::lround(clampf(c.y, 0.0f, 1.0f) * 255.0f));
            px[2] = static_cast<uchar>(std::lround(clampf(c.z, 0.0f, 1.0f) * 255.0f));
            px[3] = 255;
        }
    }
    if (!qimage.save(QString::fromStdString(path))) {
        error = "failed to write " + path;
        return false;
    }
    return true;
}

bool saveImageHdr(const std::string& path, const Image& linearImage, std::string& error) {
    return writeHdr(path, linearImage, error);
}

bool saveImageExr(const std::string& path, const Image& linearImage, std::string& error) {
#if SOLSTICE_HAVE_OPENEXR
    return writeExr(path, linearImage, error);
#else
    (void)path;
    (void)linearImage;
    error = "this build has no OpenEXR support";
    return false;
#endif
}

bool saveImageAuto(const std::string& path, const Image& linear, const RenderSettingsData& settings,
                   std::string& error) {
    const std::string ext = toLowerExtension(path);
    if (ext == "hdr") return saveImageHdr(path, linear, error);
    if (ext == "exr") return saveImageExr(path, linear, error);

    Image display(linear.width(), linear.height());
    for (int y = 0; y < linear.height(); ++y)
        for (int x = 0; x < linear.width(); ++x) display.setRgb(x, y, applyToneMap(linear.rgb(x, y), settings));
    return saveImagePng(path, display, error);
}

}  // namespace sol
