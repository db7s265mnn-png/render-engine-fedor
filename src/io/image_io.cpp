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
    result.replace(QLatin1String("<UDIM>"), number, Qt::CaseInsensitive);
    result.replace(QLatin1String("%(UDIM)d"), number);
    return result;
}

QString tokenizeUdimPathIfSequence(const QString& path) {
    static const QRegularExpression re(QStringLiteral(R"(([._])(10\d{2})(\.[^.]+)$)"));
    const QRegularExpressionMatch match = re.match(path);
    if (!match.hasMatch()) return path;
    const QString tokenized = path.left(match.capturedStart(2)) + QStringLiteral("<UDIM>") + match.captured(3);
    int found = 0;
    for (int udim = 1001; udim <= 1100; ++udim) {
        if (QFileInfo::exists(expandUdimToken(tokenized, udim))) {
            ++found;
            if (found >= 2) return tokenized;
        }
    }
    return path;
}

std::shared_ptr<Image> loadImageOrUdim(const QString& pathIn, const QString& searchDirectory, std::string& error) {
    QString path = pathIn;
    QFileInfo info(path);
    if (!info.isAbsolute() && !searchDirectory.isEmpty()) path = QDir(searchDirectory).absoluteFilePath(path);

    if (pathHasUdimToken(path)) {
        auto udimImage = std::make_shared<Image>();
        int loaded = 0;
        for (int udim = 1001; udim <= 1100; ++udim) {
            const QString tilePath = expandUdimToken(path, udim);
            if (!QFileInfo::exists(tilePath)) continue;
            auto tile = std::make_shared<Image>();
            std::string loadError;
            if (!loadImage(tilePath.toStdString(), *tile, loadError)) {
                logWarning("UDIM tile failed (" + tilePath.toStdString() + "): " + loadError);
                continue;
            }
            udimImage->addUdimTile(udim, tile);
            ++loaded;
        }
        if (loaded == 0) {
            error = "no UDIM tiles found for: " + path.toStdString();
            return nullptr;
        }
        logInfo("Loaded " + std::to_string(loaded) + " UDIM tile(s) for " + path.toStdString());
        return udimImage;
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
