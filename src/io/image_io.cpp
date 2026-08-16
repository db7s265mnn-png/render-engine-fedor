#include "io/image_io.h"

#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QRegularExpression>
#include <QString>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "core/log.h"
#include "core/math.h"
#include "io/ocio_util.h"
#include "io/tx_cache.h"
#include "io/tx_convert.h"
#include "render/framebuffer.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPENEXR
#  include <ImfChannelList.h>
#  include <ImfInputFile.h>
#  include <ImfOutputFile.h>
#  include <ImfRgbaFile.h>
#  include <ImfFrameBuffer.h>
#endif

#if SOLSTICE_HAVE_TIFF
#  include <tiffio.h>
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

bool parseRadianceResolution(const char* line, int& width, int& height, int& nSlow, int& nFast,
                             char& axisSlow, char& signSlow, char& axisFast, char& signFast) {
    char s0 = 0, a0 = 0, s1 = 0, a1 = 0;
    int n0 = 0, n1 = 0;
    if (std::sscanf(line, " %c%c %d %c%c %d", &s0, &a0, &n0, &s1, &a1, &n1) != 6) return false;
    a0 = static_cast<char>(std::toupper(static_cast<unsigned char>(a0)));
    a1 = static_cast<char>(std::toupper(static_cast<unsigned char>(a1)));
    if ((a0 != 'X' && a0 != 'Y') || (a1 != 'X' && a1 != 'Y') || a0 == a1) return false;
    if (n0 <= 0 || n1 <= 0) return false;
    signSlow = s0;
    axisSlow = a0;
    nSlow = n0;
    signFast = s1;
    axisFast = a1;
    nFast = n1;
    width = (a0 == 'X' || a1 == 'X') ? ((a0 == 'X') ? n0 : n1) : 0;
    height = (a0 == 'Y' || a1 == 'Y') ? ((a0 == 'Y') ? n0 : n1) : 0;
    return width > 0 && height > 0;
}

int radianceAxisIndex(char axis, char sign, int i, int n) {
    // Image y=0 is the top row; x=0 is the left column.
    // Radiance -Y: first scanline is the top. +Y: first scanline is the bottom.
    // Radiance +X: first pixel is the left. -X: first pixel is the right.
    if (axis == 'Y') return (sign == '-') ? i : (n - 1 - i);
    return (sign == '+') ? i : (n - 1 - i);
}

bool loadHdr(const std::string& path, Image& out, std::string& error) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) {
        error = "cannot open " + path;
        return false;
    }
    char line[512];
    bool isRadiance = false;
    while (std::fgets(line, sizeof(line), file)) {
        std::string text(line);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
        if (text.rfind("#?", 0) == 0) {
            isRadiance = true;
            continue;
        }
        if (text.empty()) break;  // end of header
        // RGBE pixel values are used as-is (stb_image / Blender / Arnold).
        // Applying EXPOSURE= from the header crushes HDR suns.
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
    int width = 0, height = 0, nSlow = 0, nFast = 0;
    char axisSlow = 0, signSlow = 0, axisFast = 0, signFast = 0;
    if (!parseRadianceResolution(line, width, height, nSlow, nFast, axisSlow, signSlow, axisFast,
                                 signFast)) {
        std::fclose(file);
        error = "unsupported HDR resolution line";
        return false;
    }

    out.resize(width, height);
    std::vector<unsigned char> scanline(size_t(nFast) * 4);
    for (int s = 0; s < nSlow; ++s) {
        if (!readHdrScanline(file, scanline, nFast)) {
            std::fclose(file);
            error = "truncated HDR scanline data";
            return false;
        }
        for (int f = 0; f < nFast; ++f) {
            const int x = (axisSlow == 'X') ? radianceAxisIndex('X', signSlow, s, nSlow)
                                            : radianceAxisIndex('X', signFast, f, nFast);
            const int y = (axisSlow == 'Y') ? radianceAxisIndex('Y', signSlow, s, nSlow)
                                            : radianceAxisIndex('Y', signFast, f, nFast);
            unsigned char rgbe[4] = {scanline[size_t(f) * 4 + 0], scanline[size_t(f) * 4 + 1],
                                     scanline[size_t(f) * 4 + 2], scanline[size_t(f) * 4 + 3]};
            out.setRgb(x, y, rgbeToLinear(rgbe), 1.0f);
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

bool writeExr(const std::string& path, const Image& image, std::string& error, int bitDepth = 16) {
    try {
        const int w = image.width();
        const int h = image.height();
        if (bitDepth >= 32) {
            Imf::Header header(w, h);
            header.channels().insert("R", Imf::Channel(Imf::FLOAT));
            header.channels().insert("G", Imf::Channel(Imf::FLOAT));
            header.channels().insert("B", Imf::Channel(Imf::FLOAT));
            header.channels().insert("A", Imf::Channel(Imf::FLOAT));
            std::vector<float> R(size_t(w) * size_t(h)), G(size_t(w) * size_t(h)), B(size_t(w) * size_t(h)),
                A(size_t(w) * size_t(h), 1.0f);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const Vec3 c = image.rgb(x, y);
                    const size_t i = size_t(y) * size_t(w) + size_t(x);
                    R[i] = c.x;
                    G[i] = c.y;
                    B[i] = c.z;
                }
            }
            Imf::OutputFile file(path.c_str(), header);
            Imf::FrameBuffer fb;
            fb.insert("R", Imf::Slice(Imf::FLOAT, (char*)R.data(), sizeof(float), sizeof(float) * size_t(w)));
            fb.insert("G", Imf::Slice(Imf::FLOAT, (char*)G.data(), sizeof(float), sizeof(float) * size_t(w)));
            fb.insert("B", Imf::Slice(Imf::FLOAT, (char*)B.data(), sizeof(float), sizeof(float) * size_t(w)));
            fb.insert("A", Imf::Slice(Imf::FLOAT, (char*)A.data(), sizeof(float), sizeof(float) * size_t(w)));
            file.setFrameBuffer(fb);
            file.writePixels(h);
            return true;
        }

        std::vector<Imf::Rgba> pixels(size_t(w) * size_t(h));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                Vec3 c = image.rgb(x, y);
                if (bitDepth <= 8) c = quantizeRgb(c, 8);
                pixels[size_t(y) * size_t(w) + size_t(x)] = Imf::Rgba(c.x, c.y, c.z, 1.0f);
            }
        }
        Imf::RgbaOutputFile file(path.c_str(), w, h, Imf::WRITE_RGBA);
        file.setFrameBuffer(pixels.data(), 1, size_t(w));
        file.writePixels(h);
        return true;
    } catch (const std::exception& e) {
        error = std::string("OpenEXR: ") + e.what();
        return false;
    }
}

bool writeExrSpectral(const std::string& path, const Image& image, int width, int height, int bins,
                      const std::vector<float>& binAccum, int sampleCount, std::string& error) {
    try {
        if (width <= 0 || height <= 0 || bins <= 0) return writeExr(path, image, error);
        Imf::Header header(width, height);
        header.channels().insert("R", Imf::Channel(Imf::FLOAT));
        header.channels().insert("G", Imf::Channel(Imf::FLOAT));
        header.channels().insert("B", Imf::Channel(Imf::FLOAT));
        for (int b = 0; b < bins; ++b) {
            const std::string name = "S" + std::to_string(b);
            header.channels().insert(name.c_str(), Imf::Channel(Imf::FLOAT));
        }
        const size_t npix = size_t(width) * size_t(height);
        std::vector<float> R(npix), G(npix), B(npix);
        std::vector<std::vector<float>> S(size_t(bins), std::vector<float>(npix, 0.0f));
        const float invS = 1.0f / float(std::max(1, sampleCount));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t i = size_t(y) * size_t(width) + size_t(x);
                const Vec3 c = (x < image.width() && y < image.height()) ? image.rgb(x, y) : Vec3(0.0f);
                R[i] = c.x;
                G[i] = c.y;
                B[i] = c.z;
                for (int b = 0; b < bins; ++b) {
                    const size_t bi = i * size_t(bins) + size_t(b);
                    if (bi < binAccum.size()) S[size_t(b)][i] = binAccum[bi] * invS;
                }
            }
        }
        Imf::OutputFile file(path.c_str(), header);
        Imf::FrameBuffer fb;
        fb.insert("R", Imf::Slice(Imf::FLOAT, (char*)R.data(), sizeof(float), sizeof(float) * size_t(width)));
        fb.insert("G", Imf::Slice(Imf::FLOAT, (char*)G.data(), sizeof(float), sizeof(float) * size_t(width)));
        fb.insert("B", Imf::Slice(Imf::FLOAT, (char*)B.data(), sizeof(float), sizeof(float) * size_t(width)));
        for (int b = 0; b < bins; ++b) {
            const std::string name = "S" + std::to_string(b);
            fb.insert(name.c_str(),
                      Imf::Slice(Imf::FLOAT, (char*)S[size_t(b)].data(), sizeof(float),
                                 sizeof(float) * size_t(width)));
        }
        file.setFrameBuffer(fb);
        file.writePixels(height);
        return true;
    } catch (const std::exception& e) {
        error = std::string("OpenEXR spectral: ") + e.what();
        return false;
    }
}
#endif

bool loadLdr(const std::string& path, Image& out, std::string& error, bool srgbColor) {
    // Prefer UTF-8 path (Windows paths from UI are UTF-8 via Qt).
    const QString qpath = QString::fromUtf8(path.data(), int(path.size()));
    QImageReader reader(qpath);
    // Qt 6 default allocation limit is 256 MB — 8K RGBA8888 is ~268 MB and would
    // fail with "unsupported or unreadable". Allow large textures (UDIM 4K/8K).
    reader.setAllocationLimit(0);
    QImage qimage = reader.read();
    if (qimage.isNull()) {
        // Fallback: QImage::load (some builds resolve format plugins differently).
        if (!qimage.load(qpath)) {
            QString formats;
            for (const QByteArray& f : QImageReader::supportedImageFormats()) {
                if (!formats.isEmpty()) formats += ',';
                formats += QString::fromLatin1(f);
            }
            error = "unsupported or unreadable image: " + path + " (" +
                    reader.errorString().toStdString() + "); Qt image formats: [" +
                    formats.toStdString() + "]";
            return false;
        }
    }
    qimage = qimage.convertToFormat(QImage::Format_RGBA8888);
    out.resize(qimage.width(), qimage.height());
    for (int y = 0; y < qimage.height(); ++y) {
        const uchar* line = qimage.constScanLine(y);
        for (int x = 0; x < qimage.width(); ++x) {
            const uchar* px = line + size_t(x) * 4;
            const float r = px[0] / 255.0f;
            const float g = px[1] / 255.0f;
            const float b = px[2] / 255.0f;
            out.setRgb(x, y,
                       srgbColor ? Vec3(srgbToLinear(r), srgbToLinear(g), srgbToLinear(b)) : Vec3(r, g, b),
                       px[3] / 255.0f);
        }
    }
    return true;
}

#if SOLSTICE_HAVE_TIFF

// IEEE half (16-bit float) → float32.
float halfBitsToFloat(uint16_t h) {
    const uint32_t sign = (uint32_t(h) >> 15) & 1u;
    uint32_t exp = (uint32_t(h) >> 10) & 0x1fu;
    uint32_t mant = uint32_t(h) & 0x3ffu;
    uint32_t fbits = 0;
    if (exp == 0) {
        if (mant == 0) {
            fbits = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            fbits = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        fbits = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        fbits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &fbits, sizeof(out));
    return out;
}

float decodeTiffChannel(const void* row, int x, int sample, uint16_t samples, uint16_t bits,
                        uint16_t sampleFormat, bool linearize) {
    if (bits == 32 && sampleFormat == SAMPLEFORMAT_IEEEFP) {
        const float* f = static_cast<const float*>(row);
        return f[size_t(x) * samples + sample];
    }
    if (bits == 16 && sampleFormat == SAMPLEFORMAT_IEEEFP) {
        const uint16_t* u = static_cast<const uint16_t*>(row);
        return halfBitsToFloat(u[size_t(x) * samples + sample]);
    }
    if (bits == 16 && sampleFormat == SAMPLEFORMAT_UINT) {
        const uint16_t* u = static_cast<const uint16_t*>(row);
        const float v = u[size_t(x) * samples + sample] / 65535.0f;
        return linearize ? srgbToLinear(v) : v;
    }
    if (bits == 8) {
        const uint8_t* u = static_cast<const uint8_t*>(row);
        const float v = u[size_t(x) * samples + sample] / 255.0f;
        return linearize ? srgbToLinear(v) : v;
    }
    return 0.0f;
}

void writeDecodedPixel(std::vector<float>& rgba, uint32_t w, uint32_t imgX, uint32_t imgY,
                       int colInRow, uint16_t samples, uint16_t bits, uint16_t sampleFormat,
                       uint16_t photometric, bool linearize, const void* rowBase) {
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 1.0f;
    // OIIO/maketx sometimes tags RGB(A) TX as MINISBLACK. Only treat true
    // greyscale (1 sample) or grey+alpha (2 samples) as luminance.
    const bool grey = samples == 1 || (photometric == PHOTOMETRIC_MINISBLACK && samples <= 2);
    if (grey) {
        r = g = b =
            decodeTiffChannel(rowBase, colInRow, 0, samples, bits, sampleFormat, linearize);
        if (samples > 1)
            a = decodeTiffChannel(rowBase, colInRow, 1, samples, bits, sampleFormat, false);
    } else {
        r = decodeTiffChannel(rowBase, colInRow, 0, samples, bits, sampleFormat, linearize);
        g = samples > 1
                ? decodeTiffChannel(rowBase, colInRow, 1, samples, bits, sampleFormat, linearize)
                : r;
        b = samples > 2
                ? decodeTiffChannel(rowBase, colInRow, 2, samples, bits, sampleFormat, linearize)
                : r;
        if (samples > 3)
            a = decodeTiffChannel(rowBase, colInRow, 3, samples, bits, sampleFormat, false);
    }
    const size_t idx = (size_t(imgY) * size_t(w) + size_t(imgX)) * 4;
    rgba[idx + 0] = r;
    rgba[idx + 1] = g;
    rgba[idx + 2] = b;
    rgba[idx + 3] = a;
}

bool readTiffDirectoryLevel(TIFF* tif, std::vector<float>& rgba, int& width, int& height, std::string& error,
                            bool srgbColor) {
    uint32_t w = 0, h = 0;
    uint16_t samples = 1, bits = 8, sampleFormat = SAMPLEFORMAT_UINT, photometric = PHOTOMETRIC_RGB;
    uint16_t planar = PLANARCONFIG_CONTIG;
    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w) || !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h) || w == 0 ||
        h == 0) {
        error = "TIFF directory missing width/height";
        return false;
    }
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &sampleFormat);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PHOTOMETRIC, &photometric);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planar);

    const bool supportedBits =
        (bits == 8) || (bits == 16 && sampleFormat == SAMPLEFORMAT_UINT) ||
        (bits == 16 && sampleFormat == SAMPLEFORMAT_IEEEFP) ||
        (bits == 32 && sampleFormat == SAMPLEFORMAT_IEEEFP);
    if (!supportedBits || samples < 1) {
        error = "unsupported TIFF sample format (need 8/16u/16f/32f, ≥1 sample)";
        return false;
    }
    // maketx / OIIO .tx may carry extra AOVs — keep the first 4 for preview.
    if (samples > 4) samples = 4;
    if (planar != PLANARCONFIG_CONTIG) {
        error = "planar (separate) TIFF/TX not supported in viewer";
        return false;
    }

    // 8/16-bit colour textures are usually sRGB; float/half .tx from maketx is linear.
    const bool isFloatBits =
        (bits == 32 && sampleFormat == SAMPLEFORMAT_IEEEFP) ||
        (bits == 16 && sampleFormat == SAMPLEFORMAT_IEEEFP);
    const bool linearize = srgbColor && !isFloatBits;

    rgba.assign(size_t(w) * size_t(h) * 4, 0.0f);
    width = int(w);
    height = int(h);

    if (TIFFIsTiled(tif)) {
        // maketx / OIIO write tiled TX — scanlines fail on these files.
        uint32_t tileW = 0, tileH = 0;
        TIFFGetFieldDefaulted(tif, TIFFTAG_TILEWIDTH, &tileW);
        TIFFGetFieldDefaulted(tif, TIFFTAG_TILELENGTH, &tileH);
        if (tileW == 0 || tileH == 0) {
            error = "tiled TIFF missing tile size";
            return false;
        }
        const tsize_t tileBytes = TIFFTileSize(tif);
        if (tileBytes <= 0) {
            error = "invalid TIFF tile size";
            return false;
        }
        std::vector<uint8_t> tile(static_cast<size_t>(tileBytes), 0);
        const size_t bytesPerSample = size_t(bits) / 8;
        const size_t bytesPerPixel = bytesPerSample * size_t(samples);
        for (uint32_t y0 = 0; y0 < h; y0 += tileH) {
            for (uint32_t x0 = 0; x0 < w; x0 += tileW) {
                if (TIFFReadTile(tif, tile.data(), x0, y0, 0, 0) < 0) {
                    error = "TIFFReadTile failed";
                    return false;
                }
                const uint32_t th = std::min(tileH, h - y0);
                const uint32_t tw = std::min(tileW, w - x0);
                for (uint32_t ty = 0; ty < th; ++ty) {
                    const uint8_t* row = tile.data() + size_t(ty) * size_t(tileW) * bytesPerPixel;
                    for (uint32_t tx = 0; tx < tw; ++tx) {
                        writeDecodedPixel(rgba, w, x0 + tx, y0 + ty, int(tx), samples, bits,
                                          sampleFormat, photometric, linearize, row);
                    }
                }
            }
        }
        return true;
    }

    const tsize_t stride = TIFFScanlineSize(tif);
    if (stride <= 0) {
        error = "invalid TIFF scanline size";
        return false;
    }
    std::vector<uint8_t> scanline(static_cast<size_t>(stride), 0);
    for (uint32_t y = 0; y < h; ++y) {
        if (TIFFReadScanline(tif, scanline.data(), y, 0) < 0) {
            error = "TIFFReadScanline failed";
            return false;
        }
        for (uint32_t x = 0; x < w; ++x) {
            writeDecodedPixel(rgba, w, x, y, int(x), samples, bits, sampleFormat, photometric,
                              linearize, scanline.data());
        }
    }
    return true;
}

bool loadTiffWithMips(const std::string& path, Image& out, std::string& error, bool srgbColor) {
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (!tif) {
        error = "cannot open TIFF/TX: " + path;
        return false;
    }

    std::vector<std::vector<float>> levels;
    std::vector<std::pair<int, int>> sizes;

    auto appendLevel = [&](TIFF* dir) -> bool {
        std::vector<float> rgba;
        int w = 0, h = 0;
        std::string levelError;
        if (!readTiffDirectoryLevel(dir, rgba, w, h, levelError, srgbColor)) {
            error = levelError;
            return false;
        }
        // Skip duplicate / ascending sizes (not a mip).
        if (!sizes.empty()) {
            if (w > sizes.back().first || h > sizes.back().second) return true;
            if (w == sizes.back().first && h == sizes.back().second) return true;
        }
        levels.push_back(std::move(rgba));
        sizes.emplace_back(w, h);
        return true;
    };

    if (!appendLevel(tif)) {
        TIFFClose(tif);
        return false;
    }

    // Prefer SUBIFD mip chain (maketx / OIIO / Arnold .tx).
    uint16_t subCount = 0;
    uint64_t* subOffsets = nullptr;
    if (TIFFGetField(tif, TIFFTAG_SUBIFD, &subCount, &subOffsets) && subCount > 0 && subOffsets) {
        const uint16_t toRead = std::min<uint16_t>(subCount, 31);
        std::vector<uint64_t> offsets(subOffsets, subOffsets + toRead);
        for (uint16_t i = 0; i < toRead; ++i) {
            if (!TIFFSetSubDirectory(tif, offsets[i])) continue;
            if (!appendLevel(tif)) {
                TIFFClose(tif);
                return false;
            }
        }
    } else {
        // Fallback: successive TIFF directories (some exporters).
        while (TIFFReadDirectory(tif)) {
            if (!appendLevel(tif)) {
                TIFFClose(tif);
                return false;
            }
            if (levels.size() >= 32) break;
        }
    }
    TIFFClose(tif);

    if (levels.empty()) {
        error = "empty TIFF/TX: " + path;
        return false;
    }

    const int width = sizes.front().first;
    const int height = sizes.front().second;
    int mipCount = int(levels.size());

    // Packed layout assumes classic half-size mips from level 0.
    bool regularPyramid = true;
    for (int i = 0; i < mipCount; ++i) {
        const int expectW = std::max(1, width >> i);
        const int expectH = std::max(1, height >> i);
        if (sizes[size_t(i)].first != expectW || sizes[size_t(i)].second != expectH) {
            regularPyramid = false;
            break;
        }
    }
    if (!regularPyramid || mipCount == 1) {
        out.setMipPyramid(std::move(levels.front()), width, height, 1);
        out.generateMipChain();
        logInfo("TX/TIFF " + path + ": " + std::string(mipCount == 1 ? "generated" : "rebuilt") + " " +
                std::to_string(out.mipCount()) + " mip levels (" + std::to_string(width) + "x" +
                std::to_string(height) + ")");
        return true;
    }

    std::vector<float> packed;
    size_t total = 0;
    for (const auto& level : levels) total += level.size();
    packed.reserve(total);
    for (auto& level : levels) {
        packed.insert(packed.end(), level.begin(), level.end());
        level.clear();
        level.shrink_to_fit();
    }
    out.setMipPyramid(std::move(packed), width, height, mipCount);
    logInfo("TX/TIFF " + path + ": loaded " + std::to_string(mipCount) + " mip levels (" +
            std::to_string(width) + "x" + std::to_string(height) + ")");
    return true;
}

#endif  // SOLSTICE_HAVE_TIFF

}  // namespace

bool imageFormatIsHdr(const std::string& path) {
    const std::string ext = toLowerExtension(path);
    return ext == "hdr" || ext == "exr" || ext == "rgbe" || ext == "pic";
}

namespace {

bool loadImageDirect(const std::string& loadPath, Image& out, std::string& error, bool srgbColor) {
    const std::string ext = toLowerExtension(loadPath);
    if (ext == "hdr" || ext == "rgbe" || ext == "pic") return loadHdr(loadPath, out, error);
    if (ext == "exr") {
#if SOLSTICE_HAVE_OPENEXR
        return loadExr(loadPath, out, error);
#else
        error = "this build has no OpenEXR support, use .hdr instead";
        return false;
#endif
    }
    if (ext == "tx" || ext == "tif" || ext == "tiff") {
#if SOLSTICE_HAVE_TIFF
        if (loadTiffWithMips(loadPath, out, error, srgbColor)) return true;
#if SOLSTICE_HAVE_OPENEXR
        // Half TX may be OpenEXR-backed (.tx with EXR payload).
        if (ext == "tx" && loadExr(loadPath, out, error)) return true;
#endif
        return false;
#else
        if (ext == "tx") {
            error = "this build has no libtiff support — cannot load .tx mipmaps";
            return false;
        }
        // Fall back to Qt for plain TIFF when libtiff is unavailable.
        return loadLdr(loadPath, out, error, srgbColor);
#endif
    }
    return loadLdr(loadPath, out, error, srgbColor);
}

void convertImageToAcescg(Image& image, const std::string& cs) {
    if (image.empty() || txSkipColorConvert(cs)) return;
    const int levels = std::max(1, image.mipCount());
    for (int level = 0; level < levels; ++level) {
        const int w = image.mipWidth(level);
        const int h = image.mipHeight(level);
        float* p = image.mipData(level);
        if (!p || w <= 0 || h <= 0) continue;
        const size_t n = size_t(w) * size_t(h);
        for (size_t i = 0; i < n; ++i, p += 4) {
            const Vec3 c = ocioConvertToAcescg(Vec3(p[0], p[1], p[2]), cs);
            p[0] = c.x;
            p[1] = c.y;
            p[2] = c.z;
        }
    }
}

}  // namespace

bool loadImage(const std::string& path, Image& out, std::string& error, bool srgbColor,
               const std::string& inputColorSpace) {
    const std::string cs = txResolveInputColorSpace(inputColorSpace, path, srgbColor);
    const bool skipConvert = txSkipColorConvert(cs);
    // TX (and OCIO) own the colour transform — load files as stored, do not
    // apply a second sRGB EOTF on 8-bit maps that will be converted to ACEScg.
    const std::string resolved = txCacheResolve(path, cs);
    const bool loadedTx = resolved != path;
    if (loadImageDirect(resolved, out, error, /*srgbColor=*/false)) {
        if (!loadedTx && !skipConvert) convertImageToAcescg(out, cs);
        return true;
    }
    if (loadedTx) {
        logWarning("tx_cache: failed to load converted texture '" + resolved + "': " + error +
                   " — falling back to " + path);
        error.clear();
        if (!loadImageDirect(path, out, error, /*srgbColor=*/false)) return false;
        if (!skipConvert) convertImageToAcescg(out, cs);
        return true;
    }
    return false;
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
                                       const std::vector<int>& explicitUdims, bool srgbColor,
                                       const std::string& inputColorSpace) {
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
            if (!loadImage(tilePath.toStdString(), *tile, loadError, srgbColor, inputColorSpace)) {
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
    if (!loadImage(path.toStdString(), *image, error, srgbColor, inputColorSpace)) return nullptr;
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

bool saveImageExr(const std::string& path, const Image& linearImage, std::string& error, int bitDepth) {
#if SOLSTICE_HAVE_OPENEXR
    return writeExr(path, linearImage, error, bitDepth);
#else
    (void)path;
    (void)linearImage;
    (void)bitDepth;
    error = "this build has no OpenEXR support";
    return false;
#endif
}

bool saveImageExrSpectral(const std::string& path, const Image& linearImage, int width, int height, int bins,
                          const std::vector<float>& binAccum, int sampleCount, std::string& error) {
#if SOLSTICE_HAVE_OPENEXR
    return writeExrSpectral(path, linearImage, width, height, bins, binAccum, sampleCount, error);
#else
    (void)width;
    (void)height;
    (void)bins;
    (void)binAccum;
    (void)sampleCount;
    return saveImageExr(path, linearImage, error);
#endif
}

bool saveImageAuto(const std::string& path, const Image& linear, const RenderSettingsData& settings,
                   std::string& error) {
    const std::string ext = toLowerExtension(path);
    if (ext == "hdr") return saveImageHdr(path, linear, error);
    if (ext == "exr") return saveImageExr(path, linear, error, settings.bitDepth);

    // LDR: write working-space values (no viewer process) — Nuke/Arnold beauty save style.
    Image out(linear.width(), linear.height());
    for (int y = 0; y < linear.height(); ++y)
        for (int x = 0; x < linear.width(); ++x)
            out.setRgb(x, y, quantizeRgb(linear.rgb(x, y), settings.bitDepth));
    return saveImagePng(path, out, error);
}

}  // namespace sol
