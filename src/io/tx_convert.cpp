#include "io/tx_convert.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QProcess>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QString>
#include <QTemporaryDir>
#include <QTextStream>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/expr_eval.h"
#include "core/log.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OPENEXR
#  include <ImfChannelList.h>
#  include <ImfHeader.h>
#  include <ImfInputFile.h>
#endif

#if SOLSTICE_HAVE_TIFF
#  include <tiffio.h>
#endif

namespace sol {
namespace {

QString toolFileName(const QString& name) {
#ifdef Q_OS_WIN
    if (!name.endsWith(QLatin1String(".exe"), Qt::CaseInsensitive))
        return name + QStringLiteral(".exe");
#endif
    return name;
}

std::string findTool(const QString& name) {
    const QString fileName = toolFileName(name);
    if (QCoreApplication::instance()) {
        const QString appDir = QCoreApplication::applicationDirPath();
        const QStringList candidates = {
            appDir + QLatin1Char('/') + fileName,
            appDir + QStringLiteral("/tools/") + fileName,
        };
        for (const QString& candidate : candidates) {
            if (QFileInfo::exists(candidate)) return QFileInfo(candidate).absoluteFilePath().toStdString();
        }
    }
    const QString path = QStandardPaths::findExecutable(name);
    return path.isEmpty() ? std::string() : path.toStdString();
}

std::string g_maketxPath;
std::string g_oiiotoolPath;
std::once_flag g_toolOnce;

void initTools() {
    std::call_once(g_toolOnce, []() {
        g_maketxPath = findTool(QStringLiteral("maketx"));
        if (g_maketxPath.empty()) g_maketxPath = findTool(QStringLiteral("maketx-2.5"));
        if (g_maketxPath.empty()) g_maketxPath = findTool(QStringLiteral("maketx-2.4"));
        g_oiiotoolPath = findTool(QStringLiteral("oiiotool"));
        if (g_oiiotoolPath.empty()) g_oiiotoolPath = findTool(QStringLiteral("oiiotool-2.5"));
        if (g_oiiotoolPath.empty()) g_oiiotoolPath = findTool(QStringLiteral("oiiotool-2.4"));
        if (!g_maketxPath.empty()) logInfo("tx_convert: maketx at " + g_maketxPath);
        if (!g_oiiotoolPath.empty()) logInfo("tx_convert: oiiotool at " + g_oiiotoolPath);
        if (g_maketxPath.empty() && g_oiiotoolPath.empty())
            logWarning("tx_convert: neither maketx nor oiiotool found next to the app or on PATH");
        else if (g_oiiotoolPath.empty())
            logWarning("tx_convert: oiiotool not found — PNG/JPG and TX reformat unavailable");
    });
}

QString replaceUdimToken(QString pattern, int udim) {
    const QString number = QString::number(udim);
    pattern.replace(QStringLiteral("<UDIM>"), number, Qt::CaseInsensitive);
    pattern.replace(QStringLiteral("%(UDIM)d"), number);
    return pattern;
}

bool looksLikeUdimPattern(const QString& path) {
    return path.contains(QStringLiteral("<UDIM>"), Qt::CaseInsensitive) ||
           path.contains(QStringLiteral("%(UDIM)d"));
}

std::string readSidecarSource(const QString& outPath) {
    QFile f(outPath + QStringLiteral(".txsrc"));
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return {};
    return QString::fromUtf8(f.readAll()).trimmed().toStdString();
}

void writeSidecarSource(const QString& outPath, const std::string& sourcePath) {
    QFile f(outPath + QStringLiteral(".txsrc"));
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) return;
    f.write(sourcePath.c_str());
    f.write("\n");
}

bool runProcess(const std::string& program, const QStringList& args, std::string& error,
                int timeoutMs = 600000) {
    QProcess proc;
    proc.setProgram(QString::fromStdString(program));
    proc.setArguments(args);
    proc.setWorkingDirectory(QFileInfo(QString::fromStdString(program)).absolutePath());
    proc.start();
    if (!proc.waitForStarted(5000)) {
        error = "failed to start " + program;
        return false;
    }
    if (!proc.waitForFinished(timeoutMs)) {
        proc.kill();
        error = "timed out: " + program;
        return false;
    }
    if (proc.exitCode() != 0) {
        const std::string err = proc.readAllStandardError().toStdString();
        error = (err.empty() ? std::string("(no stderr)") : err);
        return false;
    }
    return true;
}

bool queryImageSize(const QString& path, int& outW, int& outH, std::string& error) {
    outW = outH = 0;
    if (g_oiiotoolPath.empty()) {
        error = "oiiotool required to query image size";
        return false;
    }
    QProcess proc;
    proc.setProgram(QString::fromStdString(g_oiiotoolPath));
    proc.setArguments({path, QStringLiteral("--printinfo")});
    proc.setWorkingDirectory(QFileInfo(QString::fromStdString(g_oiiotoolPath)).absolutePath());
    proc.start();
    if (!proc.waitForFinished(60000)) {
        proc.kill();
        error = "oiiotool --printinfo timed out";
        return false;
    }
    const QString out = QString::fromUtf8(proc.readAllStandardOutput() + proc.readAllStandardError());
    const QRegularExpression re(QStringLiteral(R"((\d+)\s*[xX×]\s*(\d+))"));
    const auto m = re.match(out);
    if (!m.hasMatch()) {
        error = "could not parse image size from oiiotool";
        return false;
    }
    outW = m.captured(1).toInt();
    outH = m.captured(2).toInt();
    return outW > 0 && outH > 0;
}

QString oiioDepthArg(TxPixelType type) {
    switch (type) {
        case TxPixelType::UInt8: return QStringLiteral("uint8");
        case TxPixelType::UInt16: return QStringLiteral("uint16");
        case TxPixelType::Half: return QStringLiteral("half");
        case TxPixelType::Float:
        case TxPixelType::Original:
        default: return QStringLiteral("float");
    }
}

bool isSingleChannelMode(TxChannelMode mode) {
    return mode == TxChannelMode::R || mode == TxChannelMode::G || mode == TxChannelMode::B ||
           mode == TxChannelMode::A;
}

QString oiioChannelArg(TxChannelMode mode) {
    switch (mode) {
        case TxChannelMode::RGB: return QStringLiteral("R,G,B");
        case TxChannelMode::R: return QStringLiteral("R");
        case TxChannelMode::G: return QStringLiteral("G");
        case TxChannelMode::B: return QStringLiteral("B");
        case TxChannelMode::A: return QStringLiteral("A");
        case TxChannelMode::RGBA:
        default: return QStringLiteral("R,G,B,A");
    }
}

bool needsOiiotoolPreprocess(const TxConvertRequest& req) {
    if (req.format != TxOutputFormat::Tx) return true;
    if (req.longSide > 0) return true;
    if (req.channels != TxChannelMode::RGBA) return true;
    // Pixel type is applied by maketx -d on the final .tx.
    return false;
}

bool oiiotoolRewrite(const QString& src, const QString& dst, const TxConvertRequest& req,
                     bool applyColorConvert, std::string& error) {
    if (g_oiiotoolPath.empty()) {
        error = "oiiotool not found (required for PNG/JPG or TX reformat)";
        return false;
    }

    QStringList args;
    args << src;
    // Boolean toggle — do NOT pass "off" as an argument (oiiotool treats it as a filename).
    args << QStringLiteral("--noautocc");

    // Always select channels explicitly so outputs are truly 1/3/4-channel
    // (R alone must not become RRR RGB).
    args << QStringLiteral("--ch") << oiioChannelArg(req.channels);
    if (isSingleChannelMode(req.channels) && req.format == TxOutputFormat::Jpg) {
        // JPEG greyscale expects a single Y-like channel, not named R.
        args << QStringLiteral("--chnames") << QStringLiteral("Y");
    }

    if (req.longSide > 0) {
        int w = 0, h = 0;
        if (!queryImageSize(src, w, h, error)) return false;
        const int longSide = std::max(w, h);
        if (longSide > req.longSide) {
            const double scale = double(req.longSide) / double(longSide);
            const int tw = std::max(1, int(std::lround(w * scale)));
            const int th = std::max(1, int(std::lround(h * scale)));
            args << QStringLiteral("--resize") << QStringLiteral("%1x%2").arg(tw).arg(th);
        }
    }

    if (applyColorConvert && !txSkipColorConvert(req.inputColorSpace)) {
        if (!req.ocioConfigPath.empty())
            args << QStringLiteral("--colorconfig") << QString::fromStdString(req.ocioConfigPath);
        args << QStringLiteral("--colorconvert") << QString::fromStdString(req.inputColorSpace)
             << QStringLiteral("ACES - ACEScg");
    }

    // TX depth is applied later by maketx -d; keep preprocess temps full precision.
    if (req.format != TxOutputFormat::Tx) {
        const TxPixelType pt =
            (req.format == TxOutputFormat::Jpg) ? TxPixelType::UInt8 : req.pixelType;
        args << QStringLiteral("-d") << oiioDepthArg(pt);
    }
    args << QStringLiteral("-o") << dst;

    logInfo("tx_convert: oiiotool " + src.toStdString() + " → " + dst.toStdString());
    std::string err;
    if (!runProcess(g_oiiotoolPath, args, err)) {
        error = "oiiotool failed: " + err;
        return false;
    }
    return true;
}

bool maketxWrite(const QString& src, const QString& dst, const TxConvertRequest& req,
                 std::string& error) {
    const TxPixelType pt = req.pixelType == TxPixelType::Original ? TxPixelType::Float : req.pixelType;
    const QString depthArg = oiioDepthArg(pt);
    // half is not reliably stored in TIFF-backed .tx — use OpenEXR container.
    const bool useExrContainer = (pt == TxPixelType::Half);

    if (g_maketxPath.empty()) {
        // Fall back to oiiotool mipmaps when maketx is missing.
        if (g_oiiotoolPath.empty()) {
            error = "no maketx or oiiotool found";
            return false;
        }
        QStringList args;
        args << src << QStringLiteral("--noautocc");
        if (!txSkipColorConvert(req.inputColorSpace)) {
            if (!req.ocioConfigPath.empty())
                args << QStringLiteral("--colorconfig") << QString::fromStdString(req.ocioConfigPath);
            args << QStringLiteral("--colorconvert") << QString::fromStdString(req.inputColorSpace)
                 << QStringLiteral("ACES - ACEScg");
        }
        args << QStringLiteral("-d") << depthArg;
        args << QStringLiteral("--mipmaps");
        if (useExrContainer) {
            args << QStringLiteral("-o:format=openexr") << dst;
        } else {
            args << QStringLiteral("-o") << dst;
        }
        std::string err;
        if (!runProcess(g_oiiotoolPath, args, err)) {
            error = "oiiotool maketx-fallback failed: " + err;
            return false;
        }
        return true;
    }

    QStringList args;
    if (req.updateOnly) args << QStringLiteral("-u");
    args << QStringLiteral("--oiio");
    if (useExrContainer) args << QStringLiteral("--format") << QStringLiteral("openexr");
    args << QStringLiteral("-d") << depthArg;
    if (!txSkipColorConvert(req.inputColorSpace)) {
        if (!req.ocioConfigPath.empty())
            args << QStringLiteral("--colorconfig") << QString::fromStdString(req.ocioConfigPath);
        args << QStringLiteral("--colorconvert") << QString::fromStdString(req.inputColorSpace)
             << QStringLiteral("ACES - ACEScg");
        args << QStringLiteral("--unpremult");
    }
    args << src << QStringLiteral("-o") << dst;

    logInfo("tx_convert: maketx " + src.toStdString() + " → " + dst.toStdString() + " -d " +
            depthArg.toStdString() + (useExrContainer ? " (openexr)" : ""));
    std::string err;
    if (!runProcess(g_maketxPath, args, err)) {
        error = "maketx failed: " + err;
        return false;
    }
    return true;
}

}  // namespace

std::vector<std::string> txCuratedColorSpaces() {
    return {
        "ACES - ACEScg",
        "Utility - sRGB - Texture",
        "Utility - Linear - sRGB",
        "Utility - Raw",
        "Utility - Rec.709 - Texture",
        "Output - sRGB",
        "role_data",
        "role_color_picking",
    };
}

std::vector<std::string> txColorSpacesFromConfig(const std::string& ocioConfigPath) {
    std::vector<std::string> spaces;
    if (ocioConfigPath.empty()) return txCuratedColorSpaces();
    QFile file(QString::fromStdString(ocioConfigPath));
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return txCuratedColorSpaces();

    QTextStream in(&file);
    bool inColorSpaces = false;
    while (!in.atEnd()) {
        const QString line = in.readLine();
        const QString trimmed = line.trimmed();
        if (trimmed.startsWith(QLatin1String("colorspaces:")) ||
            trimmed.startsWith(QLatin1String("ocio_profile_version"))) {
            if (trimmed.startsWith(QLatin1String("colorspaces:"))) inColorSpaces = true;
            continue;
        }
        if (trimmed.startsWith(QLatin1String("displays:")) ||
            trimmed.startsWith(QLatin1String("roles:")) ||
            trimmed.startsWith(QLatin1String("looks:"))) {
            inColorSpaces = false;
        }
        if (!inColorSpaces && !trimmed.startsWith(QLatin1String("name:"))) continue;
        if (trimmed.startsWith(QLatin1String("name:"))) {
            QString name = trimmed.mid(5).trimmed();
            if (name.startsWith('"') && name.endsWith('"') && name.size() >= 2)
                name = name.mid(1, name.size() - 2);
            if (!name.isEmpty()) {
                const std::string n = name.toStdString();
                if (std::find(spaces.begin(), spaces.end(), n) == spaces.end()) spaces.push_back(n);
            }
        }
    }
    if (spaces.empty()) return txCuratedColorSpaces();
    std::sort(spaces.begin(), spaces.end());
    spaces.erase(std::unique(spaces.begin(), spaces.end()), spaces.end());
    return spaces;
}

std::string txResolveOcioConfig(bool useEnv, const std::string& settingsPath) {
    if (useEnv) {
        if (const char* env = std::getenv("OCIO")) {
            if (env[0] != '\0') {
                const QFileInfo info(QString::fromUtf8(env));
                if (info.exists() && info.isFile()) return info.absoluteFilePath().toStdString();
            }
        }
    }
    if (!settingsPath.empty()) {
        const QFileInfo info(QString::fromStdString(settingsPath));
        if (info.exists() && info.isFile()) return info.absoluteFilePath().toStdString();
        const QFileInfo rel(QDir::current().absoluteFilePath(QString::fromStdString(settingsPath)));
        if (rel.exists() && rel.isFile()) return rel.absoluteFilePath().toStdString();
    }
    return {};
}

bool txSkipColorConvert(const std::string& inputColorSpace) {
    if (inputColorSpace.empty()) return true;
    QString s = QString::fromStdString(inputColorSpace).trimmed();
    if (s.isEmpty()) return true;
    const QString lower = s.toLower();
    if (lower == QLatin1String("aces - acescg") || lower == QLatin1String("acescg") ||
        lower == QLatin1String("utility - raw") || lower == QLatin1String("raw") ||
        lower == QLatin1String("role_data") || lower == QLatin1String("data"))
        return true;
    return false;
}

std::string txPixelTypeOiiArg(TxPixelType type) {
    switch (type) {
        case TxPixelType::UInt8: return "uint8";
        case TxPixelType::UInt16: return "uint16";
        case TxPixelType::Half: return "half";
        case TxPixelType::Float:
        case TxPixelType::Original:
        default: return "float";
    }
}

std::string txPixelTypeDisplayLabel(TxPixelType type) {
    switch (type) {
        case TxPixelType::UInt8: return "8-bit uint";
        case TxPixelType::UInt16: return "16-bit uint";
        case TxPixelType::Half: return "16-bit half";
        case TxPixelType::Float: return "32-bit float";
        case TxPixelType::Original:
        default: return "original";
    }
}

TxPixelType txProbePixelType(const std::string& path) {
    const QString qpath = QString::fromStdString(path);
    const QString ext = QFileInfo(qpath).suffix().toLower();

#if SOLSTICE_HAVE_TIFF
    if (ext == QLatin1String("tx") || ext == QLatin1String("tif") || ext == QLatin1String("tiff")) {
        TIFF* tif = TIFFOpen(qpath.toLocal8Bit().constData(), "r");
        if (tif) {
            uint16_t bits = 0;
            uint16_t sampleFormat = SAMPLEFORMAT_UINT;
            TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits);
            TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &sampleFormat);
            TIFFClose(tif);
            if (bits <= 8) return TxPixelType::UInt8;
            if (bits <= 16) {
                if (sampleFormat == SAMPLEFORMAT_IEEEFP) return TxPixelType::Half;
                return TxPixelType::UInt16;
            }
            return TxPixelType::Float;
        }
#if SOLSTICE_HAVE_OPENEXR
        // TX may be OpenEXR-backed (half).
        try {
            Imf::InputFile file(qpath.toLocal8Bit().constData());
            const Imf::ChannelList& list = file.header().channels();
            Imf::PixelType deepest = Imf::HALF;
            bool any = false;
            for (Imf::ChannelList::ConstIterator it = list.begin(); it != list.end(); ++it) {
                any = true;
                if (it.channel().type == Imf::FLOAT) deepest = Imf::FLOAT;
                else if (it.channel().type == Imf::UINT && deepest != Imf::FLOAT) deepest = Imf::UINT;
            }
            if (any) {
                if (deepest == Imf::FLOAT) return TxPixelType::Float;
                if (deepest == Imf::UINT) return TxPixelType::UInt16;
                return TxPixelType::Half;
            }
        } catch (...) {
        }
#endif
    }
#endif

#if SOLSTICE_HAVE_OPENEXR
    if (ext == QLatin1String("exr")) {
        try {
            Imf::InputFile file(qpath.toLocal8Bit().constData());
            const Imf::ChannelList& list = file.header().channels();
            Imf::PixelType deepest = Imf::HALF;
            bool any = false;
            for (Imf::ChannelList::ConstIterator it = list.begin(); it != list.end(); ++it) {
                any = true;
                if (it.channel().type == Imf::FLOAT) deepest = Imf::FLOAT;
                else if (it.channel().type == Imf::UINT && deepest != Imf::FLOAT) deepest = Imf::UINT;
            }
            if (any) {
                if (deepest == Imf::FLOAT) return TxPixelType::Float;
                if (deepest == Imf::UINT) return TxPixelType::UInt16;
                return TxPixelType::Half;
            }
        } catch (...) {
        }
        return TxPixelType::Half;
    }
#endif

    if (ext == QLatin1String("hdr") || ext == QLatin1String("rgbe") || ext == QLatin1String("pic"))
        return TxPixelType::Float;

    QImageReader reader(qpath);
    reader.setAllocationLimit(0);
    const QImage::Format fmt = reader.imageFormat();
    if (fmt == QImage::Format_Grayscale16 || fmt == QImage::Format_RGBX64 ||
        fmt == QImage::Format_RGBA64 || fmt == QImage::Format_RGBA64_Premultiplied) {
        return TxPixelType::UInt16;
    }
    QImage img = reader.read();
    if (img.isNull()) img.load(qpath);
    if (!img.isNull()) {
        if (img.format() == QImage::Format_Grayscale16 || img.format() == QImage::Format_RGBX64 ||
            img.format() == QImage::Format_RGBA64 ||
            img.format() == QImage::Format_RGBA64_Premultiplied) {
            return TxPixelType::UInt16;
        }
        if (img.depth() > 32) return TxPixelType::UInt16;
        return TxPixelType::UInt8;
    }
    return TxPixelType::Float;
}

TxPixelType txResolvePixelType(TxPixelType selected, const std::string& sourcePath,
                               TxOutputFormat format) {
    TxPixelType t = selected;
    if (t == TxPixelType::Original) t = txProbePixelType(sourcePath);

    if (format == TxOutputFormat::Jpg) return TxPixelType::UInt8;
    if (format == TxOutputFormat::Png) {
        if (t == TxPixelType::Half || t == TxPixelType::Float) return TxPixelType::UInt16;
        if (t == TxPixelType::UInt16) return TxPixelType::UInt16;
        return TxPixelType::UInt8;
    }
    // EXR / TIFF / TX: all four concrete types allowed.
    if (t == TxPixelType::Original) return TxPixelType::Float;
    return t;
}

int txExtractFrameNumber(const std::string& path) {
    const QString base = QFileInfo(QString::fromStdString(path)).completeBaseName();
    const QRegularExpression re(QStringLiteral(R"((?:^|[._-])(\d+)$)"));
    const auto m = re.match(base);
    if (!m.hasMatch()) return 0;
    return m.captured(1).toInt();
}

TxOutputFormat txFormatFromPath(const std::string& pathOrExt) {
    QString ext = QString::fromStdString(pathOrExt).trimmed().toLower();
    if (ext.contains(QLatin1Char('/')) || ext.contains(QLatin1Char('\\')) ||
        ext.contains(QLatin1Char('.'))) {
        ext = QFileInfo(QString::fromStdString(pathOrExt)).suffix().toLower();
    }
    // Patterns like tile_<UDIM>.exr → suffix still works via QFileInfo.
    if (ext.isEmpty()) {
        const QString p = QString::fromStdString(pathOrExt).toLower();
        if (p.contains(QLatin1String(".exr"))) ext = QStringLiteral("exr");
        else if (p.contains(QLatin1String(".png"))) ext = QStringLiteral("png");
        else if (p.contains(QLatin1String(".jpg")) || p.contains(QLatin1String(".jpeg")))
            ext = QStringLiteral("jpg");
        else if (p.contains(QLatin1String(".tx"))) ext = QStringLiteral("tx");
        else if (p.contains(QLatin1String(".tif"))) ext = QStringLiteral("tif");
        else if (p.contains(QLatin1String(".hdr"))) ext = QStringLiteral("hdr");
    }
    if (ext == QLatin1String("tx")) return TxOutputFormat::Tx;
    if (ext == QLatin1String("png")) return TxOutputFormat::Png;
    if (ext == QLatin1String("jpg") || ext == QLatin1String("jpeg")) return TxOutputFormat::Jpg;
    if (ext == QLatin1String("tif") || ext == QLatin1String("tiff")) return TxOutputFormat::Tiff;
    if (ext == QLatin1String("exr") || ext == QLatin1String("hdr") || ext == QLatin1String("rgbe") ||
        ext == QLatin1String("pic"))
        return TxOutputFormat::Exr;
    // Unknown / empty source while Original is selected → treat as EXR for UI defaults.
    return TxOutputFormat::Exr;
}

TxOutputFormat txResolveFormat(TxOutputFormat selected, const std::string& sourcePath) {
    if (selected == TxOutputFormat::Original) return txFormatFromPath(sourcePath);
    return selected;
}

std::string txOutputExtension(TxOutputFormat format, const std::string& sourcePath) {
    if (format == TxOutputFormat::Original) {
        QString ext = QFileInfo(QString::fromStdString(sourcePath)).suffix().toLower();
        if (ext == QLatin1String("jpeg")) ext = QStringLiteral("jpg");
        if (ext.isEmpty()) {
            // UDIM / $F pattern: peel trailing .ext before tokens if needed.
            const QString p = QString::fromStdString(sourcePath);
            const QRegularExpression re(QStringLiteral(R"(\.([A-Za-z0-9]+)$)"));
            const auto m = re.match(p);
            if (m.hasMatch()) ext = m.captured(1).toLower();
        }
        if (ext.isEmpty()) return "exr";
        return ext.toStdString();
    }
    switch (format) {
        case TxOutputFormat::Png: return "png";
        case TxOutputFormat::Jpg: return "jpg";
        case TxOutputFormat::Exr: return "exr";
        case TxOutputFormat::Tiff: return "tif";
        case TxOutputFormat::Tx:
        default: return "tx";
    }
}

std::vector<std::string> txExpandUdimSources(const std::string& sourcePathOrPattern, int frameStart,
                                             int frameEnd) {
    const QString path = QString::fromStdString(sourcePathOrPattern);
    std::vector<std::string> out;
    if (!looksLikeUdimPattern(path)) {
        out.push_back(sourcePathOrPattern);
        return out;
    }
    const bool filter = frameStart > 0 && frameEnd > 0;
    int a = frameStart, b = frameEnd;
    if (filter && b < a) std::swap(a, b);
    for (int udim = 1001; udim <= 1100; ++udim) {
        if (filter && (udim < a || udim > b)) continue;
        const QString concrete = replaceUdimToken(path, udim);
        if (QFileInfo::exists(concrete)) out.push_back(concrete.toStdString());
    }
    return out;
}

std::vector<std::string> txExpandFrameSources(const std::string& sourcePathOrPattern, int frameStart,
                                              int frameEnd) {
    const QString path = QString::fromStdString(sourcePathOrPattern);
    std::vector<std::string> out;
    if (!path.contains(QStringLiteral("$F"))) return out;
    if (frameEnd < frameStart) std::swap(frameStart, frameEnd);
    frameStart = std::max(1, frameStart);
    frameEnd = std::max(frameStart, frameEnd);
    for (int f = frameStart; f <= frameEnd; ++f) {
        const QString concrete = resolveFramePathExisting(path, f);
        if (QFileInfo::exists(concrete)) out.push_back(concrete.toStdString());
    }
    return out;
}

std::string txAllocateOutputPath(const std::string& sourcePath, const std::string& outputDir,
                                 TxOutputFormat format) {
    const QFileInfo src(QString::fromStdString(sourcePath));
    const QString baseName = src.completeBaseName();
    const TxOutputFormat resolved = txResolveFormat(format, sourcePath);
    const QString ext = QString::fromStdString(
        format == TxOutputFormat::Original ? txOutputExtension(TxOutputFormat::Original, sourcePath)
                                           : txOutputExtension(resolved));
    QString dir = QString::fromStdString(outputDir);
    if (dir.isEmpty()) dir = QStringLiteral("tx_cache");
    const QFileInfo dirInfo(dir);
    const QString absDir = dirInfo.isAbsolute() ? dir : QDir::current().absoluteFilePath(dir);
    QDir().mkpath(absDir);

    auto candidate = [&](int copyIndex) -> QString {
        QString stem = baseName;
        if (copyIndex > 0) stem += QStringLiteral("_copy_%1").arg(copyIndex);
        return QDir(absDir).absoluteFilePath(stem + QLatin1Char('.') + ext);
    };

    for (int copy = 0; copy < 10000; ++copy) {
        const QString outPath = candidate(copy);
        const QFileInfo outInfo(outPath);
        if (!outInfo.exists()) return outPath.toStdString();
        const std::string recorded = readSidecarSource(outPath);
        if (recorded.empty() || recorded == sourcePath) return outPath.toStdString();
    }
    return candidate(9999).toStdString();
}

TxConvertResult txConvertOne(const TxConvertRequest& reqIn) {
    TxConvertResult result;
    TxConvertRequest req = reqIn;
    // Original is resolved to a concrete format before writing.
    if (req.format == TxOutputFormat::Original)
        req.format = txResolveFormat(TxOutputFormat::Original, req.sourcePath);
    req.pixelType = txResolvePixelType(req.pixelType, req.sourcePath, req.format);
    result.outputPath = req.outputPath;
    initTools();
    if (g_maketxPath.empty() && g_oiiotoolPath.empty()) {
        result.error = "no maketx or oiiotool found (expected next to the app, or on PATH)";
        return result;
    }
    if (req.sourcePath.empty() || req.outputPath.empty()) {
        result.error = "empty source or output path";
        return result;
    }
    const QFileInfo srcInfo(QString::fromStdString(req.sourcePath));
    if (!srcInfo.exists()) {
        result.error = "source not found: " + req.sourcePath;
        return result;
    }

    // Already matching TX with no reformat — return as-is only when writing onto itself.
    const QFileInfo dstInfoEarly(QString::fromStdString(req.outputPath));
    if (req.format == TxOutputFormat::Tx && srcInfo.suffix().toLower() == QLatin1String("tx") &&
        !needsOiiotoolPreprocess(req) &&
        dstInfoEarly.absoluteFilePath() == srcInfo.absoluteFilePath() &&
        txProbePixelType(req.sourcePath) == req.pixelType) {
        result.ok = true;
        result.outputPath = req.sourcePath;
        return result;
    }

    const QFileInfo dstInfo(QString::fromStdString(req.outputPath));
    QDir().mkpath(dstInfo.absolutePath());

    if (req.updateOnly && dstInfo.exists()) {
        const std::string recorded = readSidecarSource(QString::fromStdString(req.outputPath));
        if ((recorded.empty() || recorded == req.sourcePath) &&
            dstInfo.lastModified() >= srcInfo.lastModified()) {
            result.ok = true;
            return result;
        }
    }

    const QString srcQ = QString::fromStdString(req.sourcePath);
    const QString dstQ = QString::fromStdString(req.outputPath);
    std::string error;

    if (req.format == TxOutputFormat::Png || req.format == TxOutputFormat::Jpg ||
        req.format == TxOutputFormat::Exr || req.format == TxOutputFormat::Tiff) {
        // No colour-space convert — resize / bit / channels only.
        if (!oiiotoolRewrite(srcQ, dstQ, req, /*applyColorConvert=*/false, error)) {
            result.error = error;
            return result;
        }
        writeSidecarSource(dstQ, req.sourcePath);
        result.ok = true;
        return result;
    }

    // TX path.
    if (needsOiiotoolPreprocess(req)) {
        // Force cleanup even if the process is killed — keep temps out of the output folder.
        QTemporaryDir tmp;
        tmp.setAutoRemove(true);
        if (!tmp.isValid()) {
            result.error = "could not create temp dir for TX preprocess";
            return result;
        }
        const QString tempPath = tmp.filePath(QStringLiteral("pre.exr"));
        TxConvertRequest pre = req;
        if (!oiiotoolRewrite(srcQ, tempPath, pre, /*applyColorConvert=*/false, error)) {
            result.error = error;
            return result;
        }
        if (!maketxWrite(tempPath, dstQ, req, error)) {
            result.error = error;
            return result;
        }
    } else {
        if (!maketxWrite(srcQ, dstQ, req, error)) {
            result.error = error;
            return result;
        }
    }

    writeSidecarSource(dstQ, req.sourcePath);
    result.ok = true;
    return result;
}

void removeConvertSidecars(const std::vector<TxConvertResult>& results) {
    for (const TxConvertResult& r : results) {
        if (r.outputPath.empty()) continue;
        const QString side = QString::fromStdString(r.outputPath) + QStringLiteral(".txsrc");
        QFile::remove(side);
    }
}

bool txConvertPattern(const std::string& sourcePathOrPattern, const std::string& outputDir,
                      const TxConvertOptions& options, std::vector<TxConvertResult>& results,
                      std::string& error, const TxConvertProgressFn& progress) {
    results.clear();
    const QString pathQ = QString::fromStdString(sourcePathOrPattern);
    std::vector<std::string> sources;
    if (pathQ.contains(QLatin1Char('$'))) {
        sources = txExpandFrameSources(sourcePathOrPattern, options.frameStart, options.frameEnd);
    } else if (looksLikeUdimPattern(pathQ)) {
        sources =
            txExpandUdimSources(sourcePathOrPattern, options.frameStart, options.frameEnd);
    } else {
        sources = txExpandUdimSources(sourcePathOrPattern);
    }
    if (sources.empty()) {
        error = "no source files found for: " + sourcePathOrPattern;
        return false;
    }

    const int total = int(sources.size());
    results.assign(size_t(total), TxConvertResult{});
    if (progress) progress(0, total, {});

    // Estimate peak RAM per concurrent job (preprocess + maketx scratch). Conservative.
    std::int64_t perJob = 512LL * 1024 * 1024;
    if (options.longSide > 0) {
        const std::int64_t edge = std::max<std::int64_t>(1, options.longSide);
        perJob = std::max(perJob, edge * edge * 16);  // RGBA float-ish working set
    } else {
        perJob = std::max(perJob, std::int64_t(1536) * 1024 * 1024);  // ~1.5 GiB for large EXRs
    }
    const std::int64_t budget =
        options.memoryBudgetBytes > 0 ? options.memoryBudgetBytes : (32LL * 1024 * 1024 * 1024);
    int maxParallel = options.maxParallelJobs;
    if (maxParallel <= 0) maxParallel = int(std::max<std::int64_t>(1, budget / perJob));
    const int hw = std::max(1, int(std::thread::hardware_concurrency()));
    maxParallel = std::clamp(maxParallel, 1, std::min(hw, total));

    std::atomic<int> nextIdx{0};
    std::atomic<int> completed{0};
    std::mutex progressMu;
    std::mutex errorMu;
    std::atomic<bool> allOk{true};

    auto worker = [&]() {
        while (true) {
            const int i = nextIdx.fetch_add(1);
            if (i >= total) return;

            const std::string& src = sources[size_t(i)];
            TxConvertRequest req;
            req.sourcePath = src;
            req.outputPath = txAllocateOutputPath(src, outputDir, options.format);
            req.inputColorSpace = options.inputColorSpace;
            req.ocioConfigPath = options.ocioConfigPath;
            req.updateOnly = options.updateOnly;
            req.format = options.format;
            req.pixelType = options.pixelType;
            req.longSide = options.longSide;
            req.channels = options.channels;

            TxConvertResult r = txConvertOne(req);
            results[size_t(i)] = r;
            if (!r.ok) {
                allOk.store(false);
                std::lock_guard<std::mutex> lock(errorMu);
                if (error.empty()) error = r.error;
                logWarning("tx_convert: " + r.error);
            }
            const int done = completed.fetch_add(1) + 1;
            if (progress) {
                std::lock_guard<std::mutex> lock(progressMu);
                progress(done, total, src);
            }
        }
    };

    if (maxParallel <= 1) {
        worker();
    } else {
        std::vector<std::thread> threads;
        threads.reserve(size_t(maxParallel));
        for (int t = 0; t < maxParallel; ++t) threads.emplace_back(worker);
        for (auto& th : threads) th.join();
    }
    // Sidecars are only used during allocate/update in this run — don't leave .txsrc in the
    // output folder.
    removeConvertSidecars(results);
    return allOk.load();
}

bool txConvertPattern(const std::string& sourcePathOrPattern, const std::string& outputDir,
                      const std::string& inputColorSpace, const std::string& ocioConfigPath,
                      std::vector<TxConvertResult>& results, std::string& error, int frameStart,
                      int frameEnd) {
    TxConvertOptions opt;
    opt.inputColorSpace = inputColorSpace;
    opt.ocioConfigPath = ocioConfigPath;
    opt.frameStart = frameStart;
    opt.frameEnd = frameEnd;
    return txConvertPattern(sourcePathOrPattern, outputDir, opt, results, error);
}

}  // namespace sol
