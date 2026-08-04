#include "io/tx_convert.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QString>
#include <QTemporaryDir>
#include <QTextStream>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <string>

#include "core/expr_eval.h"
#include "core/log.h"

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
        if (!g_maketxPath.empty()) logInfo("tx_convert: maketx at " + g_maketxPath);
        if (!g_oiiotoolPath.empty()) logInfo("tx_convert: oiiotool at " + g_oiiotoolPath);
        if (g_maketxPath.empty() && g_oiiotoolPath.empty())
            logWarning("tx_convert: neither maketx nor oiiotool found next to the app or on PATH");
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

QString oiioDepthArg(int bitDepth, TxOutputFormat format) {
    if (format == TxOutputFormat::Jpg) return QStringLiteral("uint8");
    if (bitDepth <= 0) {
        // Leave source depth — omit -d at call site when possible; float is safe intermediate.
        return QStringLiteral("float");
    }
    if (bitDepth <= 8) return QStringLiteral("uint8");
    if (bitDepth <= 16) {
        return (format == TxOutputFormat::Png) ? QStringLiteral("uint16") : QStringLiteral("half");
    }
    return QStringLiteral("float");
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
    // Explicit bit depth for TX (0 = leave source / maketx default).
    if (req.bitDepth == 8 || req.bitDepth == 16 || req.bitDepth == 32) return true;
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
    args << QStringLiteral("--autocc") << QStringLiteral("off");

    if (req.channels != TxChannelMode::RGBA) {
        args << QStringLiteral("--ch") << oiioChannelArg(req.channels);
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

    if (req.bitDepth > 0 || req.format == TxOutputFormat::Jpg) {
        args << QStringLiteral("-d") << oiioDepthArg(req.bitDepth, req.format);
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
    if (g_maketxPath.empty()) {
        // Fall back to oiiotool mipmaps when maketx is missing.
        if (g_oiiotoolPath.empty()) {
            error = "no maketx or oiiotool found";
            return false;
        }
        QStringList args;
        args << src << QStringLiteral("--autocc") << QStringLiteral("off");
        if (!txSkipColorConvert(req.inputColorSpace)) {
            if (!req.ocioConfigPath.empty())
                args << QStringLiteral("--colorconfig") << QString::fromStdString(req.ocioConfigPath);
            args << QStringLiteral("--colorconvert") << QString::fromStdString(req.inputColorSpace)
                 << QStringLiteral("ACES - ACEScg");
        }
        args << QStringLiteral("--mipmaps") << QStringLiteral("-o") << dst;
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
    if (!txSkipColorConvert(req.inputColorSpace)) {
        if (!req.ocioConfigPath.empty())
            args << QStringLiteral("--colorconfig") << QString::fromStdString(req.ocioConfigPath);
        args << QStringLiteral("--colorconvert") << QString::fromStdString(req.inputColorSpace)
             << QStringLiteral("ACES - ACEScg");
        args << QStringLiteral("--unpremult");
    }
    args << src << QStringLiteral("-o") << dst;

    logInfo("tx_convert: maketx " + src.toStdString() + " → " + dst.toStdString());
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

int txExtractFrameNumber(const std::string& path) {
    const QString base = QFileInfo(QString::fromStdString(path)).completeBaseName();
    const QRegularExpression re(QStringLiteral(R"((?:^|[._-])(\d+)$)"));
    const auto m = re.match(base);
    if (!m.hasMatch()) return 0;
    return m.captured(1).toInt();
}

std::string txOutputExtension(TxOutputFormat format) {
    switch (format) {
        case TxOutputFormat::Png: return "png";
        case TxOutputFormat::Jpg: return "jpg";
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
    const QString ext = QString::fromStdString(txOutputExtension(format));
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

TxConvertResult txConvertOne(const TxConvertRequest& req) {
    TxConvertResult result;
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

    // Already matching TX with no reformat — return as-is.
    if (req.format == TxOutputFormat::Tx && srcInfo.suffix().toLower() == QLatin1String("tx") &&
        !needsOiiotoolPreprocess(req)) {
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

    if (req.format == TxOutputFormat::Png || req.format == TxOutputFormat::Jpg) {
        // PNG/JPG: no colour-space convert — resize / bit / channels only.
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
        QTemporaryDir tmp;
        if (!tmp.isValid()) {
            result.error = "could not create temp dir for TX preprocess";
            return result;
        }
        const QString tempPath = tmp.filePath(QStringLiteral("pre.exr"));
        TxConvertRequest pre = req;
        // Intermediate stays float-friendly; colour convert happens in maketx.
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

bool txConvertPattern(const std::string& sourcePathOrPattern, const std::string& outputDir,
                      const TxConvertOptions& options, std::vector<TxConvertResult>& results,
                      std::string& error) {
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

    bool allOk = true;
    for (const std::string& src : sources) {
        TxConvertRequest req;
        req.sourcePath = src;
        req.outputPath = txAllocateOutputPath(src, outputDir, options.format);
        req.inputColorSpace = options.inputColorSpace;
        req.ocioConfigPath = options.ocioConfigPath;
        req.updateOnly = options.updateOnly;
        req.format = options.format;
        req.bitDepth = options.bitDepth;
        req.longSide = options.longSide;
        req.channels = options.channels;
        TxConvertResult r = txConvertOne(req);
        results.push_back(r);
        if (!r.ok) {
            allOk = false;
            if (error.empty()) error = r.error;
            logWarning("tx_convert: " + r.error);
        }
    }
    return allOk;
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
