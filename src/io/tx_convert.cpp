#include "io/tx_convert.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QString>
#include <QTextStream>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <string>

#include "core/log.h"

namespace sol {
namespace {

std::string findTool(const QString& name) {
    const QString path = QStandardPaths::findExecutable(name);
    return path.isEmpty() ? std::string() : path.toStdString();
}

std::string g_toolPath;
std::string g_toolKind;
std::once_flag g_toolOnce;

void initTool() {
    std::call_once(g_toolOnce, []() {
        g_toolPath = findTool(QStringLiteral("maketx"));
        if (g_toolPath.empty()) g_toolPath = findTool(QStringLiteral("maketx-2.5"));
        if (g_toolPath.empty()) g_toolPath = findTool(QStringLiteral("maketx-2.4"));
        if (!g_toolPath.empty()) {
            g_toolKind = "maketx";
            logInfo("tx_convert: using maketx at " + g_toolPath);
            return;
        }
        g_toolPath = findTool(QStringLiteral("oiiotool"));
        if (!g_toolPath.empty()) {
            g_toolKind = "oiiotool";
            logInfo("tx_convert: using oiiotool at " + g_toolPath);
            return;
        }
        logWarning("tx_convert: neither maketx nor oiiotool found on PATH");
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

std::string readSidecarSource(const QString& txPath) {
    QFile f(txPath + QStringLiteral(".txsrc"));
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return {};
    return QString::fromUtf8(f.readAll()).trimmed().toStdString();
}

void writeSidecarSource(const QString& txPath, const std::string& sourcePath) {
    QFile f(txPath + QStringLiteral(".txsrc"));
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) return;
    f.write(sourcePath.c_str());
    f.write("\n");
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
    // OCIO YAML: lines like `  - !<ColorSpace>` followed by `    name: Foo`
    // Also match `name: Foo` under colorspaces.
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
        // Relative to cwd.
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

std::vector<std::string> txExpandUdimSources(const std::string& sourcePathOrPattern) {
    const QString path = QString::fromStdString(sourcePathOrPattern);
    std::vector<std::string> out;
    if (!looksLikeUdimPattern(path)) {
        out.push_back(sourcePathOrPattern);
        return out;
    }
    // Scan common UDIM range 1001–1100.
    for (int udim = 1001; udim <= 1100; ++udim) {
        const QString concrete = replaceUdimToken(path, udim);
        if (QFileInfo::exists(concrete)) out.push_back(concrete.toStdString());
    }
    return out;
}

std::string txAllocateOutputPath(const std::string& sourcePath, const std::string& outputDir) {
    const QFileInfo src(QString::fromStdString(sourcePath));
    const QString baseName = src.completeBaseName();  // strip extension; keep UDIM digits in name
    QString dir = QString::fromStdString(outputDir);
    if (dir.isEmpty()) dir = QStringLiteral("tx_cache");
    const QFileInfo dirInfo(dir);
    const QString absDir = dirInfo.isAbsolute() ? dir : QDir::current().absoluteFilePath(dir);
    QDir().mkpath(absDir);

    auto candidate = [&](int copyIndex) -> QString {
        QString stem = baseName;
        if (copyIndex > 0) stem += QStringLiteral("_copy_%1").arg(copyIndex);
        return QDir(absDir).absoluteFilePath(stem + QStringLiteral(".tx"));
    };

    // Prefer an existing tx that already belongs to this source (sidecar).
    for (int copy = 0; copy < 10000; ++copy) {
        const QString txPath = candidate(copy);
        const QFileInfo txInfo(txPath);
        if (!txInfo.exists()) {
            // Free slot — also check no foreign sidecar.
            return txPath.toStdString();
        }
        const std::string recorded = readSidecarSource(txPath);
        if (recorded.empty() || recorded == sourcePath) return txPath.toStdString();
        // Occupied by another source → try next _copy_N.
    }
    return candidate(9999).toStdString();
}

TxConvertResult txConvertOne(const TxConvertRequest& req) {
    TxConvertResult result;
    result.outputPath = req.outputPath;
    initTool();
    if (g_toolKind.empty()) {
        result.error = "no maketx or oiiotool on PATH";
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

    // Already .tx — copy reference.
    {
        QString ext = srcInfo.suffix().toLower();
        if (ext == QLatin1String("tx")) {
            result.ok = true;
            result.outputPath = req.sourcePath;
            return result;
        }
    }

    const QFileInfo dstInfo(QString::fromStdString(req.outputPath));
    QDir().mkpath(dstInfo.absolutePath());

    // Update-only: skip if tx exists, is newer than source, and sidecar matches.
    if (req.updateOnly && dstInfo.exists()) {
        const std::string recorded = readSidecarSource(QString::fromStdString(req.outputPath));
        if ((recorded.empty() || recorded == req.sourcePath) &&
            dstInfo.lastModified() >= srcInfo.lastModified()) {
            result.ok = true;
            return result;
        }
    }

    QStringList args;
    const QString srcQ = QString::fromStdString(req.sourcePath);
    const QString dstQ = QString::fromStdString(req.outputPath);
    const bool doConvert = !txSkipColorConvert(req.inputColorSpace);

    if (g_toolKind == "maketx") {
        if (req.updateOnly) args << QStringLiteral("-u");
        args << QStringLiteral("--oiio");
        if (doConvert) {
            if (!req.ocioConfigPath.empty())
                args << QStringLiteral("--colorconfig") << QString::fromStdString(req.ocioConfigPath);
            args << QStringLiteral("--colorconvert") << QString::fromStdString(req.inputColorSpace)
                 << QStringLiteral("ACES - ACEScg");
            args << QStringLiteral("--unpremult");
        }
        args << srcQ << QStringLiteral("-o") << dstQ;
    } else {
        // oiiotool fallback — limited color management.
        args << srcQ;
        if (doConvert && !req.ocioConfigPath.empty()) {
            args << QStringLiteral("--colorconfig") << QString::fromStdString(req.ocioConfigPath);
            args << QStringLiteral("--colorconvert") << QString::fromStdString(req.inputColorSpace)
                 << QStringLiteral("ACES - ACEScg");
        } else {
            args << QStringLiteral("--autocc") << QStringLiteral("off");
        }
        args << QStringLiteral("--mipmaps") << QStringLiteral("-o") << dstQ;
    }

    logInfo("tx_convert: " + g_toolKind + " " + req.sourcePath + " → " + req.outputPath +
            (doConvert ? (" [" + req.inputColorSpace + " → ACES - ACEScg]") : " [no colorconvert]"));

    QProcess proc;
    proc.setProgram(QString::fromStdString(g_toolPath));
    proc.setArguments(args);
    proc.start();
    if (!proc.waitForStarted(5000)) {
        result.error = g_toolKind + ": failed to start";
        return result;
    }
    if (!proc.waitForFinished(600000)) {
        proc.kill();
        result.error = g_toolKind + ": timed out";
        return result;
    }
    if (proc.exitCode() != 0) {
        const std::string err = proc.readAllStandardError().toStdString();
        result.error = g_toolKind + " failed: " + (err.empty() ? "(no stderr)" : err);
        return result;
    }

    writeSidecarSource(QString::fromStdString(req.outputPath), req.sourcePath);
    result.ok = true;
    return result;
}

bool txConvertPattern(const std::string& sourcePathOrPattern, const std::string& outputDir,
                      const std::string& inputColorSpace, const std::string& ocioConfigPath,
                      std::vector<TxConvertResult>& results, std::string& error) {
    results.clear();
    const std::vector<std::string> sources = txExpandUdimSources(sourcePathOrPattern);
    if (sources.empty()) {
        error = "no source files found for: " + sourcePathOrPattern;
        return false;
    }
    bool allOk = true;
    for (const std::string& src : sources) {
        TxConvertRequest req;
        req.sourcePath = src;
        req.outputPath = txAllocateOutputPath(src, outputDir);
        req.inputColorSpace = inputColorSpace;
        req.ocioConfigPath = ocioConfigPath;
        req.updateOnly = true;
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

}  // namespace sol
