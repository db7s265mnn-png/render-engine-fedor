#include "io/tx_cache.h"

#include <QDir>
#include <QFileInfo>
#include <QProcess>
#include <QStandardPaths>
#include <QString>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

#include "core/log.h"

namespace sol {
namespace {

// ---------------------------------------------------------------------------
// Hex hash helpers (portable, no external hashing library)
// ---------------------------------------------------------------------------

static uint64_t fnv1a64(const char* data, size_t len) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(data[i]));
        hash *= 1099511628211ULL;
    }
    return hash;
}

static std::string toHex16(uint64_t v) {
    static const char hex[] = "0123456789abcdef";
    std::string result(16, '0');
    for (int i = 15; i >= 0; --i) {
        result[size_t(i)] = hex[v & 0xF];
        v >>= 4;
    }
    return result;
}

// Stable cache key: hash of (path + mtime + size).
static std::string cacheKey(const std::string& path, qint64 mtime, qint64 size) {
    std::string combined = path + "|" + std::to_string(mtime) + "|" + std::to_string(size);
    const uint64_t h = fnv1a64(combined.c_str(), combined.size());
    return toHex16(h);
}

// ---------------------------------------------------------------------------
// Tool discovery
// ---------------------------------------------------------------------------

static std::string findTool(const QString& name) {
    const QString path = QStandardPaths::findExecutable(name);
    if (!path.isEmpty()) return path.toStdString();
    return {};
}

// Try maketx first, then oiiotool.
// Returns the tool kind: "maketx", "oiiotool", or "" (none found).
static std::string discoverTool(std::string& toolPath) {
    // maketx (OIIO)
    std::string p = findTool(QStringLiteral("maketx"));
    if (!p.empty()) {
        toolPath = p;
        return "maketx";
    }
    // Some distributions ship it as maketx-<version>
    p = findTool(QStringLiteral("maketx-2.5"));
    if (p.empty()) p = findTool(QStringLiteral("maketx-2.4"));
    if (!p.empty()) {
        toolPath = p;
        return "maketx";
    }
    // oiiotool fallback
    p = findTool(QStringLiteral("oiiotool"));
    if (!p.empty()) {
        toolPath = p;
        return "oiiotool";
    }
    return {};
}

// Cache the tool path so we only probe PATH once per process session.
static std::string g_toolPath;
static std::string g_toolKind;
static std::once_flag g_toolOnce;

static void initTool() {
    std::call_once(g_toolOnce, []() {
        g_toolKind = discoverTool(g_toolPath);
        if (g_toolKind.empty()) {
            logWarning("tx_cache: neither maketx nor oiiotool found on PATH; "
                       "TX conversion is disabled");
        } else {
            logInfo("tx_cache: using " + g_toolKind + " at " + g_toolPath);
        }
    });
}

// ---------------------------------------------------------------------------
// Process-wide active settings pointer
// ---------------------------------------------------------------------------

static std::mutex g_settingsMutex;
static const RenderSettingsData* g_activeTxSettings = nullptr;

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

// Build the output .tx filename from the stable cache key.
static std::string buildTxPath(const std::string& sourcePath, const std::string& cacheDir,
                                const std::string& key) {
    const QFileInfo src(QString::fromStdString(sourcePath));
    const QString stem = src.completeBaseName() + "_" + QString::fromStdString(key);
    const QString txName = stem + ".tx";
    if (cacheDir.empty()) {
        return QDir(QStringLiteral("tx_cache")).absoluteFilePath(txName).toStdString();
    }
    const QString dir = QString::fromStdString(cacheDir);
    const QFileInfo dirInfo(dir);
    const QString absDir = dirInfo.isAbsolute() ? dir : QDir::current().absoluteFilePath(dir);
    return QDir(absDir).absoluteFilePath(txName).toStdString();
}

static bool runConversion(const std::string& src, const std::string& dst, std::string& error) {
    initTool();
    if (g_toolKind.empty()) {
        error = "no maketx or oiiotool on PATH";
        return false;
    }

    QStringList args;
    const QString srcQ = QString::fromStdString(src);
    const QString dstQ = QString::fromStdString(dst);

    if (g_toolKind == "maketx") {
        // maketx -u: update only if out-of-date; --oiio: OIIO-compatible TX.
        args << QStringLiteral("-u") << QStringLiteral("--oiio")
             << srcQ
             << QStringLiteral("-o") << dstQ;
    } else {
        // oiiotool best-effort: write as tiled multidir TIFF with mipmaps.
        args << srcQ
             << QStringLiteral("--autocc") << QStringLiteral("off")
             << QStringLiteral("--mipmaps")
             << QStringLiteral("-o") << dstQ;
    }

    QProcess proc;
    proc.setProgram(QString::fromStdString(g_toolPath));
    proc.setArguments(args);
    proc.start();
    if (!proc.waitForStarted(5000)) {
        error = g_toolKind + ": failed to start";
        return false;
    }
    if (!proc.waitForFinished(300000)) {  // 5 min cap
        proc.kill();
        error = g_toolKind + ": timed out converting " + src;
        return false;
    }
    if (proc.exitCode() != 0) {
        const std::string stderr_out = proc.readAllStandardError().toStdString();
        error = g_toolKind + " failed (exit " + std::to_string(proc.exitCode()) + "): " +
                (stderr_out.empty() ? "(no stderr)" : stderr_out);
        return false;
    }
    return true;
}

static std::string resolveCacheDir(const char* txCacheDir) {
    if (txCacheDir && txCacheDir[0] != '\0') {
        return std::string(txCacheDir);
    }
    return "tx_cache";
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::string resolveTxCachedPath(const std::string& sourcePath, const std::string& cacheDir) {
    const QFileInfo src(QString::fromStdString(sourcePath));
    const qint64 mtime = src.exists() ? src.lastModified().toSecsSinceEpoch() : 0;
    const qint64 size  = src.exists() ? src.size() : 0;
    const std::string key = cacheKey(sourcePath, mtime, size);
    return buildTxPath(sourcePath, cacheDir, key);
}

bool ensureTxTexture(const std::string& sourcePath, const RenderSettingsData& settings,
                     std::string& outPath, std::string& error) {
    if (!settings.enableTxCache) {
        outPath = sourcePath;
        return true;
    }

    // If the source is already a .tx file, reuse it as-is (it already has mips).
    {
        const size_t dot = sourcePath.rfind('.');
        if (dot != std::string::npos) {
            std::string ext = sourcePath.substr(dot + 1);
            for (char& c : ext) c = char(std::tolower(static_cast<unsigned char>(c)));
            if (ext == "tx") {
                outPath = sourcePath;
                return true;
            }
        }
    }

    const QFileInfo srcInfo(QString::fromStdString(sourcePath));
    if (!srcInfo.exists()) {
        outPath = sourcePath;
        error = "source texture not found: " + sourcePath;
        return false;
    }

    const qint64 mtime = srcInfo.lastModified().toSecsSinceEpoch();
    const qint64 size  = srcInfo.size();
    const std::string key = cacheKey(sourcePath, mtime, size);
    const std::string cacheDir = resolveCacheDir(settings.txCacheDir);
    const std::string txPath = buildTxPath(sourcePath, cacheDir, key);

    const QFileInfo txInfo(QString::fromStdString(txPath));
    if (txInfo.exists()) {
        // Already converted and up-to-date.
        outPath = txPath;
        return true;
    }

    // Ensure cache directory exists.
    const QFileInfo txDirInfo(txInfo.dir().absolutePath());
    if (!txDirInfo.exists()) {
        if (!QDir().mkpath(txInfo.dir().absolutePath())) {
            outPath = sourcePath;
            error = "tx_cache: cannot create cache directory: " + txInfo.dir().absolutePath().toStdString();
            return false;
        }
    }

    logInfo("tx_cache: converting " + sourcePath + " → " + txPath);

    std::string convError;
    if (!runConversion(sourcePath, txPath, convError)) {
        logWarning("tx_cache: conversion failed for " + sourcePath + ": " + convError);
        outPath = sourcePath;
        error = convError;
        return false;
    }

    logInfo("tx_cache: ready " + txPath);
    outPath = txPath;
    return true;
}

void setActiveTxCacheSettings(const RenderSettingsData* settings) {
    std::lock_guard<std::mutex> lock(g_settingsMutex);
    g_activeTxSettings = settings;
}

bool txCacheActive() {
    std::lock_guard<std::mutex> lock(g_settingsMutex);
    return g_activeTxSettings != nullptr && g_activeTxSettings->enableTxCache != 0;
}

// Internal: retrieve a copy of the active settings under the lock.
// Returns false when inactive.
static bool getActiveTxSettings(RenderSettingsData& out) {
    std::lock_guard<std::mutex> lock(g_settingsMutex);
    if (!g_activeTxSettings || !g_activeTxSettings->enableTxCache) return false;
    out = *g_activeTxSettings;
    return true;
}

// Exposed for image_io.cpp — try to convert and return the tx path (or source).
std::string txCacheResolve(const std::string& sourcePath) {
    RenderSettingsData settings{};
    if (!getActiveTxSettings(settings)) return sourcePath;
    std::string outPath;
    std::string error;
    ensureTxTexture(sourcePath, settings, outPath, error);
    return outPath;
}

}  // namespace sol
