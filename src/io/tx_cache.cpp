#include "io/tx_cache.h"

#include "io/tx_convert.h"

#include <mutex>
#include <string>

#include "core/log.h"

namespace sol {
namespace {

std::mutex g_settingsMutex;
const RenderSettingsData* g_activeTxSettings = nullptr;

// Optional per-path input color space override for the next resolve (MaterialX).
std::mutex g_csMutex;
std::string g_defaultInputColorSpace = "ACES - ACEScg";

std::string resolveCacheDir(const char* txCacheDir) {
    if (txCacheDir && txCacheDir[0] != '\0') return std::string(txCacheDir);
    return "tx_cache";
}

}  // namespace

void setTxDefaultInputColorSpace(const std::string& colorSpace) {
    std::lock_guard<std::mutex> lock(g_csMutex);
    g_defaultInputColorSpace = colorSpace.empty() ? "ACES - ACEScg" : colorSpace;
}

std::string txDefaultInputColorSpace() {
    std::lock_guard<std::mutex> lock(g_csMutex);
    return g_defaultInputColorSpace;
}

std::string resolveTxCachedPath(const std::string& sourcePath, const std::string& cacheDir) {
    return txAllocateOutputPath(sourcePath, cacheDir.empty() ? "tx_cache" : cacheDir);
}

bool ensureTxTexture(const std::string& sourcePath, const RenderSettingsData& settings,
                     std::string& outPath, std::string& error) {
    return ensureTxTexture(sourcePath, settings, txDefaultInputColorSpace(), outPath, error);
}

bool ensureTxTexture(const std::string& sourcePath, const RenderSettingsData& settings,
                     const std::string& inputColorSpace, std::string& outPath, std::string& error) {
    if (!settings.enableTxCache) {
        outPath = sourcePath;
        return true;
    }

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

    const std::string cacheDir = resolveCacheDir(settings.txCacheDir);
    const std::string ocio = txResolveOcioConfig(settings.ocioUseEnv != 0, settings.ocioConfigPath);

    std::vector<TxConvertResult> results;
    std::string convError;
    const bool ok =
        txConvertPattern(sourcePath, cacheDir, inputColorSpace, ocio, results, convError);
    if (!results.empty() && results[0].ok) {
        outPath = results[0].outputPath;
        // UDIM: first tile path — loadImageOrUdim expects pattern; callers using single
        // files get the converted tx. For UDIM patterns, callers should convert tiles
        // individually via txConvertPattern before load.
        if (results.size() == 1) return true;
        // Multiple tiles: leave source pattern; tiles were written beside cache.
        // Replace is handled by converting each concrete tile path at load time.
        outPath = sourcePath;
        return ok;
    }
    outPath = sourcePath;
    error = convError.empty() ? "tx conversion failed" : convError;
    return false;
}

void setActiveTxCacheSettings(const RenderSettingsData* settings) {
    std::lock_guard<std::mutex> lock(g_settingsMutex);
    g_activeTxSettings = settings;
}

bool txCacheActive() {
    std::lock_guard<std::mutex> lock(g_settingsMutex);
    return g_activeTxSettings != nullptr && g_activeTxSettings->enableTxCache != 0;
}

static bool getActiveTxSettings(RenderSettingsData& out) {
    std::lock_guard<std::mutex> lock(g_settingsMutex);
    if (!g_activeTxSettings || !g_activeTxSettings->enableTxCache) return false;
    out = *g_activeTxSettings;
    return true;
}

std::string txCacheResolve(const std::string& sourcePath) {
    RenderSettingsData settings{};
    if (!getActiveTxSettings(settings)) return sourcePath;

    // Single concrete path (including one UDIM tile). Convert in place.
    std::string outPath;
    std::string error;
    if (!ensureTxTexture(sourcePath, settings, txDefaultInputColorSpace(), outPath, error)) {
        if (!error.empty()) logWarning("tx_cache: " + error);
        return sourcePath;
    }
    // If conversion wrote a .tx for this exact file, use it.
    if (outPath.size() >= 3 && (outPath.compare(outPath.size() - 3, 3, ".tx") == 0 ||
                                outPath.compare(outPath.size() - 3, 3, ".TX") == 0))
        return outPath;
    // Fallback: allocate and convert one.
    TxConvertRequest req;
    req.sourcePath = sourcePath;
    req.outputPath = txAllocateOutputPath(sourcePath, resolveCacheDir(settings.txCacheDir));
    req.inputColorSpace = txDefaultInputColorSpace();
    req.ocioConfigPath = txResolveOcioConfig(settings.ocioUseEnv != 0, settings.ocioConfigPath);
    const TxConvertResult r = txConvertOne(req);
    return r.ok ? r.outputPath : sourcePath;
}

}  // namespace sol
