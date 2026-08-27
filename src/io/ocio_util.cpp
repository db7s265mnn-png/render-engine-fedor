#if defined(_MSC_VER) && !defined(__clang__)
// MSVC /O2 imports OCIO vftables as data; /DELAYLOAD then dies with LNK1194.
#  pragma optimize("", off)
#  pragma auto_inline(off)
#endif

#include "io/ocio_util.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/log.h"
#include "io/tx_cache.h"
#include "io/tx_convert.h"
#include "solstice_config.h"

#if SOLSTICE_HAVE_OCIO
#  include <OpenColorIO/OpenColorIO.h>
namespace OCIO_NS = OCIO_NAMESPACE;
#endif

namespace sol {
namespace {

std::mutex g_mutex;
std::string g_loadedPath;
bool g_fromEnv = false;
bool g_tried = false;

#if SOLSTICE_HAVE_OCIO
OCIO_NS::ConstConfigRcPtr g_config;
OCIO_NS::ConstCPUProcessorRcPtr g_procSrgb;
OCIO_NS::ConstCPUProcessorRcPtr g_procRec709;
OCIO_NS::ConstCPUProcessorRcPtr g_procRec2020;
OCIO_NS::ConstCPUProcessorRcPtr g_procActive;
int g_procWorkingSpace = -1;
std::unordered_map<std::string, OCIO_NS::ConstCPUProcessorRcPtr> g_convertProcs;
#endif

int g_preparedView = -1;
int g_preparedWorking = -1;
bool g_preparedClassic = false;

float encodeGamma24(float c) {
    c = saturatef(c);
    // BT.2020 / Rec.709-style gamma 2.4 with linear toe (approx).
    constexpr float a = 1.09929682680944f;
    constexpr float b = 0.018053968645747f;
    if (c < b) return c * 4.5f;
    return a * powf(c, 0.45f) - (a - 1.0f);
}

Vec3 encodeGamma24Vec(Vec3 c) {
    return Vec3(encodeGamma24(c.x), encodeGamma24(c.y), encodeGamma24(c.z));
}

#if SOLSTICE_HAVE_OCIO
std::string pickColorSpace(const OCIO_NS::ConstConfigRcPtr& config, const std::vector<std::string>& names) {
    for (const std::string& n : names) {
        try {
            if (config->getColorSpace(n.c_str())) return n;
        } catch (...) {
        }
    }
    try {
        const char* scene = config->getCanonicalName("scene_linear");
        if (scene && scene[0]) return scene;
    } catch (...) {
    }
    if (config->getNumColorSpaces() > 0) return config->getColorSpaceNameByIndex(0);
    return {};
}

bool tryDisplayView(const OCIO_NS::ConstConfigRcPtr& config, const std::string& src,
                    const char* display, const char* view, OCIO_NS::ConstCPUProcessorRcPtr& out) {
    try {
        bool displayOk = false;
        for (int i = 0; i < config->getNumDisplays(); ++i) {
            if (std::string(config->getDisplay(i)) == display) {
                displayOk = true;
                break;
            }
        }
        if (!displayOk) return false;
        bool viewOk = false;
        for (int i = 0; i < config->getNumViews(display); ++i) {
            if (std::string(config->getView(display, i)) == view) {
                viewOk = true;
                break;
            }
        }
        if (!viewOk) return false;

        OCIO_NS::DisplayViewTransformRcPtr xf = OCIO_NS::DisplayViewTransform::Create();
        xf->setSrc(src.c_str());
        xf->setDisplay(display);
        xf->setView(view);
        OCIO_NS::ConstProcessorRcPtr proc = config->getProcessor(xf);
        out = proc->getDefaultCPUProcessor();
        return out != nullptr;
    } catch (const std::exception& ex) {
        // Do not catch OCIO::Exception by name: MSVC then imports the vftable
        // as data and /DELAYLOAD OpenColorIO fails with LNK1194.
        logDebug(std::string("OCIO DisplayView failed (") + display + "/" + view + "): " + ex.what());
        return false;
    } catch (...) {
        return false;
    }
}

enum class ViewKind { Srgb, Rec709, Rec2020 };

bool buildViewProcessor(const OCIO_NS::ConstConfigRcPtr& config, const std::string& src,
                        ViewKind kind, OCIO_NS::ConstCPUProcessorRcPtr& out, std::string& used) {
    struct Pair {
        const char* display;
        const char* view;
    };
    const Pair srgbPairs[] = {
        {"ACES", "sRGB"},
        {"ACES", "sRGB (ACES)"},
        {"sRGB", "ACES"},
        {"sRGB (ACES)", "sRGB"},
        {"ACES", "Output - sRGB"},
    };
    const Pair rec709Pairs[] = {
        {"ACES", "Rec.709"},
        {"ACES", "rec709"},
        {"ACES", "Rec709"},
        {"ACES", "rec709 (ACES)"},
        {"Rec.709", "ACES"},
        {"rec709", "ACES"},
        {"ACES", "Output - Rec.709"},
    };
    const Pair rec2020Pairs[] = {
        {"ACES", "Rec.2020"},
        {"ACES", "Rec2020"},
        {"ACES", "rec2020"},
        {"ACES", "Rec.2020 (ACES)"},
        {"Rec.2020", "ACES"},
        {"rec2020", "ACES"},
        {"ACES", "Output - Rec.2020"},
        {"ACES", "HDR Rec.2020"},
        {"ACES", "Rec.2020 ST2084"},
    };

    const Pair* pairs = srgbPairs;
    int n = int(sizeof(srgbPairs) / sizeof(srgbPairs[0]));
    if (kind == ViewKind::Rec709) {
        pairs = rec709Pairs;
        n = int(sizeof(rec709Pairs) / sizeof(rec709Pairs[0]));
    } else if (kind == ViewKind::Rec2020) {
        pairs = rec2020Pairs;
        n = int(sizeof(rec2020Pairs) / sizeof(rec2020Pairs[0]));
    }

    for (int i = 0; i < n; ++i) {
        if (tryDisplayView(config, src, pairs[i].display, pairs[i].view, out)) {
            used = std::string(pairs[i].display) + " / " + pairs[i].view;
            return true;
        }
    }

    if (config->getNumDisplays() <= 0) return false;
    const char* display = config->getDisplay(0);
    const int nViews = config->getNumViews(display);
    for (int i = 0; i < nViews; ++i) {
        const char* view = config->getView(display, i);
        const std::string v = view ? view : "";
        bool match = false;
        if (kind == ViewKind::Rec2020) {
            match = v.find("2020") != std::string::npos;
        } else if (kind == ViewKind::Rec709) {
            match = v.find("709") != std::string::npos ||
                    (v.find("Rec") != std::string::npos && v.find("2020") == std::string::npos);
        } else {
            match = v.find("sRGB") != std::string::npos || v.find("srgb") != std::string::npos;
        }
        if (!match && i + 1 < nViews) continue;
        if (tryDisplayView(config, src, display, view, out)) {
            used = std::string(display) + " / " + v;
            return true;
        }
    }
    return false;
}

void rebuildProcessors(int workingSpace) {
    g_procSrgb.reset();
    g_procRec709.reset();
    g_procRec2020.reset();
    g_procWorkingSpace = workingSpace;
    if (!g_config) return;

    const std::string src =
        (workingSpace == kWorkingSpaceAcesCg)
            ? pickColorSpace(g_config, {"ACES - ACEScg", "ACEScg", "acescg"})
            : pickColorSpace(g_config, {"Utility - Linear - sRGB", "Linear Rec.709 (sRGB)", "linear",
                                        "scene-linear Rec 709/sRGB", "lin_srgb"});
    if (src.empty()) {
        logWarning("OCIO: NOT FOUND suitable source colour space in config");
        return;
    }

    std::string usedSrgb, usedRec, used2020;
    if (buildViewProcessor(g_config, src, ViewKind::Srgb, g_procSrgb, usedSrgb))
        logInfo("OCIO: FOUND view sRGB: " + src + " → " + usedSrgb);
    else
        logWarning("OCIO: NOT FOUND Display/View for sRGB");
    if (buildViewProcessor(g_config, src, ViewKind::Rec709, g_procRec709, usedRec))
        logInfo("OCIO: FOUND view Rec.709: " + src + " → " + usedRec);
    else
        logWarning("OCIO: NOT FOUND Display/View for Rec.709");
    if (buildViewProcessor(g_config, src, ViewKind::Rec2020, g_procRec2020, used2020))
        logInfo("OCIO: FOUND view Rec.2020: " + src + " → " + used2020);
    else
        logWarning("OCIO: NOT FOUND Display/View for Rec.2020");

    if (!g_procSrgb || !g_procRec709 || !g_procRec2020) {
        std::string displays;
        for (int i = 0; i < g_config->getNumDisplays(); ++i) {
            const char* d = g_config->getDisplay(i);
            if (i) displays += "; ";
            displays += d ? d : "?";
            displays += "{";
            for (int j = 0; j < g_config->getNumViews(d); ++j) {
                if (j) displays += ",";
                displays += g_config->getView(d, j);
            }
            displays += "}";
        }
        logInfo("OCIO displays/views: " + (displays.empty() ? std::string("(none)") : displays));
    }
}

OCIO_NS::ConstCPUProcessorRcPtr processorForView(int viewTransform) {
    if (viewTransform == kViewRec709Aces) return g_procRec709;
    if (viewTransform == kViewRec2020) return g_procRec2020;
    return g_procSrgb;
}
#endif

}  // namespace

bool ocioLibraryAvailable() {
#if SOLSTICE_HAVE_OCIO
    return true;
#else
    return false;
#endif
}

OcioStatus ocioEnsureConfig(bool useEnv, const std::string& settingsPath) {
    ocioBindWindowsRuntimeDlls();
    OcioStatus st;
    st.libraryAvailable = ocioLibraryAvailable();
#if !SOLSTICE_HAVE_OCIO
    st.message = "OCIO: NOT FOUND — OpenColorIO library not linked in this build";
    return st;
#else
    std::lock_guard<std::mutex> lock(g_mutex);

    std::string path;
    bool fromEnv = false;
    if (useEnv) {
        if (const char* env = std::getenv("OCIO")) {
            if (env[0] != '\0') {
                path = env;
                fromEnv = true;
            }
        }
    } else if (!settingsPath.empty()) {
        path = settingsPath;
    }

    const bool needLoad = !g_tried || path != g_loadedPath || fromEnv != g_fromEnv || !g_config;
    if (needLoad) {
        g_tried = true;
        g_loadedPath = path;
        g_fromEnv = fromEnv;
        g_config.reset();
        g_procSrgb.reset();
        g_procRec709.reset();
        g_procRec2020.reset();
        g_procActive.reset();
        g_convertProcs.clear();
        g_procWorkingSpace = -1;
        g_preparedView = -1;
        if (path.empty()) {
            if (useEnv)
                st.message = "OCIO: NOT FOUND — Use OCIO from Environment is on, but OCIO env is empty";
            else
                st.message = "OCIO: NOT FOUND — set Film → OCIO Config (or enable Use OCIO from Environment)";
            return st;
        }
        try {
            g_config = OCIO_NS::Config::CreateFromFile(path.c_str());
        } catch (const std::exception& ex) {
            st.message = std::string("OCIO: NOT FOUND — failed to load config ") + path + " — " + ex.what();
            return st;
        } catch (...) {
            st.message = std::string("OCIO: NOT FOUND — failed to load config ") + path + " — unknown error";
            return st;
        }
    }

    st.configLoaded = g_config != nullptr;
    st.fromEnvironment = g_fromEnv;
    st.configPath = g_loadedPath;
    if (st.configLoaded) {
        st.message = "OCIO: FOUND — config " + st.configPath +
                     (st.fromEnvironment ? " [OCIO env]" : " [Film path]");
    } else if (useEnv) {
        st.message = "OCIO: NOT FOUND — Use OCIO from Environment is on, but OCIO env is empty";
    } else {
        st.message = "OCIO: NOT FOUND — set Film → OCIO Config (or enable Use OCIO from Environment)";
    }
    return st;
#endif
}

void ocioLogStatus(bool useEnv, const std::string& settingsPath) {
    const OcioStatus st = ocioEnsureConfig(useEnv, settingsPath);
    if (st.configLoaded)
        logInfo(st.message);
    else
        logWarning(st.message);
}

Vec3 classicApplyView(Vec3 linearWorking, int workingSpace, int viewTransform) {
    if (viewTransform == kViewRaw) {
        return Vec3(saturatef(linearWorking.x), saturatef(linearWorking.y),
                    saturatef(linearWorking.z));
    }
    Vec3 c = linearWorking;
    // Houdini-style without OCIO: no filmic tone map — only space convert + OETF.
    if (workingSpace == kWorkingSpaceAcesCg) c = acescgToLinearSrgb(c);
    if (viewTransform == kViewRec2020) return encodeGamma24Vec(c);
    // sRGB and Rec.709 share a close OETF; use the sRGB curve for Classic.
    return linearToSrgbVec(c);
}

bool ocioApplyView(Vec3 linearWorking, int workingSpace, int viewTransform, Vec3& outDisplay) {
    if (viewTransform == kViewRaw) {
        outDisplay = Vec3(saturatef(linearWorking.x), saturatef(linearWorking.y),
                          saturatef(linearWorking.z));
        return true;
    }
    if (!ocioPrepareView(workingSpace, viewTransform)) {
        outDisplay = classicApplyView(linearWorking, workingSpace, viewTransform);
        return false;
    }
    outDisplay = ocioApplyViewPrepared(linearWorking);
    return true;
}

bool ocioPrepareView(int workingSpace, int viewTransform) {
    g_preparedClassic = false;
    g_preparedWorking = workingSpace;
    if (viewTransform == kViewRaw) {
#if SOLSTICE_HAVE_OCIO
        std::lock_guard<std::mutex> lock(g_mutex);
        g_procActive.reset();
#endif
        g_preparedView = kViewRaw;
        return true;
    }
#if !SOLSTICE_HAVE_OCIO
    (void)workingSpace;
    return false;
#else
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_config) return false;
    if (g_procWorkingSpace != workingSpace) rebuildProcessors(workingSpace);
    g_procActive = processorForView(viewTransform);
    g_preparedView = viewTransform;
    return g_procActive != nullptr;
#endif
}

bool displayPrepareView(int workingSpace, int colorManagement, int viewTransform) {
    if (colorManagement == kColorClassic || viewTransform == kViewRaw) {
#if SOLSTICE_HAVE_OCIO
        std::lock_guard<std::mutex> lock(g_mutex);
        g_procActive.reset();
#endif
        g_preparedClassic = (viewTransform != kViewRaw) && (colorManagement == kColorClassic);
        // Raw always uses linear clamp via prepared path (classic or OCIO).
        if (viewTransform == kViewRaw) {
            g_preparedClassic = false;
            g_preparedView = kViewRaw;
            g_preparedWorking = workingSpace;
            return true;
        }
        g_preparedView = viewTransform;
        g_preparedWorking = workingSpace;
        return true;
    }
    const bool ok = ocioPrepareView(workingSpace, viewTransform);
    if (!ok) {
        // Fall back to Classic transfer so the viewport still looks reasonable.
        g_preparedClassic = true;
        g_preparedView = viewTransform;
        g_preparedWorking = workingSpace;
        return true;
    }
    g_preparedClassic = false;
    return true;
}

Vec3 ocioApplyViewPrepared(Vec3 linearWorking) {
    if (g_preparedView == kViewRaw) {
        return Vec3(saturatef(linearWorking.x), saturatef(linearWorking.y),
                    saturatef(linearWorking.z));
    }
    if (g_preparedClassic) {
        return classicApplyView(linearWorking, g_preparedWorking, g_preparedView);
    }
#if !SOLSTICE_HAVE_OCIO
    return classicApplyView(linearWorking, g_preparedWorking, g_preparedView);
#else
    if (!g_procActive) {
        return classicApplyView(linearWorking, g_preparedWorking, g_preparedView);
    }
    float rgb[3] = {linearWorking.x, linearWorking.y, linearWorking.z};
    g_procActive->applyRGB(rgb);
    return Vec3(saturatef(rgb[0]), saturatef(rgb[1]), saturatef(rgb[2]));
#endif
}

namespace {

std::string lowerCopy(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

Vec3 classicConvertToAcescg(Vec3 rgb, const std::string& inputColorSpace) {
    if (txSkipColorConvert(inputColorSpace)) return rgb;
    const std::string lower = lowerCopy(inputColorSpace);
    const bool texture = lower.find("srgb - texture") != std::string::npos ||
                         lower.find("srgb_texture") != std::string::npos ||
                         lower.find("rec.709 - texture") != std::string::npos ||
                         lower == "output - srgb" || lower == "srgb" ||
                         lower == "utility - rec.709 - texture";
    const bool linearSrgb = lower.find("linear - srgb") != std::string::npos ||
                            lower.find("lin_srgb") != std::string::npos ||
                            lower.find("linear rec.709") != std::string::npos ||
                            lower.find("scene-linear rec 709") != std::string::npos ||
                            lower == "linear";
    Vec3 c = rgb;
    if (texture)
        c = Vec3(srgbToLinearUnclamped(c.x), srgbToLinearUnclamped(c.y), srgbToLinearUnclamped(c.z));
    if (texture || linearSrgb) return linearSrgbToAcescg(c);
    return rgb;
}

#if SOLSTICE_HAVE_OCIO
OCIO_NS::ConstCPUProcessorRcPtr convertProcessor(const std::string& inputColorSpace) {
    if (!g_config || txSkipColorConvert(inputColorSpace)) return nullptr;
    auto it = g_convertProcs.find(inputColorSpace);
    if (it != g_convertProcs.end()) return it->second;

    const std::string dst = pickColorSpace(g_config, {"ACES - ACEScg", "ACEScg", "acescg"});
    if (dst.empty()) return nullptr;

    std::vector<std::string> srcNames = {inputColorSpace};
    const std::string lower = lowerCopy(inputColorSpace);
    if (lower.find("srgb - texture") != std::string::npos || lower == "srgb") {
        srcNames.insert(srcNames.end(),
                        {"Utility - sRGB - Texture", "sRGB", "srgb_texture", "Input - Generic - sRGB - Texture"});
    } else if (lower.find("linear - srgb") != std::string::npos || lower.find("lin_srgb") != std::string::npos ||
               lower == "linear") {
        srcNames.insert(srcNames.end(), {"Utility - Linear - sRGB", "Linear Rec.709 (sRGB)", "lin_srgb",
                                         "scene-linear Rec 709/sRGB", "Linear Rec.709"});
    } else if (lower.find("rec.709 - texture") != std::string::npos) {
        srcNames.insert(srcNames.end(), {"Utility - Rec.709 - Texture", "Rec.709", "rec709"});
    } else if (lower.find("output - srgb") != std::string::npos) {
        srcNames.insert(srcNames.end(), {"Output - sRGB", "sRGB"});
    }

    std::string src = pickColorSpace(g_config, srcNames);
    if (src.empty()) return nullptr;

    try {
        OCIO_NS::ColorSpaceTransformRcPtr xf = OCIO_NS::ColorSpaceTransform::Create();
        xf->setSrc(src.c_str());
        xf->setDst(dst.c_str());
        OCIO_NS::ConstProcessorRcPtr proc = g_config->getProcessor(xf);
        OCIO_NS::ConstCPUProcessorRcPtr cpu = proc->getDefaultCPUProcessor();
        g_convertProcs[inputColorSpace] = cpu;
        return cpu;
    } catch (const std::exception& ex) {
        logDebug(std::string("OCIO colorconvert failed (") + inputColorSpace + " → " + dst + "): " +
                 ex.what());
        g_convertProcs[inputColorSpace] = nullptr;
        return nullptr;
    } catch (...) {
        g_convertProcs[inputColorSpace] = nullptr;
        return nullptr;
    }
}
#endif

void ensureConvertConfigLoaded() {
    RenderSettingsData settings{};
    if (txCacheGetActiveSettings(settings)) {
        ocioEnsureConfig(settings.ocioUseEnv != 0, settings.ocioConfigPath);
        return;
    }
    ocioEnsureConfig(true, {});
}

}  // namespace

Vec3 ocioConvertToAcescg(Vec3 rgb, const std::string& inputColorSpace) {
    if (txSkipColorConvert(inputColorSpace)) return rgb;
    ensureConvertConfigLoaded();
#if SOLSTICE_HAVE_OCIO
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (OCIO_NS::ConstCPUProcessorRcPtr proc = convertProcessor(inputColorSpace)) {
            float v[3] = {rgb.x, rgb.y, rgb.z};
            proc->applyRGB(v);
            return Vec3(v[0], v[1], v[2]);
        }
    }
#endif
    return classicConvertToAcescg(rgb, inputColorSpace);
}

}  // namespace sol
