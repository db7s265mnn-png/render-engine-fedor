#include "io/ocio_util.h"

#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "core/log.h"
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
OCIO_NS::ConstCPUProcessorRcPtr g_procActive;
int g_procWorkingSpace = -1;
int g_preparedView = -1;

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
        // Validate display/view exist.
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
    } catch (const OCIO_NS::Exception& ex) {
        logDebug(std::string("OCIO DisplayView failed (") + display + "/" + view + "): " + ex.what());
        return false;
    } catch (...) {
        return false;
    }
}

bool buildViewProcessor(const OCIO_NS::ConstConfigRcPtr& config, const std::string& src,
                        bool rec709, OCIO_NS::ConstCPUProcessorRcPtr& out, std::string& used) {
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
    const Pair recPairs[] = {
        {"ACES", "Rec.709"},
        {"ACES", "rec709"},
        {"ACES", "Rec709"},
        {"ACES", "rec709 (ACES)"},
        {"Rec.709", "ACES"},
        {"rec709", "ACES"},
        {"ACES", "Output - Rec.709"},
    };
    const Pair* pairs = rec709 ? recPairs : srgbPairs;
    const int n = rec709 ? int(sizeof(recPairs) / sizeof(recPairs[0]))
                         : int(sizeof(srgbPairs) / sizeof(srgbPairs[0]));
    for (int i = 0; i < n; ++i) {
        if (tryDisplayView(config, src, pairs[i].display, pairs[i].view, out)) {
            used = std::string(pairs[i].display) + " / " + pairs[i].view;
            return true;
        }
    }

    // Last resort: first display + its default / first view containing sRGB or 709.
    if (config->getNumDisplays() <= 0) return false;
    const char* display = config->getDisplay(0);
    const int nViews = config->getNumViews(display);
    for (int i = 0; i < nViews; ++i) {
        const char* view = config->getView(display, i);
        const std::string v = view ? view : "";
        const bool match = rec709 ? (v.find("709") != std::string::npos || v.find("Rec") != std::string::npos)
                                  : (v.find("sRGB") != std::string::npos || v.find("srgb") != std::string::npos);
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
    g_procWorkingSpace = workingSpace;
    if (!g_config) return;

    const std::string src =
        (workingSpace == kWorkingSpaceAcesCg)
            ? pickColorSpace(g_config, {"ACES - ACEScg", "ACEScg", "acescg"})
            : pickColorSpace(g_config, {"Utility - Linear - sRGB", "Linear Rec.709 (sRGB)", "linear",
                                        "scene-linear Rec 709/sRGB", "lin_srgb"});
    if (src.empty()) {
        logWarning("OCIO: no suitable source colour space in config");
        return;
    }

    std::string usedSrgb, usedRec;
    if (buildViewProcessor(g_config, src, false, g_procSrgb, usedSrgb))
        logInfo("OCIO view sRGB (ACES): " + src + " → " + usedSrgb);
    else
        logWarning("OCIO: could not resolve Display/View for sRGB (ACES); available displays logged below");
    if (buildViewProcessor(g_config, src, true, g_procRec709, usedRec))
        logInfo("OCIO view rec709 (ACES): " + src + " → " + usedRec);
    else
        logWarning("OCIO: could not resolve Display/View for rec709 (ACES)");

    if (!g_procSrgb || !g_procRec709) {
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
    OcioStatus st;
    st.libraryAvailable = ocioLibraryAvailable();
#if !SOLSTICE_HAVE_OCIO
    st.message = "OpenColorIO: not available in this build (compile with OpenColorIO)";
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
    }
    if (path.empty() && !settingsPath.empty()) path = settingsPath;

    // Reload if path changed or first call.
    const bool needLoad = !g_tried || path != g_loadedPath || fromEnv != g_fromEnv || !g_config;
    if (needLoad) {
        g_tried = true;
        g_loadedPath = path;
        g_fromEnv = fromEnv;
        g_config.reset();
        g_procSrgb.reset();
        g_procRec709.reset();
        g_procActive.reset();
        g_procWorkingSpace = -1;
        g_preparedView = -1;
        if (path.empty()) {
            st.message = "OpenColorIO: library OK, but no config (set OCIO or Film → OCIO Config)";
            return st;
        }
        try {
            g_config = OCIO_NS::Config::CreateFromFile(path.c_str());
        } catch (const OCIO_NS::Exception& ex) {
            st.message = std::string("OpenColorIO: library OK, config not loaded — ") + ex.what();
            return st;
        } catch (...) {
            st.message = "OpenColorIO: library OK, config not loaded — unknown error";
            return st;
        }
    }

    st.configLoaded = g_config != nullptr;
    st.fromEnvironment = g_fromEnv;
    st.configPath = g_loadedPath;
    if (st.configLoaded) {
        st.message = "OpenColorIO: found, config " + st.configPath +
                     (st.fromEnvironment ? " [OCIO env]" : " [settings]");
    } else {
        st.message = "OpenColorIO: library OK, but no config (set OCIO or Film → OCIO Config)";
    }
    return st;
#endif
}

void ocioLogStatus(bool useEnv, const std::string& settingsPath) {
    const OcioStatus st = ocioEnsureConfig(useEnv, settingsPath);
    if (st.configLoaded)
        logInfo(st.message);
    else if (st.libraryAvailable)
        logWarning(st.message);
    else
        logWarning(st.message);
}

bool ocioApplyView(Vec3 linearWorking, int workingSpace, int viewTransform, Vec3& outDisplay) {
    if (viewTransform == kViewRaw) {
        outDisplay = Vec3(saturatef(linearWorking.x), saturatef(linearWorking.y),
                          saturatef(linearWorking.z));
        return true;
    }
    if (!ocioPrepareView(workingSpace, viewTransform)) {
        outDisplay = linearWorking;
        return false;
    }
    outDisplay = ocioApplyViewPrepared(linearWorking);
    return true;
}

bool ocioPrepareView(int workingSpace, int viewTransform) {
    if (viewTransform == kViewRaw) {
#if SOLSTICE_HAVE_OCIO
        std::lock_guard<std::mutex> lock(g_mutex);
        g_procActive.reset();
        g_preparedView = kViewRaw;
#endif
        return true;
    }
#if !SOLSTICE_HAVE_OCIO
    (void)workingSpace;
    return false;
#else
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_config) return false;
    if (g_procWorkingSpace != workingSpace) rebuildProcessors(workingSpace);
    g_procActive = (viewTransform == kViewRec709Aces) ? g_procRec709 : g_procSrgb;
    g_preparedView = viewTransform;
    return g_procActive != nullptr;
#endif
}

Vec3 ocioApplyViewPrepared(Vec3 linearWorking) {
#if !SOLSTICE_HAVE_OCIO
    return Vec3(saturatef(linearWorking.x), saturatef(linearWorking.y), saturatef(linearWorking.z));
#else
    if (g_preparedView == kViewRaw || !g_procActive) {
        return Vec3(saturatef(linearWorking.x), saturatef(linearWorking.y),
                    saturatef(linearWorking.z));
    }
    float rgb[3] = {linearWorking.x, linearWorking.y, linearWorking.z};
    g_procActive->applyRGB(rgb);
    return Vec3(saturatef(rgb[0]), saturatef(rgb[1]), saturatef(rgb[2]));
#endif
}

}  // namespace sol
