// Windows: OpenColorIO_2_3.dll imports zlib1.dll. NVIDIA optixInit does
// LoadLibrary("zlib.dll") and hangs if that file sits next to the exe.
// Keep OCIO + zlib in bin/ocio/ and delay-load them after the OptiX probe.
#if defined(_WIN32)

#include "io/ocio_util.h"

#include <delayimp.h>
#include <windows.h>

#include <cstring>
#include <string>

static std::wstring solsticeExeDirW() {
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return {};
    std::wstring p(buf, buf + n);
    const size_t slash = p.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return {};
    return p.substr(0, slash);
}

static std::wstring solsticeOcioDirW() { return solsticeExeDirW() + L"\\ocio"; }

static bool nameIsOcioRuntimeDll(const char* name) {
    if (!name || !name[0]) return false;
    return _stricmp(name, "zlib1.dll") == 0 || _stricmp(name, "zlib.dll") == 0 ||
           _stricmp(name, "zlibd.dll") == 0 || _stricmp(name, "yaml-cpp.dll") == 0 ||
           _stricmp(name, "expat.dll") == 0 || _stricmp(name, "libexpat.dll") == 0 ||
           _stricmp(name, "libexpat-1.dll") == 0 || _strnicmp(name, "OpenColorIO", 11) == 0;
}

static HMODULE loadFromOcioDir(const char* dllName) {
    if (!dllName) return nullptr;
    std::wstring path = solsticeOcioDirW() + L"\\";
    for (const char* c = dllName; *c; ++c) {
        const unsigned char ch = static_cast<unsigned char>(*c);
        if (ch >= 128) return nullptr;
        path.push_back(static_cast<wchar_t>(ch));
    }
    return LoadLibraryExW(path.c_str(), nullptr,
                          LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}

static FARPROC WINAPI solsticeDliNotifyHook(unsigned notify, PDelayLoadInfo pdli) {
    if (notify != dliNotePreLoadLibrary || !pdli || !pdli->szDll) return nullptr;
    if (!nameIsOcioRuntimeDll(pdli->szDll)) return nullptr;
    HMODULE mod = loadFromOcioDir(pdli->szDll);
    return reinterpret_cast<FARPROC>(mod);
}

extern "C" const PfnDliHook __pfnDliNotifyHook2 = solsticeDliNotifyHook;

namespace sol {

void ocioBindWindowsRuntimeDlls() {
    static bool done = false;
    if (done) return;
    done = true;
    const std::wstring dir = solsticeOcioDirW();
    if (dir.empty()) return;
    CreateDirectoryW(dir.c_str(), nullptr);
    AddDllDirectory(dir.c_str());
}

}  // namespace sol

#endif
