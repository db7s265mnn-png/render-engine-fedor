// Windows: OpenColorIO_2_3.dll imports zlib1.dll. NVIDIA optixInit
// LoadLibrary("zlib.dll") hangs if that file is next to the exe or on the
// process search path. Keep OpenColorIO + zlib1.dll in bin/ocio and delay-load
// only OpenColorIO (zlib1 comes in as its dependency from that folder).
#if defined(_WIN32)

#include "io/ocio_util.h"

// delayimp.h on some SDKs declares a writable hook; DELAYIMP_INSECURE_WRITABLE_HOOKS
// matches both old and new MSVC headers (const vs non-const __pfnDliNotifyHook2).
#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0602
#  undef _WIN32_WINNT
#  define _WIN32_WINNT 0x0A00
#endif
#define DELAYIMP_INSECURE_WRITABLE_HOOKS
#include <windows.h>
#include <delayimp.h>

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
    // Never hook zlib.dll — NVIDIA optixInit LoadLibrary("zlib.dll") must
    // resolve to System32, not bin\ocio.
    return _strnicmp(name, "OpenColorIO", 11) == 0;
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

extern "C" PfnDliHook __pfnDliNotifyHook2 = solsticeDliNotifyHook;

namespace sol {

void ocioBindWindowsRuntimeDlls() {
    static bool done = false;
    if (done) return;
    done = true;
    const std::wstring dir = solsticeOcioDirW();
    if (dir.empty()) return;
    CreateDirectoryW(dir.c_str(), nullptr);
    // Do not AddDllDirectory(ocio): that puts ocio\zlib.dll on LoadLibrary("zlib.dll")
    // and hangs NVIDIA optixInit after the startup probe already succeeded.
}

}  // namespace sol

#endif
