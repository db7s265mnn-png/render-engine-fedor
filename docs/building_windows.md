# Building a distributable Solstice.exe on Windows

## One-click (OptiX ON)

On a Windows machine with the prerequisites below already installed, double-click
`BUILD_WINDOWS.bat` — **OptiX-min** (Qt + Embree + CUDA). Output: `C:\gz-build`.

`BUILD_WINDOWS_FULL.bat` — **full app** (OptiX + OpenVDB + MaterialX + OpenPGL + Alembic +
OpenEXR + OCIO + TinyUSDZ + TX Tools). Output: `C:\gz-full`. First full run builds
Imath/OpenEXR/Alembic/TBB/OpenVDB/OCIO into `%LOCALAPPDATA%\grendizer-deps`.

**Visual Studio 2026** (MSVC 14.50+) needs **CUDA 13.2 or newer**. CUDA 12.0 cannot
parse that STL (`type_traits` / `aligned_storage` / `result_of` while compiling
OptiX wavefront kernels). CUDA 13 dropped Pascal (`compute_60`); the script passes
`-DSOLSTICE_OPTIX_ARCH=compute_75`.

GPU OptiX is a **wavefront** of small modules (`init_from_camera`, `intersect_*`, `shade_*`,
`hit_miss`). It does not compile `integrator.h`, volumes, SSS, MaterialX procedurals, or
polynomial optics. Kill any old 3h `cicc` job, delete `C:\gz-build\generated`, re-run
`BUILD_WINDOWS.bat`.

**Visual Studio 2022** can still use CUDA 12.x and `compute_60`.

If both CUDA 12.0 and 13.2 are installed, the script **always uses 13.2**. The
build tree is `C:\gz-build` (a GitHub zip under `Downloads` is too long for MSVC
object files). After a failed or CUDA-upgraded run, delete `C:\gz-build` — keep
`%LOCALAPPDATA%\grendizer-deps`.

Override auto-detected paths with environment variables if needed: `QT_ROOT`,
`CUDA_PATH`, `OptiX_ROOT`, `GRENDIZER_BUILD_DIR`, `GRENDIZER_DEPS`.

To skip the Embree download, put `embree-4.4.0.x64.windows.zip` in
`%USERPROFILE%\Downloads`, `C:\grendizer-deps`, or a `deps-cache` folder next to
`BUILD_WINDOWS.bat` ([Embree 4.4.0 zip](https://github.com/RenderKit/embree/releases/download/v4.4.0/embree-4.4.0.x64.windows.zip)).
After one successful run, compiled deps live in `%LOCALAPPDATA%\grendizer-deps`.

The exe lands in `C:\gz-build\bin\` (Ninja). A leftover `bin\Release` folder from an older
VS tree is ignored. Engine → Render Device should list `GPU (OptiX)` without “not in this
build”.

## Prerequisites

* Visual Studio 2022 (C++ desktop) **or** Visual Studio 2026
* CMake 3.20+
* Qt 6 MSVC kit (`msvc2022_64` or `msvc2019_64`). Any 6.x is fine, including 6.11.1
  at `C:\Qt\6.11.1\msvc2022_64`. MinGW kits cannot be used with this Visual Studio build.
* CUDA Toolkit: **13.2+** with VS 2026, or 12.x with VS 2022. OptiX headers are cloned
  automatically if the SDK is missing.

## Dependencies through vcpkg

```powershell
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg install embree:x64-windows alembic:x64-windows openexr:x64-windows
```

The repository also ships a `vcpkg.json`, so `cmake` with the vcpkg toolchain installs these
in manifest mode on its own. The `embree` port carries Embree 4; older vcpkg checkouts where
it still resolves to Embree 3 will fail at configure time with a version error.

## Configure and build

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
      -DCMAKE_PREFIX_PATH=C:/Qt/6.11.1/msvc2022_64
cmake --build build --config Release
```

The executable lands in `build\bin\Release\Solstice.exe`.

## Making it self contained

```powershell
cmake --build build --config Release --target deploy
```

The `deploy` target runs `windeployqt`, which copies the Qt DLLs, platform plugins and image
format plugins next to the executable. Copy the vcpkg runtime DLLs as well — the easiest way
is to let vcpkg's applocal deployment handle it (it runs automatically for Release builds) or
to copy them manually:

```powershell
copy C:\vcpkg\installed\x64-windows\bin\embree4.dll build\bin\Release\
copy C:\vcpkg\installed\x64-windows\bin\tbb12.dll build\bin\Release\
copy C:\vcpkg\installed\x64-windows\bin\Alembic.dll build\bin\Release\
copy C:\vcpkg\installed\x64-windows\bin\Imath*.dll build\bin\Release\
copy C:\vcpkg\installed\x64-windows\bin\OpenEXR*.dll build\bin\Release\
copy C:\vcpkg\installed\x64-windows\bin\Iex*.dll build\bin\Release\
copy C:\vcpkg\installed\x64-windows\bin\IlmThread*.dll build\bin\Release\
```

Zip `build\bin\Release` and the result runs on any Windows machine without an installer.

## Enabling OptiX

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
      -DCMAKE_PREFIX_PATH=C:/Qt/6.11.1/msvc2022_64 ^
      -DSOLSTICE_ENABLE_OPTIX=ON ^
      -DOptiX_ROOT="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"
```

The OptiX device programs are compiled to PTX by `nvcc` and embedded into the executable, so
nothing extra needs to be shipped. Only the NVIDIA display driver is required at runtime;
users without an NVIDIA GPU keep the Embree backend.

If `nvcc` refuses your host compiler, pass `-DSOLSTICE_CUDA_HOST_COMPILER=<path to cl.exe>`,
and use `-DSOLSTICE_OPTIX_ARCH=75-virtual` (or another virtual architecture) to control the
PTX target.

## Troubleshooting

* **"Qt platform plugin windows not found"** — the deploy step did not run; use
  `windeployqt.exe build\bin\Release\Solstice.exe`.
* **Alembic files fail to load** — confirm the build printed `Alembic support : 1` while
  configuring; without it the import node reports that the feature is missing.
* **The GPU backend falls back to Embree** — check the log panel, it prints why OptiX was
  unavailable (no CUDA device, driver too old, or a build without OptiX support).
* **nvcc fails on `type_traits` / `aligned_storage` / `result_of`** — CUDA 12.0 was used
  with VS 2026. Install CUDA 13.2, delete `C:\gz-build`, re-run `BUILD_WINDOWS.bat`.
  The log must show `nvcc release 13.2` and `OptiX PTX arch: compute_75`.
* **`Unsupported gpu architecture 'compute_60'`** — leftover CMake cache from CUDA 12.
  Delete `C:\gz-build` and re-run. Do not delete `%LOCALAPPDATA%\grendizer-deps`.
* **`fatal error C1083` / OpenPGL path longer than 250 characters** — the GitHub zip
  under `Downloads` is too deep. Current one-click builds in `C:\gz-build`. Replace
  `BUILD_WINDOWS.bat` + `tools\build_windows.ps1` and re-run.
