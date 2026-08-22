# Building a distributable Solstice.exe on Windows

## One-click (OptiX ON)

On a Windows machine with the prerequisites below already installed, double-click
`BUILD_WINDOWS.bat` in the repo root. It finds Visual Studio, Qt, CUDA, OptiX and
vcpkg, configures `-DSOLSTICE_ENABLE_OPTIX=ON`, builds Release and runs `deploy`.

Override auto-detected paths with environment variables if needed: `QT_ROOT`,
`VCPKG_ROOT`, `CUDA_PATH`, `OptiX_ROOT`, `GRENDIZER_BUILD_DIR`.

The exe lands in `build-windows\bin\Release\`. Engine → Render Backend should list
`GPU (OptiX)` without “not in this build”.

## Prerequisites

* Visual Studio 2022 with the C++ desktop workload
* CMake 3.20+
* Qt 6 MSVC kit (`msvc2022_64` or `msvc2019_64`). Any 6.x is fine, including 6.11.1
  at `C:\Qt\6.11.1\msvc2022_64`. MinGW kits cannot be used with this Visual Studio build.
* [vcpkg](https://vcpkg.io) for Embree, Alembic and OpenEXR
* Optional: CUDA Toolkit 12.x and the OptiX SDK 7.7 or newer for GPU rendering

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
