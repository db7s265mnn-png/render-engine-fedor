@echo off
title Grendizer Render - Windows FULL build
cd /d "%~dp0"

echo.
echo Double-click this file for the FULL app (OptiX + VDB + MaterialX + Alembic + EXR).
echo Visual Studio 2026 needs CUDA 13.2. CUDA 12.0 can stay installed.
echo Output: C:\gz-full
echo Close Grendizer_Render first or the linker cannot overwrite the exe (LNK1168).
echo First full run builds OpenEXR/Alembic/OpenVDB (cached after that).
echo OpenColorIO is NOT linked on Windows OptiX: its zlib.dll hangs NVIDIA optixInit
echo (HUD stuck on OptiX checking). Display/View uses Classic.
echo GPU probe runs before the window so Intel iGPU cannot hide the NVIDIA card.
echo Deploy: embree/tbb/openvdb/cudart next to the exe. No zlib.dll, no OpenColorIO.dll.
echo TinyUSDZ linking can sit with no new lines for a long time - wait.
echo Keep C:\gz-full and %%LOCALAPPDATA%%\grendizer-deps
echo Do not delete C:\gz-full (MaterialX / TinyUSDZ / OpenPGL live in _deps).
echo.

where powershell >nul 2>&1
if errorlevel 1 (
    echo PowerShell not found.
    goto :end
)

set GRENDIZER_FULL_DEPS=1
set GRENDIZER_BUILD_DIR=C:\gz-full
if not exist "%GRENDIZER_BUILD_DIR%" mkdir "%GRENDIZER_BUILD_DIR%"

powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build_windows.ps1"
if errorlevel 1 (
    echo.
    echo BUILD FAILED. Scroll up for the error.
) else (
    echo.
    echo BUILD OK.
)

:end
echo.
pause
