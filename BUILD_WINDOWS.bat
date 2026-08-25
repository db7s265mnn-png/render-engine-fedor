@echo off
title Grendizer Render - Windows OptiX build
cd /d "%~dp0"

echo.
echo Double-click this file to compile (Release + OptiX, MINIMAL).
echo Visual Studio 2026 needs CUDA 13.2. CUDA 12.0 can stay installed.
echo Output: C:\gz-build
echo Close Grendizer_Render first or the linker cannot overwrite the exe (LNK1168).
echo This is the fast GPU OptiX build (no VDB/MaterialX/Alembic/...).
echo Full app: BUILD_WINDOWS_FULL.bat  -^>  C:\gz-full
echo Keep %%LOCALAPPDATA%%\grendizer-deps (Embree cache).
echo Deleting C:\gz-build is OK - this script creates it again.
echo.

where powershell >nul 2>&1
if errorlevel 1 (
    echo PowerShell not found.
    goto :end
)

if not exist "C:\gz-build" mkdir "C:\gz-build"

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
