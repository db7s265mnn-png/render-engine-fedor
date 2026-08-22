@echo off
title Grendizer Render - Windows OptiX build
cd /d "%~dp0"

echo.
echo Double-click this file to compile (Release + OptiX).
echo Visual Studio 2026 needs CUDA 13.2. CUDA 12.0 can stay installed.
echo Output goes to C:\gz-build  (NOT inside Downloads - Windows path limit).
echo OptiX-min: no MaterialX, OpenPGL, TinyUSDZ, OpenVDB, Alembic, tests.
echo Keep %%LOCALAPPDATA%%\grendizer-deps (Embree cache).
echo.

where powershell >nul 2>&1
if errorlevel 1 (
    echo PowerShell not found.
    goto :end
)

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
