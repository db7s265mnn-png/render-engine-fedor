@echo off
title Grendizer Render - Windows OptiX build
cd /d "%~dp0"

echo.
echo Double-click this file to compile (Release + OptiX).
echo First run can take a long time (vcpkg + nvcc).
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
