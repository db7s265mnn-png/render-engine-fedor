# One-click Windows Release build with OptiX.
# Invoked by BUILD_WINDOWS.bat at the repo root. Do not require the user to
# pass flags; override paths with env vars if auto-detect is wrong:
#   QT_ROOT, VCPKG_ROOT, CUDA_PATH, OptiX_ROOT, GRENDIZER_BUILD_DIR
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "ОШИБКА: $Message" -ForegroundColor Red
    exit 1
}

function Info([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}

function Find-VsWhere {
    $p = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $p) { return $p }
    return $null
}

function Import-VcVars64 {
    $vswhere = Find-VsWhere
    if (-not $vswhere) {
        Fail "Не найден Visual Studio Installer (vswhere). Поставь VS 2022 с workload «Desktop development with C++»."
    }
    $vsRoot = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsRoot) {
        Fail "Visual Studio найдена, но нет C++ toolset. В VS Installer включи «Desktop development with C++»."
    }
    $script:VsYear = "2022"
    $year = & $vswhere -latest -products * -property catalog_productLineVersion | Select-Object -First 1
    if ($year) { $script:VsYear = "$year".Trim() }
    $vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        Fail "Нет vcvars64.bat в $vsRoot"
    }
    Info "Visual Studio: $vsRoot ($script:VsYear)"
    $tmp = [IO.Path]::GetTempFileName()
    cmd.exe /c "`"$vcvars`" >nul && set" | Set-Content -Path $tmp -Encoding ascii
    Get-Content $tmp | ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') {
            Set-Item -LiteralPath "Env:$($matches[1])" -Value $matches[2]
        }
    }
    Remove-Item $tmp -ErrorAction SilentlyContinue
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        Fail "После vcvars64 не найден cl.exe"
    }
}

function Find-CMake {
    $cmd = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $guess = @(
        "${env:ProgramFiles}\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\CMake\bin\cmake.exe"
    )
    foreach ($g in $guess) {
        if (Test-Path $g) { return $g }
    }
    $vswhere = Find-VsWhere
    if ($vswhere) {
        $fromVs = & $vswhere -latest -products * -find "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" |
            Select-Object -First 1
        if ($fromVs -and (Test-Path $fromVs)) { return $fromVs }
    }
    Fail "CMake не найден. Поставь https://cmake.org/download/ (и галку Add CMake to PATH) либо компонент CMake в VS."
}

function Find-Git {
    $cmd = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($g in @(
        "${env:ProgramFiles}\Git\cmd\git.exe",
        "${env:LocalAppData}\Programs\Git\cmd\git.exe"
    )) {
        if (Test-Path $g) { return $g }
    }
    Fail "Git не найден. Поставь https://git-scm.com/download/win"
}

function Find-QtPrefix {
    foreach ($envName in @("QT_ROOT", "QT_ROOT_DIR", "QTDIR")) {
        $v = [Environment]::GetEnvironmentVariable($envName)
        if ($v -and (Test-Path (Join-Path $v "lib\cmake\Qt6\Qt6Config.cmake"))) { return $v }
    }
    $roots = @("C:\Qt", "D:\Qt", "E:\Qt", (Join-Path $env:USERPROFILE "Qt"))
    $kits = @("msvc2022_64", "msvc2019_64")
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $versions = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^6\.\d+' } |
            Sort-Object { [version]($_.Name -replace '[^\d.].*$','') } -Descending
        foreach ($ver in $versions) {
            foreach ($kit in $kits) {
                $p = Join-Path $ver.FullName $kit
                if (Test-Path (Join-Path $p "lib\cmake\Qt6\Qt6Config.cmake")) { return $p }
            }
        }
    }
    Fail @"
Qt 6 (msvc2019_64 / msvc2022_64) не найден.
Поставь Qt 6 через https://www.qt.io/download-qt-installer в C:\Qt
или задай переменную QT_ROOT, например:
  set QT_ROOT=C:\Qt\6.7.2\msvc2019_64
"@
}

function Find-CudaPath {
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
        return $env:CUDA_PATH
    }
    $base = "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $base) {
        $vers = Get-ChildItem $base -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^v12\.' -or $_.Name -match '^v13\.' } |
            Sort-Object Name -Descending
        foreach ($v in $vers) {
            if (Test-Path (Join-Path $v.FullName "bin\nvcc.exe")) { return $v.FullName }
        }
    }
    Fail @"
CUDA Toolkit не найден (нужен nvcc).
Поставь CUDA 12.x: https://developer.nvidia.com/cuda-downloads
После установки закрой это окно и запусти BUILD_WINDOWS.bat снова.
"@
}

function Find-OptiXRoot([string]$Git) {
    foreach ($envName in @("OptiX_ROOT", "OptiX_INSTALL_DIR")) {
        $v = [Environment]::GetEnvironmentVariable($envName)
        if ($v -and (Test-Path (Join-Path $v "include\optix.h"))) { return $v }
    }
    $nvidia = Join-Path $env:ProgramData "NVIDIA Corporation"
    if (Test-Path $nvidia) {
        $sdks = Get-ChildItem $nvidia -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "OptiX SDK*" } |
            Sort-Object Name -Descending
        foreach ($s in $sdks) {
            if (Test-Path (Join-Path $s.FullName "include\optix.h")) { return $s.FullName }
        }
    }
    foreach ($p in @("C:\optix-dev", (Join-Path $Root "third_party\optix-dev"))) {
        if (Test-Path (Join-Path $p "include\optix.h")) { return $p }
    }
    $local = Join-Path $env:LOCALAPPDATA "grendizer-optix-dev"
    if (Test-Path (Join-Path $local "include\optix.h")) { return $local }
    Info "OptiX SDK не найден — качаю публичные заголовки NVIDIA/optix-dev v7.7.0 ..."
    if (Test-Path $local) { Remove-Item -Recurse -Force $local }
    & $Git clone --depth 1 --branch v7.7.0 https://github.com/NVIDIA/optix-dev.git $local
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $local "include\optix.h"))) {
        Fail "Не удалось скачать OptiX headers. Поставь OptiX SDK 7.7+ с NVIDIA Developer или проверь сеть."
    }
    return $local
}

function Find-VcpkgRoot([string]$Git) {
    if ($env:VCPKG_ROOT -and (Test-Path (Join-Path $env:VCPKG_ROOT "scripts\buildsystems\vcpkg.cmake"))) {
        return $env:VCPKG_ROOT
    }
    foreach ($p in @(
        "C:\vcpkg",
        "D:\vcpkg",
        (Join-Path $Root "vcpkg"),
        (Join-Path (Split-Path $Root -Parent) "vcpkg")
    )) {
        if (Test-Path (Join-Path $p "scripts\buildsystems\vcpkg.cmake")) { return $p }
    }
    $local = Join-Path $env:LOCALAPPDATA "grendizer-vcpkg"
    if (Test-Path (Join-Path $local "scripts\buildsystems\vcpkg.cmake")) { return $local }
    Info "vcpkg не найден — клонирую в $local (первый раз долго) ..."
    & $Git clone --depth 1 https://github.com/microsoft/vcpkg.git $local
    if ($LASTEXITCODE -ne 0) { Fail "Не удалось клонировать vcpkg. Проверь сеть / GitHub." }
    $boot = Join-Path $local "bootstrap-vcpkg.bat"
    cmd.exe /c "`"$boot`" -disableMetrics"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $local "vcpkg.exe"))) {
        Fail "bootstrap-vcpkg.bat не удался"
    }
    return $local
}

Write-Host ""
Write-Host "=== Grendizer Render — Windows сборка с OptiX ===" -ForegroundColor Green
Write-Host "Репозиторий: $Root"
Write-Host "Первый запуск может занять долго (vcpkg + nvcc PTX + Release)."
Write-Host ""

Import-VcVars64
$CMake = Find-CMake
$Git = Find-Git
$Qt = Find-QtPrefix
$Cuda = Find-CudaPath
$env:CUDA_PATH = $Cuda
$env:PATH = "$(Join-Path $Cuda 'bin');$env:PATH"
$OptiX = Find-OptiXRoot $Git
$Vcpkg = Find-VcpkgRoot $Git
$Cl = (Get-Command cl.exe).Source
$Nvcc = Join-Path $Cuda "bin\nvcc.exe"
$BuildDir = if ($env:GRENDIZER_BUILD_DIR) { $env:GRENDIZER_BUILD_DIR } else { Join-Path $Root "build-windows" }

Info "CMake:  $CMake"
Info "Qt:     $Qt"
Info "CUDA:   $Cuda"
Info "nvcc:   $Nvcc"
Info "cl.exe: $Cl"
Info "OptiX:  $OptiX"
Info "vcpkg:  $Vcpkg"
Info "Build:  $BuildDir"
Write-Host ""

if (-not (Test-Path $Nvcc)) { Fail "nvcc.exe нет по пути $Nvcc" }

$Generator = "Visual Studio 17 2022"
if ($script:VsYear -eq "2019") { $Generator = "Visual Studio 16 2019" }
Info "CMake generator: $Generator"

$Toolchain = Join-Path $Vcpkg "scripts\buildsystems\vcpkg.cmake"
& $CMake -S $Root -B $BuildDir -G $Generator -A x64 `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    "-DCMAKE_PREFIX_PATH=$Qt" `
    "-DCMAKE_CUDA_COMPILER=$Nvcc" `
    "-DCMAKE_CUDA_HOST_COMPILER=$Cl" `
    "-DSOLSTICE_CUDA_HOST_COMPILER=$Cl" `
    "-DSOLSTICE_ENABLE_OPTIX=ON" `
    "-DOptiX_ROOT=$OptiX" `
    "-DSOLSTICE_ENABLE_OCIO=ON" `
    "-DSOLSTICE_ENABLE_OPENVDB=ON" `
    "-DSOLSTICE_MODERN_CPU=ON"
if ($LASTEXITCODE -ne 0) { Fail "cmake configure не удался. Смотри лог выше (Qt / CUDA / OptiX / vcpkg)." }

$Cfg = Join-Path $BuildDir "generated\solstice_config.h"
if (-not (Test-Path $Cfg) -or -not (Select-String -Path $Cfg -Pattern "SOLSTICE_HAVE_OPTIX 1" -Quiet)) {
    Fail "OptiX не включился (SOLSTICE_HAVE_OPTIX != 1). Проверь, что nvcc и OptiX_ROOT видны в логе cmake."
}
Info "OptiX вшит в эту сборку (SOLSTICE_HAVE_OPTIX 1)."

Write-Host ""
Info "Компиляция Release (PTX через nvcc может занять несколько минут) ..."
& $CMake --build $BuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) { Fail "Сборка не удалась." }

Write-Host ""
Info "Копирую Qt DLL рядом с exe (deploy) ..."
& $CMake --build $BuildDir --config Release --target deploy
if ($LASTEXITCODE -ne 0) {
    Write-Host "Предупреждение: target deploy не собрался. exe может требовать Qt в PATH." -ForegroundColor Yellow
}

$Bin = Join-Path $BuildDir "bin\Release"
if (-not (Test-Path $Bin)) { $Bin = Join-Path $BuildDir "bin" }
$Cudart = Get-ChildItem (Join-Path $Cuda "bin") -Filter "cudart64_*.dll" -ErrorAction SilentlyContinue
if ($Cudart -and (Test-Path $Bin)) {
    $Cudart | ForEach-Object {
        Copy-Item $_.FullName $Bin -Force
        Info "Скопирован $($_.Name)"
    }
}

$Exe = Get-ChildItem $Bin -Filter "Grendizer_Render*.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $Exe) {
    $Exe = Get-ChildItem $Bin -Filter "Solstice*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (-not $Exe) { Fail "exe не найден в $Bin" }

Write-Host ""
Write-Host "ГОТОВО: $($Exe.FullName)" -ForegroundColor Green
Write-Host "В приложении: Engine → Render Backend → GPU (OptiX)"
Write-Host "Без хвоста «not in this build». Нужна NVIDIA-видеокарта и свежий драйвер."
Write-Host ""

try { Invoke-Item $Bin } catch { }
exit 0
