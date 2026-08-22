# One-click Windows Release build with OptiX.
# ASCII-only: Windows PowerShell 5.1 on a Russian locale mis-parses UTF-8
# without BOM and treats Cyrillic bytes as quotes / broken tokens.
# Override paths with env vars if auto-detect is wrong:
#   QT_ROOT, VCPKG_ROOT, CUDA_PATH, OptiX_ROOT, GRENDIZER_BUILD_DIR
$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root

function Fail([string]$Message) {
    Write-Host ''
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Info([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}

function Find-VsWhere {
    $pf86 = ${env:ProgramFiles(x86)}
    $p = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $p) { return $p }
    return $null
}

function Import-VcVars64 {
    $vswhere = Find-VsWhere
    if (-not $vswhere) {
        Fail 'Visual Studio Installer (vswhere) not found. Install VS 2022 with Desktop development with C++.'
    }
    $vsRoot = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsRoot) {
        Fail 'Visual Studio found, but the C++ toolset is missing. Enable Desktop development with C++.'
    }
    $script:VsYear = '2022'
    $year = & $vswhere -latest -products * -property catalog_productLineVersion | Select-Object -First 1
    if ($year) { $script:VsYear = ("$year").Trim() }
    $vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path -LiteralPath $vcvars)) {
        Fail "vcvars64.bat missing under $vsRoot"
    }
    Info "Visual Studio: $vsRoot ($script:VsYear)"
    $tmp = [IO.Path]::GetTempFileName()
    cmd.exe /c "`"$vcvars`" >nul && set" | Set-Content -Path $tmp -Encoding ascii
    Get-Content -Path $tmp | ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') {
            Set-Item -LiteralPath ("Env:" + $matches[1]) -Value $matches[2]
        }
    }
    Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        Fail 'cl.exe not on PATH after vcvars64'
    }
}

function Find-CMake {
    $cmd = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $guess = @(
        (Join-Path $env:ProgramFiles 'CMake\bin\cmake.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'CMake\bin\cmake.exe')
    )
    foreach ($g in $guess) {
        if ($g -and (Test-Path -LiteralPath $g)) { return $g }
    }
    $vswhere = Find-VsWhere
    if ($vswhere) {
        $fromVs = & $vswhere -latest -products * -find 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' |
            Select-Object -First 1
        if ($fromVs -and (Test-Path -LiteralPath $fromVs)) { return $fromVs }
    }
    Fail 'CMake not found. Install https://cmake.org/download/ (Add CMake to PATH) or the VS CMake component.'
}

function Find-Git {
    $cmd = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($g in @(
        (Join-Path $env:ProgramFiles 'Git\cmd\git.exe'),
        (Join-Path $env:LocalAppData 'Programs\Git\cmd\git.exe')
    )) {
        if (Test-Path -LiteralPath $g) { return $g }
    }
    Fail 'Git not found. Install https://git-scm.com/download/win'
}

function Test-QtPrefix([string]$Prefix) {
    if (-not $Prefix) { return $false }
    $cfg = Join-Path $Prefix 'lib\cmake\Qt6\Qt6Config.cmake'
    return (Test-Path -LiteralPath $cfg)
}

function Find-QtKitsUnder([string]$Dir) {
    $found = @()
    if (-not (Test-Path -LiteralPath $Dir)) { return $found }
    if (Test-QtPrefix $Dir) {
        $found += $Dir
        return $found
    }
    Get-ChildItem -LiteralPath $Dir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        if (Test-QtPrefix $_.FullName) {
            $found += $_.FullName
            return
        }
        Get-ChildItem -LiteralPath $_.FullName -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            if (Test-QtPrefix $_.FullName) { $found += $_.FullName }
        }
    }
    return $found
}

function Select-MsvcQtKit([object]$Kits) {
    $list = @($Kits | Where-Object { $_ })
    if ($list.Count -eq 0) { return $null }
    $msvc = @($list | Where-Object { $_ -match 'msvc' })
    if ($msvc.Count -eq 0) { return $null }
    $sorted = @($msvc | Sort-Object {
        $ver = [version]'0.0.0'
        if ($_ -match '\\(6\.\d+(?:\.\d+)?)\\') {
            try { $ver = [version]$matches[1] } catch { }
        }
        $msvcRank = 0
        if ($_ -match 'msvc2022') { $msvcRank = 2 }
        elseif ($_ -match 'msvc2019') { $msvcRank = 1 }
        '{0:D2}.{1}' -f $msvcRank, $ver
    } -Descending)
    return $sorted[0]
}

function Find-Python {
    foreach ($name in @('py', 'python', 'python3')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

function Install-MsvcQtKit {
    $outDir = 'C:\Qt'
    $want = Join-Path $outDir '6.8.3\msvc2022_64'
    if (Test-QtPrefix $want) { return $want }

    $py = Find-Python
    if (-not $py) {
        return $null
    }

    Info 'Only MinGW Qt is installed. Downloading Qt 6.8.3 MSVC 2022 64-bit into C:\Qt (about 1 GB, one-time) ...'
    Info 'Your existing C:\Qt\6.11.1\mingw_64 is left untouched.'
    $aqtArgs = @('-m', 'pip', 'install', '--user', '--upgrade', 'aqtinstall')
    if ($py -match '\\py.exe$') {
        & $py -3 @aqtArgs
    } else {
        & $py @aqtArgs
    }
    $aqtInstall = @(
        '-m', 'aqt', 'install-qt',
        'windows', 'desktop', '6.8.3', 'win64_msvc2022_64',
        '--outputdir', $outDir
    )
    if ($py -match '\\py.exe$') {
        & $py -3 @aqtInstall
    } else {
        & $py @aqtInstall
    }
    if (Test-QtPrefix $want) { return $want }
    return $null
}

function Find-QtPrefix {
    foreach ($envName in @('QT_ROOT', 'QT_ROOT_DIR', 'QTDIR')) {
        $v = [Environment]::GetEnvironmentVariable($envName)
        if (-not $v) { continue }
        if (Test-QtPrefix $v) { return $v }
        $fromEnv = Select-MsvcQtKit (Find-QtKitsUnder $v)
        if ($fromEnv) { return $fromEnv }
    }
    $all = @()
    foreach ($root in @('C:\Qt', 'D:\Qt', 'E:\Qt', (Join-Path $env:USERPROFILE 'Qt'))) {
        $all += Find-QtKitsUnder $root
    }
    $all = @($all | Select-Object -Unique)
    $pick = Select-MsvcQtKit $all
    if ($pick) { return $pick }

    $downloaded = Install-MsvcQtKit
    if ($downloaded) { return $downloaded }

    $hint = 'none'
    if ($all.Count -gt 0) { $hint = ($all -join ', ') }
    Fail @"
No Qt MSVC kit. You currently have MinGW only (C:\Qt\6.11.1\mingw_64).
MinGW cannot be linked with this Visual Studio + OptiX build.

Do this once, then re-run BUILD_WINDOWS.bat:

1. Run C:\Qt\MaintenanceTool.exe
2. Choose Add or remove components
3. Open Qt -> Qt 6.11.1
4. Check MSVC 2022 64-bit (NOT MinGW)
5. Next / Update
6. Confirm folder exists: C:\Qt\6.11.1\msvc2022_64

Kits found: $hint
"@
}

function Find-CudaPath {
    if ($env:CUDA_PATH) {
        $nvcc = Join-Path $env:CUDA_PATH (Join-Path 'bin' 'nvcc.exe')
        if (Test-Path -LiteralPath $nvcc) { return $env:CUDA_PATH }
    }
    $base = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $base) {
        $vers = Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^v1[23]\.' } |
            Sort-Object Name -Descending
        foreach ($v in $vers) {
            $nvcc = Join-Path $v.FullName (Join-Path 'bin' 'nvcc.exe')
            if (Test-Path -LiteralPath $nvcc) { return $v.FullName }
        }
    }
    Fail 'CUDA Toolkit not found (need nvcc). Install CUDA 12.x from https://developer.nvidia.com/cuda-downloads then re-run BUILD_WINDOWS.bat'
}

function Find-OptiXRoot([string]$GitExe) {
    foreach ($envName in @('OptiX_ROOT', 'OptiX_INSTALL_DIR')) {
        $v = [Environment]::GetEnvironmentVariable($envName)
        if ($v) {
            $hdr = Join-Path $v (Join-Path 'include' 'optix.h')
            if (Test-Path -LiteralPath $hdr) { return $v }
        }
    }
    $nvidia = Join-Path $env:ProgramData 'NVIDIA Corporation'
    if (Test-Path -LiteralPath $nvidia) {
        $sdks = Get-ChildItem -LiteralPath $nvidia -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like 'OptiX SDK*' } |
            Sort-Object Name -Descending
        foreach ($s in $sdks) {
            $hdr = Join-Path $s.FullName (Join-Path 'include' 'optix.h')
            if (Test-Path -LiteralPath $hdr) { return $s.FullName }
        }
    }
    foreach ($p in @('C:\optix-dev', (Join-Path $Root 'third_party\optix-dev'))) {
        $hdr = Join-Path $p (Join-Path 'include' 'optix.h')
        if (Test-Path -LiteralPath $hdr) { return $p }
    }
    $local = Join-Path $env:LOCALAPPDATA 'grendizer-optix-dev'
    $hdrLocal = Join-Path $local (Join-Path 'include' 'optix.h')
    if (Test-Path -LiteralPath $hdrLocal) { return $local }
    Info 'OptiX SDK not found - cloning NVIDIA/optix-dev v7.7.0 headers ...'
    if (Test-Path -LiteralPath $local) { Remove-Item -LiteralPath $local -Recurse -Force }
    & $GitExe clone --depth 1 --branch v7.7.0 https://github.com/NVIDIA/optix-dev.git $local
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $hdrLocal)) {
        Fail 'Failed to fetch OptiX headers. Install OptiX SDK 7.7+ or check the network.'
    }
    return $local
}

function Find-VcpkgRoot([string]$GitExe) {
    if ($env:VCPKG_ROOT) {
        $tc = Join-Path $env:VCPKG_ROOT 'scripts\buildsystems\vcpkg.cmake'
        if (Test-Path -LiteralPath $tc) { return $env:VCPKG_ROOT }
    }
    foreach ($p in @(
        'C:\vcpkg',
        'D:\vcpkg',
        (Join-Path $Root 'vcpkg'),
        (Join-Path (Split-Path $Root -Parent) 'vcpkg')
    )) {
        $tc = Join-Path $p 'scripts\buildsystems\vcpkg.cmake'
        if (Test-Path -LiteralPath $tc) { return $p }
    }
    $local = Join-Path $env:LOCALAPPDATA 'grendizer-vcpkg'
    $tcLocal = Join-Path $local 'scripts\buildsystems\vcpkg.cmake'
    if (Test-Path -LiteralPath $tcLocal) { return $local }
    Info "vcpkg not found - cloning into $local (first run is slow) ..."
    & $GitExe clone --depth 1 https://github.com/microsoft/vcpkg.git $local
    if ($LASTEXITCODE -ne 0) { Fail 'Failed to clone vcpkg. Check network / GitHub.' }
    $boot = Join-Path $local 'bootstrap-vcpkg.bat'
    cmd.exe /c "`"$boot`" -disableMetrics"
    $vcpkgExe = Join-Path $local 'vcpkg.exe'
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $vcpkgExe)) {
        Fail 'bootstrap-vcpkg.bat failed'
    }
    return $local
}

Write-Host ''
Write-Host '=== Grendizer Render - Windows OptiX build ===' -ForegroundColor Green
Write-Host "Repo: $Root"
Write-Host 'First run can take a long time (vcpkg + nvcc PTX + Release).'
Write-Host ''

Import-VcVars64
$CMake = Find-CMake
$Git = Find-Git
$Qt = Find-QtPrefix
$Cuda = Find-CudaPath
$env:CUDA_PATH = $Cuda
$cudaBin = Join-Path $Cuda 'bin'
$env:PATH = "$cudaBin;$env:PATH"
$OptiX = Find-OptiXRoot $Git
$Vcpkg = Find-VcpkgRoot $Git
$Cl = (Get-Command cl.exe).Source
$Nvcc = Join-Path $cudaBin 'nvcc.exe'
if ($env:GRENDIZER_BUILD_DIR) {
    $BuildDir = $env:GRENDIZER_BUILD_DIR
} else {
    $BuildDir = Join-Path $Root 'build-windows'
}

Info "CMake:  $CMake"
Info "Qt:     $Qt"
Info "CUDA:   $Cuda"
Info "nvcc:   $Nvcc"
Info "cl.exe: $Cl"
Info "OptiX:  $OptiX"
Info "vcpkg:  $Vcpkg"
Info "Build:  $BuildDir"
Write-Host ''

if (-not (Test-Path -LiteralPath $Nvcc)) { Fail "nvcc.exe missing: $Nvcc" }

$Generator = 'Visual Studio 17 2022'
if ($script:VsYear -eq '2019') { $Generator = 'Visual Studio 16 2019' }
Info "CMake generator: $Generator"

$Toolchain = Join-Path $Vcpkg 'scripts\buildsystems\vcpkg.cmake'
& $CMake -S $Root -B $BuildDir -G $Generator -A x64 `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    "-DCMAKE_PREFIX_PATH=$Qt" `
    "-DCMAKE_CUDA_COMPILER=$Nvcc" `
    "-DCMAKE_CUDA_HOST_COMPILER=$Cl" `
    "-DSOLSTICE_CUDA_HOST_COMPILER=$Cl" `
    '-DSOLSTICE_ENABLE_OPTIX=ON' `
    "-DOptiX_ROOT=$OptiX" `
    '-DSOLSTICE_ENABLE_OCIO=ON' `
    '-DSOLSTICE_ENABLE_OPENVDB=ON' `
    '-DSOLSTICE_MODERN_CPU=ON'
if ($LASTEXITCODE -ne 0) { Fail 'cmake configure failed. Check Qt / CUDA / OptiX / vcpkg in the log above.' }

$Cfg = Join-Path $BuildDir 'generated\solstice_config.h'
if (-not (Test-Path -LiteralPath $Cfg)) {
    Fail "missing $Cfg after configure"
}
if (-not (Select-String -Path $Cfg -Pattern 'SOLSTICE_HAVE_OPTIX 1' -Quiet)) {
    Fail 'OptiX did not enable (SOLSTICE_HAVE_OPTIX != 1). Check nvcc and OptiX_ROOT in the cmake log.'
}
Info 'OptiX is compiled into this build (SOLSTICE_HAVE_OPTIX 1).'

Write-Host ''
Info 'Building Release (nvcc PTX can take several minutes) ...'
& $CMake --build $BuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) { Fail 'Build failed.' }

Write-Host ''
Info 'Deploying Qt DLLs next to the exe ...'
& $CMake --build $BuildDir --config Release --target deploy
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Warning: deploy target failed. The exe may need Qt on PATH.' -ForegroundColor Yellow
}

$Bin = Join-Path $BuildDir (Join-Path 'bin' 'Release')
if (-not (Test-Path -LiteralPath $Bin)) { $Bin = Join-Path $BuildDir 'bin' }
$cudartDir = Join-Path $Cuda 'bin'
if (Test-Path -LiteralPath $cudartDir) {
    $Cudart = Get-ChildItem -LiteralPath $cudartDir -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue
    if ($Cudart -and (Test-Path -LiteralPath $Bin)) {
        $Cudart | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $Bin -Force
            Info ("Copied " + $_.Name)
        }
    }
}

$Exe = Get-ChildItem -LiteralPath $Bin -Filter 'Grendizer_Render*.exe' -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $Exe) {
    $Exe = Get-ChildItem -LiteralPath $Bin -Filter 'Solstice*.exe' -ErrorAction SilentlyContinue |
        Select-Object -First 1
}
if (-not $Exe) { Fail "exe not found in $Bin" }

Write-Host ''
Write-Host ("DONE: " + $Exe.FullName) -ForegroundColor Green
Write-Host 'In the app: Engine -> Render Backend -> GPU (OptiX)'
Write-Host 'It must NOT say "not in this build". Needs an NVIDIA GPU + current driver.'
Write-Host ''

try { Invoke-Item $Bin } catch { }
exit 0
