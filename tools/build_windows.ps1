# One-click Windows Release build with OptiX.
# ASCII-only: Windows PowerShell 5.1 on a Russian locale mis-parses UTF-8
# without BOM and treats Cyrillic bytes as quotes / broken tokens.
# Override paths with env vars if auto-detect is wrong:
#   QT_ROOT, VCPKG_ROOT, CUDA_PATH, OptiX_ROOT, GRENDIZER_BUILD_DIR
$ErrorActionPreference = 'Stop'
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch { }

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
    if ($env:VCPKG_ROOT -and ($env:VCPKG_ROOT -match 'Visual Studio\\.*\\VC\\vcpkg')) {
        Info 'Ignoring Visual Studio bundled vcpkg (it requires builtin-baseline).'
        Remove-Item Env:\VCPKG_ROOT
    }
    # Never let a random C:\vcpkg toolchain hijack this build (hwloc fails on VS 2026).
    Remove-Item Env:\VCPKG_ROOT -ErrorAction SilentlyContinue
    if ($env:CMAKE_TOOLCHAIN_FILE -and ($env:CMAKE_TOOLCHAIN_FILE -match 'vcpkg')) {
        Remove-Item Env:\CMAKE_TOOLCHAIN_FILE
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
    $candidates = @()
    $base = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $base) {
        Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^v1[23]\.' } |
            ForEach-Object {
                $nvcc = Join-Path $_.FullName (Join-Path 'bin' 'nvcc.exe')
                if (Test-Path -LiteralPath $nvcc) { $candidates += $_.FullName }
            }
    }
    if ($env:CUDA_PATH) {
        $nvcc = Join-Path $env:CUDA_PATH (Join-Path 'bin' 'nvcc.exe')
        if (Test-Path -LiteralPath $nvcc) { $candidates += $env:CUDA_PATH }
    }
    $candidates = @($candidates | Select-Object -Unique | Sort-Object -Descending)
    if ($candidates.Count -gt 0) { return $candidates[0] }
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

function Invoke-GitClone([string]$Url, [string]$Branch, [string]$Dest) {
    if (Test-Path -LiteralPath $Dest) { return }
    New-Item -ItemType Directory -Force -Path (Split-Path $Dest -Parent) | Out-Null
    & $Git clone --depth 1 --branch $Branch $Url $Dest
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed: $Url" }
}

function Invoke-DepCMakeInstall([string]$Src, [string]$Name, [string[]]$Extra, [switch]$AllowFail) {
    $b = Join-Path $Src 'build'
    Info "Building $Name ..."
    $cfg = @(
        '-S', $Src, '-B', $b, '-G', 'Ninja',
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$script:DepsPrefix",
        "-DCMAKE_PREFIX_PATH=$script:DepsPrefix",
        "-DCMAKE_MAKE_PROGRAM=$Ninja",
        "-DCMAKE_C_COMPILER=$Cl",
        "-DCMAKE_CXX_COMPILER=$Cl"
    ) + $Extra
    & $CMake @cfg
    if ($LASTEXITCODE -ne 0) {
        if ($AllowFail) {
            Write-Host "Warning: $Name configure failed (optional)." -ForegroundColor Yellow
            return
        }
        Fail "$Name cmake configure failed"
    }
    & $CMake --build $b --target install --parallel
    if ($LASTEXITCODE -ne 0) {
        if ($AllowFail) {
            Write-Host "Warning: $Name install failed (optional)." -ForegroundColor Yellow
            return
        }
        Fail "$Name install failed"
    }
}

function Find-EmbreeZip {
    $name = 'embree-4.4.0.x64.windows.zip'
    $dirs = @(
        $script:DepsPrefix,
        (Join-Path $Root 'deps-cache'),
        'C:\grendizer-deps',
        (Join-Path $env:USERPROFILE 'Downloads'),
        $env:TEMP
    )
    if ($env:GRENDIZER_DEPS) { $dirs = @($env:GRENDIZER_DEPS) + $dirs }
    foreach ($d in $dirs) {
        if (-not $d) { continue }
        $p = Join-Path $d $name
        if (Test-Path -LiteralPath $p) { return $p }
    }
    return $null
}

function Ensure-NativeDeps {
    if ($env:GRENDIZER_DEPS) {
        $script:DepsPrefix = $env:GRENDIZER_DEPS
    } else {
        $script:DepsPrefix = Join-Path $env:LOCALAPPDATA 'grendizer-deps'
    }
    $stamp = Join-Path $script:DepsPrefix 'stamp-native-1.txt'
    $srcRoot = Join-Path $env:LOCALAPPDATA 'grendizer-deps-src'
    if ($env:GRENDIZER_DEPS_SRC) { $srcRoot = $env:GRENDIZER_DEPS_SRC }
    New-Item -ItemType Directory -Force -Path $script:DepsPrefix | Out-Null
    New-Item -ItemType Directory -Force -Path $srcRoot | Out-Null

    $embreeCmake = Get-ChildItem -Path $script:DepsPrefix -Recurse -Filter 'embree-config.cmake' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $embreeCmake) {
        $embreeCmake = Get-ChildItem -Path $script:DepsPrefix -Recurse -Filter 'EmbreeConfig.cmake' -ErrorAction SilentlyContinue |
            Select-Object -First 1
    }
    if (-not $embreeCmake) {
        $zip = Find-EmbreeZip
        if ($zip) {
            Info "Using local Embree zip: $zip"
        } else {
            Info 'Downloading Embree 4.4.0 Windows zip (one-time, then reused) ...'
            $zip = Join-Path $script:DepsPrefix 'embree-4.4.0.x64.windows.zip'
            $url = 'https://github.com/RenderKit/embree/releases/download/v4.4.0/embree-4.4.0.x64.windows.zip'
            $oldPref = $ProgressPreference
            $ProgressPreference = 'SilentlyContinue'
            try {
                Invoke-WebRequest -Uri $url -OutFile $zip
            } finally {
                $ProgressPreference = $oldPref
            }
        }
        $embreeDest = Join-Path $script:DepsPrefix 'embree'
        if (Test-Path -LiteralPath $embreeDest) { Remove-Item -LiteralPath $embreeDest -Recurse -Force }
        Expand-Archive -Path $zip -DestinationPath $embreeDest -Force
        $embreeCmake = Get-ChildItem -Path $embreeDest -Recurse -Filter 'embree-config.cmake' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $embreeCmake) {
            Fail 'Embree cmake config missing after unzip'
        }
    }

    if (Test-Path -LiteralPath $stamp) {
        Info "Native deps already installed: $script:DepsPrefix"
        return
    }

    Invoke-GitClone 'https://github.com/AcademySoftwareFoundation/Imath.git' 'v3.1.12' (Join-Path $srcRoot 'imath')
    Invoke-DepCMakeInstall (Join-Path $srcRoot 'imath') 'Imath' @(
        '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTING=OFF'
    )

    Invoke-GitClone 'https://github.com/AcademySoftwareFoundation/openexr.git' 'v3.2.4' (Join-Path $srcRoot 'openexr')
    Invoke-DepCMakeInstall (Join-Path $srcRoot 'openexr') 'OpenEXR' @(
        '-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTING=OFF',
        '-DOPENEXR_BUILD_TOOLS=OFF', '-DOPENEXR_INSTALL_EXAMPLES=OFF',
        '-DOPENEXR_FORCE_INTERNAL_DEFLATE=ON'
    )

    Invoke-GitClone 'https://github.com/alembic/alembic.git' '1.8.6' (Join-Path $srcRoot 'alembic')
    Invoke-DepCMakeInstall (Join-Path $srcRoot 'alembic') 'Alembic' @(
        '-DUSE_HDF5=OFF', '-DALEMBIC_SHARED_LIBS=OFF', '-DUSE_TESTS=OFF',
        '-DUSE_BINARIES=OFF', '-DUSE_EXAMPLES=OFF', '-DALEMBIC_BUILD_LIBS=ON'
    )

    Invoke-GitClone 'https://github.com/oneapi-src/oneTBB.git' 'v2021.12.0' (Join-Path $srcRoot 'onetbb')
    Invoke-DepCMakeInstall (Join-Path $srcRoot 'onetbb') 'oneTBB' @(
        '-DTBB_TEST=OFF', '-DTBB_STRICT=OFF', '-DBUILD_SHARED_LIBS=ON'
    )

    Invoke-GitClone 'https://github.com/AcademySoftwareFoundation/openvdb.git' 'v12.0.1' (Join-Path $srcRoot 'openvdb')
    Invoke-DepCMakeInstall (Join-Path $srcRoot 'openvdb') 'OpenVDB' @(
        "-DTBB_ROOT=$script:DepsPrefix",
        '-DOPENVDB_BUILD_CORE=ON', '-DOPENVDB_BUILD_BINARIES=OFF',
        '-DOPENVDB_BUILD_VDB_PRINT=OFF', '-DOPENVDB_BUILD_VDB_LOD=OFF',
        '-DOPENVDB_BUILD_VDB_RENDER=OFF', '-DOPENVDB_BUILD_VDB_VIEW=OFF',
        '-DOPENVDB_BUILD_PYTHON_MODULE=OFF', '-DOPENVDB_BUILD_UNITTESTS=OFF',
        '-DOPENVDB_BUILD_NANOVDB=OFF', '-DOPENVDB_CORE_SHARED=ON',
        '-DOPENVDB_CORE_STATIC=OFF', '-DUSE_BLOSC=OFF', '-DUSE_ZLIB=OFF',
        '-DUSE_EXR=OFF', '-DUSE_IMATH_HALF=OFF',
        '-DOPENVDB_USE_DELAYED_LOADING=OFF'
    )

    Invoke-GitClone 'https://github.com/AcademySoftwareFoundation/OpenColorIO.git' 'v2.3.2' (Join-Path $srcRoot 'ocio')
    Invoke-DepCMakeInstall (Join-Path $srcRoot 'ocio') 'OpenColorIO' @(
            '-DBUILD_SHARED_LIBS=ON', '-DOCIO_BUILD_APPS=OFF', '-DOCIO_BUILD_TESTS=OFF',
            '-DOCIO_BUILD_GPU_TESTS=OFF', '-DOCIO_BUILD_PYTHON=OFF', '-DOCIO_BUILD_JAVA=OFF',
            '-DOCIO_BUILD_DOCS=OFF', '-DOCIO_INSTALL_EXT_PACKAGES=ALL'
        ) -AllowFail

    Set-Content -Path $stamp -Value 'ok' -Encoding ascii
    Info "Native deps installed: $script:DepsPrefix"
}

function Resolve-EmbreePrefix {
    $f = Get-ChildItem -Path $script:DepsPrefix -Recurse -Filter 'embree-config.cmake' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $f) {
        $f = Get-ChildItem -Path $script:DepsPrefix -Recurse -Filter 'EmbreeConfig.cmake' -ErrorAction SilentlyContinue |
            Select-Object -First 1
    }
    if (-not $f) { Fail 'embree-config.cmake not found under deps' }
    return $f.Directory.Parent.Parent.Parent.FullName
}

function Find-Ninja {
    $cmd = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $vswhere = Find-VsWhere
    if ($vswhere) {
        $fromVs = & $vswhere -latest -products * -find 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe' |
            Select-Object -First 1
        if ($fromVs -and (Test-Path -LiteralPath $fromVs)) { return $fromVs }
        $fromVs2 = & $vswhere -latest -products * -find '**/ninja.exe' |
            Select-Object -First 1
        if ($fromVs2 -and (Test-Path -LiteralPath $fromVs2)) { return $fromVs2 }
    }
    $local = Join-Path $env:LOCALAPPDATA 'grendizer-ninja'
    $exe = Join-Path $local 'ninja.exe'
    if (Test-Path -LiteralPath $exe) { return $exe }
    Info 'Ninja not found - downloading ninja-win.zip ...'
    New-Item -ItemType Directory -Force -Path $local | Out-Null
    $zip = Join-Path $env:TEMP 'grendizer-ninja-win.zip'
    $url = 'https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-win.zip'
    Invoke-WebRequest -Uri $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $local -Force
    if (-not (Test-Path -LiteralPath $exe)) { Fail 'Failed to download ninja.exe' }
    return $exe
}

Write-Host ''
Write-Host '=== Grendizer Render - Windows OptiX build ===' -ForegroundColor Green
Write-Host "Repo: $Root"
Write-Host 'First run can take a long time (Embree zip + Imath/OpenEXR/Alembic/TBB/OpenVDB + nvcc).'
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
$Ninja = Find-Ninja
$ninjaDir = Split-Path $Ninja -Parent
$env:PATH = "$ninjaDir;$env:PATH"
$Cl = (Get-Command cl.exe).Source
$Nvcc = Join-Path $cudaBin 'nvcc.exe'
Ensure-NativeDeps
$embreeRoot = Resolve-EmbreePrefix
if ($env:GRENDIZER_BUILD_DIR) {
    $BuildDir = $env:GRENDIZER_BUILD_DIR
} else {
    $BuildDir = Join-Path $Root 'build-windows'
}

$Prefix = "$Qt;$script:DepsPrefix;$embreeRoot"

Info "CMake:  $CMake"
Info "Qt:     $Qt"
Info "CUDA:   $Cuda"
Info "nvcc:   $Nvcc"
Info "cl.exe: $Cl"
Info "OptiX:  $OptiX"
Info "Deps:   $script:DepsPrefix"
Info "Ninja:  $Ninja"
Info "Build:  $BuildDir"
Write-Host ''

if (-not (Test-Path -LiteralPath $Nvcc)) { Fail "nvcc.exe missing: $Nvcc" }

# Ninja + cl.exe works on VS 2022 and VS 2026. The VS 17 generator cannot see VS 18.
$Generator = 'Ninja'
Info "CMake generator: $Generator (cl.exe from VS $script:VsYear)"

$cache = Join-Path $BuildDir 'CMakeCache.txt'
if (Test-Path -LiteralPath $cache) {
    $stale = $false
    if (Select-String -Path $cache -Pattern 'vcpkg' -Quiet) { $stale = $true }
    $oldGen = Select-String -Path $cache -Pattern '^CMAKE_GENERATOR:INTERNAL=(.+)$' | Select-Object -First 1
    if ($oldGen -and $oldGen.Matches[0].Groups[1].Value -ne $Generator) { $stale = $true }
    $errLog = Join-Path $BuildDir 'CMakeFiles\CMakeError.log'
    if (Test-Path -LiteralPath $errLog) { $stale = $true }
    if ($stale) {
        Info "Clearing $BuildDir (old vcpkg / generator cache)"
        Remove-Item -LiteralPath $BuildDir -Recurse -Force
    }
}

$env:NVCC_PREPEND_FLAGS = '--allow-unsupported-compiler'
$env:NVCC_APPEND_FLAGS = '--allow-unsupported-compiler'

& $CMake -S $Root -B $BuildDir -G $Generator `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DCMAKE_MAKE_PROGRAM=$Ninja" `
    "-DCMAKE_C_COMPILER=$Cl" `
    "-DCMAKE_CXX_COMPILER=$Cl" `
    "-DCMAKE_PREFIX_PATH=$Prefix" `
    "-DCMAKE_CUDA_COMPILER=$Nvcc" `
    "-DCMAKE_CUDA_HOST_COMPILER=$Cl" `
    "-DCMAKE_CUDA_COMPILER_ID=NVIDIA" `
    "-DCMAKE_CUDA_COMPILER_FORCED=TRUE" `
    "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler" `
    "-DCMAKE_CUDA_COMPILER_ID_FLAGS=--allow-unsupported-compiler" `
    "-DSOLSTICE_CUDA_HOST_COMPILER=$Cl" `
    '-DSOLSTICE_ENABLE_OPTIX=ON' `
    "-DOptiX_ROOT=$OptiX" `
    '-DSOLSTICE_ENABLE_OCIO=ON' `
    '-DSOLSTICE_ENABLE_OPENVDB=ON' `
    '-DSOLSTICE_MODERN_CPU=ON' `
    '-DSOLSTICE_BUILD_TX_TOOLS_ALPHA=OFF' `
    '-DSOLSTICE_BUILD_TX_TOOLS_OMEGA=OFF'
if ($LASTEXITCODE -ne 0) { Fail 'cmake configure failed. Check Qt / CUDA / OptiX / deps in the log above.' }

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
& $CMake --build $BuildDir --parallel
if ($LASTEXITCODE -ne 0) { Fail 'Build failed.' }

Write-Host ''
Info 'Deploying Qt DLLs next to the exe ...'
& $CMake --build $BuildDir --target deploy
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
if ($script:DepsPrefix -and (Test-Path -LiteralPath $Bin)) {
    foreach ($dir in @(
        (Join-Path $script:DepsPrefix 'bin'),
        (Join-Path $script:DepsPrefix 'embree\bin'),
        (Join-Path $script:DepsPrefix 'embree')
    )) {
        if (-not (Test-Path -LiteralPath $dir)) { continue }
        foreach ($pat in @('embree*.dll', 'tbb*.dll', 'tbbmalloc*.dll', 'openvdb*.dll', 'OpenColorIO*.dll')) {
            Get-ChildItem -LiteralPath $dir -Filter $pat -ErrorAction SilentlyContinue | ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination $Bin -Force
                Info ("Copied " + $_.Name)
            }
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
