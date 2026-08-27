# One-click Windows Release build with OptiX.
# ASCII-only: Windows PowerShell 5.1 on a Russian locale mis-parses UTF-8
# without BOM and treats Cyrillic bytes as quotes / broken tokens.
# Override paths with env vars if auto-detect is wrong:
#   QT_ROOT, CUDA_PATH, OptiX_ROOT, GRENDIZER_BUILD_DIR, GRENDIZER_DEPS
# FULL compiles the OpenColorIO library from source. OCIO env / Film path is
# the .ocio colour config, not the SDK.
# Default build dir is C:\gz-build (GitHub zip under Downloads is too long for MSVC).
# CUDA 13.2+ is required for Visual Studio 2026 (MSVC 14.50+). CUDA 12.x is
# enough only with VS 2022 / MSVC 14.44. If both are installed, 13.2 is used.
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

# Grendizer_Render keeps tbb12.dll / embree loaded. A full rebuild already
# wrote the new exe; failing Copy-Item after that is a false BUILD FAILED.
function Copy-RuntimeFile([string]$Src, [string]$DestDir) {
    $name = [IO.Path]::GetFileName($Src)
    $dest = Join-Path $DestDir $name
    try {
        Copy-Item -LiteralPath $Src -Destination $DestDir -Force -ErrorAction Stop
        Info ("Copied " + $name)
    } catch {
        if (Test-Path -LiteralPath $dest) {
            Write-Host ("Warning: " + $name + " is in use (close Grendizer_Render / TX Tools). Left the existing DLL.") -ForegroundColor Yellow
        } else {
            Fail ("Could not copy " + $name + " to " + $DestDir + ": " + $_.Exception.Message)
        }
    }
}

function Find-VsWhere {
    $pf86 = ${env:ProgramFiles(x86)}
    $p = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $p) { return $p }
    return $null
}

function Get-MsvcToolsetDirFromCl([string]$ClPath) {
    if (-not $ClPath) { return $null }
    return [IO.Path]::GetFullPath((Join-Path $ClPath '..\..\..\..'))
}

function Find-MsvcToolsets {
    $hits = @()
    $bases = @(
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio')
    )
    if (${env:ProgramFiles(x86)}) {
        $bases += (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio')
    }
    foreach ($base in $bases) {
        if (-not (Test-Path -LiteralPath $base)) { continue }
        foreach ($year in (Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue)) {
            foreach ($ed in (Get-ChildItem -LiteralPath $year.FullName -Directory -ErrorAction SilentlyContinue)) {
                $msvcRoot = Join-Path $ed.FullName 'VC\Tools\MSVC'
                if (-not (Test-Path -LiteralPath $msvcRoot)) { continue }
                foreach ($ts in (Get-ChildItem -LiteralPath $msvcRoot -Directory -ErrorAction SilentlyContinue)) {
                    $cl = Join-Path $ts.FullName 'bin\Hostx64\x64\cl.exe'
                    if (-not (Test-Path -LiteralPath $cl)) { continue }
                    $ver = $null
                    try { $ver = [version]$ts.Name } catch { continue }
                    $hits += [pscustomobject]@{
                        Version = $ver
                        Dir     = $ts.FullName
                        Cl      = $cl
                        VsRoot  = $ed.FullName
                    }
                }
            }
        }
    }
    return $hits
}

function Find-CudaFriendlyMsvcToolset {
    $hits = @(Find-MsvcToolsets)
    $friendly = @($hits | Where-Object { $_.Version -lt [version]'14.50' } | Sort-Object Version -Descending)
    if ($friendly.Count -gt 0) { return $friendly[0] }
    return $null
}

function Switch-MsvcToolsetDir([string]$FromDir, [string]$ToDir) {
    $from = $FromDir.TrimEnd('\')
    $to = $ToDir.TrimEnd('\')
    if (-not $from -or -not $to -or ($from -eq $to)) { return }
    foreach ($var in @('INCLUDE', 'LIB', 'LIBPATH', 'PATH', 'Path')) {
        $val = [Environment]::GetEnvironmentVariable($var)
        if ($val -and $val.Contains($from)) {
            Set-Item -LiteralPath ("Env:" + $var) -Value ($val.Replace($from, $to))
        }
    }
    $hostBin = Join-Path $to 'bin\Hostx64\x64'
    $env:PATH = "$hostBin;$env:PATH"
    $env:VCToolsInstallDir = $to + '\'
}

function Find-VsInstallRoot {
    # CUDA 12.0 can parse MSVC 14.44 / VS 2022 headers, not VS 2026 14.51 STL.
    $friendly = Find-CudaFriendlyMsvcToolset
    if ($friendly) { return $friendly.VsRoot }

    foreach ($year in @('2022', '2019')) {
        foreach ($ed in @('Community', 'Professional', 'Enterprise', 'BuildTools')) {
            $root = Join-Path $env:ProgramFiles "Microsoft Visual Studio\$year\$ed"
            $vcvars = Join-Path $root 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path -LiteralPath $vcvars) { return $root }
        }
    }
    $vswhere = Find-VsWhere
    if ($vswhere) {
        $vsRoot = & $vswhere -latest -version "[17.0,18.0)" -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        if ($vsRoot) { return ("$vsRoot").Trim() }
        $vsRoot = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        if ($vsRoot) { return ("$vsRoot").Trim() }
    }
    return $null
}

function Import-EnvFromVcVars([string]$Vcvars, [string]$VcvarsVer) {
    $extra = ''
    if ($VcvarsVer) { $extra = " -vcvars_ver=$VcvarsVer" }
    $tmp = [IO.Path]::GetTempFileName()
    cmd.exe /c "`"$Vcvars`"$extra >nul && set" | Set-Content -Path $tmp -Encoding ascii
    Get-Content -Path $tmp | ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') {
            Set-Item -LiteralPath ("Env:" + $matches[1]) -Value $matches[2]
        }
    }
    Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
}

function Get-ClToolsetVersion {
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if (-not $cl) { return $null }
    if ($cl.Source -match '\\MSVC\\(14\.\d+)') { return $matches[1] }
    return $null
}

function Import-VcVars64 {
    $vsRoot = Find-VsInstallRoot
    if (-not $vsRoot) {
        Fail 'Visual Studio with C++ is not installed. Install VS 2022 Desktop development with C++.'
    }
    $vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path -LiteralPath $vcvars)) {
        Fail "vcvars64.bat missing under $vsRoot"
    }
    $script:VsYear = 'unknown'
    if ($vsRoot -match '\\2022\\') { $script:VsYear = '2022' }
    elseif ($vsRoot -match '\\2019\\') { $script:VsYear = '2019' }
    elseif ($vsRoot -match '\\18\\') { $script:VsYear = '2026' }

    Info "Visual Studio: $vsRoot"
    Import-EnvFromVcVars $vcvars $null
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        Fail 'cl.exe not on PATH after vcvars64'
    }

    $toolset = Get-ClToolsetVersion
    Info ("cl.exe toolset: " + $toolset + " (" + (Get-Command cl.exe).Source + ")")

    # Side-by-side VS 2022 toolset (14.4x) can host CUDA 12. Skip if none is installed.
    if ($toolset -and ([version]$toolset -ge [version]'14.50')) {
        foreach ($ver in @('14.44', '14.43', '14.42', '14.41', '14.40', '14.39', '14.38')) {
            Info "Trying MSVC $ver with vcvars (CUDA 12 compatible) ..."
            try {
                Import-EnvFromVcVars $vcvars $ver
            } catch { continue }
            $toolset = Get-ClToolsetVersion
            if ($toolset -and ([version]$toolset -lt [version]'14.50')) {
                Info "Pinned MSVC toolset $toolset"
                break
            }
        }
    }

    $toolset = Get-ClToolsetVersion
    if ($toolset -and ([version]$toolset -ge [version]'14.50')) {
        $friendly = Find-CudaFriendlyMsvcToolset
        if ($friendly) {
            $fromDir = Get-MsvcToolsetDirFromCl ((Get-Command cl.exe).Source)
            Info ("Switching host MSVC " + $toolset + " -> " + $friendly.Version)
            Switch-MsvcToolsetDir $fromDir $friendly.Dir
            $toolset = Get-ClToolsetVersion
        }
    }

    Info ("nvcc host cl.exe: " + (Get-Command cl.exe).Source)
    $script:MsvcToolset = $toolset

    Remove-Item Env:\VCPKG_ROOT -ErrorAction SilentlyContinue
    if ($env:CMAKE_TOOLCHAIN_FILE -and ($env:CMAKE_TOOLCHAIN_FILE -match 'vcpkg')) {
        Remove-Item Env:\CMAKE_TOOLCHAIN_FILE
    }
}

function Get-NvccRelease([string]$NvccPath) {
    $out = & $NvccPath --version 2>&1 | Out-String
    if ($out -match 'release\s+(\d+)\.(\d+)') {
        return [version]($matches[1] + '.' + $matches[2])
    }
    return [version]'0.0'
}

function Test-NvccHostStl([string]$Nvcc, [string]$Cl, [string]$Arch) {
    $dir = Join-Path $env:TEMP 'grendizer-nvcc-probe'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $cu = Join-Path $dir 'probe.cu'
    $ptx = Join-Path $dir 'probe.ptx'
    Set-Content -LiteralPath $cu -Encoding ascii -Value @"
#include <type_traits>
__device__ int probe() { return (int)sizeof(std::integral_constant<int, 1>::value); }
"@
    if (Test-Path -LiteralPath $ptx) { Remove-Item -LiteralPath $ptx -Force }
    Info "Probing nvcc vs MSVC STL (must pass before the OptiX kernel build) ..."
    & $Nvcc -ptx -std=c++17 --allow-unsupported-compiler `
        -ccbin $Cl "-arch=$Arch" `
        -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH `
        -D_ENABLE_EXTENDED_ALIGNED_STORAGE `
        -DNOMINMAX `
        -o $ptx $cu
    return (($LASTEXITCODE -eq 0) -and (Test-Path -LiteralPath $ptx))
}

function Add-CudaCandidate {
    param(
        [System.Collections.Generic.List[object]]$List,
        [hashtable]$Seen,
        [string]$Root
    )
    if ([string]::IsNullOrWhiteSpace($Root)) { return }
    if (-not (Test-Path -LiteralPath $Root)) { return }
    $full = $null
    try { $full = [IO.Path]::GetFullPath($Root) } catch { return }
    if (-not $full) { return }
    if ($Seen.ContainsKey($full)) { return }
    $nvcc = Join-Path $full 'bin\nvcc.exe'
    if (-not (Test-Path -LiteralPath $nvcc)) { return }
    $Seen[$full] = $true
    $ver = Get-NvccRelease $nvcc
    Info ("Found CUDA " + $ver.ToString() + "  " + $full)
    [void]$List.Add([pscustomobject]@{ Path = $full; Version = $ver; Nvcc = $nvcc })
}

function Resolve-CudaForOptix {
    $script:OptixArch = 'compute_60'
    $script:CudaRelease = [version]'0.0'
    $script:CudaRoot = $null

    $list = New-Object 'System.Collections.Generic.List[object]'
    $seen = @{}

    $base = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $base) {
        foreach ($d in (Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue)) {
            Add-CudaCandidate $list $seen $d.FullName
        }
    }
    Add-CudaCandidate $list $seen $env:CUDA_PATH
    Add-CudaCandidate $list $seen $env:CUDA_PATH_V13_2
    Add-CudaCandidate $list $seen $env:CUDA_PATH_V13_1
    Add-CudaCandidate $list $seen (Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA\v13.2')

    $pickPath = $null
    $pickVer = [version]'0.0'
    $i = 0
    while ($i -lt $list.Count) {
        $item = $list[$i]
        $i = $i + 1
        $v = [version]$item.Version
        $p = [string]$item.Path
        if ($v -ge [version]'13.2') {
            if (($pickVer -lt [version]'13.2') -or ($v -gt $pickVer)) {
                $pickVer = $v
                $pickPath = $p
            }
        }
    }
    if (-not $pickPath) {
        $i = 0
        while ($i -lt $list.Count) {
            $item = $list[$i]
            $i = $i + 1
            $v = [version]$item.Version
            $p = [string]$item.Path
            if ($v -gt $pickVer) {
                $pickVer = $v
                $pickPath = $p
            }
        }
    }

    if (-not $pickPath) {
        Fail @"
CUDA Toolkit not found (nvcc.exe).
Visual Studio 2026 needs CUDA 13.2 at:
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
Download: https://developer.nvidia.com/cuda-downloads
Close this window, open a NEW cmd, then re-run BUILD_WINDOWS.bat.
"@
    }

    $toolset = Get-ClToolsetVersion
    $toolsetVer = [version]'0.0'
    if ($toolset) { $toolsetVer = [version]$toolset }
    if (($toolsetVer -ge [version]'14.50') -and ($pickVer -lt [version]'13.2')) {
        Fail @"
MSVC $toolset (VS 2026) cannot compile OptiX PTX with CUDA $($pickVer.ToString()).
Picked: $pickPath
Need:   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe
Do NOT delete %LOCALAPPDATA%\grendizer-deps
"@
    }

    $script:CudaRoot = $pickPath
    $script:CudaRelease = $pickVer
    if ($pickVer -ge [version]'13.0') {
        $script:OptixArch = 'compute_75'
    } else {
        $script:OptixArch = 'compute_60'
    }
    Info ("Using CUDA " + $pickVer.ToString() + " at " + $pickPath)
    Info ("OptiX PTX arch: " + $script:OptixArch)
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
    $cm = Join-Path $Dest 'CMakeLists.txt'
    if (Test-Path -LiteralPath $cm) { return }
    if (Test-Path -LiteralPath $Dest) { Remove-Item -LiteralPath $Dest -Recurse -Force }
    New-Item -ItemType Directory -Force -Path (Split-Path $Dest -Parent) | Out-Null
    & $Git clone --depth 1 --branch $Branch $Url $Dest
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $cm)) { Fail "git clone failed: $Url" }
}

function Test-OcioSafeToRecurse([string]$Prefix) {
    try { $full = [IO.Path]::GetFullPath($Prefix) } catch { return $false }
    $norm = $full.TrimEnd('\', '/')
    foreach ($b in @(
        $env:SystemRoot, 'C:\Windows', 'C:\Windows\System32',
        $env:ProgramFiles, ${env:ProgramFiles(x86)}, 'C:\'
    )) {
        if (-not $b) { continue }
        try {
            $bn = [IO.Path]::GetFullPath($b).TrimEnd('\', '/')
            if ($norm -ieq $bn) { return $false }
        } catch { }
    }
    $leaf = Split-Path $norm -Leaf
    if ($leaf -match '(?i)ocio|opencolorio|grendizer|deps') { return $true }
    foreach ($sub in @('include', 'lib', 'lib64', 'cmake', 'share')) {
        if (Test-Path -LiteralPath (Join-Path $norm $sub)) { return $true }
    }
    return $false
}

function Find-OcioConfigCMake([string]$Prefix) {
    if (-not $Prefix -or -not (Test-Path -LiteralPath $Prefix)) { return $null }
    foreach ($rel in @(
        'lib\cmake\OpenColorIO\OpenColorIOConfig.cmake',
        'lib64\cmake\OpenColorIO\OpenColorIOConfig.cmake',
        'cmake\OpenColorIO\OpenColorIOConfig.cmake',
        'share\OpenColorIO\OpenColorIOConfig.cmake',
        'share\opencolorio\OpenColorIOConfig.cmake',
        'OpenColorIOConfig.cmake'
    )) {
        $p = Join-Path $Prefix $rel
        if (Test-Path -LiteralPath $p) { return Get-Item -LiteralPath $p }
    }
    if (-not (Test-OcioSafeToRecurse $Prefix)) { return $null }
    return Get-ChildItem -LiteralPath $Prefix -Recurse -Depth 6 -Filter 'OpenColorIOConfig.cmake' -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Test-OcioInstallPrefix([string]$root) {
    if (-not $root) { return $null }
    $root = $root.Trim().TrimEnd('\', '/')
    if (-not (Test-Path -LiteralPath $root)) { return $null }
    if (Test-Path -LiteralPath $root -PathType Leaf) {
        $name = [IO.Path]::GetFileName($root)
        if ($name -eq 'OpenColorIOConfig.cmake') {
            $root = Split-Path $root
        } else {
            return $null
        }
    }
    $cfg = Find-OcioConfigCMake $root
    $hdr = Join-Path $root 'include\OpenColorIO\OpenColorIO.h'
    $hasHdr = Test-Path -LiteralPath $hdr
    if (-not $hasHdr -and $cfg) {
        $walk = $cfg.Directory.FullName
        for ($i = 0; $i -lt 6; $i++) {
            $try = Join-Path $walk 'include\OpenColorIO\OpenColorIO.h'
            if (Test-Path -LiteralPath $try) {
                $root = $walk
                $hasHdr = $true
                break
            }
            $parent = Split-Path $walk
            if (-not $parent -or $parent -eq $walk) { break }
            $walk = $parent
        }
    }
    $lib = $null
    foreach ($libRel in @('lib', 'lib64')) {
        $libDir = Join-Path $root $libRel
        if (-not (Test-Path -LiteralPath $libDir)) { continue }
        $lib = Get-ChildItem -LiteralPath $libDir -Filter 'OpenColorIO*.lib' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($lib) { break }
    }
    if (-not $lib -and $cfg) {
        $walk = $cfg.Directory.FullName
        for ($i = 0; $i -lt 6; $i++) {
            foreach ($libRel in @('lib', 'lib64')) {
                $libDir = Join-Path $walk $libRel
                if (-not (Test-Path -LiteralPath $libDir)) { continue }
                $lib = Get-ChildItem -LiteralPath $libDir -Filter 'OpenColorIO*.lib' -ErrorAction SilentlyContinue |
                    Select-Object -First 1
                if ($lib) { break }
            }
            if ($lib) { break }
            $parent = Split-Path $walk
            if (-not $parent -or $parent -eq $walk) { break }
            $walk = $parent
        }
    }
    # Need a real linkable install. A stray CMake file from a failed tree is not enough.
    if (-not (($hasHdr -and $lib) -or ($cfg -and $hasHdr) -or ($cfg -and $lib))) { return $null }
    $dll = $null
    foreach ($binRel in @('bin', 'lib')) {
        $binDir = Join-Path $root $binRel
        if (-not (Test-Path -LiteralPath $binDir)) { continue }
        $dll = Get-ChildItem -LiteralPath $binDir -Filter 'OpenColorIO*.dll' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($dll) { break }
    }
    return [pscustomobject]@{
        Prefix   = $root
        CMakeDir = $(if ($cfg) { $cfg.Directory.FullName } else { $null })
        Include  = $(if ($hasHdr) { Join-Path $root 'include' } else { $null })
        Library  = $(if ($lib) { $lib.FullName } else { $null })
        Dll      = $(if ($dll) { $dll.FullName } else { $null })
    }
}

function Find-SystemOpenColorIO {
    $candidates = New-Object System.Collections.Generic.List[string]
    # Prefer an explicit SDK prefix. Do not treat OCIO=config.ocio as a library root.
    foreach ($e in @('OpenColorIO_DIR', 'OpenColorIO_ROOT', 'OCIO_ROOT', 'OCIO_HOME')) {
        $v = [Environment]::GetEnvironmentVariable($e)
        if ($v) { [void]$candidates.Add($v) }
    }
    foreach ($pf in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if (-not $pf) { continue }
        [void]$candidates.Add((Join-Path $pf 'OpenColorIO'))
        [void]$candidates.Add((Join-Path $pf 'Academy Software Foundation\OpenColorIO'))
        Get-ChildItem -LiteralPath $pf -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match 'OpenColorIO|OCIO' } |
            ForEach-Object { [void]$candidates.Add($_.FullName) }
    }
    if ($env:VCPKG_ROOT) {
        [void]$candidates.Add((Join-Path $env:VCPKG_ROOT 'installed\x64-windows'))
        [void]$candidates.Add((Join-Path $env:VCPKG_ROOT 'installed\x64-windows-release'))
    }
    if ($env:CONDA_PREFIX) { [void]$candidates.Add($env:CONDA_PREFIX) }
    foreach ($extra in @('C:\ocio', 'C:\OpenColorIO', (Join-Path $env:LOCALAPPDATA 'OpenColorIO'))) {
        [void]$candidates.Add($extra)
    }
    if ($env:PATH) {
        foreach ($dir in ($env:PATH -split ';')) {
            if (-not $dir) { continue }
            $dll = Get-ChildItem -LiteralPath $dir -Filter 'OpenColorIO*.dll' -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($dll) {
                [void]$candidates.Add($dir)
                $parent = Split-Path $dir
                if ($parent) { [void]$candidates.Add($parent) }
            }
        }
    }
    if ($script:DepsPrefix) { [void]$candidates.Add($script:DepsPrefix) }
    $seen = @{}
    foreach ($c in $candidates) {
        if (-not $c) { continue }
        $key = $c.ToLowerInvariant()
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true
        $hit = Test-OcioInstallPrefix $c
        if ($hit) { return $hit }
    }
    return $null
}

function Get-VsGeneratorArgs {
    # Prefer the VS year we already imported vcvars from, then cmake --help.
    # OCIO MUST use a VS generator (CI). Ninja nested ExternalProject breaks minizip-ng.
    if ($script:VsYear -eq '2026') {
        return @('-G', 'Visual Studio 18 2026', '-A', 'x64')
    }
    if ($script:VsYear -eq '2022') {
        return @('-G', 'Visual Studio 17 2022', '-A', 'x64')
    }
    if ($script:VsYear -eq '2019') {
        return @('-G', 'Visual Studio 16 2019', '-A', 'x64')
    }
    $help = & $CMake --help 2>&1 | Out-String
    if ($help -match 'Visual Studio 18 2026') { return @('-G', 'Visual Studio 18 2026', '-A', 'x64') }
    if ($help -match 'Visual Studio 17 2022') { return @('-G', 'Visual Studio 17 2022', '-A', 'x64') }
    if ($help -match 'Visual Studio 16 2019') { return @('-G', 'Visual Studio 16 2019', '-A', 'x64') }
    return @()
}

function Test-ZlibInDeps {
    $hdr = Join-Path $script:DepsPrefix 'include\zlib.h'
    if (-not (Test-Path -LiteralPath $hdr)) { return $false }
    foreach ($libRel in @('lib', 'lib64')) {
        $libDir = Join-Path $script:DepsPrefix $libRel
        if (-not (Test-Path -LiteralPath $libDir)) { continue }
        $hit = Get-ChildItem -LiteralPath $libDir -Filter 'zlib*.lib' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $hit) {
            $hit = Get-ChildItem -LiteralPath $libDir -Filter 'z.lib' -ErrorAction SilentlyContinue |
                Select-Object -First 1
        }
        if ($hit) { return $true }
    }
    return $false
}

function Test-MinizipInDeps {
    foreach ($rel in @(
        'include\minizip-ng\mz.h',
        'include\minizip\mz.h',
        'include\mz.h'
    )) {
        if (Test-Path -LiteralPath (Join-Path $script:DepsPrefix $rel)) { return $true }
    }
    return $false
}

function Test-YamlCppInDeps {
    $hdr = Join-Path $script:DepsPrefix 'include\yaml-cpp\yaml.h'
    if (-not (Test-Path -LiteralPath $hdr)) { return $false }
    foreach ($libRel in @('lib', 'lib64')) {
        $libDir = Join-Path $script:DepsPrefix $libRel
        if (-not (Test-Path -LiteralPath $libDir)) { continue }
        $hit = Get-ChildItem -LiteralPath $libDir -Filter 'yaml-cpp*.lib' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($hit) { return $true }
    }
    return $false
}

function Test-ExpatInDeps {
    $hdr = Join-Path $script:DepsPrefix 'include\expat.h'
    if (-not (Test-Path -LiteralPath $hdr)) { return $false }
    foreach ($libRel in @('lib', 'lib64')) {
        $libDir = Join-Path $script:DepsPrefix $libRel
        if (-not (Test-Path -LiteralPath $libDir)) { continue }
        $hit = Get-ChildItem -LiteralPath $libDir -Filter '*expat*.lib' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($hit) { return $true }
    }
    return $false
}

function Test-PystringInDeps {
    foreach ($rel in @(
        'include\pystring\pystring.h',
        'include\pystring.h'
    )) {
        if (Test-Path -LiteralPath (Join-Path $script:DepsPrefix $rel)) { return $true }
    }
    return $false
}

function Install-OcioSdkDeps {
    # Prebuild every OCIO ExternalProject dep with Ninja + POLICY 3.5.
    # OCIO's nested VS ExternalProjects (yaml-cpp_install, minizip-ng_install, …)
    # die on CMake 4.x (cmake_minimum < 3.5) and zlib race. MISSING then finds
    # these and skips ExternalProject entirely.
    if (-not (Test-ZlibInDeps)) {
        $zlibSrc = Join-Path $script:DepsSrc 'zlib'
        if (Test-Path -LiteralPath $zlibSrc) {
            Info 'Removing previous zlib sources (need v1.3.1 for CMake 4.x).'
            Remove-Item -LiteralPath $zlibSrc -Recurse -Force
        }
        Invoke-GitClone 'https://github.com/madler/zlib.git' 'v1.3.1' $zlibSrc
        $zb = Join-Path $zlibSrc 'build'
        if (Test-Path -LiteralPath $zb) { Remove-Item -LiteralPath $zb -Recurse -Force }
        Invoke-DepCMakeInstall $zlibSrc 'zlib' @(
            '-DBUILD_SHARED_LIBS=OFF', '-DZLIB_BUILD_EXAMPLES=OFF'
        )
        if (-not (Test-ZlibInDeps)) {
            Fail "zlib install finished but zlib.h / zlib*.lib missing under $script:DepsPrefix"
        }
    } else {
        Info 'zlib already in deps — skipping'
    }
    if (-not (Test-MinizipInDeps)) {
        $mzSrc = Join-Path $script:DepsSrc 'minizip-ng'
        if (Test-Path -LiteralPath $mzSrc) {
            Info 'Removing previous minizip-ng sources (need 3.0.7 for OCIO 2.3.2).'
            Remove-Item -LiteralPath $mzSrc -Recurse -Force
        }
        Invoke-GitClone 'https://github.com/zlib-ng/minizip-ng.git' '3.0.7' $mzSrc
        $mb = Join-Path $mzSrc 'build'
        if (Test-Path -LiteralPath $mb) { Remove-Item -LiteralPath $mb -Recurse -Force }
        Invoke-DepCMakeInstall $mzSrc 'minizip-ng' @(
            "-DCMAKE_PREFIX_PATH=$script:DepsPrefix",
            "-DZLIB_ROOT=$script:DepsPrefix",
            '-DBUILD_SHARED_LIBS=OFF',
            '-DMZ_OPENSSL=OFF', '-DMZ_LIBBSD=OFF', '-DMZ_BUILD_TESTS=OFF',
            '-DMZ_COMPAT=OFF', '-DMZ_BZIP2=OFF', '-DMZ_LZMA=OFF',
            '-DMZ_LIBCOMP=OFF', '-DMZ_ZSTD=OFF', '-DMZ_PKCRYPT=OFF',
            '-DMZ_WZAES=OFF', '-DMZ_SIGNING=OFF', '-DMZ_ZLIB=ON',
            '-DMZ_ICONV=OFF', '-DMZ_FETCH_LIBS=OFF', '-DMZ_FORCE_FETCH_LIBS=OFF'
        )
        if (-not (Test-MinizipInDeps)) {
            Fail "minizip-ng install finished but mz.h missing under $script:DepsPrefix"
        }
    } else {
        Info 'minizip-ng already in deps — skipping'
    }
    if (-not (Test-YamlCppInDeps)) {
        # yaml-cpp 0.7.0: cmake_minimum_required(VERSION 3.4) — CMake 4 rejects it
        # inside OCIO's yaml-cpp_install ExternalProject. Prebuild with POLICY 3.5.
        $ySrc = Join-Path $script:DepsSrc 'yaml-cpp'
        if (Test-Path -LiteralPath $ySrc) { Remove-Item -LiteralPath $ySrc -Recurse -Force }
        Invoke-GitClone 'https://github.com/jbeder/yaml-cpp.git' 'yaml-cpp-0.7.0' $ySrc
        $yb = Join-Path $ySrc 'build'
        if (Test-Path -LiteralPath $yb) { Remove-Item -LiteralPath $yb -Recurse -Force }
        Invoke-DepCMakeInstall $ySrc 'yaml-cpp' @(
            '-DBUILD_SHARED_LIBS=OFF',
            '-DYAML_BUILD_SHARED_LIBS=OFF',
            '-DYAML_CPP_BUILD_TESTS=OFF',
            '-DYAML_CPP_BUILD_TOOLS=OFF',
            '-DYAML_CPP_BUILD_CONTRIB=OFF'
        )
        if (-not (Test-YamlCppInDeps)) {
            Fail "yaml-cpp install finished but yaml.h / yaml-cpp*.lib missing under $script:DepsPrefix"
        }
    } else {
        Info 'yaml-cpp already in deps — skipping'
    }
    if (-not (Test-ExpatInDeps)) {
        $eSrc = Join-Path $script:DepsSrc 'expat'
        if (Test-Path -LiteralPath $eSrc) { Remove-Item -LiteralPath $eSrc -Recurse -Force }
        Invoke-GitClone 'https://github.com/libexpat/libexpat.git' 'R_2_5_0' $eSrc
        $eb = Join-Path $eSrc 'build'
        if (Test-Path -LiteralPath $eb) { Remove-Item -LiteralPath $eb -Recurse -Force }
        $eCmake = Join-Path $eSrc 'expat'
        if (-not (Test-Path -LiteralPath (Join-Path $eCmake 'CMakeLists.txt'))) {
            Fail "libexpat R_2_5_0 missing expat/CMakeLists.txt under $eSrc"
        }
        Invoke-DepCMakeInstall $eSrc 'expat' @(
            '-DEXPAT_BUILD_DOCS=OFF', '-DEXPAT_BUILD_EXAMPLES=OFF',
            '-DEXPAT_BUILD_TESTS=OFF', '-DEXPAT_BUILD_TOOLS=OFF',
            '-DEXPAT_SHARED_LIBS=OFF'
        ) -SourceDir $eCmake
        if (-not (Test-ExpatInDeps)) {
            Fail "expat install finished but expat.h missing under $script:DepsPrefix"
        }
    } else {
        Info 'expat already in deps — skipping'
    }
    if (-not (Test-PystringInDeps)) {
        # Upstream pystring has no CMakeLists — use OCIO's Buildpystring.cmake.
        $pSrc = Join-Path $script:DepsSrc 'pystring'
        if (Test-Path -LiteralPath $pSrc) { Remove-Item -LiteralPath $pSrc -Recurse -Force }
        New-Item -ItemType Directory -Force -Path (Split-Path $pSrc -Parent) | Out-Null
        & $Git clone --depth 1 --branch 'v1.1.3' 'https://github.com/imageworks/pystring.git' $pSrc
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $pSrc 'pystring.cpp'))) {
            Fail 'git clone pystring v1.1.3 failed'
        }
        $pCm = Join-Path $pSrc 'CMakeLists.txt'
        @'
project(pystring)
cmake_minimum_required(VERSION 3.10)
include(GNUInstallDirs)
set(HEADERS pystring.h)
set(SOURCES pystring.cpp)
add_library(${PROJECT_NAME} STATIC ${HEADERS} ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pystring)
'@ | Set-Content -Path $pCm -Encoding ascii
        $pb = Join-Path $pSrc 'build'
        if (Test-Path -LiteralPath $pb) { Remove-Item -LiteralPath $pb -Recurse -Force }
        Invoke-DepCMakeInstall $pSrc 'pystring' @('-DBUILD_SHARED_LIBS=OFF')
        if (-not (Test-PystringInDeps)) {
            Fail "pystring install finished but pystring.h missing under $script:DepsPrefix"
        }
    } else {
        Info 'pystring already in deps — skipping'
    }
}

function Install-OpenColorIO {
    # FULL downloads and compiles the OpenColorIO SDK (library) into
    # %LOCALAPPDATA%\grendizer-deps. The OCIO env / Film path is only the
    # .ocio colour config — that is not this step and never replaces the SDK.
    #
    # Do not use EXT_PACKAGES=ALL for minizip: OCIO's ExternalProject races zlib
    # (C1083 zlib.h) and Ninja nested minizip dies on OpenSSL defaults.
    # Prebuild zlib + minizip-ng 3.0.7, then VS + MISSING; yaml-cpp/expat/pystring
    # still come from OCIO's VS ExternalProject.
    $src = Join-Path $script:DepsSrc 'ocio'
    $b = Join-Path $src 'build'
    Info 'OpenColorIO SDK: git clone v2.3.2 + compile into deps (FULL requires the library).'
    Info 'Not the .ocio config file — that stays on OCIO env / Film.'
    $candidates = New-Object System.Collections.Generic.List[object]
    $primary = Get-VsGeneratorArgs
    if ($primary.Count -ge 2) { [void]$candidates.Add($primary) }
    foreach ($fallback in @(
        @('-G', 'Visual Studio 18 2026', '-A', 'x64'),
        @('-G', 'Visual Studio 17 2022', '-A', 'x64'),
        @('-G', 'Visual Studio 16 2019', '-A', 'x64')
    )) {
        $key = $fallback -join ' '
        $dup = $false
        foreach ($c in $candidates) {
            if (($c -join ' ') -eq $key) { $dup = $true; break }
        }
        if (-not $dup) { [void]$candidates.Add($fallback) }
    }
    if ($candidates.Count -eq 0) {
        Fail @"
OpenColorIO SDK needs the Visual Studio CMake generator.
Install VS 2022 or VS 2026 Desktop development with C++, then re-run BUILD_WINDOWS_FULL.bat.
Keep C:\gz-full and %LOCALAPPDATA%\grendizer-deps.
"@
    }
    Info 'OpenColorIO SDK deps: zlib, minizip-ng, yaml-cpp, expat, pystring (prebuilt; no nested ExternalProject).'
    Install-OcioSdkDeps
    # Fresh clone if the tree looks incomplete (failed prior attempt).
    $cm = Join-Path $src 'CMakeLists.txt'
    $hdrProbe = Join-Path $src 'include\OpenColorIO\OpenColorIO.h'
    if ((Test-Path -LiteralPath $src) -and (-not (Test-Path -LiteralPath $cm) -or -not (Test-Path -LiteralPath $hdrProbe))) {
        Info 'Removing incomplete OpenColorIO source tree.'
        Remove-Item -LiteralPath $src -Recurse -Force
    }
    Info 'Downloading OpenColorIO SDK sources (AcademySoftwareFoundation/OpenColorIO v2.3.2) ...'
    Invoke-GitClone 'https://github.com/AcademySoftwareFoundation/OpenColorIO.git' 'v2.3.2' $src
    if (Test-Path -LiteralPath $b) {
        Info 'Removing previous OCIO build tree (wipe broken yaml-cpp/minizip ExternalProject leftovers).'
        Remove-Item -LiteralPath $b -Recurse -Force
    }
    $ocioFlags = @(
        "-DCMAKE_INSTALL_PREFIX=$script:DepsPrefix",
        "-DCMAKE_PREFIX_PATH=$script:DepsPrefix",
        '-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_SHARED_LIBS=ON',
        '-DOCIO_BUILD_APPS=OFF',
        '-DOCIO_BUILD_TESTS=OFF',
        '-DOCIO_BUILD_GPU_TESTS=OFF',
        '-DOCIO_BUILD_PYTHON=OFF',
        '-DOCIO_BUILD_JAVA=OFF',
        '-DOCIO_BUILD_DOCS=OFF',
        '-DOCIO_INSTALL_EXT_PACKAGES=MISSING',
        "-DZLIB_ROOT=$script:DepsPrefix",
        "-Dminizip-ng_ROOT=$script:DepsPrefix",
        '-Dminizip-ng_STATIC_LIBRARY=ON',
        "-Dyaml-cpp_ROOT=$script:DepsPrefix",
        "-Dexpat_ROOT=$script:DepsPrefix",
        "-Dpystring_ROOT=$script:DepsPrefix",
        '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'
    )
    $usedGen = $null
    foreach ($vsGen in $candidates) {
        if (Test-Path -LiteralPath $b) { Remove-Item -LiteralPath $b -Recurse -Force }
        Info ('Compiling OpenColorIO SDK: ' + ($vsGen -join ' ') + ' + MISSING (all ext deps prebuilt)')
        $cfg = @('-S', $src, '-B', $b) + $vsGen + $ocioFlags
        & $CMake @cfg
        if ($LASTEXITCODE -eq 0) {
            $usedGen = $vsGen
            break
        }
        Write-Host ('Warning: OpenColorIO configure rejected generator ' + ($vsGen -join ' ')) -ForegroundColor Yellow
    }
    if (-not $usedGen) {
        Fail 'OpenColorIO SDK cmake configure failed for every Visual Studio generator. See log above.'
    }
    Info ('Installing OpenColorIO SDK with ' + ($usedGen -join ' ') + ' (--target install) into ' + $script:DepsPrefix)
    & $CMake --build $b --config Release --target install --parallel
    if ($LASTEXITCODE -ne 0) {
        Fail @"
OpenColorIO SDK MSBuild install failed.
This compiles the OpenColorIO library (headers + OpenColorIO.dll), not the .ocio colour config.
If the log still shows minizip-ng_install / zlib.h: keep %LOCALAPPDATA%\grendizer-deps and re-run.
Close Grendizer if a DLL is locked.
"@
    }
    $cfgFile = Find-OcioConfigCMake $script:DepsPrefix
    if (-not $cfgFile) {
        Fail "OpenColorIO SDK install finished but OpenColorIOConfig.cmake is not under $($script:DepsPrefix)."
    }
    $hit = Test-OcioInstallPrefix $script:DepsPrefix
    if (-not $hit) {
        Fail "OpenColorIO SDK incomplete under $($script:DepsPrefix) (need headers + import lib / DLL)."
    }
    if (-not $hit.Dll) {
        $dllProbe = Get-ChildItem -LiteralPath (Join-Path $script:DepsPrefix 'bin') -Filter 'OpenColorIO*.dll' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $dllProbe) {
            Fail "OpenColorIO SDK missing OpenColorIO*.dll under $($script:DepsPrefix)\bin. FULL Display/View needs that DLL."
        }
    }
    $script:OcioInstall = $hit
    Info ("OpenColorIO SDK ready: " + $cfgFile.FullName)
    if ($hit.Dll) { Info ("OpenColorIO DLL: " + $hit.Dll) }
}

function Invoke-DepCMakeInstall([string]$Src, [string]$Name, [string[]]$Extra, [switch]$AllowFail, [string]$SourceDir = '') {
    $cmakeSrc = $Src
    if ($SourceDir) { $cmakeSrc = $SourceDir }
    $b = Join-Path $Src 'build'
    Info "Building $Name ..."
    $cfg = @(
        '-S', $cmakeSrc, '-B', $b, '-G', 'Ninja',
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$script:DepsPrefix",
        "-DCMAKE_PREFIX_PATH=$script:DepsPrefix",
        "-DCMAKE_MAKE_PROGRAM=$Ninja",
        "-DCMAKE_C_COMPILER=$Cl",
        "-DCMAKE_CXX_COMPILER=$Cl",
        # CMake 4.x removed compat with cmake_minimum_required < 3.5 (zlib 1.2.x, yaml-cpp 0.7, etc.).
        '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'
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
    $script:DepsSrc = Join-Path $env:LOCALAPPDATA 'grendizer-deps-src'
    if ($env:GRENDIZER_DEPS_SRC) { $script:DepsSrc = $env:GRENDIZER_DEPS_SRC }
    $srcRoot = $script:DepsSrc
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

    # OptiX one-click only needs the Embree zip (TBB is inside it). Skip
    # Imath/OpenEXR/Alembic/oneTBB/OpenVDB/OCIO source builds.
    if (-not $env:GRENDIZER_FULL_DEPS) {
        Info "OptiX-min deps ready (Embree only): $script:DepsPrefix"
        return
    }

    # Download + compile the OpenColorIO SDK into deps. A .ocio file on OCIO=
    # is the colour config at runtime, never a substitute for this library.
    $script:OcioInstall = Test-OcioInstallPrefix $script:DepsPrefix
    if ((Test-Path -LiteralPath $stamp) -and $script:OcioInstall -and $script:OcioInstall.Dll) {
        Info "Native deps already installed (incl. OpenColorIO SDK): $script:DepsPrefix"
        return
    }
    if (Test-Path -LiteralPath $stamp) {
        Info "Deps stamp exists but OpenColorIO SDK is missing — downloading/compiling OCIO only (keeping Embree/OpenEXR/OpenVDB)."
        Install-OpenColorIO
        $script:OcioInstall = Test-OcioInstallPrefix $script:DepsPrefix
        if (-not $script:OcioInstall) {
            Fail "OpenColorIO SDK still missing under $script:DepsPrefix after rebuild. FULL requires SOLSTICE_HAVE_OCIO 1."
        }
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

    Install-OpenColorIO
    $script:OcioInstall = Test-OcioInstallPrefix $script:DepsPrefix
    if (-not $script:OcioInstall) {
        Fail "OpenColorIO SDK missing under $script:DepsPrefix after install. FULL requires SOLSTICE_HAVE_OCIO 1."
    }
    if (-not $script:OcioInstall.Dll) {
        Fail "OpenColorIO SDK installed but OpenColorIO*.dll missing under $script:DepsPrefix\bin."
    }

    Set-Content -Path $stamp -Value 'ok' -Encoding ascii
    Info "Native deps installed (OpenColorIO SDK included): $script:DepsPrefix"
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

function Test-WritableDir([string]$Dir) {
    try {
        New-Item -ItemType Directory -Force -Path $Dir | Out-Null
        $probe = Join-Path $Dir '.__gz_write'
        [IO.File]::WriteAllText($probe, 'ok')
        Remove-Item -LiteralPath $probe -Force
        return $true
    } catch {
        return $false
    }
}

function Resolve-BuildDir {
    if ($env:GRENDIZER_BUILD_DIR) {
        $d = $env:GRENDIZER_BUILD_DIR
        if (-not (Test-WritableDir $d)) { Fail "GRENDIZER_BUILD_DIR is not writable: $d" }
        return $d
    }
    # GitHub zip in Downloads\...\name (7)\name\build-windows blows MSVC's 250-char
    # object path (C1083 / OpenPGL CMAKE warning). Keep the build tree short.
    $short = 'C:\gz-build'
    $fallbackName = 'gz-build'
    if ($env:GRENDIZER_FULL_DEPS) {
        $short = 'C:\gz-full'
        $fallbackName = 'gz-full'
    }
    foreach ($d in @($short, (Join-Path $env:LOCALAPPDATA $fallbackName))) {
        if (Test-WritableDir $d) { return $d }
    }
    Fail "Cannot create $short. Set GRENDIZER_BUILD_DIR to a short path (example C:\g)."
}

Write-Host ''
Write-Host '=== Grendizer Render - Windows OptiX build ===' -ForegroundColor Green
Write-Host "Repo: $Root"
$script:FullBuild = $false
if ($env:GRENDIZER_FULL_DEPS) { $script:FullBuild = $true }
if ($script:FullBuild) {
    Write-Host 'Mode: FULL (VDB, MaterialX, OpenPGL, Alembic, OpenEXR, OCIO, TinyUSDZ, TX Tools).'
    Write-Host 'Output: C:\gz-full   (BUILD_WINDOWS_FULL.bat)'
    Write-Host 'TinyUSDZ lib.exe can sit quiet for many minutes - that is not a hang.'
} else {
    Write-Host 'Mode: OPTIX-MIN (Qt + Embree + CUDA). Fast path to GPU OptiX.'
    Write-Host 'Output: C:\gz-build'
    Write-Host 'Full app (VDB/MaterialX/Alembic/...): double-click BUILD_WINDOWS_FULL.bat'
}

Import-VcVars64
$CMake = Find-CMake
$Git = Find-Git
$Qt = Find-QtPrefix
$Cl = (Get-Command cl.exe).Source
Resolve-CudaForOptix
$Cuda = [string]$script:CudaRoot
$env:CUDA_PATH = $Cuda
$env:CUDA_HOME = $Cuda
$cudaBin = Join-Path $Cuda 'bin'
$env:PATH = "$cudaBin;$env:PATH"
$Nvcc = Join-Path $cudaBin 'nvcc.exe'
if (-not (Test-Path -LiteralPath $Nvcc)) { Fail "nvcc.exe missing: $Nvcc" }
if (-not (Test-NvccHostStl $Nvcc $Cl $script:OptixArch)) {
    Fail @"
nvcc cannot parse this MSVC STL (type_traits).
cl.exe: $Cl
nvcc:   $Nvcc

Need CUDA 13.2+ with Visual Studio 2026.
Confirm: `"$Nvcc`" --version   shows release 13.2
Then delete C:\gz-build (keep %LOCALAPPDATA%\grendizer-deps) and re-run.
"@
}
$OptiX = Find-OptiXRoot $Git
$Ninja = Find-Ninja
$ninjaDir = Split-Path $Ninja -Parent
$env:PATH = "$ninjaDir;$env:PATH"
$script:OcioInstall = $null
Ensure-NativeDeps
$embreeRoot = Resolve-EmbreePrefix
$BuildDir = Resolve-BuildDir
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $BuildDir 'CMakeCache.txt'))) {
    Info "No CMake cache in $BuildDir - clean configure (expected after deleting the build folder)."
}

$Prefix = "$Qt;$script:DepsPrefix;$embreeRoot"
if ($script:OcioInstall -and $script:OcioInstall.Prefix) {
    $Prefix = "$($script:OcioInstall.Prefix);$Prefix"
    Info ("OpenColorIO prefix: " + $script:OcioInstall.Prefix)
}

Info "CMake:  $CMake"
Info "Qt:     $Qt"
Info "CUDA:   $Cuda"
Info "nvcc:   $Nvcc"
Info "arch:   $script:OptixArch"
Info "cl.exe: $Cl"
Info "OptiX:  $OptiX"
Info "Deps:   $script:DepsPrefix"
Info "Ninja:  $Ninja"
Info "Build:  $BuildDir"
Write-Host ''

# Ninja + cl.exe works on VS 2022 and VS 2026. The VS 17 generator cannot see VS 18.
$Generator = 'Ninja'
Info "CMake generator: $Generator (cl.exe from VS $script:VsYear)"

function Normalize-CmakeSrc([string]$p) {
    if (-not $p) { return '' }
    return ($p.Trim().TrimEnd('\', '/') -replace '\\', '/').ToLowerInvariant()
}

$cache = Join-Path $BuildDir 'CMakeCache.txt'
if (Test-Path -LiteralPath $cache) {
    $stale = $false
    $staleWhy = ''
    if (Select-String -Path $cache -Pattern 'vcpkg' -Quiet) {
        $stale = $true
        $staleWhy = 'vcpkg cache'
    }
    $oldGen = Select-String -Path $cache -Pattern '^CMAKE_GENERATOR:INTERNAL=(.+)$' | Select-Object -First 1
    if ($oldGen -and $oldGen.Matches[0].Groups[1].Value -ne $Generator) {
        $stale = $true
        $staleWhy = 'CMake generator changed'
    }
    $oldNvcc = Select-String -Path $cache -Pattern '^CMAKE_CUDA_COMPILER:FILEPATH=(.+)$' | Select-Object -First 1
    if ($oldNvcc) {
        $oldNvccPath = $oldNvcc.Matches[0].Groups[1].Value.Replace('/', '\')
        if ($oldNvccPath -ne $Nvcc) {
            $stale = $true
            $staleWhy = 'nvcc path changed'
        }
    }
    if ($script:CudaRelease -ge [version]'13.0') {
        if (Select-String -Path $cache -Pattern 'compute_60|CUDA\\\\v12|CUDA/v12' -Quiet) {
            $stale = $true
            $staleWhy = 'old CUDA 12 cache'
        }
    }
    # GitHub zip extracts as "...-6909 (8)" vs "(44)". C:\gz-full still points at the old tree.
    $oldHome = Select-String -Path $cache -Pattern '^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$' | Select-Object -First 1
    if ($oldHome) {
        $cachedSrc = Normalize-CmakeSrc $oldHome.Matches[0].Groups[1].Value
        $currentSrc = Normalize-CmakeSrc $Root
        if ($cachedSrc -and $currentSrc -and ($cachedSrc -ne $currentSrc)) {
            $stale = $true
            $staleWhy = "source moved (`n  cache: $cachedSrc`n  now:   $currentSrc)"
        }
    }
    # CMake 4.4 on Windows turned file:///C:/… into /C:/… and poisoned FetchContent.
    if (Select-String -Path $cache -Pattern 'file:///C:|not found: /C:|openpgl-v0\.7\.1\.tgz' -Quiet) {
        $stale = $true
        $staleWhy = 'broken FetchContent file:///C: URL in cache'
    }
    if ($stale) {
        # Keep _deps so MaterialX / TinyUSDZ / OpenPGL git clones are not wiped
        # when a new GitHub zip extracts to a different folder name.
        Info "Resetting CMake cache in $BuildDir ($staleWhy). Keeping _deps."
        foreach ($n in @(
            'CMakeCache.txt', 'CMakeFiles', 'cmake_install.cmake',
            'CTestTestfile.cmake', 'build.ninja', 'CMakeError.log',
            'CMakeOutput.log', 'CMakeConfigureLog.yaml'
        )) {
            $p = Join-Path $BuildDir $n
            if (Test-Path -LiteralPath $p) {
                Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

$env:NVCC_PREPEND_FLAGS = '--allow-unsupported-compiler'
$env:NVCC_APPEND_FLAGS = '--allow-unsupported-compiler'

$featureFlags = @(
    '-DSOLSTICE_ENABLE_OPTIX=ON',
    "-DSOLSTICE_OPTIX_ARCH=$script:OptixArch",
    "-DOptiX_ROOT=$OptiX",
    '-DSOLSTICE_MODERN_CPU=ON',
    '-DSOLSTICE_BUILD_TESTS=OFF',
    '-DSOLSTICE_BUILD_TOOLS=OFF'
)
if ($script:FullBuild) {
    function Ensure-GitDepSrc([string]$Url, [string]$Tag, [string]$Dest, [string]$Name) {
        $cm = Join-Path $Dest 'CMakeLists.txt'
        if (Test-Path -LiteralPath $cm) {
            Info ($Name + ' source: ' + $Dest)
            return
        }
        if (Test-Path -LiteralPath $Dest) {
            Remove-Item -LiteralPath $Dest -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path (Split-Path $Dest -Parent) | Out-Null
        Info ("Cloning " + $Name + " " + $Tag + " (git, not GitHub tarball — CMake 4.x 403)")
        & $Git clone --depth 1 --branch $Tag $Url $Dest
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $cm)) {
            Fail ("git clone " + $Name + " " + $Tag + " failed. FULL needs " + $Name + ". Check github.com.")
        }
    }
    $depsSrc = Join-Path $BuildDir '_deps'
    Ensure-GitDepSrc 'https://github.com/AcademySoftwareFoundation/MaterialX.git' 'v1.39.4' (Join-Path $depsSrc 'materialx-src') 'MaterialX'
    Ensure-GitDepSrc 'https://github.com/RenderKit/openpgl.git' 'v0.7.1' (Join-Path $depsSrc 'openpgl-src') 'OpenPGL'
    Ensure-GitDepSrc 'https://github.com/lighttransport/tinyusdz.git' 'v0.9.4' (Join-Path $depsSrc 'tinyusdz-src') 'TinyUSDZ'
    $featureFlags += @(
        '-DSOLSTICE_ENABLE_ALEMBIC=ON',
        '-DSOLSTICE_ENABLE_OPENEXR=ON',
        '-DSOLSTICE_ENABLE_TIFF=ON',
        '-DSOLSTICE_ENABLE_MATERIALX=ON',
        '-DSOLSTICE_ENABLE_OPENPGL=ON',
        '-DSOLSTICE_ENABLE_OPENSUBDIV=ON',
        '-DSOLSTICE_ENABLE_OPENVDB=ON',
        '-DSOLSTICE_ENABLE_OCIO=ON',
        '-DSOLSTICE_ENABLE_TINYUSDZ=ON',
        '-DSOLSTICE_BUILD_TX_TOOLS_ALPHA=ON',
        '-DSOLSTICE_BUILD_TX_TOOLS_OMEGA=ON'
    )
} else {
    $featureFlags += @(
        '-DSOLSTICE_ENABLE_ALEMBIC=OFF',
        '-DSOLSTICE_ENABLE_OPENEXR=OFF',
        '-DSOLSTICE_ENABLE_TIFF=OFF',
        '-DSOLSTICE_ENABLE_MATERIALX=OFF',
        '-DSOLSTICE_ENABLE_OPENPGL=OFF',
        '-DSOLSTICE_ENABLE_OPENSUBDIV=OFF',
        '-DSOLSTICE_ENABLE_OPENVDB=OFF',
        '-DSOLSTICE_ENABLE_OCIO=OFF',
        '-DSOLSTICE_ENABLE_TINYUSDZ=OFF',
        '-DSOLSTICE_BUILD_TX_TOOLS_ALPHA=OFF',
        '-DSOLSTICE_BUILD_TX_TOOLS_OMEGA=OFF'
    )
}

$ocioDir = $null
if ($script:OcioInstall -and $script:OcioInstall.CMakeDir) {
    $ocioCfgFile = Join-Path $script:OcioInstall.CMakeDir 'OpenColorIOConfig.cmake'
    if (Test-Path -LiteralPath $ocioCfgFile) {
        $ocioDir = $script:OcioInstall.CMakeDir
    }
}
if (-not $ocioDir) {
    $ocioCmake = $null
    if ($script:OcioInstall -and $script:OcioInstall.Prefix) {
        $ocioCmake = Find-OcioConfigCMake $script:OcioInstall.Prefix
    }
    if (-not $ocioCmake) { $ocioCmake = Find-OcioConfigCMake $script:DepsPrefix }
    if ($ocioCmake) { $ocioDir = $ocioCmake.Directory.FullName }
}
if ($ocioDir) {
    $featureFlags += "-DOpenColorIO_DIR=$ocioDir"
    $env:OpenColorIO_DIR = $ocioDir
    Info "OpenColorIO_DIR: $ocioDir"
}
if ($script:OcioInstall) {
    $featureFlags += "-DOpenColorIO_ROOT=$($script:OcioInstall.Prefix)"
    $env:OpenColorIO_ROOT = $script:OcioInstall.Prefix
    if ($script:OcioInstall.Include) {
        $featureFlags += "-DOpenColorIO_INCLUDE_DIR=$($script:OcioInstall.Include)"
    }
    if ($script:OcioInstall.Library) {
        $featureFlags += "-DOpenColorIO_LIBRARY=$($script:OcioInstall.Library)"
    }
}

& $CMake -S $Root -B $BuildDir -G $Generator `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DCMAKE_MAKE_PROGRAM=$Ninja" `
    "-DCMAKE_C_COMPILER=$Cl" `
    "-DCMAKE_CXX_COMPILER=$Cl" `
    "-DCMAKE_PREFIX_PATH=$Prefix" `
    "-DCMAKE_CUDA_COMPILER=$Nvcc" `
    "-DCMAKE_CUDA_HOST_COMPILER=$Cl" `
    "-DCMAKE_CUDA_COMPILER_ID=NVIDIA" `
    "-DCMAKE_EXE_LINKER_FLAGS=/IGNORE:4006" `
    "-DCMAKE_STATIC_LINKER_FLAGS=/IGNORE:4006" `
    "-DCMAKE_SHARED_LINKER_FLAGS=/IGNORE:4006" `
    "-DCUDAToolkit_ROOT=$Cuda" `
    "-DCUDA_TOOLKIT_ROOT_DIR=$Cuda" `
    "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler" `
    "-DCMAKE_CUDA_COMPILER_ID_FLAGS=--allow-unsupported-compiler" `
    "-DSOLSTICE_CUDA_HOST_COMPILER=$Cl" `
    @featureFlags
if ($LASTEXITCODE -ne 0) { Fail 'cmake configure failed. Check Qt / CUDA / OptiX / deps in the log above.' }

$Cfg = Join-Path $BuildDir 'generated\solstice_config.h'
if (-not (Test-Path -LiteralPath $Cfg)) {
    Fail "missing $Cfg after configure"
}
if (-not (Select-String -Path $Cfg -Pattern 'SOLSTICE_HAVE_OPTIX 1' -Quiet)) {
    Fail 'OptiX did not enable (SOLSTICE_HAVE_OPTIX != 1). Check nvcc and OptiX_ROOT in the cmake log.'
}
Info 'OptiX is compiled into this build (SOLSTICE_HAVE_OPTIX 1).'
if ($script:FullBuild) {
    # FULL must not ship an exe that silently dropped a feature (MaterialX
    # was lost that way). OpenSubdiv stays off on Windows (MSVC climits).
    $required = @(
        @('SOLSTICE_HAVE_MATERIALX 1', 'MaterialX'),
        @('SOLSTICE_HAVE_OPENPGL 1', 'OpenPGL'),
        @('SOLSTICE_HAVE_TINYUSDZ 1', 'TinyUSDZ'),
        @('SOLSTICE_HAVE_ALEMBIC 1', 'Alembic'),
        @('SOLSTICE_HAVE_OPENEXR 1', 'OpenEXR'),
        @('SOLSTICE_HAVE_OPENVDB 1', 'OpenVDB'),
        @('SOLSTICE_HAVE_TIFF 1', 'libtiff'),
        @('SOLSTICE_HAVE_OCIO 1', 'OpenColorIO')
    )
    foreach ($req in $required) {
        if (-not (Select-String -Path $Cfg -Pattern $req[0] -Quiet)) {
            Fail ($req[1] + ' did not enable (' + $req[0] + ' missing). FULL requires it — not a skip. CMAKE_PREFIX_PATH=' + $Prefix + '. See the cmake log.')
        }
        Info ($req[1] + ' is compiled into this build.')
    }
}

function Find-RenderExe([string]$Dir) {
    $binRoot = Join-Path $Dir 'bin'
    $dirs = @(
        $binRoot,
        (Join-Path $binRoot 'Release'),
        (Join-Path $binRoot 'RelWithDebInfo'),
        (Join-Path $binRoot 'Debug')
    )
    foreach ($d in $dirs) {
        if (-not (Test-Path -LiteralPath $d)) { continue }
        foreach ($pat in @('Grendizer_Render*.exe', 'Solstice*.exe')) {
            $hit = Get-ChildItem -LiteralPath $d -Filter $pat -File -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($hit) { return $hit }
        }
    }
    if (Test-Path -LiteralPath $binRoot) {
        foreach ($pat in @('Grendizer_Render*.exe', 'Solstice*.exe')) {
            $hit = Get-ChildItem -LiteralPath $binRoot -Recurse -Filter $pat -File -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($hit) { return $hit }
        }
    }
    return $null
}

function Try-UnlockRenderExe([string]$Dir) {
    $old = Find-RenderExe $Dir
    if (-not $old) { return }
    try {
        Remove-Item -LiteralPath $old.FullName -Force -ErrorAction Stop
        Info ("Removed old exe so the linker can write it: " + $old.Name)
    } catch {
        Fail @"
Cannot overwrite $($old.FullName)
Close Grendizer_Render (and TX Tools) and run the bat again. This is LNK1168, not nvcc/sobol.
"@
    }
}

Write-Host ''
Info 'Building in parallel: nvcc PTX uses 1 core; MSVC compiles the rest on the other cores.'
Info 'GPU OptiX is wavefront modules (init/intersect/shade), not integrator.h. ninja compiles them in parallel.'
Info 'Ninja [n/N] is real progress. Do not close Grendizer_Render while linking (LNK1168).'
$j = $env:NUMBER_OF_PROCESSORS
if (-not $j) { $j = '8' }
Try-UnlockRenderExe $BuildDir
# Call cmake in-process. Start-Process left ExitCode $null after the last
# link step, and the script printed BUILD FAILED with no LNK line even when
# ninja had already written bin\Grendizer_Render-*.exe.
$buildStarted = Get-Date
& $CMake --build $BuildDir --parallel $j
$code = 0
if ($null -ne $LASTEXITCODE) { $code = [int]$LASTEXITCODE }
$ExeAfter = Find-RenderExe $BuildDir
if ($code -ne 0) {
    $fresh = $false
    if ($ExeAfter) {
        $fresh = $ExeAfter.LastWriteTime -ge $buildStarted.AddSeconds(-5)
    }
    if ($fresh) {
        Info ("cmake exit code $code but ninja wrote a fresh exe - continuing: " + $ExeAfter.Name)
    } else {
        Fail @"
Compile/link failed (ninja exit $code). Scroll up for the first error from cl/link/nvcc.
C4244 / C4996 above are warnings, not the failure.
If the last line was Linking CXX executable: close Grendizer_Render and retry (exe locked).
If sobol.h / undefined in device code: that nvcc bug is already fixed in this zip.
Look in $BuildDir\bin for Grendizer_Render-0.9.3-*.exe - a false FAIL used to hide a finished link.
Deleting $BuildDir is OK. Keep %LOCALAPPDATA%\grendizer-deps.
"@
    }
}

$Exe = Find-RenderExe $BuildDir
if (-not $Exe) {
    Fail @"
Compile reported success but Grendizer_Render*.exe is not under $BuildDir\bin.
Ninja writes the exe to bin\ (not bin\Release). Do not treat MaterialX .mtlx files as the app.
"@
}

Write-Host ''
Info 'Deploying Qt DLLs next to the exe ...'
& $CMake --build $BuildDir --target deploy
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Warning: deploy target failed. The exe may need Qt on PATH.' -ForegroundColor Yellow
}
$Bin = $Exe.DirectoryName
Info ("Found exe: " + $Exe.FullName)

$cudartDir = Join-Path $Cuda 'bin'
if (Test-Path -LiteralPath $cudartDir) {
    $Cudart = Get-ChildItem -LiteralPath $cudartDir -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue
    if ($Cudart) {
        $Cudart | ForEach-Object { Copy-RuntimeFile $_.FullName $Bin }
    }
}
if ($script:DepsPrefix) {
    foreach ($dir in @(
        (Join-Path $script:DepsPrefix 'bin'),
        (Join-Path $script:DepsPrefix 'embree\bin'),
        (Join-Path $script:DepsPrefix 'embree')
    )) {
        if (-not (Test-Path -LiteralPath $dir)) { continue }
        foreach ($pat in @('embree*.dll', 'tbb*.dll', 'tbbmalloc*.dll', 'openvdb*.dll', 'OpenColorIO*.dll', 'yaml-cpp*.dll', 'expat.dll', 'libexpat*.dll')) {
            Get-ChildItem -LiteralPath $dir -Filter $pat -ErrorAction SilentlyContinue | ForEach-Object {
                Copy-RuntimeFile $_.FullName $Bin
            }
        }
    }
}
if ($script:OcioInstall) {
    if ($script:OcioInstall.Dll) {
        Copy-RuntimeFile $script:OcioInstall.Dll $Bin
    }
    foreach ($dir in @(
        (Join-Path $script:OcioInstall.Prefix 'bin'),
        (Join-Path $script:OcioInstall.Prefix 'lib')
    )) {
        if (-not (Test-Path -LiteralPath $dir)) { continue }
        foreach ($pat in @('OpenColorIO*.dll', 'yaml-cpp*.dll', 'expat.dll', 'libexpat*.dll')) {
            Get-ChildItem -LiteralPath $dir -Filter $pat -ErrorAction SilentlyContinue | ForEach-Object {
                Copy-RuntimeFile $_.FullName $Bin
            }
        }
    }
}

Write-Host ''
Write-Host ("DONE: " + $Exe.FullName) -ForegroundColor Green
Write-Host 'In the app: Engine -> Render Device -> GPU (OptiX)'
Write-Host 'It must NOT say "not in this build". Needs an NVIDIA GPU + current driver.'
Write-Host ''

try { Invoke-Item $Bin } catch { }
exit 0
