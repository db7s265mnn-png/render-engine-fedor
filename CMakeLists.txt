cmake_minimum_required(VERSION 3.20)

# CUDA 12.0 nvcc rejects newer MSVC (VS 2022 17.10+ / VS 2026) unless this
# flag is visible during compiler identification, which runs at enable_language.
if(WIN32)
    set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} --allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
    set(CMAKE_CUDA_COMPILER_ID_FLAGS "--allow-unsupported-compiler")
endif()

project(Solstice VERSION 0.9.0 LANGUAGES CXX)

set(SOLSTICE_APP_DISPLAY_NAME "Grendizer_Render" CACHE STRING "User-visible application name")
set(SOLSTICE_TX_TOOLS_ALPHA_NAME "Grendizer_TX_Tools_Alpha" CACHE STRING "TX Tools Alpha (Qt viewport) name")
set(SOLSTICE_TX_TOOLS_OMEGA_NAME "Grendizer_TX_Tools_Omega" CACHE STRING "TX Tools Omega (DX12 viewport) name")
# Legacy alias kept for older scripts; Alpha/Omega names are the real outputs.
set(SOLSTICE_TX_CONVERTER_NAME "${SOLSTICE_TX_TOOLS_ALPHA_NAME}" CACHE STRING "User-visible TX Tools app name")

# TX Tools editions (both ON by default; Omega is Windows-only).
option(SOLSTICE_BUILD_TX_TOOLS_ALPHA "Build TX Tools Alpha (Qt viewport)" ON)
option(SOLSTICE_BUILD_TX_TOOLS_OMEGA "Build TX Tools Omega (DX12 viewport, Windows)" ON)

# Build identity: PROJECT_VERSION + short git SHA so exe / Help can be traced.
set(SOLSTICE_GIT_SHA "unknown")
set(SOLSTICE_GIT_DIRTY "")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short=7 HEAD
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE SOLSTICE_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" status --porcelain
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE _solstice_git_status
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_solstice_git_status)
        set(SOLSTICE_GIT_DIRTY "-dirty")
    endif()
endif()
string(TIMESTAMP SOLSTICE_BUILD_DATE "%Y%m%d" UTC)
set(SOLSTICE_BUILD_ID "${PROJECT_VERSION}-${SOLSTICE_GIT_SHA}${SOLSTICE_GIT_DIRTY}")
# Human-readable string for Help → About / window title.
set(SOLSTICE_VERSION_FULL "${PROJECT_VERSION} (${SOLSTICE_GIT_SHA}${SOLSTICE_GIT_DIRTY}, ${SOLSTICE_BUILD_DATE})")
# Exe / artifact basename, e.g. Grendizer_Render-0.9.0-dc870a4
set(SOLSTICE_OUTPUT_NAME "${SOLSTICE_APP_DISPLAY_NAME}-${SOLSTICE_BUILD_ID}")
message(STATUS "Build id: ${SOLSTICE_BUILD_ID}")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SOLSTICE_ENABLE_OPTIX "Build the OptiX GPU path tracing backend" OFF)
set(SOLSTICE_OPTIX_NVCC_OPT "1" CACHE STRING
    "nvcc -O level for OptiX PTX (1 = faster compile; OptiX still optimizes at module create)")
option(SOLSTICE_ENABLE_ALEMBIC "Build with Alembic geometry import" ON)
option(SOLSTICE_ENABLE_OPENEXR "Build with OpenEXR image support" ON)
option(SOLSTICE_ENABLE_TIFF "Build with libtiff (.tx / .tif mipmap) support" ON)
option(SOLSTICE_ENABLE_MATERIALX "Build with Academy Software Foundation MaterialX" ON)
option(SOLSTICE_ENABLE_OPENPGL "Build with Intel OpenPGL path guiding (CPU)" ON)
option(SOLSTICE_ENABLE_OPENSUBDIV "Build with OpenSubdiv Catmull-Clark" ON)
option(SOLSTICE_ENABLE_OPENVDB "Build with OpenVDB volume support (optional)" ON)
option(SOLSTICE_BUILD_TESTS "Build unit tests" ON)
option(SOLSTICE_BUILD_APP "Build the Qt desktop application" ON)
option(SOLSTICE_BUILD_TOOLS "Build helper tools (test scene generators)" ON)
option(SOLSTICE_ENABLE_OCIO "Build with OpenColorIO display views (system lib)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optimize for modern x86-64 (AVX2 / BMI1/BMI2 / FMA / …). Drops CPUs older than
# roughly Haswell / Excavator / Zen 1 (~2013–2015). Embree still does its own
# runtime ISA dispatch; this mainly accelerates our integrator / shading code.
option(SOLSTICE_MODERN_CPU "Target x86-64-v3 (AVX2+); faster, not for ancient CPUs" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(MSVC)
    # Also true for clang-cl (MSVC-compatible driver).
    add_compile_options(/W3 /bigobj)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/MP)
    endif()
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN)
    # TinyUSDZ ships the same template instantiations in two .cc files; MSVC then
    # prints thousands of LNK4006 lines and the link looks hung in cmd.exe.
    add_link_options(/IGNORE:4006)
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /IGNORE:4006")
    if(SOLSTICE_MODERN_CPU)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # clang-cl: forward -march to the Clang CC1.
            add_compile_options(/clang:-march=x86-64-v3)
        else()
            add_compile_options(/arch:AVX2)
        endif()
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
    if(SOLSTICE_MODERN_CPU)
        add_compile_options(-march=x86-64-v3)
    endif()
endif()

if(SOLSTICE_MODERN_CPU)
    message(STATUS "CPU baseline: x86-64-v3 / AVX2 (SOLSTICE_MODERN_CPU=ON)")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
qt_standard_project_setup()

find_package(embree 4 REQUIRED CONFIG)

set(SOLSTICE_HAVE_ALEMBIC_01 0)
if(SOLSTICE_ENABLE_ALEMBIC)
    find_package(Alembic CONFIG)
    if(Alembic_FOUND)
        set(SOLSTICE_HAVE_ALEMBIC_01 1)
    else()
        message(WARNING "Alembic not found - .abc import will be disabled")
    endif()
endif()

set(SOLSTICE_HAVE_OPENEXR_01 0)
if(SOLSTICE_ENABLE_OPENEXR)
    find_package(OpenEXR CONFIG)
    if(OpenEXR_FOUND)
        set(SOLSTICE_HAVE_OPENEXR_01 1)
    else()
        message(WARNING "OpenEXR not found - .exr images will be disabled")
    endif()
endif()

# OpenColorIO — Nuke-style Display/View for the viewport (optional).
# Windows CI builds/links OpenColorIO.dll; config/LUTs stay on the user machine.
set(SOLSTICE_HAVE_OCIO_01 0)
set(SOLSTICE_OCIO_TARGET "")
if(SOLSTICE_ENABLE_OCIO)
    find_package(OpenColorIO CONFIG QUIET)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OCIO_PKG QUIET IMPORTED_TARGET OpenColorIO)
    endif()
    if(TARGET OpenColorIO::OpenColorIO)
        # Debian/Ubuntu OpenColorIOConfig can leave include as "/include".
        get_target_property(_sol_ocio_incs OpenColorIO::OpenColorIO INTERFACE_INCLUDE_DIRECTORIES)
        set(_sol_ocio_inc_ok FALSE)
        if(_sol_ocio_incs)
            foreach(_d IN LISTS _sol_ocio_incs)
                if(EXISTS "${_d}/OpenColorIO/OpenColorIO.h")
                    set(_sol_ocio_inc_ok TRUE)
                endif()
            endforeach()
        endif()
        if(NOT _sol_ocio_inc_ok)
            find_path(_sol_ocio_hdr_dir OpenColorIO/OpenColorIO.h)
            if(_sol_ocio_hdr_dir)
                set_property(TARGET OpenColorIO::OpenColorIO APPEND PROPERTY
                             INTERFACE_INCLUDE_DIRECTORIES "${_sol_ocio_hdr_dir}")
                set(_sol_ocio_inc_ok TRUE)
            elseif(EXISTS "/usr/include/OpenColorIO/OpenColorIO.h")
                set_property(TARGET OpenColorIO::OpenColorIO APPEND PROPERTY
                             INTERFACE_INCLUDE_DIRECTORIES "/usr/include")
                set(_sol_ocio_inc_ok TRUE)
            elseif(WIN32)
                # Trust the imported target from a CMAKE_PREFIX_PATH install.
                set(_sol_ocio_inc_ok TRUE)
            endif()
        endif()
        if(_sol_ocio_inc_ok)
            set(SOLSTICE_HAVE_OCIO_01 1)
            set(SOLSTICE_OCIO_TARGET OpenColorIO::OpenColorIO)
        endif()
    endif()
    if(NOT SOLSTICE_HAVE_OCIO_01 AND TARGET PkgConfig::OCIO_PKG)
        set(SOLSTICE_HAVE_OCIO_01 1)
        set(SOLSTICE_OCIO_TARGET PkgConfig::OCIO_PKG)
    endif()
    if(SOLSTICE_HAVE_OCIO_01)
        message(STATUS "OpenColorIO found — Display/View enabled")
    else()
        message(STATUS "OpenColorIO not found — viewport Display/View falls back to passthrough")
    endif()
endif()

# libtiff — Arnold/OIIO .tx textures with embedded mipmaps (SUBIFD / multi-dir TIFF).
set(SOLSTICE_HAVE_TIFF_01 0)
if(SOLSTICE_ENABLE_TIFF)
    find_package(TIFF QUIET)
    if(TIFF_FOUND)
        set(SOLSTICE_HAVE_TIFF_01 1)
    else()
        include(FetchContent)
        message(STATUS "libtiff not found — fetching libtiff 4.6.0 for .tx mipmaps")
        set(tiff-tools OFF CACHE BOOL "" FORCE)
        set(tiff-tests OFF CACHE BOOL "" FORCE)
        set(tiff-docs OFF CACHE BOOL "" FORCE)
        set(tiff-contrib OFF CACHE BOOL "" FORCE)
        set(tiff-install OFF CACHE BOOL "" FORCE)
        set(_solstice_saved_shared_for_tiff ${BUILD_SHARED_LIBS})
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(libtiff
            URL https://gitlab.com/libtiff/libtiff/-/archive/v4.6.0/libtiff-v4.6.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(libtiff)
        set(BUILD_SHARED_LIBS ${_solstice_saved_shared_for_tiff} CACHE BOOL "" FORCE)
        if(TARGET TIFF::TIFF)
            set(SOLSTICE_HAVE_TIFF_01 1)
        elseif(TARGET tiff)
            if(NOT TARGET TIFF::TIFF)
                add_library(TIFF::TIFF ALIAS tiff)
            endif()
            set(SOLSTICE_HAVE_TIFF_01 1)
        endif()
    endif()
    if(NOT SOLSTICE_HAVE_TIFF_01)
        message(WARNING "libtiff not available — .tx mipmap loading disabled")
    endif()
endif()

# OpenVDB — optional heterogeneous volumes (falls back to homogeneous σ when absent).
set(SOLSTICE_HAVE_OPENVDB_01 0)
if(SOLSTICE_ENABLE_OPENVDB)
    # Windows CI installs OpenVDB into DEPS_PREFIX; Embree's TBB lives under embree/.
    if(DEFINED ENV{DEPS_PREFIX} AND EXISTS "$ENV{DEPS_PREFIX}")
        list(APPEND CMAKE_PREFIX_PATH "$ENV{DEPS_PREFIX}" "$ENV{DEPS_PREFIX}/embree")
    endif()
    # Prefer Config package from CMAKE_PREFIX_PATH (Windows deps prefix / vcpkg).
    find_package(OpenVDB CONFIG QUIET)
    if(NOT OpenVDB_FOUND)
        list(APPEND CMAKE_MODULE_PATH "/usr/lib/x86_64-linux-gnu/cmake/OpenVDB")
        find_package(OpenVDB QUIET)
    endif()
    if(OpenVDB_FOUND)
        set(SOLSTICE_HAVE_OPENVDB_01 1)
        message(STATUS "OpenVDB found (package)")
    else()
        # Debian/Ubuntu / manual prefix: FindOpenVDB may be missing; probe libs directly.
        find_library(OPENVDB_LIBRARY NAMES openvdb openvdb_d
            PATHS "$ENV{DEPS_PREFIX}/lib" "$ENV{DEPS_PREFIX}/lib64"
            NO_DEFAULT_PATH)
        if(NOT OPENVDB_LIBRARY)
            find_library(OPENVDB_LIBRARY NAMES openvdb openvdb_d)
        endif()
        find_path(OPENVDB_INCLUDE_DIR openvdb/openvdb.h
            PATHS "$ENV{DEPS_PREFIX}/include"
            NO_DEFAULT_PATH)
        if(NOT OPENVDB_INCLUDE_DIR)
            find_path(OPENVDB_INCLUDE_DIR openvdb/openvdb.h)
        endif()
        if(OPENVDB_LIBRARY AND OPENVDB_INCLUDE_DIR)
            set(SOLSTICE_HAVE_OPENVDB_01 1)
            if(NOT TARGET OpenVDB::openvdb)
                add_library(OpenVDB::openvdb UNKNOWN IMPORTED)
                set_target_properties(OpenVDB::openvdb PROPERTIES
                    IMPORTED_LOCATION "${OPENVDB_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${OPENVDB_INCLUDE_DIR}")
            endif()
            set(OpenVDB_FOUND TRUE)
            message(STATUS "OpenVDB found: ${OPENVDB_LIBRARY}")
        else()
            message(STATUS "OpenVDB not found — VDB media use homogeneous σa/σs fallback")
        endif()
    endif()
endif()

set(SOLSTICE_HAVE_OPTIX_01 0)
set(SOLSTICE_OPTIX_IR_PATH "")
if(SOLSTICE_ENABLE_OPTIX)
    include(cmake/OptiXBackend.cmake)
endif()

# MaterialX (https://github.com/AcademySoftwareFoundation/MaterialX)
set(SOLSTICE_HAVE_MATERIALX_01 0)
set(SOLSTICE_MATERIALX_LIBRARIES "" CACHE PATH "Path to MaterialX libraries/ (stdlib, pbrlib, bxdf)")
if(SOLSTICE_ENABLE_MATERIALX)
    list(APPEND CMAKE_PREFIX_PATH "/tmp/mtlx-install")
    find_package(MaterialX CONFIG QUIET)
    if(NOT MaterialX_FOUND)
        include(FetchContent)
        set(MATERIALX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_VIEWER OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GRAPH_EDITOR OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_JS OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_GLSL OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_OSL OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_MDL OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_GEN_MSL OFF CACHE BOOL "" FORCE)
        set(MATERIALX_BUILD_RENDER OFF CACHE BOOL "" FORCE)
        set(MATERIALX_INSTALL_PYTHON OFF CACHE BOOL "" FORCE)
        set(MATERIALX_INSTALL_RESOURCES OFF CACHE BOOL "" FORCE)
        if(EXISTS "/tmp/mtlx.tgz")
            set(_solstice_mtlx_url "file:///tmp/mtlx.tgz")
        else()
            set(_solstice_mtlx_url
                "https://github.com/AcademySoftwareFoundation/MaterialX/archive/refs/tags/v1.39.4.tar.gz")
        endif()
        FetchContent_Declare(MaterialX
            URL ${_solstice_mtlx_url}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(MaterialX)
        set(MaterialX_FOUND TRUE)
    endif()
    if(MaterialX_FOUND)
        set(SOLSTICE_HAVE_MATERIALX_01 1)
        if(NOT SOLSTICE_MATERIALX_LIBRARIES)
            if(DEFINED materialx_SOURCE_DIR AND EXISTS "${materialx_SOURCE_DIR}/libraries")
                set(SOLSTICE_MATERIALX_LIBRARIES "${materialx_SOURCE_DIR}/libraries" CACHE PATH "" FORCE)
            elseif(EXISTS "/tmp/mtlx-src/libraries")
                set(SOLSTICE_MATERIALX_LIBRARIES "/tmp/mtlx-src/libraries" CACHE PATH "" FORCE)
            elseif(EXISTS "${CMAKE_PREFIX_PATH}/libraries/stdlib")
                set(SOLSTICE_MATERIALX_LIBRARIES "${CMAKE_PREFIX_PATH}/libraries" CACHE PATH "" FORCE)
            endif()
        endif()
        # Also expose libraries next to the runtime binary for shipped builds.
        if(SOLSTICE_MATERIALX_LIBRARIES AND EXISTS "${SOLSTICE_MATERIALX_LIBRARIES}")
            file(COPY "${SOLSTICE_MATERIALX_LIBRARIES}/"
                 DESTINATION "${CMAKE_BINARY_DIR}/bin/materialx/libraries")
            # Multi-config generators (VS) put the exe under bin/Release.
            file(COPY "${SOLSTICE_MATERIALX_LIBRARIES}/"
                 DESTINATION "${CMAKE_BINARY_DIR}/bin/Release/materialx/libraries")
        endif()
    else()
        message(WARNING "MaterialX not found - Material Network will be disabled")
    endif()
endif()

# OpenPGL (https://github.com/RenderKit/openpgl) — CPU path guiding
set(SOLSTICE_HAVE_OPENPGL_01 0)
if(SOLSTICE_ENABLE_OPENPGL)
    include(FetchContent)
    find_package(TBB QUIET)
    if(NOT TBB_FOUND AND NOT TARGET TBB::tbb)
        message(STATUS "TBB not found — fetching oneTBB for OpenPGL")
        set(TBB_TEST OFF CACHE BOOL "" FORCE)
        set(TBB_STRICT OFF CACHE BOOL "" FORCE)
        set(_solstice_saved_build_shared ${BUILD_SHARED_LIBS})
        set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
        if(EXISTS "/tmp/onetbb.tgz")
            set(_solstice_tbb_url "file:///tmp/onetbb.tgz")
        else()
            set(_solstice_tbb_url
                "https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.12.0.tar.gz")
        endif()
        FetchContent_Declare(onetbb
            URL ${_solstice_tbb_url}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(onetbb)
        set(BUILD_SHARED_LIBS ${_solstice_saved_build_shared} CACHE BOOL "" FORCE)
        # OpenPGL's FindTBB cannot parse generator expressions on in-tree TBB
        # alias targets — give it a concrete include root instead.
        FetchContent_GetProperties(onetbb SOURCE_DIR _solstice_onetbb_src)
        if(_solstice_onetbb_src)
            set(TBB_INCLUDE_DIR "${_solstice_onetbb_src}/include" CACHE PATH "" FORCE)
            set(OPENPGL_TBB_ROOT "${_solstice_onetbb_src}" CACHE PATH "" FORCE)
            set(TBB_ROOT "${_solstice_onetbb_src}" CACHE PATH "" FORCE)
        endif()
    endif()
    if(TBB_FOUND OR TARGET TBB::tbb)
        set(OPENPGL_BUILD_STATIC ON CACHE BOOL "" FORCE)
        set(OPENPGL_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(OPENPGL_BUILD_CHECK_TOOL OFF CACHE BOOL "" FORCE)
        set(OPENPGL_ISA_AVX512 OFF CACHE BOOL "" FORCE)
        if(EXISTS "/tmp/openpgl.tgz")
            set(_solstice_openpgl_url "file:///tmp/openpgl.tgz")
        else()
            set(_solstice_openpgl_url
                "https://github.com/RenderKit/openpgl/archive/refs/tags/v0.7.1.tar.gz")
        endif()
        FetchContent_Declare(openpgl
            URL ${_solstice_openpgl_url}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(openpgl)
        if(TARGET openpgl)
            set(SOLSTICE_HAVE_OPENPGL_01 1)
        else()
            message(WARNING "OpenPGL target missing — path guiding disabled")
        endif()
    else()
        message(WARNING "TBB not available — OpenPGL path guiding disabled")
    endif()
endif()

# OpenSubdiv — Catmull-Clark for displacement tessellation
set(SOLSTICE_HAVE_OPENSUBDIV_01 0)
if(SOLSTICE_ENABLE_OPENSUBDIV AND NOT WIN32)
    include(FetchContent)
    message(STATUS "Fetching OpenSubdiv for Catmull-Clark tessellation")
    set(NO_TUTORIALS ON CACHE BOOL "" FORCE)
    set(NO_EXAMPLES ON CACHE BOOL "" FORCE)
    set(NO_REGRESSION ON CACHE BOOL "" FORCE)
    set(NO_DOC ON CACHE BOOL "" FORCE)
    set(NO_OMP ON CACHE BOOL "" FORCE)
    set(NO_TBB ON CACHE BOOL "" FORCE)
    set(NO_CUDA ON CACHE BOOL "" FORCE)
    set(NO_OPENCL ON CACHE BOOL "" FORCE)
    set(NO_OPENGL ON CACHE BOOL "" FORCE)
    set(NO_METAL ON CACHE BOOL "" FORCE)
    set(NO_DX ON CACHE BOOL "" FORCE)
    set(NO_TESTS ON CACHE BOOL "" FORCE)
    set(NO_GLTESTS ON CACHE BOOL "" FORCE)
    set(NO_CLEW ON CACHE BOOL "" FORCE)
    set(NO_PTEX ON CACHE BOOL "" FORCE)
    if(EXISTS "/tmp/opensubdiv.tgz")
        set(_solstice_osd_url "file:///tmp/opensubdiv.tgz")
    else()
        set(_solstice_osd_url
            "https://github.com/PixarAnimationStudios/OpenSubdiv/archive/refs/tags/v3_6_1.tar.gz")
    endif()
    FetchContent_Declare(opensubdiv
        URL ${_solstice_osd_url}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(opensubdiv)
    if(TARGET osd_static_cpu OR TARGET osdCPU)
        set(SOLSTICE_HAVE_OPENSUBDIV_01 1)
    else()
        message(WARNING "OpenSubdiv targets missing — catclark falls back to linear")
    endif()
elseif(SOLSTICE_ENABLE_OPENSUBDIV AND WIN32)
    message(STATUS "OpenSubdiv disabled on Windows (clang-cl/MSVC climits conflict) — catclark → linear")
endif()

# TinyUSDZ — binary USD (.usdc / crate .usd / .usdz) → USDA text for our importer
set(SOLSTICE_HAVE_TINYUSDZ_01 0)
option(SOLSTICE_ENABLE_TINYUSDZ "Build with TinyUSDZ for binary USD import" ON)
if(SOLSTICE_ENABLE_TINYUSDZ)
    include(FetchContent)
    message(STATUS "Fetching TinyUSDZ for binary USD (.usdc) support")
    set(TINYUSDZ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_EXR OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_TIFF OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_OPENSUBDIV OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_USDMTLX OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_AUDIO OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_TYDRA OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_C_API OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_USD_TO_GLTF OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_PXR_COMPAT_API OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_BUILTIN_IMAGE_LOADER OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_USDOBJ OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_USDFBX OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_USDVOX OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_WITH_PYTHON OFF CACHE BOOL "" FORCE)
    set(TINYUSDZ_NO_WERROR ON CACHE BOOL "" FORCE)
    set(TINYUSDZ_USE_CCACHE OFF CACHE BOOL "" FORCE)
    if(EXISTS "/tmp/tinyusdz.tgz")
        set(_solstice_tinyusdz_url "file:///tmp/tinyusdz.tgz")
    else()
        set(_solstice_tinyusdz_url
            "https://github.com/lighttransport/tinyusdz/archive/refs/tags/v0.9.4.tar.gz")
    endif()
    FetchContent_Declare(tinyusdz
        URL ${_solstice_tinyusdz_url}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    # TinyUSDZ headers must not be scanned by Qt AUTOMOC.
    set(_solstice_saved_automoc ${CMAKE_AUTOMOC})
    set(_solstice_saved_autouic ${CMAKE_AUTOUIC})
    set(_solstice_saved_autorcc ${CMAKE_AUTORCC})
    set(CMAKE_AUTOMOC OFF)
    set(CMAKE_AUTOUIC OFF)
    set(CMAKE_AUTORCC OFF)
    FetchContent_MakeAvailable(tinyusdz)
    set(CMAKE_AUTOMOC ${_solstice_saved_automoc})
    set(CMAKE_AUTOUIC ${_solstice_saved_autouic})
    set(CMAKE_AUTORCC ${_solstice_saved_autorcc})
    if(TARGET tinyusdz_static)
        set(SOLSTICE_HAVE_TINYUSDZ_01 1)
        FetchContent_GetProperties(tinyusdz SOURCE_DIR _solstice_tinyusdz_src)
        set_target_properties(tinyusdz_static PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
        if(MSVC)
            target_compile_options(tinyusdz_static PRIVATE /W0)
        endif()
    else()
        message(WARNING "TinyUSDZ target missing — binary USD import disabled")
    endif()
endif()

configure_file(${CMAKE_SOURCE_DIR}/cmake/solstice_config.h.in
               ${CMAKE_BINARY_DIR}/generated/solstice_config.h @ONLY)

# ---------------------------------------------------------------------------
# Core library: math, scene, IO, renderer and node graph
# ---------------------------------------------------------------------------
set(SOLSTICE_LIB_SOURCES
    src/core/image.cpp
    src/core/log.cpp
    src/core/expr_eval.cpp
    src/scene/scene.cpp
    src/scene/displace.cpp
    src/scene/tessellate.cpp
    src/scene/triangulate.cpp
    src/scene/volume_grid.cpp
    src/scene/dcsdd_contouring.cpp
    src/io/image_io.cpp
    src/io/tx_cache.cpp
    src/io/tx_convert.cpp
    src/io/ocio_util.cpp
    src/io/alembic_loader.cpp
    src/io/usd_loader.cpp
    src/io/materialx_graph.cpp
    src/io/materialx_bake.cpp
    src/io/materialx_compile.cpp
    src/render/framebuffer.cpp
    src/render/rgb_spectrum_tables.cpp
    src/render/render_session.cpp
    src/render/scene_picker.cpp
    src/render/motion_blur.cpp
    src/render/volume_vdb.cpp
    src/render/physical_sky.cpp
    third_party/hosek_wilkie/ArHosekSkyModel.cc
    src/render/cpu/embree_device.cpp
    src/render/cpu/polynomial_optics.cpp
    src/render/cpu/path_guiding.cpp
    src/render/optix/optix_device_stub.cpp
    src/nodes/parameter.cpp
    src/nodes/node.cpp
    src/nodes/stage.cpp
    src/nodes/node_graph.cpp
    src/nodes/node_registry.cpp
    src/nodes/nodes_library.cpp
    src/nodes/nodes_vdb.cpp
    src/app/default_scene.cpp
    src/app/document.cpp
    src/app/headless.cpp
)

set(SOLSTICE_LIB_HEADERS
    src/core/math.h
    src/core/units.h
    src/core/rng.h
    src/core/image.h
    src/core/log.h
    src/core/expr_eval.h
    src/core/thread_pool.h
    src/scene/types.h
    src/scene/scene.h
    src/scene/displace.h
    src/scene/tessellate.h
    src/scene/triangulate.h
    src/scene/volume_grid.h
    src/scene/dcsdd_contouring.h
    src/io/image_io.h
    src/io/tx_cache.h
    src/io/tx_convert.h
    src/io/ocio_util.h
    src/io/alembic_loader.h
    src/io/usd_loader.h
    src/io/materialx_graph.h
    src/io/materialx_bake.h
    src/io/materialx_compile.h
    src/render/procedural.h
    src/render/shading.h
    src/render/lights.h
    src/render/physical_sky.h
    src/render/integrator.h
    src/render/photon_map.h
    src/render/framebuffer.h
    src/render/render_device.h
    src/render/render_session.h
    src/render/scene_picker.h
    src/nodes/parameter.h
    src/nodes/node.h
    src/nodes/stage.h
    src/nodes/node_graph.h
    src/nodes/node_registry.h
    src/app/default_scene.h
    src/app/document.h
    src/app/headless.h
)

add_library(solstice_lib STATIC ${SOLSTICE_LIB_SOURCES} ${SOLSTICE_LIB_HEADERS})
target_include_directories(solstice_lib PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/generated)
target_include_directories(solstice_lib PRIVATE ${CMAKE_SOURCE_DIR}/third_party/lentil)
target_include_directories(solstice_lib PRIVATE ${CMAKE_SOURCE_DIR}/third_party/earcut)
target_include_directories(solstice_lib PRIVATE ${CMAKE_SOURCE_DIR}/third_party/hosek_wilkie)
target_link_libraries(solstice_lib PUBLIC Qt6::Core Qt6::Gui embree)

if(SOLSTICE_HAVE_ALEMBIC_01)
    target_link_libraries(solstice_lib PUBLIC Alembic::Alembic)
endif()
if(SOLSTICE_HAVE_OPENVDB_01)
    target_link_libraries(solstice_lib PUBLIC OpenVDB::openvdb)
    # OpenVDB pulls TBB / Boost / Blosc transitively on most distros.
    find_package(TBB QUIET)
    if(TBB_FOUND)
        target_link_libraries(solstice_lib PUBLIC TBB::tbb)
    endif()
endif()
if(SOLSTICE_HAVE_OPENEXR_01)
    target_link_libraries(solstice_lib PUBLIC OpenEXR::OpenEXR)
endif()
if(SOLSTICE_HAVE_OCIO_01 AND SOLSTICE_OCIO_TARGET)
    target_link_libraries(solstice_lib PUBLIC ${SOLSTICE_OCIO_TARGET})
endif()
if(SOLSTICE_HAVE_TIFF_01)
    target_link_libraries(solstice_lib PUBLIC TIFF::TIFF)
endif()
if(SOLSTICE_HAVE_MATERIALX_01)
    if(TARGET MaterialX::MaterialXCore)
        target_link_libraries(solstice_lib PUBLIC MaterialX::MaterialXCore MaterialX::MaterialXFormat)
    elseif(TARGET MaterialXCore)
        target_link_libraries(solstice_lib PUBLIC MaterialXCore MaterialXFormat)
    else()
        target_include_directories(solstice_lib PUBLIC /tmp/mtlx-install/include)
        target_link_directories(solstice_lib PUBLIC /tmp/mtlx-install/lib)
        target_link_libraries(solstice_lib PUBLIC MaterialXFormat MaterialXCore)
    endif()
endif()
if(SOLSTICE_HAVE_OPTIX_01)
    target_sources(solstice_lib PRIVATE
        src/render/optix/optix_device.cpp
        ${SOLSTICE_OPTIX_EMBED_SOURCES})
    target_link_libraries(solstice_lib PUBLIC CUDA::cudart CUDA::cuda_driver)
    target_include_directories(solstice_lib PUBLIC ${OptiX_INCLUDE_DIR})
    add_dependencies(solstice_lib solstice_optix_programs)
endif()

if(SOLSTICE_HAVE_OPENPGL_01)
    target_link_libraries(solstice_lib PUBLIC openpgl)
    if(TARGET TBB::tbb)
        target_link_libraries(solstice_lib PUBLIC TBB::tbb)
    endif()
endif()

if(SOLSTICE_HAVE_OPENSUBDIV_01)
    if(TARGET osd_static_cpu)
        target_link_libraries(solstice_lib PUBLIC osd_static_cpu)
    elseif(TARGET osdCPU)
        target_link_libraries(solstice_lib PUBLIC osdCPU)
    endif()
    if(DEFINED opensubdiv_SOURCE_DIR)
        target_include_directories(solstice_lib PUBLIC ${opensubdiv_SOURCE_DIR})
    endif()
endif()

if(SOLSTICE_HAVE_TINYUSDZ_01)
    target_link_libraries(solstice_lib PRIVATE tinyusdz_static)
    if(_solstice_tinyusdz_src)
        target_include_directories(solstice_lib PRIVATE ${_solstice_tinyusdz_src}/src)
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(solstice_lib PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
set(SOLSTICE_APP_SOURCES
    src/main.cpp
    src/ui/main_window.cpp
    src/ui/material_network_view.cpp
    src/ui/material_wire_item.cpp
    src/ui/node_graph_view.cpp
    src/ui/node_item.cpp
    src/ui/node_icons.cpp
    src/ui/connection_item.cpp
    src/ui/parameter_panel.cpp
    src/ui/render_view.cpp
    src/ui/scene_graph_panel.cpp
    src/ui/log_panel.cpp
    src/ui/theme.cpp
    src/ui/timeline_bar.cpp
    src/ui/texture_file_dialog.cpp
)

set(SOLSTICE_APP_HEADERS
    src/ui/main_window.h
    src/ui/material_network_view.h
    src/ui/material_wire_item.h
    src/ui/node_graph_view.h
    src/ui/node_item.h
    src/ui/node_icons.h
    src/ui/connection_item.h
    src/ui/parameter_panel.h
    src/ui/render_view.h
    src/ui/scene_graph_panel.h
    src/ui/log_panel.h
    src/ui/theme.h
    src/ui/timeline_bar.h
    src/ui/texture_file_dialog.h
)

if(SOLSTICE_BUILD_APP)
    set(CMAKE_AUTORCC ON)
    set(SOLSTICE_APP_ICON_SOURCES resources/icons.qrc)
    if(WIN32)
        # Embed multi-size .ico so Explorer / taskbar / shortcuts use the app icon.
        list(APPEND SOLSTICE_APP_ICON_SOURCES resources/app_icon.rc)
    endif()
    qt_add_executable(Solstice WIN32 MACOSX_BUNDLE
        ${SOLSTICE_APP_SOURCES}
        ${SOLSTICE_APP_HEADERS}
        ${SOLSTICE_APP_ICON_SOURCES})
    target_link_libraries(Solstice PRIVATE solstice_lib Qt6::Widgets)
    set_target_properties(Solstice PROPERTIES
        OUTPUT_NAME "${SOLSTICE_OUTPUT_NAME}"
        MACOSX_BUNDLE_BUNDLE_NAME "${SOLSTICE_APP_DISPLAY_NAME}"
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.grendizer.render")
    # Write build id next to the binary so packaging / CI can discover the exe name.
    file(WRITE "${CMAKE_BINARY_DIR}/solstice_build_id.txt" "${SOLSTICE_BUILD_ID}\n")
    file(WRITE "${CMAKE_BINARY_DIR}/solstice_output_name.txt" "${SOLSTICE_OUTPUT_NAME}\n")

    # Standalone TX Tools (maketx/OCIO core + texture viewer).
    # Alpha = Qt FloatPreviewCanvas; Omega = same UI, DX12 viewport (Windows only).
    set(TX_CONVERTER_ICON_SOURCES resources/icons.qrc)
    if(WIN32)
        list(APPEND TX_CONVERTER_ICON_SOURCES resources/app_icon.rc)
    endif()
    set(TX_TOOLS_COMMON_SOURCES
        TX_Converter/main.cpp
        TX_Converter/texture_viewer.cpp
        TX_Converter/texture_viewer.h
        src/ui/theme.cpp
        src/ui/timeline_bar.cpp
        src/ui/timeline_bar.h
        src/ui/texture_file_dialog.cpp
        src/ui/texture_file_dialog.h
        ${TX_CONVERTER_ICON_SOURCES})

    set(TX_TOOLS_DEPLOY_TARGETS)

    if(SOLSTICE_BUILD_TX_TOOLS_ALPHA)
        qt_add_executable(TX_Converter_Alpha WIN32 MACOSX_BUNDLE
            ${TX_TOOLS_COMMON_SOURCES})
        target_include_directories(TX_Converter_Alpha PRIVATE ${CMAKE_SOURCE_DIR}/TX_Converter)
        target_link_libraries(TX_Converter_Alpha PRIVATE solstice_lib Qt6::Widgets)
        set_target_properties(TX_Converter_Alpha PROPERTIES
            OUTPUT_NAME "${SOLSTICE_TX_TOOLS_ALPHA_NAME}"
            MACOSX_BUNDLE_BUNDLE_NAME "${SOLSTICE_TX_TOOLS_ALPHA_NAME}"
            MACOSX_BUNDLE_GUI_IDENTIFIER "com.grendizer.txtools.alpha")
        target_compile_definitions(TX_Converter_Alpha PRIVATE
            SOLSTICE_TX_CONVERTER_NAME="${SOLSTICE_TX_TOOLS_ALPHA_NAME}"
            SOLSTICE_TX_VIEWER_DX12=0)
        list(APPEND TX_TOOLS_DEPLOY_TARGETS TX_Converter_Alpha)
        # Back-compat alias for scripts that still expect TX_Converter.
        add_executable(TX_Converter ALIAS TX_Converter_Alpha)
    endif()

    if(WIN32 AND SOLSTICE_BUILD_TX_TOOLS_OMEGA)
        qt_add_executable(TX_Converter_Omega WIN32
            ${TX_TOOLS_COMMON_SOURCES}
            TX_Converter/dx12_preview_canvas.cpp
            TX_Converter/dx12_preview_canvas.h)
        target_include_directories(TX_Converter_Omega PRIVATE ${CMAKE_SOURCE_DIR}/TX_Converter)
        target_link_libraries(TX_Converter_Omega PRIVATE solstice_lib Qt6::Widgets
            d3d12 dxgi d3dcompiler)
        set_target_properties(TX_Converter_Omega PROPERTIES
            OUTPUT_NAME "${SOLSTICE_TX_TOOLS_OMEGA_NAME}")
        target_compile_definitions(TX_Converter_Omega PRIVATE
            SOLSTICE_TX_CONVERTER_NAME="${SOLSTICE_TX_TOOLS_OMEGA_NAME}"
            SOLSTICE_TX_VIEWER_DX12=1)
        list(APPEND TX_TOOLS_DEPLOY_TARGETS TX_Converter_Omega)
    endif()
endif()

# ---------------------------------------------------------------------------
# Tools and tests
# ---------------------------------------------------------------------------
if(SOLSTICE_BUILD_TOOLS AND SOLSTICE_HAVE_ALEMBIC_01)
    add_executable(sol_make_test_abc tools/make_test_abc.cpp)
    target_link_libraries(sol_make_test_abc PRIVATE solstice_lib)
endif()

if(SOLSTICE_BUILD_TESTS)
    enable_testing()
    add_executable(solstice_tests tests/test_main.cpp)
    target_link_libraries(solstice_tests PRIVATE solstice_lib)
    add_test(NAME solstice_tests COMMAND solstice_tests)
endif()

# ---------------------------------------------------------------------------
# Install / deploy
# ---------------------------------------------------------------------------
if(SOLSTICE_BUILD_APP)
    install(TARGETS Solstice
            BUNDLE DESTINATION .
            RUNTIME DESTINATION bin)
endif()

if(WIN32 AND SOLSTICE_BUILD_APP)
    # Produces a self contained folder with the Qt runtime next to the executable.
    get_target_property(SOLSTICE_QMAKE_EXECUTABLE Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(SOLSTICE_QT_BIN_DIR "${SOLSTICE_QMAKE_EXECUTABLE}" DIRECTORY)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${SOLSTICE_QT_BIN_DIR}" ${Qt6_DIR}/../../../bin)
    if(WINDEPLOYQT_EXECUTABLE)
        set(_sol_deploy_cmds
            COMMAND ${CMAKE_COMMAND} -E echo "Deploying Qt runtime next to ${SOLSTICE_OUTPUT_NAME} + TX Tools"
            COMMAND ${WINDEPLOYQT_EXECUTABLE} --release --no-translations --no-system-d3d-compiler
                    --compiler-runtime $<TARGET_FILE:Solstice>)
        set(_sol_deploy_deps Solstice)
        foreach(_tx ${TX_TOOLS_DEPLOY_TARGETS})
            list(APPEND _sol_deploy_cmds
                COMMAND ${WINDEPLOYQT_EXECUTABLE} --release --no-translations --no-system-d3d-compiler
                        $<TARGET_FILE:${_tx}>)
            list(APPEND _sol_deploy_deps ${_tx})
        endforeach()
        add_custom_target(deploy ${_sol_deploy_cmds} DEPENDS ${_sol_deploy_deps})
    endif()
endif()

message(STATUS "")
message(STATUS "${SOLSTICE_APP_DISPLAY_NAME} ${SOLSTICE_VERSION_FULL}")
message(STATUS "  Output name     : ${SOLSTICE_OUTPUT_NAME}")
message(STATUS "  Compiler        : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Qt              : ${Qt6_VERSION}")
message(STATUS "  Embree          : ${EMBREE_VERSION}")
message(STATUS "  Alembic support : ${SOLSTICE_HAVE_ALEMBIC_01}")
message(STATUS "  OpenEXR support : ${SOLSTICE_HAVE_OPENEXR_01}")
message(STATUS "  OpenColorIO     : ${SOLSTICE_HAVE_OCIO_01}")
message(STATUS "  TIFF /.tx mips  : ${SOLSTICE_HAVE_TIFF_01}")
message(STATUS "  OpenVDB volumes : ${SOLSTICE_HAVE_OPENVDB_01}")
message(STATUS "  MaterialX       : ${SOLSTICE_HAVE_MATERIALX_01}")
message(STATUS "  OptiX backend   : ${SOLSTICE_HAVE_OPTIX_01}")
message(STATUS "  OpenPGL guiding : ${SOLSTICE_HAVE_OPENPGL_01}")
message(STATUS "  OpenSubdiv      : ${SOLSTICE_HAVE_OPENSUBDIV_01}")
message(STATUS "  TinyUSDZ / USDC : ${SOLSTICE_HAVE_TINYUSDZ_01}")
message(STATUS "  Modern CPU ISA  : ${SOLSTICE_MODERN_CPU}")
message(STATUS "")

# OBJ -> Alembic converter for shipping demo assets.
if(SOLSTICE_BUILD_TOOLS AND SOLSTICE_HAVE_ALEMBIC_01)
    add_executable(sol_obj_to_abc tools/obj_to_abc.cpp)
    target_link_libraries(sol_obj_to_abc PRIVATE Alembic::Alembic)
endif()
