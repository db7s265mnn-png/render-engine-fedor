# Configures the optional OptiX GPU backend: finds CUDA and the OptiX SDK,
# compiles the device programs to PTX and embeds them into the executable.
#
# Point the build at the SDK with -DOptiX_ROOT=/path/to/OptiX-SDK or by setting
# the OptiX_INSTALL_DIR environment variable.
#
# PTX is produced via an explicit nvcc custom command (not CUDA_PTX_COMPILATION
# OBJECT libraries). The VS generator otherwise injects host flags like /W3 /MP
# into the nvcc cmdline and fails with:
#   "A single input file is required for a non-link phase..."
#
# On Windows, nvcc must use MSVC cl.exe as the host compiler. Passing clang-cl
# via -ccbin has hung CI for tens of minutes.

set(SOLSTICE_CUDA_HOST_COMPILER "" CACHE FILEPATH
    "Host compiler for nvcc (-ccbin). On Windows this should be MSVC cl.exe.")

if(WIN32)
    if(SOLSTICE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${SOLSTICE_CUDA_HOST_COMPILER}" CACHE FILEPATH "" FORCE)
    elseif(NOT CMAKE_CUDA_HOST_COMPILER AND CMAKE_CXX_COMPILER MATCHES "clang-cl")
        find_program(_solstice_cl NAMES cl.exe)
        if(_solstice_cl)
            set(CMAKE_CUDA_HOST_COMPILER "${_solstice_cl}" CACHE FILEPATH "" FORCE)
            set(SOLSTICE_CUDA_HOST_COMPILER "${_solstice_cl}" CACHE FILEPATH "" FORCE)
            message(STATUS "nvcc host compiler (auto cl.exe): ${_solstice_cl}")
        else()
            message(WARNING "clang-cl build with OptiX: cl.exe not on PATH. Pass -DSOLSTICE_CUDA_HOST_COMPILER=<cl.exe>")
        endif()
    endif()
endif()

include(CheckLanguage)
if(WIN32)
    # CUDA 12.0 nvcc rejects VS 2022 17.10+ / VS 2026 during CMake compiler ID.
    # We compile PTX via a custom nvcc command and only need CUDAToolkit for
    # headers + cudart, so skip enable_language(CUDA) on Windows.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
    set(CMAKE_CUDA_COMPILER_ID_FLAGS "--allow-unsupported-compiler")
endif()
if(NOT CMAKE_CUDA_COMPILER)
    check_language(CUDA)
endif()
if(NOT CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER NAMES nvcc nvcc.exe)
endif()
if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "OptiX backend requested but no CUDA compiler was found - disabling it")
    return()
endif()

if(NOT WIN32)
    enable_language(CUDA)
endif()
find_package(CUDAToolkit REQUIRED)

set(_solstice_default_optix_arch "compute_60")
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
    set(_solstice_default_optix_arch "compute_75")
endif()
set(SOLSTICE_OPTIX_ARCH "${_solstice_default_optix_arch}" CACHE STRING
    "Virtual CUDA arch for OptiX PTX (--gpu-architecture)")
# CUDA 13 dropped Pascal (compute_60). A leftover cache from CUDA 12 would fail nvcc.
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0 AND SOLSTICE_OPTIX_ARCH MATCHES "compute_6")
    message(STATUS "CUDA ${CUDAToolkit_VERSION} dropped ${SOLSTICE_OPTIX_ARCH}; OptiX PTX uses compute_75")
    set(SOLSTICE_OPTIX_ARCH "compute_75")
endif()
message(STATUS "OptiX PTX arch: ${SOLSTICE_OPTIX_ARCH} (CUDA ${CUDAToolkit_VERSION})")

find_path(OptiX_INCLUDE_DIR
    NAMES optix.h
    HINTS
        ${OptiX_ROOT}
        $ENV{OptiX_ROOT}
        ${OptiX_INSTALL_DIR}
        $ENV{OptiX_INSTALL_DIR}
        /usr/local/optix
        /opt/optix
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0"
    PATH_SUFFIXES include
)

if(NOT OptiX_INCLUDE_DIR)
    message(WARNING "OptiX headers not found - set OptiX_ROOT to the SDK directory. GPU backend disabled.")
    return()
endif()

message(STATUS "OptiX headers: ${OptiX_INCLUDE_DIR}")

set(_solstice_optix_dir ${CMAKE_SOURCE_DIR}/src/render/optix)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated)

if(NOT DEFINED SOLSTICE_OPTIX_NVCC_OPT)
    set(SOLSTICE_OPTIX_NVCC_OPT "1")
endif()
message(STATUS "OptiX PTX: wavefront modules (init/intersect/shade), nvcc -O${SOLSTICE_OPTIX_NVCC_OPT}")

set(_solstice_nvcc_inc_flags)
foreach(_inc IN LISTS CUDAToolkit_INCLUDE_DIRS)
    list(APPEND _solstice_nvcc_inc_flags "-I${_inc}")
endforeach()

set(_solstice_nvcc_ccbin)
if(SOLSTICE_CUDA_HOST_COMPILER)
    # Two-arg form so paths with spaces (Program Files) stay one nvcc flag value.
    set(_solstice_nvcc_ccbin -ccbin "${SOLSTICE_CUDA_HOST_COMPILER}")
elseif(WIN32 AND CMAKE_CUDA_HOST_COMPILER)
    set(_solstice_nvcc_ccbin -ccbin "${CMAKE_CUDA_HOST_COMPILER}")
elseif(MSVC AND CMAKE_CXX_COMPILER AND NOT CMAKE_CXX_COMPILER MATCHES "clang-cl")
    # Help nvcc find cl.exe when driven outside the VS CUDA .targets path.
    set(_solstice_nvcc_ccbin -ccbin "${CMAKE_CXX_COMPILER}")
endif()

# -lineinfo makes the huge shared integrator PTX much slower to compile; keep it
# for Debug so GPU faults still have source maps.
set(_solstice_nvcc_lineinfo)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_solstice_nvcc_lineinfo -lineinfo)
endif()

# VS 2026 STL (yvals_core.h STL1002) requires CUDA 13.2; CUDA 12.0 needs this define.
set(_solstice_nvcc_unsupported)
if(WIN32)
    set(_solstice_nvcc_unsupported
        --allow-unsupported-compiler
        -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
        -D_ENABLE_EXTENDED_ALIGNED_STORAGE
        -Xcompiler=/bigobj,/nologo)
endif()

# Cycles: each integrator stage is its own kernel (intersect_closest, shade_surface,
# …) so cicc never sees optixTrace + BSDF + lights in one megakernel. ninja compiles
# these TUs in parallel. -O1: OptiX re-optimizes at optixModuleCreate.
set(_solstice_nvcc_ptx_common
    ${_solstice_nvcc_ccbin}
    -ptx
    -std=c++17
    -O${SOLSTICE_OPTIX_NVCC_OPT}
    --use_fast_math
    --disable-warnings
    ${_solstice_nvcc_lineinfo}
    ${_solstice_nvcc_unsupported}
    -arch=${SOLSTICE_OPTIX_ARCH}
    -D_USE_MATH_DEFINES
    -DNOMINMAX
    -DWIN32_LEAN_AND_MEAN
    -I${CMAKE_SOURCE_DIR}/src
    -I${CMAKE_BINARY_DIR}/generated
    -I${OptiX_INCLUDE_DIR}
    ${_solstice_nvcc_inc_flags})

set(_solstice_optix_base
    ${_solstice_optix_dir}/optix_common.cuh
    ${_solstice_optix_dir}/optix_wavefront.cuh
    ${_solstice_optix_dir}/launch_params.h
    ${_solstice_optix_dir}/path_state.h
    ${CMAKE_SOURCE_DIR}/src/scene/types.h
    ${CMAKE_SOURCE_DIR}/src/core/math.h
    ${CMAKE_SOURCE_DIR}/src/core/rng.h)

set(SOLSTICE_OPTIX_EMBED_SOURCES "")

function(solstice_optix_kernel name source symbol)
    cmake_parse_arguments(K "LIGHTS" "" "DEPENDS" ${ARGN})
    set(ptx "${CMAKE_BINARY_DIR}/generated/optix_${name}.ptx")
    set(embed "${CMAKE_BINARY_DIR}/generated/solstice_optix_${name}_ir.cpp")
    set(defs)
    if(K_LIGHTS)
        list(APPEND defs -DSOLSTICE_OPTIX_KERNEL=1)
    endif()
    add_custom_command(
        OUTPUT ${ptx}
        COMMAND ${CMAKE_COMMAND} -E echo "nvcc PTX ${name} start"
        COMMAND ${CMAKE_CUDA_COMPILER}
                ${_solstice_nvcc_ptx_common}
                ${defs}
                -o ${ptx}
                ${source}
        COMMAND ${CMAKE_COMMAND} -E echo "nvcc PTX ${name} done"
        DEPENDS ${source} ${K_DEPENDS}
        COMMENT "nvcc OptiX PTX (${name})"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${embed}
        COMMAND ${CMAKE_COMMAND}
                -DINPUT=${ptx}
                -DOUTPUT=${embed}
                -DSYMBOL=${symbol}
                -P ${CMAKE_SOURCE_DIR}/cmake/embed_binary.cmake
        DEPENDS ${ptx}
                ${CMAKE_SOURCE_DIR}/cmake/embed_binary.cmake
                ${CMAKE_SOURCE_DIR}/cmake/embed_binary.py
        COMMENT "Embedding OptiX PTX (${name})"
        VERBATIM
    )
    set(_acc "${SOLSTICE_OPTIX_EMBED_SOURCES}")
    list(APPEND _acc ${embed})
    set(SOLSTICE_OPTIX_EMBED_SOURCES "${_acc}" PARENT_SCOPE)
    set_source_files_properties(${embed} PROPERTIES GENERATED TRUE)
endfunction()

solstice_optix_kernel(hit_miss
    ${_solstice_optix_dir}/optix_hit_miss.cu
    solsticeOptixHitIr
    DEPENDS ${_solstice_optix_base})

solstice_optix_kernel(init_from_camera
    ${_solstice_optix_dir}/optix_init_from_camera.cu
    solsticeOptixInitIr
    DEPENDS ${_solstice_optix_base}
            ${_solstice_optix_dir}/optix_geom.cuh
            ${_solstice_optix_dir}/optix_volume.cuh
            ${CMAKE_SOURCE_DIR}/src/render/blue_noise.h
            ${CMAKE_SOURCE_DIR}/src/render/volume.h
            ${CMAKE_SOURCE_DIR}/src/render/volume_track.h)

solstice_optix_kernel(intersect_closest
    ${_solstice_optix_dir}/optix_intersect_closest.cu
    solsticeOptixIntersectClosestIr
    DEPENDS ${_solstice_optix_base} ${_solstice_optix_dir}/optix_trace.cuh)

solstice_optix_kernel(intersect_shadow
    ${_solstice_optix_dir}/optix_intersect_shadow.cu
    solsticeOptixIntersectShadowIr
    DEPENDS ${_solstice_optix_base} ${_solstice_optix_dir}/optix_trace.cuh)

solstice_optix_kernel(shade_surface
    ${_solstice_optix_dir}/optix_shade_surface.cu
    solsticeOptixShadeSurfaceIr
    LIGHTS
    DEPENDS ${_solstice_optix_base}
            ${_solstice_optix_dir}/optix_geom.cuh
            ${_solstice_optix_dir}/optix_bsdf.cuh
            ${_solstice_optix_dir}/optix_volume.cuh
            ${CMAKE_SOURCE_DIR}/src/render/lights.h
            ${CMAKE_SOURCE_DIR}/src/render/volume.h
            ${CMAKE_SOURCE_DIR}/src/render/volume_track.h)

solstice_optix_kernel(shade_background
    ${_solstice_optix_dir}/optix_shade_background.cu
    solsticeOptixShadeBackgroundIr
    LIGHTS
    DEPENDS ${_solstice_optix_base} ${CMAKE_SOURCE_DIR}/src/render/lights.h)

solstice_optix_kernel(shade_shadow
    ${_solstice_optix_dir}/optix_shade_shadow.cu
    solsticeOptixShadeShadowIr
    DEPENDS ${_solstice_optix_base}
            ${_solstice_optix_dir}/optix_volume.cuh
            ${CMAKE_SOURCE_DIR}/src/render/volume.h
            ${CMAKE_SOURCE_DIR}/src/render/volume_track.h)

solstice_optix_kernel(shade_volume
    ${_solstice_optix_dir}/optix_shade_volume.cu
    solsticeOptixShadeVolumeIr
    LIGHTS
    DEPENDS ${_solstice_optix_base}
            ${_solstice_optix_dir}/optix_volume.cuh
            ${CMAKE_SOURCE_DIR}/src/render/volume.h
            ${CMAKE_SOURCE_DIR}/src/render/volume_track.h
            ${CMAKE_SOURCE_DIR}/src/render/lights.h)

add_custom_target(solstice_optix_programs DEPENDS ${SOLSTICE_OPTIX_EMBED_SOURCES})
set_source_files_properties(${SOLSTICE_OPTIX_EMBED_SOURCES} PROPERTIES GENERATED TRUE)

set(SOLSTICE_HAVE_OPTIX_01 1)
