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

set(SOLSTICE_OPTIX_ARCH "compute_60" CACHE STRING "Virtual CUDA arch for OptiX PTX (--gpu-architecture)")

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

set(SOLSTICE_OPTIX_CU ${CMAKE_SOURCE_DIR}/src/render/optix/optix_programs.cu)
set(SOLSTICE_OPTIX_PTX ${CMAKE_BINARY_DIR}/generated/optix_programs.ptx)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated)

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

# VS 2026 (cl 19.50+) is newer than CUDA 12.0's supported host list.
set(_solstice_nvcc_unsupported)
if(WIN32)
    set(_solstice_nvcc_unsupported --allow-unsupported-compiler)
endif()

add_custom_command(
    OUTPUT ${SOLSTICE_OPTIX_PTX}
    COMMAND ${CMAKE_CUDA_COMPILER}
            ${_solstice_nvcc_ccbin}
            -ptx
            -std=c++17
            --use_fast_math
            --expt-relaxed-constexpr
            ${_solstice_nvcc_lineinfo}
            ${_solstice_nvcc_unsupported}
            -arch=${SOLSTICE_OPTIX_ARCH}
            -D_USE_MATH_DEFINES
            -DNOMINMAX
            -DWIN32_LEAN_AND_MEAN
            -I${CMAKE_SOURCE_DIR}/src
            -I${CMAKE_BINARY_DIR}/generated
            -I${OptiX_INCLUDE_DIR}
            ${_solstice_nvcc_inc_flags}
            -o ${SOLSTICE_OPTIX_PTX}
            ${SOLSTICE_OPTIX_CU}
    DEPENDS
        ${SOLSTICE_OPTIX_CU}
        ${CMAKE_SOURCE_DIR}/src/render/blue_noise.h
        ${CMAKE_SOURCE_DIR}/src/render/integrator.h
        ${CMAKE_SOURCE_DIR}/src/render/optix/launch_params.h
        ${CMAKE_BINARY_DIR}/generated/solstice_config.h
    COMMENT "Compiling OptiX device programs to PTX"
    VERBATIM
)

# --- embed the PTX ----------------------------------------------------------
set(SOLSTICE_OPTIX_EMBED_SOURCE ${CMAKE_BINARY_DIR}/generated/solstice_optix_ir.cpp)
add_custom_command(
    OUTPUT ${SOLSTICE_OPTIX_EMBED_SOURCE}
    COMMAND ${CMAKE_COMMAND}
            -DINPUT=${SOLSTICE_OPTIX_PTX}
            -DOUTPUT=${SOLSTICE_OPTIX_EMBED_SOURCE}
            -DSYMBOL=solsticeOptixIr
            -P ${CMAKE_SOURCE_DIR}/cmake/embed_binary.cmake
    DEPENDS ${SOLSTICE_OPTIX_PTX} ${CMAKE_SOURCE_DIR}/cmake/embed_binary.cmake
    COMMENT "Embedding OptiX PTX"
    VERBATIM
)
add_custom_target(solstice_optix_programs DEPENDS ${SOLSTICE_OPTIX_EMBED_SOURCE})
set_source_files_properties(${SOLSTICE_OPTIX_EMBED_SOURCE} PROPERTIES GENERATED TRUE)

set(SOLSTICE_HAVE_OPTIX_01 1)
