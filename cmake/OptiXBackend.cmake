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

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "OptiX backend requested but no CUDA compiler was found - disabling it")
    return()
endif()

enable_language(CUDA)
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
    list(APPEND _solstice_nvcc_inc_flags -I${_inc})
endforeach()

set(_solstice_nvcc_ccbin)
if(SOLSTICE_CUDA_HOST_COMPILER)
    set(_solstice_nvcc_ccbin -ccbin=${SOLSTICE_CUDA_HOST_COMPILER})
elseif(MSVC AND CMAKE_CXX_COMPILER)
    # Help nvcc find cl.exe when driven outside the VS CUDA .targets path.
    set(_solstice_nvcc_ccbin -ccbin=${CMAKE_CXX_COMPILER})
endif()

add_custom_command(
    OUTPUT ${SOLSTICE_OPTIX_PTX}
    COMMAND ${CMAKE_CUDA_COMPILER}
            ${_solstice_nvcc_ccbin}
            -ptx
            -std=c++17
            --use_fast_math
            --expt-relaxed-constexpr
            -lineinfo
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
