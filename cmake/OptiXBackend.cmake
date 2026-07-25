# Configures the optional OptiX GPU backend: finds CUDA and the OptiX SDK,
# compiles the device programs to PTX and embeds them into the executable.
#
# Point the build at the SDK with -DOptiX_ROOT=/path/to/OptiX-SDK or by setting
# the OptiX_INSTALL_DIR environment variable.

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "OptiX backend requested but no CUDA compiler was found - disabling it")
    return()
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(SOLSTICE_OPTIX_ARCH "60-virtual" CACHE STRING "Virtual CUDA architecture used for the OptiX PTX")

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

# --- device programs -> PTX -------------------------------------------------
add_library(solstice_optix_ptx OBJECT ${CMAKE_SOURCE_DIR}/src/render/optix/optix_programs.cu)
set_target_properties(solstice_optix_ptx PROPERTIES
    # Qt's automatic moc would add a second translation unit and therefore a
    # second .ptx file to this target.
    AUTOMOC OFF
    AUTOUIC OFF
    AUTORCC OFF
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES "${SOLSTICE_OPTIX_ARCH}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE OFF
)
target_include_directories(solstice_optix_ptx PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/generated
    ${OptiX_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(solstice_optix_ptx PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
if(SOLSTICE_CUDA_HOST_COMPILER)
    target_compile_options(solstice_optix_ptx PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-ccbin=${SOLSTICE_CUDA_HOST_COMPILER}>)
endif()

# --- embed the PTX ----------------------------------------------------------
set(SOLSTICE_OPTIX_EMBED_SOURCE ${CMAKE_BINARY_DIR}/generated/solstice_optix_ir.cpp)
add_custom_command(
    OUTPUT ${SOLSTICE_OPTIX_EMBED_SOURCE}
    COMMAND ${CMAKE_COMMAND}
            -DINPUT=$<TARGET_OBJECTS:solstice_optix_ptx>
            -DOUTPUT=${SOLSTICE_OPTIX_EMBED_SOURCE}
            -DSYMBOL=solsticeOptixIr
            -P ${CMAKE_SOURCE_DIR}/cmake/embed_binary.cmake
    DEPENDS solstice_optix_ptx $<TARGET_OBJECTS:solstice_optix_ptx>
            ${CMAKE_SOURCE_DIR}/cmake/embed_binary.cmake
    COMMENT "Embedding OptiX PTX"
    VERBATIM
)
add_custom_target(solstice_optix_programs DEPENDS ${SOLSTICE_OPTIX_EMBED_SOURCE})
set_source_files_properties(${SOLSTICE_OPTIX_EMBED_SOURCE} PROPERTIES GENERATED TRUE)

set(SOLSTICE_HAVE_OPTIX_01 1)
