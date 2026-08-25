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
    set(SOLSTICE_OPTIX_NVCC_OPT "3")
endif()
message(STATUS "OptiX PTX: Iray wavefront modules (init/intersect/shade/tail), nvcc -O${SOLSTICE_OPTIX_NVCC_OPT}")

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
    if(SOLSTICE_IEEE_FP32)
        set(_solstice_nvcc_xcompiler -Xcompiler=/bigobj,/nologo,/fp:precise)
    else()
        set(_solstice_nvcc_xcompiler -Xcompiler=/bigobj,/nologo)
    endif()
    set(_solstice_nvcc_unsupported
        --allow-unsupported-compiler
        -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
        -D_ENABLE_EXTENDED_ALIGNED_STORAGE
        ${_solstice_nvcc_xcompiler})
endif()

# --use_fast_math ⇒ ftz + approx div/sqrt + fmad, and isfinite() is compiled out.
set(_solstice_nvcc_fp)
if(SOLSTICE_IEEE_FP32)
    set(_solstice_nvcc_fp --ftz=false --prec-div=true --prec-sqrt=true --fmad=true)
    message(STATUS "OptiX PTX: IEEE nvcc, --fmad=true, no --use_fast_math")
else()
    set(_solstice_nvcc_fp --use_fast_math)
endif()

# Iray: thin wavefront stages (intersect vs shade) plus a separate tail megakernel
# so cicc never sees optixTrace + BSDF in the interactive pipeline. path_tail is
# slower to nvcc; give it a longer timeout.
#
# Do NOT give ninja 16 separate custom commands. On Windows CI, after the
# shade_surface embed finished, ninja never started [15/16] (shade_background)
# and the step sat silent until the 45-minute cap. One Python process compiles
# and embeds every kernel in order, with flushed timestamps and a heartbeat.
set(_solstice_nvcc_ptx_common
    ${_solstice_nvcc_ccbin}
    -ptx
    -std=c++17
    -O${SOLSTICE_OPTIX_NVCC_OPT}
    ${_solstice_nvcc_fp}
    --disable-warnings
    ${_solstice_nvcc_lineinfo}
    ${_solstice_nvcc_unsupported}
    -arch=${SOLSTICE_OPTIX_ARCH}
    -D_USE_MATH_DEFINES
    -DNOMINMAX
    -DWIN32_LEAN_AND_MEAN
    -DSOL_RNG_NO_SOBOL
    -I${CMAKE_SOURCE_DIR}/src
    -I${CMAKE_BINARY_DIR}/generated
    -I${OptiX_INCLUDE_DIR}
    ${_solstice_nvcc_inc_flags})

set(_solstice_optix_base
    ${_solstice_optix_dir}/optix_common.cuh
    ${_solstice_optix_dir}/optix_wavefront.cuh
    ${_solstice_optix_dir}/optix_work.cuh
    ${_solstice_optix_dir}/launch_params.h
    ${_solstice_optix_dir}/path_state.h
    ${CMAKE_SOURCE_DIR}/src/scene/types.h
    ${CMAKE_SOURCE_DIR}/src/core/math.h
    ${CMAKE_SOURCE_DIR}/src/core/rng.h)

# Never use the Windows `py` launcher: cmake -P + py.exe hangs after the script
# prints (pipe/handle leak) and starved ninja so shade_background never started.
find_program(SOLSTICE_EMBED_PYTHON NAMES python3 python python3.exe python.exe
    REQUIRED
    DOC "Python used to embed OptiX PTX (not the Windows py launcher)")
message(STATUS "OptiX PTX embed python: ${SOLSTICE_EMBED_PYTHON}")

function(_solstice_json_escape out str)
    string(REPLACE "\\" "\\\\" str "${str}")
    string(REPLACE "\"" "\\\"" str "${str}")
    string(REPLACE "\t" "\\t" str "${str}")
    string(REPLACE "\n" "\\n" str "${str}")
    string(REPLACE "\r" "\\r" str "${str}")
    set(${out} "${str}" PARENT_SCOPE)
endfunction()

function(solstice_optix_kernel name source symbol)
    cmake_parse_arguments(K "LIGHTS" "TIMEOUT" "DEPENDS" ${ARGN})
    set(ptx "${CMAKE_BINARY_DIR}/generated/optix_${name}.ptx")
    set(embed "${CMAKE_BINARY_DIR}/generated/solstice_optix_${name}_ir.cpp")
    set(extra)
    if(K_LIGHTS)
        list(APPEND extra "-DSOLSTICE_OPTIX_KERNEL=1")
    endif()
    set_property(DIRECTORY APPEND PROPERTY SOLSTICE_OPTIX_KERNELS ${name})
    set_property(DIRECTORY PROPERTY SOLSTICE_OPTIX_SRC_${name} "${source}")
    set_property(DIRECTORY PROPERTY SOLSTICE_OPTIX_SYM_${name} "${symbol}")
    set_property(DIRECTORY PROPERTY SOLSTICE_OPTIX_PTX_${name} "${ptx}")
    set_property(DIRECTORY PROPERTY SOLSTICE_OPTIX_EMBED_${name} "${embed}")
    set_property(DIRECTORY PROPERTY SOLSTICE_OPTIX_DEFS_${name} "${extra}")
    if(K_TIMEOUT)
        set_property(DIRECTORY PROPERTY SOLSTICE_OPTIX_TIMEOUT_${name} "${K_TIMEOUT}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY SOLSTICE_OPTIX_ALL_DEPS "${source}")
    foreach(_d IN LISTS K_DEPENDS)
        set_property(DIRECTORY APPEND PROPERTY SOLSTICE_OPTIX_ALL_DEPS "${_d}")
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY SOLSTICE_OPTIX_EMBED_SOURCES "${embed}")
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
            ${_solstice_optix_dir}/optix_spawn.cuh
            ${CMAKE_SOURCE_DIR}/src/render/camera_sample.h
            ${CMAKE_SOURCE_DIR}/src/render/volume.h
            ${CMAKE_SOURCE_DIR}/src/render/volume_track.h)

solstice_optix_kernel(intersect_closest
    ${_solstice_optix_dir}/optix_intersect_closest.cu
    solsticeOptixIntersectClosestIr
    DEPENDS ${_solstice_optix_base} ${_solstice_optix_dir}/optix_trace.cuh
            ${_solstice_optix_dir}/optix_work.cuh)

solstice_optix_kernel(intersect_shadow
    ${_solstice_optix_dir}/optix_intersect_shadow.cu
    solsticeOptixIntersectShadowIr
    DEPENDS ${_solstice_optix_base} ${_solstice_optix_dir}/optix_trace.cuh
            ${_solstice_optix_dir}/optix_work.cuh)

solstice_optix_kernel(shade_surface
    ${_solstice_optix_dir}/optix_shade_surface.cu
    solsticeOptixShadeSurfaceIr
    LIGHTS
    DEPENDS ${_solstice_optix_base}
            ${_solstice_optix_dir}/optix_geom.cuh
            ${_solstice_optix_dir}/optix_bsdf.cuh
            ${_solstice_optix_dir}/optix_volume.cuh
            ${_solstice_optix_dir}/optix_spawn.cuh
            ${CMAKE_SOURCE_DIR}/src/render/lights.h
            ${CMAKE_SOURCE_DIR}/src/render/volume.h
            ${CMAKE_SOURCE_DIR}/src/render/volume_track.h)

solstice_optix_kernel(shade_background
    ${_solstice_optix_dir}/optix_shade_background.cu
    solsticeOptixShadeBackgroundIr
    LIGHTS
    DEPENDS ${_solstice_optix_base} ${CMAKE_SOURCE_DIR}/src/render/lights.h
            ${_solstice_optix_dir}/optix_spawn.cuh)

solstice_optix_kernel(shade_shadow
    ${_solstice_optix_dir}/optix_shade_shadow.cu
    solsticeOptixShadeShadowIr
    DEPENDS ${_solstice_optix_base}
            ${_solstice_optix_dir}/optix_volume.cuh
            ${_solstice_optix_dir}/optix_spawn.cuh
            ${CMAKE_SOURCE_DIR}/src/render/volume.h
            ${CMAKE_SOURCE_DIR}/src/render/volume_track.h)

solstice_optix_kernel(shade_volume
    ${_solstice_optix_dir}/optix_shade_volume.cu
    solsticeOptixShadeVolumeIr
    LIGHTS
    DEPENDS ${_solstice_optix_base}
            ${_solstice_optix_dir}/optix_volume.cuh
            ${_solstice_optix_dir}/optix_spawn.cuh
            ${CMAKE_SOURCE_DIR}/src/render/volume.h
            ${CMAKE_SOURCE_DIR}/src/render/volume_track.h
            ${CMAKE_SOURCE_DIR}/src/render/lights.h)

solstice_optix_kernel(path_tail
    ${_solstice_optix_dir}/optix_path_tail.cu
    solsticeOptixPathTailIr
    LIGHTS
    TIMEOUT 1800
    DEPENDS ${_solstice_optix_base}
            ${_solstice_optix_dir}/optix_trace.cuh
            ${_solstice_optix_dir}/optix_geom.cuh
            ${_solstice_optix_dir}/optix_bsdf.cuh
            ${_solstice_optix_dir}/optix_volume.cuh
            ${_solstice_optix_dir}/optix_work.cuh
            ${_solstice_optix_dir}/optix_spawn.cuh
            ${_solstice_optix_dir}/optix_intersect_closest.cu
            ${_solstice_optix_dir}/optix_intersect_shadow.cu
            ${_solstice_optix_dir}/optix_shade_surface.cu
            ${_solstice_optix_dir}/optix_shade_volume.cu
            ${_solstice_optix_dir}/optix_shade_background.cu
            ${_solstice_optix_dir}/optix_shade_shadow.cu
            ${CMAKE_SOURCE_DIR}/src/render/lights.h
            ${CMAKE_SOURCE_DIR}/src/render/volume.h
            ${CMAKE_SOURCE_DIR}/src/render/volume_track.h)

get_property(_solstice_optix_kernels DIRECTORY PROPERTY SOLSTICE_OPTIX_KERNELS)
get_property(SOLSTICE_OPTIX_EMBED_SOURCES DIRECTORY PROPERTY SOLSTICE_OPTIX_EMBED_SOURCES)
get_property(_solstice_optix_all_deps DIRECTORY PROPERTY SOLSTICE_OPTIX_ALL_DEPS)
if(_solstice_optix_all_deps)
    list(REMOVE_DUPLICATES _solstice_optix_all_deps)
endif()

set(_solstice_json_flags "")
foreach(_f IN LISTS _solstice_nvcc_ptx_common)
    _solstice_json_escape(_ef "${_f}")
    if(_solstice_json_flags STREQUAL "")
        set(_solstice_json_flags "\"${_ef}\"")
    else()
        string(APPEND _solstice_json_flags ", \"${_ef}\"")
    endif()
endforeach()

set(_solstice_json_kernels "")
set(_solstice_first_kernel TRUE)
foreach(_name IN LISTS _solstice_optix_kernels)
    get_property(_src DIRECTORY PROPERTY SOLSTICE_OPTIX_SRC_${_name})
    get_property(_sym DIRECTORY PROPERTY SOLSTICE_OPTIX_SYM_${_name})
    get_property(_ptx DIRECTORY PROPERTY SOLSTICE_OPTIX_PTX_${_name})
    get_property(_embed DIRECTORY PROPERTY SOLSTICE_OPTIX_EMBED_${_name})
    get_property(_defs DIRECTORY PROPERTY SOLSTICE_OPTIX_DEFS_${_name})
    get_property(_timeout DIRECTORY PROPERTY SOLSTICE_OPTIX_TIMEOUT_${_name})
    if(NOT _timeout)
        set(_timeout 0)
    endif()
    _solstice_json_escape(_ename "${_name}")
    _solstice_json_escape(_esrc "${_src}")
    _solstice_json_escape(_esym "${_sym}")
    _solstice_json_escape(_eptx "${_ptx}")
    _solstice_json_escape(_eembed "${_embed}")
    set(_eflags "[")
    set(_efirst TRUE)
    foreach(_d IN LISTS _defs)
        _solstice_json_escape(_ed "${_d}")
        if(NOT _efirst)
            string(APPEND _eflags ", ")
        endif()
        set(_efirst FALSE)
        string(APPEND _eflags "\"${_ed}\"")
    endforeach()
    string(APPEND _eflags "]")
    if(NOT _solstice_first_kernel)
        string(APPEND _solstice_json_kernels ",\n")
    endif()
    set(_solstice_first_kernel FALSE)
    string(APPEND _solstice_json_kernels
        "    {\"name\": \"${_ename}\", \"source\": \"${_esrc}\", \"ptx\": \"${_eptx}\", \"embed\": \"${_eembed}\", \"symbol\": \"${_esym}\", \"timeout_sec\": ${_timeout}, \"extra_flags\": ${_eflags}}")
endforeach()

_solstice_json_escape(_envcc "${CMAKE_CUDA_COMPILER}")
_solstice_json_escape(_epy "${SOLSTICE_EMBED_PYTHON}")
_solstice_json_escape(_eembed_py "${CMAKE_SOURCE_DIR}/cmake/embed_binary.py")
set(_solstice_ptx_log "${CMAKE_BINARY_DIR}/generated/ptx_steps.log")
_solstice_json_escape(_elog "${_solstice_ptx_log}")

set(_solstice_manifest "${CMAKE_BINARY_DIR}/generated/optix_ptx_manifest.json")
set(_solstice_manifest_tmp "${CMAKE_BINARY_DIR}/generated/optix_ptx_manifest.json.tmp")
file(WRITE "${_solstice_manifest_tmp}"
"{
  \"nvcc\": \"${_envcc}\",
  \"python\": \"${_epy}\",
  \"embed_script\": \"${_eembed_py}\",
  \"log\": \"${_elog}\",
  \"timeout_sec\": 600,
  \"common_flags\": [${_solstice_json_flags}],
  \"kernels\": [
${_solstice_json_kernels}
  ]
}
")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${_solstice_manifest_tmp}" "${_solstice_manifest}")
file(REMOVE "${_solstice_manifest_tmp}")

set(_solstice_ptx_stamp "${CMAKE_BINARY_DIR}/generated/optix_ptx.stamp")
add_custom_command(
    OUTPUT ${SOLSTICE_OPTIX_EMBED_SOURCES} ${_solstice_ptx_stamp}
    COMMAND ${CMAKE_COMMAND} -E echo "OptiX PTX sequential driver starting"
    COMMAND ${CMAKE_COMMAND} -E env PYTHONUNBUFFERED=1
            ${SOLSTICE_EMBED_PYTHON}
            ${CMAKE_SOURCE_DIR}/cmake/build_optix_ptx.py
            --manifest ${_solstice_manifest}
            --stamp ${_solstice_ptx_stamp}
    DEPENDS
        ${_solstice_optix_all_deps}
        ${_solstice_manifest}
        ${CMAKE_SOURCE_DIR}/cmake/build_optix_ptx.py
        ${CMAKE_SOURCE_DIR}/cmake/embed_binary.py
    COMMENT "OptiX PTX: sequential nvcc + embed"
    VERBATIM
)
add_custom_target(solstice_optix_programs DEPENDS ${SOLSTICE_OPTIX_EMBED_SOURCES} ${_solstice_ptx_stamp})
set_source_files_properties(${SOLSTICE_OPTIX_EMBED_SOURCES} PROPERTIES GENERATED TRUE)

set(SOLSTICE_HAVE_OPTIX_01 1)
