# Turns a binary file (OptiX PTX) into a C++ translation unit so the renderer
# stays a single self contained executable.
#
# Usage: cmake -DINPUT=<file> -DOUTPUT=<file.cpp> -DSYMBOL=<name> -P embed_binary.cmake
#
# Do not hex-dump with CMake REGEX REPLACE or string(APPEND): both are quadratic
# on multi-megabyte shade_surface PTX and hung Windows CI (45-minute cap).
# Python writes the translation unit in one linear pass.

if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED SYMBOL)
    message(FATAL_ERROR "embed_binary.cmake requires INPUT, OUTPUT and SYMBOL")
endif()

if(NOT EXISTS "${INPUT}")
    message(FATAL_ERROR "embed_binary.cmake: INPUT does not exist: ${INPUT}")
endif()

get_filename_component(_embed_dir "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(_embed_py "${_embed_dir}/embed_binary.py")
if(NOT EXISTS "${_embed_py}")
    message(FATAL_ERROR "embed_binary.cmake: missing ${_embed_py}")
endif()

file(SIZE "${INPUT}" BYTE_COUNT)
message(STATUS "embed ${SYMBOL}: ${BYTE_COUNT} bytes via python")

# Do not probe `py`: the Windows launcher hangs under cmake -P after the script
# exits, which left ninja waiting so later kernels never launched.
set(_embed_python "")
foreach(_cand IN ITEMS python3 python python3.exe python.exe)
    execute_process(
        COMMAND ${_cand} --version
        RESULT_VARIABLE _py_rc
        OUTPUT_QUIET
        ERROR_QUIET)
    if(_py_rc EQUAL 0)
        set(_embed_python ${_cand})
        break()
    endif()
endforeach()
if(_embed_python STREQUAL "")
    message(FATAL_ERROR "embed_binary.cmake: python3/python not found (needed to embed OptiX PTX)")
endif()

execute_process(
    COMMAND ${_embed_python} "${_embed_py}" "${INPUT}" "${OUTPUT}" "${SYMBOL}"
    RESULT_VARIABLE _embed_rc
    OUTPUT_VARIABLE _embed_out
    ERROR_VARIABLE _embed_err)
if(NOT _embed_rc EQUAL 0)
    message(FATAL_ERROR "embed_binary.py failed (${_embed_rc}): ${_embed_out} ${_embed_err}")
endif()
if(_embed_out)
    message(STATUS "${_embed_out}")
endif()
