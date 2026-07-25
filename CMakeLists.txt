cmake_minimum_required(VERSION 3.20)
project(Solstice VERSION 0.9.0 LANGUAGES CXX)

set(SOLSTICE_APP_DISPLAY_NAME "Sonya_Render" CACHE STRING "User-visible application name")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SOLSTICE_ENABLE_OPTIX "Build the OptiX GPU path tracing backend" OFF)
option(SOLSTICE_ENABLE_ALEMBIC "Build with Alembic geometry import" ON)
option(SOLSTICE_ENABLE_OPENEXR "Build with OpenEXR image support" ON)
option(SOLSTICE_ENABLE_MATERIALX "Build with Academy Software Foundation MaterialX" ON)
option(SOLSTICE_BUILD_TESTS "Build unit tests" ON)
option(SOLSTICE_BUILD_APP "Build the Qt desktop application" ON)
option(SOLSTICE_BUILD_TOOLS "Build helper tools (test scene generators)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(MSVC)
    add_compile_options(/W3 /MP /bigobj)
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
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

configure_file(${CMAKE_SOURCE_DIR}/cmake/solstice_config.h.in
               ${CMAKE_BINARY_DIR}/generated/solstice_config.h @ONLY)

# ---------------------------------------------------------------------------
# Core library: math, scene, IO, renderer and node graph
# ---------------------------------------------------------------------------
set(SOLSTICE_LIB_SOURCES
    src/core/image.cpp
    src/core/log.cpp
    src/scene/scene.cpp
    src/io/image_io.cpp
    src/io/alembic_loader.cpp
    src/io/usd_loader.cpp
    src/io/materialx_graph.cpp
    src/render/framebuffer.cpp
    src/render/render_session.cpp
    src/render/scene_picker.cpp
    src/render/cpu/embree_device.cpp
    src/render/optix/optix_device_stub.cpp
    src/nodes/parameter.cpp
    src/nodes/node.cpp
    src/nodes/stage.cpp
    src/nodes/node_graph.cpp
    src/nodes/node_registry.cpp
    src/nodes/nodes_library.cpp
    src/app/default_scene.cpp
    src/app/document.cpp
    src/app/headless.cpp
)

set(SOLSTICE_LIB_HEADERS
    src/core/math.h
    src/core/rng.h
    src/core/image.h
    src/core/log.h
    src/core/thread_pool.h
    src/scene/types.h
    src/scene/scene.h
    src/io/image_io.h
    src/io/alembic_loader.h
    src/io/usd_loader.h
    src/io/materialx_graph.h
    src/render/shading.h
    src/render/lights.h
    src/render/integrator.h
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
target_link_libraries(solstice_lib PUBLIC Qt6::Core Qt6::Gui embree)

if(SOLSTICE_HAVE_ALEMBIC_01)
    target_link_libraries(solstice_lib PUBLIC Alembic::Alembic)
endif()
if(SOLSTICE_HAVE_OPENEXR_01)
    target_link_libraries(solstice_lib PUBLIC OpenEXR::OpenEXR)
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
    target_sources(solstice_lib PRIVATE src/render/optix/optix_device.cpp ${SOLSTICE_OPTIX_EMBED_SOURCE})
    target_link_libraries(solstice_lib PUBLIC CUDA::cudart CUDA::cuda_driver)
    target_include_directories(solstice_lib PUBLIC ${OptiX_INCLUDE_DIR})
    add_dependencies(solstice_lib solstice_optix_programs)
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
    src/ui/connection_item.cpp
    src/ui/parameter_panel.cpp
    src/ui/render_view.cpp
    src/ui/scene_graph_panel.cpp
    src/ui/log_panel.cpp
    src/ui/theme.cpp
)

set(SOLSTICE_APP_HEADERS
    src/ui/main_window.h
    src/ui/material_network_view.h
    src/ui/material_wire_item.h
    src/ui/node_graph_view.h
    src/ui/node_item.h
    src/ui/connection_item.h
    src/ui/parameter_panel.h
    src/ui/render_view.h
    src/ui/scene_graph_panel.h
    src/ui/log_panel.h
    src/ui/theme.h
)

if(SOLSTICE_BUILD_APP)
    qt_add_executable(Solstice WIN32 MACOSX_BUNDLE ${SOLSTICE_APP_SOURCES} ${SOLSTICE_APP_HEADERS})
    target_link_libraries(Solstice PRIVATE solstice_lib Qt6::Widgets)
    set_target_properties(Solstice PROPERTIES
        OUTPUT_NAME "${SOLSTICE_APP_DISPLAY_NAME}"
        MACOSX_BUNDLE_BUNDLE_NAME "${SOLSTICE_APP_DISPLAY_NAME}"
        MACOSX_BUNDLE_GUI_IDENTIFIER "com.sonya.render")
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
        add_custom_target(deploy
            COMMAND ${CMAKE_COMMAND} -E echo "Deploying Qt runtime next to ${SOLSTICE_APP_DISPLAY_NAME}"
            COMMAND ${WINDEPLOYQT_EXECUTABLE} --release --no-translations --no-system-d3d-compiler
                    --compiler-runtime $<TARGET_FILE:Solstice>
            DEPENDS Solstice)
    endif()
endif()

message(STATUS "")
message(STATUS "${SOLSTICE_APP_DISPLAY_NAME} ${PROJECT_VERSION}")
message(STATUS "  Qt              : ${Qt6_VERSION}")
message(STATUS "  Embree          : ${EMBREE_VERSION}")
message(STATUS "  Alembic support : ${SOLSTICE_HAVE_ALEMBIC_01}")
message(STATUS "  OpenEXR support : ${SOLSTICE_HAVE_OPENEXR_01}")
message(STATUS "  MaterialX       : ${SOLSTICE_HAVE_MATERIALX_01}")
message(STATUS "  OptiX backend   : ${SOLSTICE_HAVE_OPTIX_01}")
message(STATUS "")

# OBJ -> Alembic converter for shipping demo assets.
if(SOLSTICE_BUILD_TOOLS AND SOLSTICE_HAVE_ALEMBIC_01)
    add_executable(sol_obj_to_abc tools/obj_to_abc.cpp)
    target_link_libraries(sol_obj_to_abc PRIVATE Alembic::Alembic)
endif()
