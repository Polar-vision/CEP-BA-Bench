cmake_minimum_required(VERSION 3.21)

# Keep the documented example_v2 build entry point, while also allowing the
# entire benchmark to be configured from the repository root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(CEP_BA_Bench LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/../cmake/CEPBuildSettings.cmake")
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../ba_v2" ba_v2)
endif()

add_executable(example example.cpp)
target_compile_features(example PRIVATE cxx_std_17)
target_link_libraries(example PRIVATE ba)

if(MSVC)
    target_compile_options(example PRIVATE /utf-8)
endif()

# Imported shared-library targets can provide their runtime DLL locations.
# The helper also handles a Ceres installation with no runtime DLLs to copy.
if(WIN32)
    add_custom_command(TARGET example POST_BUILD
        COMMAND "${CMAKE_COMMAND}"
            "-DDLLS=$<TARGET_RUNTIME_DLLS:example>"
            "-DOUTPUT_DIR=$<TARGET_FILE_DIR:example>"
            -P "${CMAKE_CURRENT_LIST_DIR}/../cmake/CopyRuntimeDlls.cmake"
        VERBATIM
    )
endif()
