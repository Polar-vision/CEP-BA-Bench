cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(CEP_BA LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/../cmake/CEPBuildSettings.cmake")
endif()

add_library(ba STATIC
    src/BAExporter_v2.cpp
    src/IBA_v2.cpp
    src/PBAImp_v2.cpp
    include/BAExporter_v2.h
    include/IBA_v2.h
    include/PBAImp_v2.h
    include/CameraFrameParameterizations.h
)

target_compile_features(ba PUBLIC cxx_std_17)
target_compile_definitions(ba PUBLIC BA_STATIC)
target_include_directories(ba PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(ba PUBLIC Ceres::ceres Eigen3::Eigen)

if(MSVC)
    target_compile_options(ba PRIVATE /utf-8 /bigobj)
endif()
