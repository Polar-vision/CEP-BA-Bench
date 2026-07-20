cmake_minimum_required(VERSION 3.10)
project(ba)

set(CEP_THIRDPARTY_DIR "E:/zuo/projects/backup/3rdparty" CACHE PATH "Directory containing eigen3 and Sophus")
set(EIGEN3_INCLUDE_DIR "${CEP_THIRDPARTY_DIR}/eigen3" CACHE PATH "Eigen include directory")
set(SOPHUS_INCLUDE_DIR "${CEP_THIRDPARTY_DIR}/Sophus/install/include" CACHE PATH "Sophus include directory")
set(CEP_VCPKG_INSTALLED_DIR "E:/zuo/vcpkg/installed" CACHE PATH "vcpkg installed directory")
set(CEP_VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
set(cudss_DIR "E:/zuo/Program Files/NVIDIA cuDSS/v0.5/lib/12/cmake/cudss" CACHE PATH "cuDSS CMake package directory")
set(Ceres_DIR "E:/zuo/projects/ceres-solver/install/lib/cmake/Ceres" CACHE PATH "Ceres CMake package directory")

list(INSERT CMAKE_PREFIX_PATH 0 "${CEP_VCPKG_INSTALLED_DIR}/${CEP_VCPKG_TARGET_TRIPLET}")
list(INSERT CMAKE_MODULE_PATH 0 "${CEP_VCPKG_INSTALLED_DIR}/${CEP_VCPKG_TARGET_TRIPLET}/share")

if (MSVC)
    message(STATUS "Using MSVC compiler")
    add_compile_options(/utf-8)
    string(FIND "${CMAKE_CXX_FLAGS_RELEASE}" "/O2" O2_FOUND)
    if(O2_FOUND EQUAL -1)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    endif()
    message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")

    string(FIND "${CMAKE_C_FLAGS_RELEASE}" "/O2" O2_FOUND)
    if(O2_FOUND EQUAL -1)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2")
    endif()
    message(STATUS "CMAKE_C_FLAGS_RELEASE = ${CMAKE_C_FLAGS_RELEASE}")
endif()

find_package(Ceres REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 源文件
set(SOURCES
    "src/BAExporter_v2.cpp"
    "src/IBA_v2.cpp"
    "src/PBAImp_v2.cpp"
)

# 头文件路径
set(BA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
message(STATUS "BA_INCLUDE_DIR路径: ${BA_INCLUDE_DIR}")
# ----------------------------
# 1. 生成静态库（.lib）
# ----------------------------
add_library(ba STATIC ${SOURCES})
target_compile_definitions(ba PUBLIC BA_STATIC)  # 静态库不需要导出符号
target_include_directories(ba PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${BA_INCLUDE_DIR}
)
target_include_directories(ba PRIVATE
    ${CEP_VCPKG_INSTALLED_DIR}/${CEP_VCPKG_TARGET_TRIPLET}/include
    ${EIGEN3_INCLUDE_DIR}
    ${SOPHUS_INCLUDE_DIR}
)
target_link_libraries(ba PRIVATE ${CERES_LIBRARIES})
