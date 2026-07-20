cmake_minimum_required(VERSION 3.10)
project(example)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CEP_THIRDPARTY_DIR "E:/zuo/projects/backup/3rdparty" CACHE PATH "Directory containing third-party headers")
set(CEP_VCPKG_INSTALLED_DIR "E:/zuo/vcpkg/installed" CACHE PATH "vcpkg installed directory")
set(CEP_VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
set(cudss_DIR "E:/zuo/Program Files/NVIDIA cuDSS/v0.5/lib/12/cmake/cudss" CACHE PATH "cuDSS CMake package directory")
set(Ceres_DIR "E:/zuo/projects/ceres-solver/install/lib/cmake/Ceres" CACHE PATH "Ceres CMake package directory")

set(CEP_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(BA_INCLUDE_DIR "${CEP_ROOT}/ba_v2/include")

# 确保 Vcpkg 的路径优先于系统路径
list(INSERT CMAKE_PREFIX_PATH 0 "${CEP_VCPKG_INSTALLED_DIR}/${CEP_VCPKG_TARGET_TRIPLET}")
list(INSERT CMAKE_MODULE_PATH 0 "${CEP_VCPKG_INSTALLED_DIR}/${CEP_VCPKG_TARGET_TRIPLET}/share")
# 查找 Ceres
find_package(Ceres REQUIRED)


# 源文件
set(SOURCES
    example.cpp
)

add_subdirectory("${CEP_ROOT}/ba_v2" "${CMAKE_BINARY_DIR}/ba_v2_build")

# 创建可执行程序
add_executable(example ${SOURCES})
set_target_properties(example PROPERTIES LINK_FLAGS "/SAFESEH:NO")

# 头文件路径
target_include_directories(example PRIVATE
    ${BA_INCLUDE_DIR}
    ${CEP_THIRDPARTY_DIR}
    ${CEP_VCPKG_INSTALLED_DIR}/${CEP_VCPKG_TARGET_TRIPLET}/include
)

target_link_libraries(example PRIVATE 
    ba
    ${CERES_LIBRARIES}
)

add_custom_command(TARGET example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_RUNTIME_DLLS:example> $<TARGET_FILE_DIR:example>
    COMMAND_EXPAND_LISTS
)
