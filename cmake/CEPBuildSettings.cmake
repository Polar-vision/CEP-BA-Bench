# Resolve dependencies from their installed CMake packages. Supply Ceres_DIR,
# Eigen3_DIR, or CMAKE_PREFIX_PATH when they are not on CMake's search path.
# Any additional dependencies of the selected Ceres installation are resolved
# by CeresConfig.cmake; this project does not download third-party libraries.
include_guard(GLOBAL)

if(NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Ceres 2.3 CONFIG REQUIRED)
find_package(Eigen3 3.3 CONFIG REQUIRED)

message(STATUS "CEP-BA-Bench: Ceres ${Ceres_VERSION}; Eigen ${Eigen3_VERSION}")
