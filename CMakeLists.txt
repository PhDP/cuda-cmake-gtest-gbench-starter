cmake_minimum_required(VERSION 3.25)
project(deepgreen
  DESCRIPTION "Simple template for C++/CUDA projects."
  HOMEPAGE_URL "https://github.com/PhDP/cuda-cmake-gtest-gbench-starter"
  LANGUAGES C CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set build type to Release by default:
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, default to Release.")
  set(CMAKE_BUILD_TYPE "Release")
endif ()

option(DG_TEST        "Enables testing with Google Test." ON)
option(DG_BENCH       "Enables benchmarking with Google Benchmark." ON)
option(DG_STATIC      "Builds a static library instead of a shared one." OFF)
option(DG_SANITIZE    "Adds sanitize flags." OFF)

if (DG_STATIC)
  set(DG_LIB_NAME deepgreen_static)
  set(DG_LIB_TYPE STATIC)
else ()
  set(DG_LIB_NAME deepgreen)
  set(DG_LIB_TYPE SHARED)
endif ()

include(FetchContent)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (DG_SANITIZE)
  add_compile_options(-fsanitize=address,undefined)
  add_link_options(-fsanitize=address,undefined)
endif ()

add_subdirectory(src)

if (DG_TEST)
  enable_testing()
  add_subdirectory(test)
endif ()

if (DG_BENCH)
  add_subdirectory(bench)
endif ()

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library name         : ${DG_LIB_NAME}")
message(STATUS "  Library type         : ${DG_LIB_TYPE}")
message(STATUS "  CUDA                 : ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA include dirs    : ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "  CUDA library dir     : ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "  Boost include dirs   : ${Boost_INCLUDE_DIRS}")
message(STATUS "  Boost version        : ${Boost_VERSION}")
message(STATUS "  Build tests          : ${DG_TEST}")
message(STATUS "  Build benchmarks     : ${DG_BENCH}")
message(STATUS "  Add sanitize flags   : ${DG_SANITIZE}")
message(STATUS "")
