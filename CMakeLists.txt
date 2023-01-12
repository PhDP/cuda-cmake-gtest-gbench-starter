cmake_minimum_required(VERSION 3.2)
project(deepgreen)

# Set build type to RELEASE by default:
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, default to RELEASE.")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif ()

option(DG_TEST        "DG_TEST"       ON)
option(DG_BENCH       "DG_BENCH"      ON)
option(DG_STATIC      "DG_STATIC"     OFF)
option(DG_SANITIZE    "DG_SANITIZE"   OFF)

if (DG_STATIC)
  set(DG_LIB_NAME deepgreen_static)
  set(DG_LIB_TYPE STATIC)
else ()
  set(DG_LIB_NAME deepgreen)
  set(DG_LIB_TYPE SHARED)
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost)
find_package(MATH)

include_directories(. include ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (DG_SANITIZE)
  set(CMAKE_CXX_FLAGS "-fsanitize=address -fsanitize=undefined ${CMAKE_CXX_FLAGS}")
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
message(STATUS "  Build tests          : ${DG_TEST}")
message(STATUS "  Build benchmarks     : ${DG_BENCH}")
message(STATUS "  Add sanitize flags   : ${DG_SANITIZE}")
message(STATUS "  FoundCUDA            : ${CUDA_FOUND}")
message(STATUS "  Boost include dirs   : ${Boost_INCLUDE_DIRS}")
message(STATUS "")
