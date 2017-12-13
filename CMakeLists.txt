cmake_minimum_required(VERSION 3.0)
project(deepgreen)

# Set build type to RELEASE by default:
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, default to RELEASE.")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif ()

option(BuildTest "BuildTest" ON)
option(BuildBench "BuildBench" ON)
option(BuildShared "BuildShared" OFF)
option(Sanitize "Sanitize" OFF)

if (BuildShared)
  set(LIB_TYPE SHARED)
else ()
  set(LIB_TYPE STATIC)
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost)
find_package(Math)

include_directories(. include ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (Sanitize)
  set(CMAKE_CXX_FLAGS "-fsanitize=address -fsanitize=undefined ${CMAKE_CXX_FLAGS}")
endif ()

add_subdirectory(src)

if (BuildTest)
  enable_testing()
  add_subdirectory(test)
endif ()

if (BuildBench)
  add_subdirectory(bench)
endif ()

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library type         : ${LIB_TYPE}")
message(STATUS "  Build tests          : ${BuildTest}")
message(STATUS "  Build benchmarks     : ${BuildBench}")
message(STATUS "  Sanitize flags       : ${Sanitize}")
message(STATUS "  FoundCUDA            : ${CUDA_FOUND}")
message(STATUS "  Boost include dirs   : ${Boost_INCLUDE_DIRS}")
message(STATUS "")
