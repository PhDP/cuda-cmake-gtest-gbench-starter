set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           eddb0241389718a23a42db6af5f0164b6e0139af # version 1.9.4
)

FetchContent_MakeAvailable(
  googlebenchmark
)

set(bench_cc
  mm.cc
)

add_executable(bench_deepgreen ${bench_cc})

target_link_libraries(bench_deepgreen
  benchmark
  ${DG_LIB_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBRARIES}
)
