set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           d572f4777349d43653b21d6c2fc63020ab326db2 # version 1.7.1
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
