set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           192ef10025eb2c4cdd392bc502f0c852196baa48 # version 1.9.5
  SYSTEM
)

FetchContent_MakeAvailable(googlebenchmark)

set(bench_cc
  mm.cc
)

add_executable(bench_deepgreen ${bench_cc})

target_link_libraries(bench_deepgreen PRIVATE
  benchmark::benchmark_main
  ${DG_LIB_NAME}
)
