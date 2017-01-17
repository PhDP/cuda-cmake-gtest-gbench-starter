set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)

add_subdirectory(benchmark)
include_directories(benchmark/include)

set(bench_cc
  mm.cc
)

add_executable(deepgreen_benchmark ${bench_cc})

target_link_libraries(deepgreen_benchmark
  benchmark
  deepgreen
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBS}
)
