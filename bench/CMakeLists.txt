set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)

add_subdirectory(benchmark)
include_directories(benchmark/include)

set(bench_cc
  mm.cc
)

add_executable(bench_deepgreen ${bench_cc})

target_link_libraries(bench_deepgreen
  benchmark
  ${DG_LIB_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBS}
)
