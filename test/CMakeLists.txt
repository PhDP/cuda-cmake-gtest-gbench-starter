add_library(gtest ./gtest/gtest-all.cc)
include_directories(.)
target_link_libraries(gtest ${CMAKE_THREAD_LIBS_INIT})

set(test_src
  run_all.cc
  matrix_test.cc
  matrix_mult_test.cc
)

add_executable(deepgreen_tests ${test_src})

target_link_libraries(deepgreen_tests
  gtest
  deepgreen
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBS}
)

add_test(
  NAME tests
  COMMAND deepgreen_tests
)
