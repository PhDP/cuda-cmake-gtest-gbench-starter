add_subdirectory(gtest)

include_directories(./gtest/googletest/include)

set(test_src
  run_all.cc
  matrix_test.cc
  matrix_mult_test.cc
)

add_executable(test_deepgreen ${test_src})

target_link_libraries(test_deepgreen
  gtest
  ${DG_LIB_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBRARIES}
)

add_test(
  NAME tests
  COMMAND test_deepgreen
)
