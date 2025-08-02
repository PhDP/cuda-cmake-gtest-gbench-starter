FetchContent_Declare(
  googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           52eb8108c5bdec04579160ae17225d66034bd723 # version 1.17.0
)

FetchContent_MakeAvailable(
  googletest
)

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

