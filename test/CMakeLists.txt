FetchContent_Declare(
  googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           52eb8108c5bdec04579160ae17225d66034bd723 # version 1.17.0
  SYSTEM
)

FetchContent_MakeAvailable(googletest)

set(test_src
  matrix_test.cc
  matrix_mult_test.cc
)

add_executable(test_deepgreen ${test_src})

target_link_libraries(test_deepgreen PRIVATE
  gtest_main
  ${DG_LIB_NAME}
)

add_test(
  NAME tests
  COMMAND test_deepgreen
)
