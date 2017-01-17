set(deepgreen_src
  matrix_mm.cu
)

cuda_add_library(deepgreen ${LIB_TYPE} ${deepgreen_src})

target_link_libraries(deepgreen
  ${CUDA_CUBLAS_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBS}
)
