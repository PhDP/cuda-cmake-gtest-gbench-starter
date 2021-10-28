set(deepgreen_src
  matrix_mm.cu
)

cuda_add_library(${DG_LIB_NAME} ${DG_LIB_TYPE} ${deepgreen_src})

target_link_libraries(${DG_LIB_NAME}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBRARIES}
)

install(TARGETS ${DG_LIB_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ../include/deepgreen DESTINATION include)
