set(deepgreen_src
  matrix_mm.cu
)

add_library(${DG_LIB_NAME} ${DG_LIB_TYPE} ${deepgreen_src})
#link_directories(${CUDAToolkit_LIBRARY_DIR}$)

target_link_libraries(${DG_LIB_NAME}
  CUDA::cudart
  CUDA::cublas
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBRARIES}
)

set_property(TARGET ${DG_LIB_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

install(TARGETS ${DG_LIB_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ../include/deepgreen DESTINATION include)

