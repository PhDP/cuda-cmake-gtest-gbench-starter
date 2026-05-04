set(deepgreen_src
  matrix_mm.cu
)

add_library(${DG_LIB_NAME} ${DG_LIB_TYPE} ${deepgreen_src})

target_include_directories(${DG_LIB_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Boost_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${DG_LIB_NAME} PRIVATE
  CUDA::cudart
  CUDA::cublas
  Threads::Threads
  m
)

set_property(TARGET ${DG_LIB_NAME} PROPERTY CUDA_ARCHITECTURES native)

install(TARGETS ${DG_LIB_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ../include/deepgreen DESTINATION include)
