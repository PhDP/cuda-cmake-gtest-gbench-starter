#include <cublas_v2.h>
#include "deepgreen/matrix.hh"
#include "deepgreen/matrix_mm.cuh"

namespace deepgreen {

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
  int lda = m,ldb = k,ldc = m;
  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Do the actual multiplication
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

  // Destroy the handle
  cublasDestroy(handle);
}

auto cuda_mm(matrix<float> const& a, matrix<float> const& b) -> matrix<float> {
  size_t const m = a.rows(), n = b.cols(), k = a.cols();
  size_t const c_bytes = m * n * sizeof(float);
  float* elems = (float*)std::malloc(c_bytes);

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, a.bytes());
  cudaMalloc(&d_B, b.bytes());
  cudaMalloc(&d_C, c_bytes);

  cudaMemcpy(d_A, a.data(), a.bytes(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, b.data(), b.bytes(), cudaMemcpyHostToDevice);

  gpu_blas_mmul(d_A, d_B, d_C, m, k, n);

  cudaMemcpy(elems, d_C, c_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return matrix<float>(m, n, elems);
}

} /* end namespace deepgreen */
