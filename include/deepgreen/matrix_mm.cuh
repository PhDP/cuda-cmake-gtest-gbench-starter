#ifndef MATRIX_MM_CUH_
#define MATRIX_MM_CUH_

#include "deepgreen/matrix.hh"

namespace deepgreen {

/**
  \brief Matrix multiplication on the GPU with cublas.
 */
auto cuda_mm(matrix<float> const& a, matrix<float> const& b) -> matrix<float>;

} /* end namespace deepgreen */

#endif

