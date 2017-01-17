#ifndef MATRIX_MM_HH_
#define MATRIX_MM_HH_

#include <cstdlib>
#include <cassert>
#include "deepgreen/matrix.hh"

namespace deepgreen {

/**
  \brief Naive matrix multiplication (A_mk * B_kn = C_mn)
 */
template<typename T>
auto naive_mm(matrix<T> const& a, matrix<T> const& b) -> matrix<T> {
  size_t const m = a.rows(), n = b.cols(), k = a.cols();
  assert(k == b.rows());
  auto c = matrix<T>(m, n);
  for (size_t row = 0; row < m; ++row) {
    for (size_t col = 0; col < n; ++col) {
      c(row, col) = 0.0;
      for (size_t i = 0; i < k; ++i) {
        c(row, col) += a(row, i) * b(i, col);
      }
    }
  }
  return c;
}

} /* end namespace deepgreen */

#endif

