#include <iostream>
#include "gtest/gtest.h"
#include "deepgreen/matrix.hh"
#include "deepgreen/matrix_mm.hh"
#include "deepgreen/matrix_mm.cuh"

TEST(DeepgreenMatrixMult, NaiveMM) {
  // [8.0   3.0   0.0   1.0]    [5.0 8.0 0.0 6.6]   [ 53    82.5    11.5    60.5]
  // [2.0   5.0   4.0   9.0]  * [4.0 6.0 3.5 0.1] = [ 51    78.5    36.1   118.3]
  // [7.0   6.0   10.   13.]    [3.0 7.0 2.4 9.5]   [102   168.5    58.0   238.0]
  //                            [1.0 0.5 1.0 7.4]
  // >>> import numpy as np
  // >>> a = [[8.0, 3.0, 0.0, 1.0], [2.0, 5.0, 4.0, 9.0], [7.0, 6.0, 10., 13.]]
  // >>> b = [[5.0, 8.0, 0.0, 6.6], [4.0, 6.0, 3.5, 0.1], [3.0, 7.0, 2.4, 9.5], [1.0, 0.5, 1.0, 7.4]]
  // >>> np.matmul(a, b)

  auto a = deepgreen::matrix<float>(3, 4, {
    8.0, 2.0, 7.0, 3.0, 5.0, 6.0, 0.0, 4.0, 10.0, 1.0, 9.0, 13.0
  });

  auto b = deepgreen::matrix<float>(4, 4, {
    5.0, 4.0, 3.0, 1.0, 8.0, 6.0, 7.0, 0.5, 0.0, 3.5, 2.4, 1.0, 6.6, 0.1, 9.5, 7.4
  });

  auto c = deepgreen::naive_mm(a, b);

  EXPECT_EQ(size_t{12}, c.size());
  EXPECT_EQ(size_t{3}, c.rows());
  EXPECT_EQ(size_t{4}, c.cols());
  EXPECT_FLOAT_EQ(53.0f, c(0, 0));
  EXPECT_FLOAT_EQ(78.5f, c(1, 1));
  EXPECT_FLOAT_EQ(58.0f, c(2, 2));
}

TEST(DeepgreenMatrixMult, CudaMM) {
  // [1.8   7.0   2.8   3.0]    [5.1 2.4]   [56.04,  98.74]
  // [2.0   5.2   4.7   4.1]  * [4.6 6.1] = [56.95, 115.85]
  // [4.0   1.2   5.0   8.0]    [3.2 9.9]   [57.12, 130.42]
  //                            [1.9 8.0]
  // >>> import numpy as np
  // >>> a = [[1.8, 7.0, 2.8, 3.0], [2.0, 5.2, 4.7, 4.1], [4.0, 1.2, 5.0, 8.0]]
  // >>> b = [[5.1, 2.4], [4.6, 6.1], [3.2, 9.9], [1.9, 8.0]]
  // >>> np.matmul(a, b)

  auto a = deepgreen::matrix<float>(3, 4, {
    1.8, 2.0, 4.0, 7.0, 5.2, 1.2, 2.8, 4.7, 5.0, 3.0, 4.1, 8.0
  });

  auto b = deepgreen::matrix<float>(4, 2, {
    5.1, 4.6, 3.2, 1.9, 2.4, 6.1, 9.9, 8.0
  });

  auto c = deepgreen::cuda_mm(a, b);

  EXPECT_EQ(size_t{6}, c.size());
  EXPECT_EQ(size_t{3}, c.rows());
  EXPECT_EQ(size_t{2}, c.cols());
  EXPECT_FLOAT_EQ(56.04f, c(0, 0));
  EXPECT_FLOAT_EQ(115.85f, c(1, 1));
  EXPECT_FLOAT_EQ(130.42f, c(2, 1));
}

TEST(DeepgreenMatrixMult, CudaMMSameAsNaive) {
  // Two random matrices:
  auto a = deepgreen::matrix<float>(10, 8, 42, 1.5, 2.5);
  auto b = deepgreen::matrix<float>(8, 20, 42, 2.5, 5.0);

  auto c_naive = deepgreen::naive_mm(a, b);
  auto c_cuda = deepgreen::cuda_mm(a, b);

  EXPECT_EQ(c_naive.size(), c_cuda.size());
  EXPECT_EQ(c_naive.rows(), c_cuda.rows());
  EXPECT_EQ(c_naive.cols(), c_cuda.cols());
  for (auto i = 0u; i < c_naive.size(); ++i)
    EXPECT_FLOAT_EQ(c_naive[i], c_cuda[i]);
}
