#include <iostream>
#include "gtest/gtest.h"

#include "deepgreen/matrix.hh"

TEST(DeepgreenMatrix, BasicConstructor) {
  auto x = deepgreen::matrix<float>(6, 2);
  for (auto i = 0u; i < 12; ++i)
    x(i) = (float)i;

  EXPECT_EQ(size_t{12}, x.size());
  EXPECT_EQ(size_t{12 * sizeof(float)}, x.bytes());
  EXPECT_EQ(size_t{6}, x.rows());
  EXPECT_EQ(size_t{2}, x.cols());
  EXPECT_FLOAT_EQ(10.0f, x(10));
  EXPECT_FLOAT_EQ(0.0f, x(0, 0));
  EXPECT_FLOAT_EQ(8.0f, x(2, 1));
  EXPECT_FLOAT_EQ(11.0f, x(5, 1));
}

TEST(DeepgreenMatrix, InitializerListConstructor) {
  // [8.0   3.0   0.0   1.0]
  // [2.0   5.0   4.0   9.0]
  // [7.0   6.0   10.   13.]

  auto x = deepgreen::matrix<float>(3, 4, {
    8.0, 2.0, 7.0, 3.0, 5.0, 6.0, 0.0, 4.0, 10.0, 1.0, 9.0, 13.0
  });

  EXPECT_EQ(size_t{12}, x.size());
  EXPECT_EQ(size_t{12 * sizeof(float)}, x.bytes());
  EXPECT_EQ(size_t{3}, x.rows());
  EXPECT_EQ(size_t{4}, x.cols());
  EXPECT_FLOAT_EQ(8.0f, x(0, 0));
  EXPECT_FLOAT_EQ(1.0f, x(0, 3));
  EXPECT_FLOAT_EQ(10.0f, x(2, 2));
}

TEST(DeepgreenMatrix, RandomMatrixConstructor) {
  size_t const seed = 8;
  double const minimum = 1.0545;
  double const maximum = 28.1;
  auto x = deepgreen::matrix<double>(9, 8, seed, minimum, maximum);

  EXPECT_EQ(size_t{9 * 8}, x.size());
  EXPECT_EQ(size_t{9 * 8 * sizeof(double)}, x.bytes());
  EXPECT_EQ(size_t{9}, x.rows());
  EXPECT_EQ(size_t{8}, x.cols());

  for (auto i = 0u; i < x.size(); ++i) {
    double const val = x(i);
    EXPECT_TRUE((val >= minimum && val <= maximum));
  }
}
