// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cfloat>
#include <cstdint>
#include <random>
#include <type_traits>

#include "framework/utils.h"
#include "kleidicv/kleidicv.h"

namespace {
// Google Mock custom matcher to check that two numbers are the same as each
// other to customisable precision. This is similar to gmock's standard
// FloatNear except that maximum absolute error is calculated relative to the
// magnitude of the arguments instead of being a fixed value.
MATCHER_P(FloatSimilar, max_relative_error, "") {
  auto [a, b] = arg;
  float max_abs_error = std::max(std::abs(a), std::abs(b)) * max_relative_error;
  return std::abs(a - b) <= max_abs_error;
}
}  // namespace

// Check that FloatSimilar behaves as designed.
TEST(FloatSimilar, FloatSimilar) {
  std::vector<float> eq_a = {0.0F, 1.000009F, 0.001000009F, 1e10F};
  std::vector<float> eq_b = {0.0F, 1.0F, 0.001F, 1.000009e10F};
  EXPECT_THAT(eq_a, ::testing::Pointwise(FloatSimilar(1e-5F), eq_b));

  std::vector<float> ne_a = {1.0F, 1.000011F, 0.001F, 1.0001e10F};
  std::vector<float> ne_b = {-1.0F, 1.0F, 0.001000011F, 1e10F};
  EXPECT_THAT(ne_a,
              ::testing::Pointwise(::testing::Not(FloatSimilar(1e-5F)), ne_b));
}

template <typename T>
static void resize_linear_unaccelerated_generic_upscale(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height) {
  src_stride /= sizeof(T);
  dst_stride /= sizeof(T);
  size_t scale_width = dst_width / src_width;
  size_t scale_height = dst_height / src_height;

  auto lerp1d_horizontal = [scale_width](T coeff_a, T a, T coeff_b, T b) {
    if constexpr (std::is_floating_point_v<T>) {
      float divisor = static_cast<float>(2 * scale_width);
      return (coeff_a / divisor * a + coeff_b / divisor * b);
    } else {
      T bias = scale_width;
      return (coeff_a * a + coeff_b * b + bias) / (2 * scale_width);
    }
  };

  auto lerp1d_vertical = [scale_height](T coeff_a, T a, T coeff_b, T b) {
    if constexpr (std::is_floating_point_v<T>) {
      float divisor = static_cast<float>(2 * scale_height);
      return (coeff_a / divisor * a + coeff_b / divisor * b);
    } else {
      T bias = scale_height;
      return (coeff_a * a + coeff_b * b + bias) / (2 * scale_height);
    }
  };

  auto lerp2d = [scale_width, scale_height](T coeff_a, T a, T coeff_b, T b,
                                            T coeff_c, T c, T coeff_d, T d) {
    if constexpr (std::is_floating_point_v<T>) {
      float divisor = static_cast<float>(4 * scale_height * scale_width);
      return (coeff_a / divisor * a + coeff_b / divisor * b +
              coeff_c / divisor * c + coeff_d / divisor * d);
    } else {
      T bias = 2 * scale_height * scale_width;
      return (coeff_a * a + coeff_b * b + coeff_c * c + coeff_d * d + bias) /
             (4 * scale_height * scale_width);
    }
  };
  // Handle top or bottom edge
  auto process_edge_row = [src_width, dst_width, scale_width, scale_height,
                           lerp1d_horizontal](const T *src_row, T *dst_row,
                                              size_t dst_stride) {
    for (size_t y = 0; y < scale_height / 2; ++y) {
      for (size_t x = 0; x < scale_width / 2; ++x) {
        // Left elements
        dst_row[x] = src_row[0];
        // Right elements
        dst_row[dst_width - scale_width / 2 + x] = src_row[src_width - 1];
      }
      // Middle elements
      for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
        size_t dst_x = src_x * scale_width + scale_width / 2;
        const T a = src_row[src_x], b = src_row[src_x + 1];
        for (size_t x = 0; x < scale_width; ++x) {
          dst_row[dst_x + x] =
              lerp1d_horizontal((scale_width - x) * 2 - 1, a, x * 2 + 1, b);
        }
      }
      dst_row += dst_stride;
    }
  };

  auto process_row = [src_width, dst_width, scale_width, scale_height,
                      lerp1d_vertical,
                      lerp2d](const T *src_row0, const T *src_row1, T *dst_row,
                              size_t dst_stride) {
    for (size_t y = 0; y < scale_height; ++y) {
      for (size_t x = 0; x < scale_width / 2; ++x) {
        // Left elements
        dst_row[x] = lerp1d_vertical((scale_height - y) * 2 - 1, src_row0[0],
                                     y * 2 + 1, src_row1[0]);
        // Right elements
        dst_row[dst_width - scale_width / 2 + x] =
            lerp1d_vertical((scale_height - y) * 2 - 1, src_row0[src_width - 1],
                            y * 2 + 1, src_row1[src_width - 1]);
      }
      // Middle elements
      for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
        size_t dst_x = src_x * scale_width + scale_width / 2;
        const T a = src_row0[src_x], b = src_row0[src_x + 1];
        const T c = src_row1[src_x], d = src_row1[src_x + 1];
        for (size_t x = 0; x < scale_width; ++x) {
          dst_row[dst_x + x] =
              lerp2d(((scale_width - x) * 2 - 1) * ((scale_height - y) * 2 - 1),
                     a, (x * 2 + 1) * ((scale_height - y) * 2 - 1), b,
                     ((scale_width - x) * 2 - 1) * (y * 2 + 1), c,
                     (x * 2 + 1) * (y * 2 + 1), d);
        }
      }
      dst_row += dst_stride;
    }
  };

  // Top rows
  process_edge_row(src, dst, dst_stride);

  // Middle rows
  for (size_t src_y = 0; src_y + 1 < src_height; ++src_y) {
    size_t dst_y = src_y * scale_height + scale_height / 2;
    const T *src_row0 = src + src_stride * src_y;
    const T *src_row1 = src_row0 + src_stride;
    process_row(src_row0, src_row1, dst + dst_stride * dst_y, dst_stride);
  }

  // Bottom rows
  process_edge_row(src + src_stride * (src_height - 1),
                   dst + dst_stride * (dst_height - scale_height / 2),
                   dst_stride);
}

template <typename T>
static void resize_linear_unaccelerated(const T *src, size_t src_stride,
                                        size_t src_width, size_t src_height,
                                        T *dst, size_t dst_stride,
                                        size_t dst_width, size_t dst_height) {
  // even integer scaling
  if (src_width > 0 && src_height > 0 && dst_width % src_width == 0 &&
      dst_height % src_height == 0 && (dst_width / src_width) % 2 == 0 &&
      (dst_height / src_height) % 2 == 0) {
    resize_linear_unaccelerated_generic_upscale(src, src_stride, src_width,
                                                src_height, dst, dst_stride,
                                                dst_width, dst_height);
  }
}

// Provides an alternative type instead of char to avoid values being printed as
// char.
template <typename T>
struct PrintTypeGetter {
  using type = T;
};

template <>
struct PrintTypeGetter<int8_t> {
  using type = int;
};

template <>
struct PrintTypeGetter<uint8_t> {
  using type = unsigned;
};

inline kleidicv_error_t kleidicv_resize_linear(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height) {
  return kleidicv_resize_linear_u8(src, src_stride, src_width, src_height, dst,
                                   dst_stride, dst_width, dst_height);
}

inline kleidicv_error_t kleidicv_resize_linear(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    float *dst, size_t dst_stride, size_t dst_width, size_t dst_height) {
  return kleidicv_resize_linear_f32(src, src_stride, src_width, src_height, dst,
                                    dst_stride, dst_width, dst_height);
}

template <typename T>
class ResizeLinear : public testing::Test {};

using ResizeLinearTypes = testing::Types<uint8_t, float>;
TYPED_TEST_SUITE(ResizeLinear, ResizeLinearTypes);

TYPED_TEST(ResizeLinear, NotImplemented) {
  const TypeParam src[1] = {};
  TypeParam dst[4];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_resize_linear(src, sizeof(TypeParam), 1, 1, dst,
                                   sizeof(TypeParam) * 2, 2, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_resize_linear(src, sizeof(TypeParam), 1, 1, dst,
                                   sizeof(TypeParam), 1, 2));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_resize_linear(src, sizeof(TypeParam), 1, 1, dst,
                                   sizeof(TypeParam) * 4, 4, 2));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_resize_linear(src, sizeof(TypeParam), 1, 1, dst,
                                   sizeof(TypeParam) * 2, 2, 4));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_resize_linear(src, sizeof(TypeParam), 1, 1, dst,
                                   sizeof(TypeParam) * 2, 8, 4));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_resize_linear(src, sizeof(TypeParam), 1, 1, dst,
                                   sizeof(TypeParam) * 2, 4, 8));
}

TYPED_TEST(ResizeLinear, NullPointer) {
  const TypeParam src[1] = {};
  TypeParam dst[4];
  kleidicv_error_t (*f)(const TypeParam *, size_t, size_t, size_t, TypeParam *,
                        size_t, size_t, size_t) = kleidicv_resize_linear;
  test::test_null_args(f, src, sizeof(TypeParam), 1, 1, dst,
                       sizeof(TypeParam) * 2, 2, 2);
}

TYPED_TEST(ResizeLinear, InvalidImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[4];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_resize_linear(
                src, sizeof(TypeParam) * ((KLEIDICV_MAX_IMAGE_PIXELS + 1) / 2),
                (KLEIDICV_MAX_IMAGE_PIXELS + 1) / 2, 1, dst,
                sizeof(TypeParam) * (KLEIDICV_MAX_IMAGE_PIXELS + 1),
                KLEIDICV_MAX_IMAGE_PIXELS + 1, 2));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_resize_linear(
                src, sizeof(TypeParam) * ((KLEIDICV_MAX_IMAGE_PIXELS - 1) / 2),
                (KLEIDICV_MAX_IMAGE_PIXELS - 1) / 2,
                (KLEIDICV_MAX_IMAGE_PIXELS - 1) / 2, dst,
                sizeof(TypeParam) * (KLEIDICV_MAX_IMAGE_PIXELS - 1),
                KLEIDICV_MAX_IMAGE_PIXELS - 1, KLEIDICV_MAX_IMAGE_PIXELS - 1));
}

TYPED_TEST(ResizeLinear, ZeroImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[1];
  EXPECT_EQ(KLEIDICV_OK, kleidicv_resize_linear(src, 0, 0, 0, dst, 0, 0, 0));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_resize_linear(src, sizeof(TypeParam), 1, 0, dst,
                                   sizeof(TypeParam) * 2, 2, 0));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_resize_linear(src, 0, 0, 1, dst, 0, 0, 2));
}

template <typename T>
static void do_large_dimensions_test(size_t x_scale, size_t y_scale) {
  size_t src_width = 2049;
  size_t src_height = 5;
  size_t src_stride_pixels = src_width + 6;
  size_t dst_width = src_width * x_scale;
  size_t dst_height = src_height * y_scale;
  size_t dst_stride_pixels = dst_width + 3;

  std::vector<T> src, dst, expected_data;
  src.resize(src_stride_pixels * src_height);
  dst.resize(dst_stride_pixels * dst_height);
  expected_data.resize(dst_stride_pixels * dst_height);
  std::mt19937 generator{
      static_cast<std::mt19937::result_type>(test::Options::seed())};
  std::generate(src.begin(), src.end(), generator);
  resize_linear_unaccelerated(src.data(), src_stride_pixels * sizeof(T),
                              src_width, src_height, expected_data.data(),
                              dst_stride_pixels * sizeof(T), dst_width,
                              dst_height);
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_resize_linear(src.data(), src_stride_pixels * sizeof(T),
                                   src_width, src_height, dst.data(),
                                   dst_stride_pixels * sizeof(T), dst_width,
                                   dst_height));

  for (size_t y = 0; y < dst_height; ++y) {
    std::vector<typename PrintTypeGetter<T>::type> actual{
        dst.begin() + static_cast<ptrdiff_t>(y * dst_stride_pixels),
        dst.begin() +
            static_cast<ptrdiff_t>(y * dst_stride_pixels + dst_width)},
        expected{expected_data.begin() +
                     static_cast<ptrdiff_t>(y * dst_stride_pixels),
                 expected_data.begin() +
                     static_cast<ptrdiff_t>(y * dst_stride_pixels + dst_width)};
    if constexpr (std::is_floating_point_v<T>) {
      EXPECT_THAT(actual, ::testing::Pointwise(FloatSimilar(2e-6F), expected))
          << "Row #" << y;
    } else {
      EXPECT_THAT(actual, ::testing::ElementsAreArray(expected))
          << "Row #" << y;
    }
  }
}

TYPED_TEST(ResizeLinear, LargeDimensions2x2) {
  do_large_dimensions_test<TypeParam>(2, 2);
}

TYPED_TEST(ResizeLinear, LargeDimensions4x4) {
  do_large_dimensions_test<TypeParam>(4, 4);
}

TEST(ResizeLinearFloat, LargeDimensions8x8) {
  do_large_dimensions_test<float>(8, 8);
}

TEST(ResizeLinearU8, NotImplemented_8x8) {
  const uint8_t src[1] = {};
  uint8_t dst[4];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_resize_linear(src, sizeof(uint8_t), 1, 1, dst,
                                   sizeof(uint8_t) * 8, 8, 8));
}

// Parameterised tests
template <typename T>
struct ResizeTestParams {
  std::vector<std::vector<T>> src;
  std::vector<std::vector<T>> expected;

  friend void PrintTo(const ResizeTestParams &v, std::ostream *os) {
    *os << "([\n";
    for (size_t y = 0; y < v.src.size(); ++y) {
      const auto &row = v.src[y];
      *os << "  [";
      for (size_t x = 0; x < row.size(); ++x) {
        *os << std::setw(3) << typename PrintTypeGetter<T>::type{row[x]};
        if (x + 1 != row.size()) {
          *os << ", ";
        }
      }
      *os << "]";
      if (y + 1 != v.src.size()) {
        *os << ",\n";
      }
    }
    *os << "], " << v.expected.size() << ")";
  }
};

template <typename T>
void do_linear_resize_test(const ResizeTestParams<T> &param, size_t src_padding,
                           size_t dst_padding) {
  size_t src_width = param.src[0].size();
  size_t src_height = param.src.size();
  size_t src_stride_pixels = src_width + src_padding;
  size_t dst_width = param.expected[0].size();
  size_t dst_height = param.expected.size();
  size_t dst_stride_pixels = dst_width + dst_padding;

  auto flatten = [](const std::vector<std::vector<T>> &vec2d, size_t padding) {
    std::vector<T> result;
    for (const auto &row : vec2d) {
      result.insert(result.end(), row.begin(), row.end());
      result.resize(result.size() + padding);
    }
    return result;
  };
  std::vector<T> src = flatten(param.src, src_padding), dst;
  dst.resize(dst_stride_pixels * dst_height);

  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_resize_linear(src.data(), src_stride_pixels * sizeof(T),
                                   src_width, src_height, dst.data(),
                                   dst_stride_pixels * sizeof(T), dst_width,
                                   dst_height));
  for (size_t y = 0; y < dst_height; ++y) {
    std::vector<typename PrintTypeGetter<T>::type> actual{
        dst.begin() + static_cast<ptrdiff_t>(y * dst_stride_pixels),
        dst.begin() +
            static_cast<ptrdiff_t>(y * dst_stride_pixels + dst_width)},
        expected{param.expected[y].begin(), param.expected[y].end()};
    if constexpr (std::is_floating_point_v<T>) {
      EXPECT_THAT(actual, ::testing::Pointwise(FloatSimilar(2e-6F), expected))
          << "Row #" << y;
    } else {
      EXPECT_THAT(actual, ::testing::ElementsAreArray(expected))
          << "Row #" << y;
    }
  }
}

class ResizeLinearU8
    : public testing::TestWithParam<ResizeTestParams<uint8_t>> {};

TEST_P(ResizeLinearU8, ResizeNoPadding) {
  do_linear_resize_test(GetParam(), 0, 0);
}

TEST_P(ResizeLinearU8, ResizeWithPadding) {
  do_linear_resize_test(GetParam(), 1, 2);
}

TEST_P(ResizeLinearU8, ResizePadDst) {
  do_linear_resize_test(GetParam(), 0, 3);
}

TEST_P(ResizeLinearU8, ResizePadSrc) {
  do_linear_resize_test(GetParam(), 4, 0);
}

using Pu8 = ResizeTestParams<uint8_t>;

INSTANTIATE_TEST_SUITE_P(
    ResizeLinear, ResizeLinearU8,
    testing::Values(
        // 1*1 -> 2*2
        Pu8{{{123}}, {{123, 123}, {123, 123}}},
        // 2*1 -> 4*2
        Pu8{{{0, 255}}, {{0, 64, 191, 255}, {0, 64, 191, 255}}},
        // 2*1 -> 4*2. Check rounding behaviour.
        Pu8{{{1, 63}}, {{1, 17, 48, 63}, {1, 17, 48, 63}}},
        // 2*2 -> 4*4
        Pu8{{{0, 255}, {100, 8}},
            {{0, 64, 191, 255},
             {25, 67, 151, 193},
             {75, 74, 71, 70},
             {100, 77, 31, 8}}},
        // 3*3 -> 6*6
        Pu8{{{1, 63, 164}, {28, 251, 35}, {218, 64, 99}},
            {{1, 17, 48, 88, 139, 164},
             {8, 33, 84, 115, 126, 132},
             {21, 67, 158, 170, 101, 67},
             {76, 108, 172, 166, 89, 51},
             {171, 156, 126, 104, 90, 83},
             {218, 180, 103, 73, 90, 99}}},
        // 4*4 -> 8*8
        Pu8{{{10, 30, 5, 70},
             {255, 11, 11, 12},
             {127, 127, 128, 0},
             {200, 100, 150, 50}},
            {{10, 15, 25, 24, 11, 21, 54, 70},
             {71, 60, 37, 21, 11, 19, 43, 56},
             {194, 149, 60, 14, 11, 14, 22, 27},
             {223, 177, 86, 40, 40, 32, 17, 9},
             {159, 144, 113, 98, 99, 75, 27, 3},
             {145, 139, 127, 124, 130, 103, 43, 13},
             {182, 163, 126, 116, 135, 118, 64, 38},
             {200, 175, 125, 113, 138, 125, 75, 50}}},
        // 35*2 -> 70*4
        Pu8{{{0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  82,
              155, 104, 108, 227, 46,  162, 21,  220, 235, 183, 113, 225,
              146, 196, 144, 104, 148, 19,  126, 172, 9,   12,  61},
             {4,  5,   6,   7,   8,  9,   10, 11,  12,  13,  14, 193,
              44, 105, 191, 106, 73, 148, 13, 161, 118, 21,  3,  34,
              40, 150, 120, 68,  75, 14,  31, 124, 221, 214, 146}},
            {{0,   0,   1,   1,   2,   2,   3,   3,   4,   4,   5,   5,
              6,   6,   7,   7,   8,   8,   9,   9,   10,  28,  64,  100,
              137, 142, 117, 105, 107, 138, 197, 182, 91,  75,  133, 127,
              56,  71,  170, 224, 231, 222, 196, 166, 131, 141, 197, 205,
              166, 159, 184, 183, 157, 134, 114, 115, 137, 116, 51,  46,
              99,  138, 161, 131, 50,  10,  11,  24,  49,  61},
             {1,   1,   2,   2,   3,   3,   4,   4,   5,   5,   6,   6,
              7,   7,   8,   8,   9,   9,   10,  10,  11,  36,  85,  114,
              123, 122, 110, 110, 123, 146, 180, 161, 89,  79,  132, 124,
              54,  66,  159, 205, 206, 190, 158, 128, 100, 108, 154, 163,
              134, 136, 168, 173, 150, 127, 106, 104, 121, 102, 46,  39,
              81,  117, 146, 136, 87,  62,  62,  67,  77,  82},
             {3,  3,  4,   4,   5,   5,   6,   6,   7,   7,  8,   8,
              9,  9,  10,  10,  11,  11,  12,  12,  13,  51, 127, 142,
              95, 80, 97,  121, 154, 162, 145, 119, 84,  88, 130, 117,
              49, 55, 136, 169, 154, 126, 83,  54,  38,  43, 69,  78,
              70, 90, 138, 153, 135, 114, 89,  81,  89,  74, 35,  25,
              45, 75, 116, 144, 160, 167, 165, 154, 134, 125},
             {4,  4,  5,   5,   6,   6,   7,   7,   8,   8,  9,   9,
              10, 10, 11,  11,  12,  12,  13,  13,  14,  59, 148, 156,
              81, 59, 90,  127, 170, 170, 127, 98,  81,  92, 129, 114,
              47, 50, 124, 150, 129, 94,  45,  17,  8,   11, 26,  36,
              39, 68, 123, 143, 128, 107, 81,  70,  73,  60, 29,  18,
              27, 54, 101, 148, 197, 219, 216, 197, 163, 146}}},
        // 2*2 -> 8*8
        Pu8{{{0, 255}, {128, 124}},
            {{0, 0, 32, 96, 159, 223, 255, 255},
             {0, 0, 32, 96, 159, 223, 255, 255},
             {16, 16, 44, 99, 155, 211, 239, 239},
             {48, 48, 68, 107, 147, 186, 206, 206},
             {80, 80, 92, 115, 138, 161, 173, 173},
             {112, 112, 116, 123, 130, 137, 140, 140},
             {128, 128, 128, 127, 126, 125, 124, 124},
             {128, 128, 128, 127, 126, 125, 124, 124}}},
        // 35*2 -> 140*8
        Pu8{{{0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  82,
              155, 104, 108, 227, 46,  162, 21,  220, 235, 183, 113, 225,
              146, 196, 144, 104, 148, 19,  126, 172, 9,   12,  61},
             {4,  5,   6,   7,   8,  9,   10, 11,  12,  13,  14, 193,
              44, 105, 191, 106, 73, 148, 13, 161, 118, 21,  3,  34,
              40, 150, 120, 68,  75, 14,  31, 124, 221, 214, 146}},
            {{0,   0,   0,   0,   1,   1,   1,   1,   2,   2,   2,   2,   3,
              3,   3,   3,   4,   4,   4,   4,   5,   5,   5,   5,   6,   6,
              6,   6,   7,   7,   7,   7,   8,   8,   8,   8,   9,   9,   9,
              9,   10,  10,  19,  37,  55,  73,  91,  109, 128, 146, 149, 136,
              123, 110, 105, 106, 107, 108, 123, 153, 182, 212, 204, 159, 114,
              69,  61,  90,  119, 148, 144, 109, 74,  39,  46,  96,  145, 195,
              222, 226, 229, 233, 229, 216, 203, 190, 174, 157, 139, 122, 127,
              155, 183, 211, 215, 195, 176, 156, 152, 165, 177, 190, 190, 177,
              164, 151, 139, 129, 119, 109, 110, 121, 132, 143, 132, 100, 67,
              35,  32,  59,  86,  113, 132, 143, 155, 166, 152, 111, 70,  29,
              9,   10,  11,  12,  18,  30,  43,  55,  61,  61},
             {0,   0,   0,   0,   1,   1,   1,   1,   2,   2,   2,   2,   3,
              3,   3,   3,   4,   4,   4,   4,   5,   5,   5,   5,   6,   6,
              6,   6,   7,   7,   7,   7,   8,   8,   8,   8,   9,   9,   9,
              9,   10,  10,  19,  37,  55,  73,  91,  109, 128, 146, 149, 136,
              123, 110, 105, 106, 107, 108, 123, 153, 182, 212, 204, 159, 114,
              69,  61,  90,  119, 148, 144, 109, 74,  39,  46,  96,  145, 195,
              222, 226, 229, 233, 229, 216, 203, 190, 174, 157, 139, 122, 127,
              155, 183, 211, 215, 195, 176, 156, 152, 165, 177, 190, 190, 177,
              164, 151, 139, 129, 119, 109, 110, 121, 132, 143, 132, 100, 67,
              35,  32,  59,  86,  113, 132, 143, 155, 166, 152, 111, 70,  29,
              9,   10,  11,  12,  18,  30,  43,  55,  61,  61},
             {1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   3,   3,   3,
              3,   4,   4,   4,   4,   5,   5,   5,   5,   6,   6,   6,   6,
              7,   7,   7,   7,   8,   8,   8,   8,   9,   9,   9,   9,   10,
              10,  10,  10,  21,  43,  64,  85,  102, 113, 124, 135, 137, 127,
              118, 109, 106, 109, 113, 117, 130, 153, 177, 200, 192, 151, 110,
              70,  63,  91,  119, 146, 143, 108, 73,  38,  44,  92,  140, 189,
              214, 216, 217, 219, 213, 199, 184, 170, 155, 139, 123, 107, 112,
              137, 163, 188, 193, 175, 158, 141, 140, 154, 169, 183, 184, 172,
              159, 147, 136, 125, 115, 105, 104, 114, 124, 134, 124, 94,  64,
              33,  30,  54,  78,  102, 121, 134, 147, 160, 150, 117, 84,  52,
              36,  36,  37,  37,  42,  50,  59,  67,  72,  72},
             {2,   2,   2,   2,   2,   2,   3,   3,   3,   3,   4,   4,   4,
              4,   5,   5,   5,   5,   6,   6,   6,   6,   7,   7,   7,   7,
              8,   8,   8,   8,   9,   9,   9,   9,   10,  10,  10,  10,  11,
              11,  11,  11,  26,  54,  82,  110, 122, 120, 117, 115, 112, 110,
              108, 106, 109, 117, 126, 135, 144, 155, 166, 176, 166, 135, 103,
              72,  69,  94,  119, 144, 139, 105, 70,  35,  40,  85,  130, 175,
              197, 195, 194, 192, 183, 165, 148, 131, 116, 103, 91,  78,  82,
              102, 123, 143, 147, 136, 124, 112, 115, 133, 152, 170, 173, 162,
              151, 140, 129, 118, 107, 96,  94,  102, 109, 117, 108, 82,  56,
              30,  26,  45,  63,  81,  98,  114, 130, 146, 146, 129, 113, 97,
              88,  88,  88,  88,  88,  90,  91,  92,  93,  93},
             {3,   3,   3,   3,   3,   3,   4,   4,   4,   4,   5,   5,   5,
              5,   6,   6,   6,   6,   7,   7,   7,   7,   8,   8,   8,   8,
              9,   9,   9,   9,   10,  10,  10,  10,  11,  11,  11,  11,  12,
              12,  12,  12,  30,  65,  99,  134, 143, 127, 110, 94,  88,  93,
              98,  102, 112, 125, 139, 153, 159, 157, 155, 152, 140, 118, 96,
              74,  74,  97,  119, 142, 136, 102, 67,  33,  37,  79,  120, 162,
              180, 175, 170, 165, 152, 132, 112, 92,  77,  68,  58,  49,  52,
              67,  83,  98,  102, 96,  89,  83,  91,  113, 134, 156, 162, 153,
              143, 134, 123, 111, 99,  87,  84,  89,  95,  100, 92,  70,  48,
              27,  22,  35,  48,  60,  76,  95,  114, 133, 142, 142, 142, 142,
              141, 140, 139, 139, 135, 129, 123, 117, 114, 114},
             {4,   4,   4,   4,   4,   4,   5,   5,   5,   5,   6,   6,   6,
              6,   7,   7,   7,   7,   8,   8,   8,   8,   9,   9,   9,   9,
              10,  10,  10,  10,  11,  11,  11,  11,  12,  12,  12,  12,  13,
              13,  13,  13,  34,  76,  117, 158, 164, 134, 103, 73,  64,  76,
              87,  99,  114, 133, 152, 171, 173, 158, 143, 129, 115, 102, 89,
              76,  80,  100, 120, 140, 133, 99,  65,  31,  33,  72,  110, 149,
              164, 155, 146, 137, 121, 98,  76,  53,  38,  32,  26,  20,  22,
              32,  42,  53,  57,  56,  55,  54,  66,  92,  117, 143, 152, 143,
              135, 127, 117, 104, 91,  79,  74,  77,  80,  83,  75,  58,  41,
              23,  18,  25,  32,  39,  54,  76,  97,  119, 138, 154, 170, 186,
              194, 192, 191, 189, 182, 169, 155, 142, 135, 135},
             {4,   4,   4,   4,   5,   5,   5,   5,   6,   6,   6,   6,   7,
              7,   7,   7,   8,   8,   8,   8,   9,   9,   9,   9,   10,  10,
              10,  10,  11,  11,  11,  11,  12,  12,  12,  12,  13,  13,  13,
              13,  14,  14,  36,  81,  126, 171, 174, 137, 100, 63,  52,  67,
              82,  97,  116, 137, 159, 180, 180, 159, 138, 117, 102, 94,  85,
              77,  82,  101, 120, 139, 131, 97,  64,  30,  32,  69,  106, 143,
              156, 145, 134, 123, 106, 82,  57,  33,  19,  14,  10,  5,   7,
              15,  22,  30,  35,  36,  38,  39,  54,  81,  109, 136, 146, 139,
              131, 124, 114, 101, 88,  75,  69,  71,  72,  74,  67,  52,  37,
              22,  16,  20,  25,  29,  43,  66,  89,  112, 136, 160, 185, 209,
              220, 218, 217, 215, 206, 189, 172, 155, 146, 146},
             {4,   4,   4,   4,   5,   5,   5,   5,   6,   6,   6,   6,   7,
              7,   7,   7,   8,   8,   8,   8,   9,   9,   9,   9,   10,  10,
              10,  10,  11,  11,  11,  11,  12,  12,  12,  12,  13,  13,  13,
              13,  14,  14,  36,  81,  126, 171, 174, 137, 100, 63,  52,  67,
              82,  97,  116, 137, 159, 180, 180, 159, 138, 117, 102, 94,  85,
              77,  82,  101, 120, 139, 131, 97,  64,  30,  32,  69,  106, 143,
              156, 145, 134, 123, 106, 82,  57,  33,  19,  14,  10,  5,   7,
              15,  22,  30,  35,  36,  38,  39,  54,  81,  109, 136, 146, 139,
              131, 124, 114, 101, 88,  75,  69,  71,  72,  74,  67,  52,  37,
              22,  16,  20,  25,  29,  43,  66,  89,  112, 136, 160, 185, 209,
              220, 218, 217, 215, 206, 189, 172, 155, 146, 146}}}));

class ResizeLinearF32 : public testing::TestWithParam<ResizeTestParams<float>> {
};

TEST_P(ResizeLinearF32, ResizeNoPadding) {
  do_linear_resize_test(GetParam(), 0, 0);
}

TEST_P(ResizeLinearF32, ResizeWithPadding) {
  do_linear_resize_test(GetParam(), 1, 2);
}

TEST_P(ResizeLinearF32, ResizePadDst) {
  do_linear_resize_test(GetParam(), 0, 3);
}

TEST_P(ResizeLinearF32, ResizePadSrc) {
  do_linear_resize_test(GetParam(), 4, 0);
}

using Pf32 = ResizeTestParams<float>;

INSTANTIATE_TEST_SUITE_P(
    ResizeLinear, ResizeLinearF32,
    testing::Values(
        // 1*1 -> 2*2
        Pf32{{{123}}, {{123, 123}, {123, 123}}},
        // 2*1 -> 4*2
        Pf32{{{0, 255}},
             {{0, 63.75F, 191.25F, 255}, {0, 63.75F, 191.25F, 255}}},
        // 2*2 -> 4*4
        Pf32{{{FLT_MAX, 1e38F}, {0, FLT_TRUE_MIN}},
             {{FLT_MAX, 2.8021173e38F, 1.6007058e38F, 1e38F},
              {2.5521173e38F, 2.101588e38F, 1.2005294e38F, 7.5e37F},
              {8.5070577e37F, 7.0052933e37F, 4.0017645e37F, 2.5e37F},
              {0, 0, FLT_TRUE_MIN, FLT_TRUE_MIN}}},
        // 3*3 -> 6*6
        Pf32{{{1, 63, 164}, {28, 251, 35}, {218, 64, 99}},
             {{1, 16.5F, 47.5F, 88.25F, 138.75F, 164},
              {7.75F, 33.3125F, 84.4375F, 115.4375F, 126.3125F, 131.75F},
              {21.25F, 66.9375F, 158.3125F, 169.8125F, 101.4375F, 67.25F},
              {75.5F, 107.6875F, 172.0625F, 165.9375F, 89.3125F, 51},
              {170.5F, 155.5625F, 125.6875F, 103.8125F, 89.9375F, 83},
              {218, 179.5F, 102.5F, 72.75F, 90.25F, 99}}},
        // 4*4 -> 8*8
        Pf32{
            {{10, 30, 5, 70},
             {255, 11, 11, 12},
             {127, 127, 128, 0},
             {200, 100, 150, 50}},
            {{10, 15, 25, 23.75F, 11.25F, 21.25F, 53.75F, 70},
             {71.25F, 59.75F, 36.75F, 20.5625F, 11.1875F, 18.75F, 43.25F,
              55.5F},
             {193.75F, 149.25F, 60.25F, 14.1875F, 11.0625F, 13.75F, 22.25F,
              26.5F},
             {223, 177.25F, 85.75F, 40.0625F, 40.1875F, 32.4375F, 16.8125F, 9},
             {159, 143.75F, 113.25F, 98.1875F, 98.5625F, 74.8125F, 26.9375F, 3},
             {145.25F, 139, 126.5F, 123.5625F, 130.1875F, 103.25F, 42.75F,
              12.5F},
             {181.75F, 163, 125.5F, 116.1875F, 135.0625F, 117.75F, 64.25F,
              37.5F},
             {200, 175, 125, 112.5F, 137.5F, 125, 75, 50}}},
        // 35*2 -> 70*4
        Pf32{{{0,   1,   2,       3,       4,   5,   6,   7,   8,
               9,   10,  FLT_MAX, FLT_MAX, 104, 108, 227, 46,  162,
               21,  220, 235,     183,     113, 225, 146, 196, 144,
               104, 148, 19,      126,     172, 9,   12,  61},
              {4,       5,   6,   7,   8,  9,   10, 11,  12,  13,  14, FLT_MAX,
               FLT_MAX, 105, 191, 106, 73, 148, 13, 161, 118, 21,  3,  34,
               40,      150, 120, 68,  75, 14,  31, 124, 221, 214, 146}},
             {{0.0F,         0.25F,         0.75F,         1.25F,
               1.75F,        2.25F,         2.75F,         3.25F,
               3.75F,        4.25F,         4.75F,         5.25F,
               5.75F,        6.25F,         6.75F,         7.25F,
               7.75F,        8.25F,         8.75F,         9.25F,
               9.75F,        8.50705e+37F,  2.552115e+38F, 3.40282e+38F,
               3.40282e+38F, 2.552115e+38F, 8.50705e+37F,  105.0F,
               107.0F,       137.75F,       197.25F,       181.75F,
               91.25F,       75.0F,         133.0F,        126.75F,
               56.25F,       70.75F,        170.25F,       223.75F,
               231.25F,      222.0F,        196.0F,        165.5F,
               130.5F,       141.0F,        197.0F,        205.25F,
               165.75F,      158.5F,        183.5F,        183.0F,
               157.0F,       134.0F,        114.0F,        115.0F,
               137.0F,       115.75F,       51.25F,        45.75F,
               99.25F,       137.5F,        160.5F,        131.25F,
               49.75F,       9.75F,         11.25F,        24.25F,
               48.75F,       61.0F},
              {1.0F,         1.25F,         1.75F,         2.25F,
               2.75F,        3.25F,         3.75F,         4.25F,
               4.75F,        5.25F,         5.75F,         6.25F,
               6.75F,        7.25F,         7.75F,         8.25F,
               8.75F,        9.25F,         9.75F,         10.25F,
               10.75F,       8.50705e+37F,  2.552115e+38F, 3.40282e+38F,
               3.40282e+38F, 2.552115e+38F, 8.50705e+37F,  110.375F,
               122.625F,     145.75F,       179.75F,       160.75F,
               88.75F,       79.1875F,      132.0625F,     123.625F,
               53.875F,      65.5625F,      158.6875F,     205.375F,
               205.625F,     189.9375F,     158.3125F,     128.25F,
               99.75F,       108.4375F,     154.3125F,     162.8125F,
               133.9375F,    135.75F,       168.25F,       172.875F,
               149.625F,     127.25F,       105.75F,       103.6875F,
               121.0625F,    101.75F,       45.75F,        38.875F,
               81.125F,      116.6875F,     145.5625F,     135.5F,
               86.5F,        62.125F,       62.375F,       67.4375F,
               77.3125F,     82.25F},
              {3.0F,         3.25F,         3.75F,         4.25F,
               4.75F,        5.25F,         5.75F,         6.25F,
               6.75F,        7.25F,         7.75F,         8.25F,
               8.75F,        9.25F,         9.75F,         10.25F,
               10.75F,       11.25F,        11.75F,        12.25F,
               12.75F,       8.50705e+37F,  2.552115e+38F, 3.40282e+38F,
               3.40282e+38F, 2.552115e+38F, 8.50705e+37F,  121.125F,
               153.875F,     161.75F,       144.75F,       118.75F,
               83.75F,       87.5625F,      130.1875F,     117.375F,
               49.125F,      55.1875F,      135.5625F,     168.625F,
               154.375F,     125.8125F,     82.9375F,      53.75F,
               38.25F,       43.3125F,      68.9375F,      77.9375F,
               70.3125F,     90.25F,        137.75F,       152.625F,
               134.875F,     113.75F,       89.25F,        81.0625F,
               89.1875F,     73.75F,        34.75F,        25.125F,
               44.875F,      75.0625F,      115.6875F,     144.0F,
               160.0F,       166.875F,      164.625F,      153.8125F,
               134.4375F,    124.75F},
              {4.0F,         4.25F,         4.75F,         5.25F,
               5.75F,        6.25F,         6.75F,         7.25F,
               7.75F,        8.25F,         8.75F,         9.25F,
               9.75F,        10.25F,        10.75F,        11.25F,
               11.75F,       12.25F,        12.75F,        13.25F,
               13.75F,       8.50705e+37F,  2.552115e+38F, 3.40282e+38F,
               3.40282e+38F, 2.552115e+38F, 8.50705e+37F,  126.5F,
               169.5F,       169.75F,       127.25F,       97.75F,
               81.25F,       91.75F,        129.25F,       114.25F,
               46.75F,       50.0F,         124.0F,        150.25F,
               128.75F,      93.75F,        45.25F,        16.5F,
               7.5F,         10.75F,        26.25F,        35.5F,
               38.5F,        67.5F,         122.5F,        142.5F,
               127.5F,       107.0F,        81.0F,         69.75F,
               73.25F,       59.75F,        29.25F,        18.25F,
               26.75F,       54.25F,        100.75F,       148.25F,
               196.75F,      219.25F,       215.75F,       197.0F,
               163.0F,       146.0F}}},
        // 2*2 -> 8*8
        Pf32{{{FLT_MAX, 1e38F}, {0, FLT_TRUE_MIN}},
             {{FLT_MAX, FLT_MAX, 3.10247e38F, 2.5017644e38F, 1.9010587e38F,
               1.3003528e38F, 1e38F, 1e38F},
              {FLT_MAX, FLT_MAX, 3.1024702e38F, 2.5017644e38F, 1.9010587e38F,
               1.3003528e38F, 1e38F, 1e38F},
              {2.9774701e38F, 2.9774701e38F, 2.7146614e38F, 2.189044e38F,
               1.6634263e38F, 1.1378087e38F, 8.75e37F, 8.75e37F},
              {2.1267644e38F, 2.1267644e38F, 1.9390438e38F, 1.5636028e38F,
               1.1881617e38F, 8.1272051e37F, 6.25e37F, 6.25e37F},
              {1.2760587e38F, 1.2760587e38F, 1.1634263e38F, 9.3816164e37F,
               7.1289698e37F, 4.8763233e37F, 3.75e37F, 3.75e37F},
              {4.2535288e37F, 4.2535288e37F, 3.8780877e37F, 3.1272055e37F,
               2.3763234e37F, 1.6254411e37F, 1.25e37F, 1.25e37F},
              {0, 0, 0, 0, FLT_TRUE_MIN, FLT_TRUE_MIN, FLT_TRUE_MIN,
               FLT_TRUE_MIN},
              {0, 0, 0, 0, FLT_TRUE_MIN, FLT_TRUE_MIN, FLT_TRUE_MIN,
               FLT_TRUE_MIN}}},
        // 35*2 -> 140*8
        Pf32{{{0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  82,
               155, 104, 108, 227, 46,  162, 21,  220, 235, 183, 113, 225,
               146, 196, 144, 104, 148, 19,  126, 172, 9,   12,  61},
              {4,  5,   6,   7,   8,  9,   10, 11,  12,  13,  14, 193,
               44, 105, 191, 106, 73, 148, 13, 161, 118, 21,  3,  34,
               40, 150, 120, 68,  75, 14,  31, 124, 221, 214, 146}},
             {{0,        0,        0.125F,   0.375F,   0.625F,   0.875F,
               1.125F,   1.375F,   1.625F,   1.875F,   2.125F,   2.375F,
               2.625F,   2.875F,   3.125F,   3.375F,   3.625F,   3.875F,
               4.125F,   4.375F,   4.625F,   4.875F,   5.125F,   5.375F,
               5.625F,   5.875F,   6.125F,   6.375F,   6.625F,   6.875F,
               7.125F,   7.375F,   7.625F,   7.875F,   8.125F,   8.375F,
               8.625F,   8.875F,   9.125F,   9.375F,   9.625F,   9.875F,
               19,       37,       55,       73,       91.125F,  109.375F,
               127.625F, 145.875F, 148.625F, 135.875F, 123.125F, 110.375F,
               104.5F,   105.5F,   106.5F,   107.5F,   122.875F, 152.625F,
               182.375F, 212.125F, 204.375F, 159.125F, 113.875F, 68.625F,
               60.5F,    89.5F,    118.5F,   147.5F,   144.375F, 109.125F,
               73.875F,  38.625F,  45.875F,  95.625F,  145.375F, 195.125F,
               221.875F, 225.625F, 229.375F, 233.125F, 228.5F,   215.5F,
               202.5F,   189.5F,   174.25F,  156.75F,  139.25F,  121.75F,
               127,      155,      183,      211,      215.125F, 195.375F,
               175.625F, 155.875F, 152.25F,  164.75F,  177.25F,  189.75F,
               189.5F,   176.5F,   163.5F,   150.5F,   139,      129,
               119,      109,      109.5F,   120.5F,   131.5F,   142.5F,
               131.875F, 99.625F,  67.375F,  35.125F,  32.375F,  59.125F,
               85.875F,  112.625F, 131.75F,  143.25F,  154.75F,  166.25F,
               151.625F, 110.875F, 70.125F,  29.375F,  9.375F,   10.125F,
               10.875F,  11.625F,  18.125F,  30.375F,  42.625F,  54.875F,
               61,       61},
              {0,        0,        0.125F,   0.375F,   0.625F,   0.875F,
               1.125F,   1.375F,   1.625F,   1.875F,   2.125F,   2.375F,
               2.625F,   2.875F,   3.125F,   3.375F,   3.625F,   3.875F,
               4.125F,   4.375F,   4.625F,   4.875F,   5.125F,   5.375F,
               5.625F,   5.875F,   6.125F,   6.375F,   6.625F,   6.875F,
               7.125F,   7.375F,   7.625F,   7.875F,   8.125F,   8.375F,
               8.625F,   8.875F,   9.125F,   9.375F,   9.625F,   9.875F,
               19,       37,       55,       73,       91.125F,  109.375F,
               127.625F, 145.875F, 148.625F, 135.875F, 123.125F, 110.375F,
               104.5F,   105.5F,   106.5F,   107.5F,   122.875F, 152.625F,
               182.375F, 212.125F, 204.375F, 159.125F, 113.875F, 68.625F,
               60.5F,    89.5F,    118.5F,   147.5F,   144.375F, 109.125F,
               73.875F,  38.625F,  45.875F,  95.625F,  145.375F, 195.125F,
               221.875F, 225.625F, 229.375F, 233.125F, 228.5F,   215.5F,
               202.5F,   189.5F,   174.25F,  156.75F,  139.25F,  121.75F,
               127,      155,      183,      211,      215.125F, 195.375F,
               175.625F, 155.875F, 152.25F,  164.75F,  177.25F,  189.75F,
               189.5F,   176.5F,   163.5F,   150.5F,   139,      129,
               119,      109,      109.5F,   120.5F,   131.5F,   142.5F,
               131.875F, 99.625F,  67.375F,  35.125F,  32.375F,  59.125F,
               85.875F,  112.625F, 131.75F,  143.25F,  154.75F,  166.25F,
               151.625F, 110.875F, 70.125F,  29.375F,  9.375F,   10.125F,
               10.875F,  11.625F,  18.125F,  30.375F,  42.625F,  54.875F,
               61,       61},
              {0.5F,        0.5F,        0.625F,      0.875F,      1.125F,
               1.375F,      1.625F,      1.875F,      2.125F,      2.375F,
               2.625F,      2.875F,      3.125F,      3.375F,      3.625F,
               3.875F,      4.125F,      4.375F,      4.625F,      4.875F,
               5.125F,      5.375F,      5.625F,      5.875F,      6.125F,
               6.375F,      6.625F,      6.875F,      7.125F,      7.375F,
               7.625F,      7.875F,      8.125F,      8.375F,      8.625F,
               8.875F,      9.125F,      9.375F,      9.625F,      9.875F,
               10.125F,     10.375F,     21.171875F,  42.515625F,  63.859375F,
               85.203125F,  101.53125F,  112.84375F,  124.15625F,  135.46875F,
               136.5F,      127.25F,     118,         108.75F,     105.90625F,
               109.46875F,  113.03125F,  116.59375F,  130.0625F,   153.4375F,
               176.8125F,   200.1875F,   191.5625F,   150.9375F,   110.3125F,
               69.6875F,    63.234375F,  90.953125F,  118.671875F, 146.39062F,
               142.71875F,  107.65625F,  72.59375F,   37.53125F,   44.078125F,
               92.234375F,  140.39062F,  188.54688F,  213.59375F,  215.53125F,
               217.46875F,  219.40625F,  213.17188F,  198.76562F,  184.35938F,
               169.95312F,  154.8125F,   138.9375F,   123.0625F,   107.1875F,
               111.984375F, 137.45312F,  162.92188F,  188.39062F,  192.57812F,
               175.48438F,  158.39062F,  141.29688F,  139.9375F,   154.3125F,
               168.6875F,   183.0625F,   184.09375F,  171.78125F,  159.46875F,
               147.15625F,  135.8125F,   125.4375F,   115.0625F,   104.6875F,
               104.421875F, 114.265625F, 124.109375F, 133.95312F,  123.8125F,
               93.6875F,    63.5625F,    33.4375F,    30.34375F,   54.28125F,
               78.21875F,   102.15625F,  120.609375F, 133.57812F,  146.54688F,
               159.51562F,  149.6875F,   117.0625F,   84.4375F,    51.8125F,
               35.71875F,   36.15625F,   36.59375F,   37.03125F,   41.546875F,
               50.140625F,  58.734375F,  67.328125F,  71.625F,     71.625F},
              {1.5F,        1.5F,        1.625F,      1.875F,      2.125F,
               2.375F,      2.625F,      2.875F,      3.125F,      3.375F,
               3.625F,      3.875F,      4.125F,      4.375F,      4.625F,
               4.875F,      5.125F,      5.375F,      5.625F,      5.875F,
               6.125F,      6.375F,      6.625F,      6.875F,      7.125F,
               7.375F,      7.625F,      7.875F,      8.125F,      8.375F,
               8.625F,      8.875F,      9.125F,      9.375F,      9.625F,
               9.875F,      10.125F,     10.375F,     10.625F,     10.875F,
               11.125F,     11.375F,     25.515625F,  53.546875F,  81.578125F,
               109.609375F, 122.34375F,  119.78125F,  117.21875F,  114.65625F,
               112.25F,     110,         107.75F,     105.5F,      108.71875F,
               117.40625F,  126.09375F,  134.78125F,  144.4375F,   155.0625F,
               165.6875F,   176.3125F,   165.9375F,   134.5625F,   103.1875F,
               71.8125F,    68.703125F,  93.859375F,  119.015625F, 144.17188F,
               139.40625F,  104.71875F,  70.03125F,   35.34375F,   40.484375F,
               85.453125F,  130.42188F,  175.39062F,  197.03125F,  195.34375F,
               193.65625F,  191.96875F,  182.51562F,  165.29688F,  148.07812F,
               130.85938F,  115.9375F,   103.3125F,   90.6875F,    78.0625F,
               81.953125F,  102.359375F, 122.765625F, 143.17188F,  147.48438F,
               135.70312F,  123.921875F, 112.140625F, 115.3125F,   133.4375F,
               151.5625F,   169.6875F,   173.28125F,  162.34375F,  151.40625F,
               140.46875F,  129.4375F,   118.3125F,   107.1875F,   96.0625F,
               94.265625F,  101.796875F, 109.328125F, 116.859375F, 107.6875F,
               81.8125F,    55.9375F,    30.0625F,    26.28125F,   44.59375F,
               62.90625F,   81.21875F,   98.328125F,  114.234375F, 130.14062F,
               146.04688F,  145.8125F,   129.4375F,   113.0625F,   96.6875F,
               88.40625F,   88.21875F,   88.03125F,   87.84375F,   88.390625F,
               89.671875F,  90.953125F,  92.234375F,  92.875F,     92.875F},
              {2.5F,       2.5F,        2.625F,      2.875F,      3.125F,
               3.375F,     3.625F,      3.875F,      4.125F,      4.375F,
               4.625F,     4.875F,      5.125F,      5.375F,      5.625F,
               5.875F,     6.125F,      6.375F,      6.625F,      6.875F,
               7.125F,     7.375F,      7.625F,      7.875F,      8.125F,
               8.375F,     8.625F,      8.875F,      9.125F,      9.375F,
               9.625F,     9.875F,      10.125F,     10.375F,     10.625F,
               10.875F,    11.125F,     11.375F,     11.625F,     11.875F,
               12.125F,    12.375F,     29.859375F,  64.578125F,  99.296875F,
               134.01562F, 143.15625F,  126.71875F,  110.28125F,  93.84375F,
               88,         92.75F,      97.5F,       102.25F,     111.53125F,
               125.34375F, 139.15625F,  152.96875F,  158.8125F,   156.6875F,
               154.5625F,  152.4375F,   140.3125F,   118.1875F,   96.0625F,
               73.9375F,   74.171875F,  96.765625F,  119.359375F, 141.95312F,
               136.09375F, 101.78125F,  67.46875F,   33.15625F,   36.890625F,
               78.671875F, 120.453125F, 162.23438F,  180.46875F,  175.15625F,
               169.84375F, 164.53125F,  151.85938F,  131.82812F,  111.796875F,
               91.765625F, 77.0625F,    67.6875F,    58.3125F,    48.9375F,
               51.921875F, 67.265625F,  82.609375F,  97.953125F,  102.390625F,
               95.921875F, 89.453125F,  82.984375F,  90.6875F,    112.5625F,
               134.4375F,  156.3125F,   162.46875F,  152.90625F,  143.34375F,
               133.78125F, 123.0625F,   111.1875F,   99.3125F,    87.4375F,
               84.109375F, 89.328125F,  94.546875F,  99.765625F,  91.5625F,
               69.9375F,   48.3125F,    26.6875F,    22.21875F,   34.90625F,
               47.59375F,  60.28125F,   76.046875F,  94.890625F,  113.734375F,
               132.57812F, 141.9375F,   141.8125F,   141.6875F,   141.5625F,
               141.09375F, 140.28125F,  139.46875F,  138.65625F,  135.23438F,
               129.20312F, 123.171875F, 117.140625F, 114.125F,    114.125F},
              {3.5F,        3.5F,        3.625F,      3.875F,      4.125F,
               4.375F,      4.625F,      4.875F,      5.125F,      5.375F,
               5.625F,      5.875F,      6.125F,      6.375F,      6.625F,
               6.875F,      7.125F,      7.375F,      7.625F,      7.875F,
               8.125F,      8.375F,      8.625F,      8.875F,      9.125F,
               9.375F,      9.625F,      9.875F,      10.125F,     10.375F,
               10.625F,     10.875F,     11.125F,     11.375F,     11.625F,
               11.875F,     12.125F,     12.375F,     12.625F,     12.875F,
               13.125F,     13.375F,     34.203125F,  75.609375F,  117.015625F,
               158.42188F,  163.96875F,  133.65625F,  103.34375F,  73.03125F,
               63.75F,      75.5F,       87.25F,      99,          114.34375F,
               133.28125F,  152.21875F,  171.15625F,  173.1875F,   158.3125F,
               143.4375F,   128.5625F,   114.6875F,   101.8125F,   88.9375F,
               76.0625F,    79.640625F,  99.671875F,  119.703125F, 139.73438F,
               132.78125F,  98.84375F,   64.90625F,   30.96875F,   33.296875F,
               71.890625F,  110.484375F, 149.07812F,  163.90625F,  154.96875F,
               146.03125F,  137.09375F,  121.203125F, 98.359375F,  75.515625F,
               52.671875F,  38.1875F,    32.0625F,    25.9375F,    19.8125F,
               21.890625F,  32.171875F,  42.453125F,  52.734375F,  57.296875F,
               56.140625F,  54.984375F,  53.828125F,  66.0625F,    91.6875F,
               117.3125F,   142.9375F,   151.65625F,  143.46875F,  135.28125F,
               127.09375F,  116.6875F,   104.0625F,   91.4375F,    78.8125F,
               73.953125F,  76.859375F,  79.765625F,  82.671875F,  75.4375F,
               58.0625F,    40.6875F,    23.3125F,    18.15625F,   25.21875F,
               32.28125F,   39.34375F,   53.765625F,  75.546875F,  97.328125F,
               119.109375F, 138.0625F,   154.1875F,   170.3125F,   186.4375F,
               193.78125F,  192.34375F,  190.90625F,  189.46875F,  182.07812F,
               168.73438F,  155.39062F,  142.04688F,  135.375F,    135.375F},
              {4,        4,        4.125F,   4.375F,   4.625F,   4.875F,
               5.125F,   5.375F,   5.625F,   5.875F,   6.125F,   6.375F,
               6.625F,   6.875F,   7.125F,   7.375F,   7.625F,   7.875F,
               8.125F,   8.375F,   8.625F,   8.875F,   9.125F,   9.375F,
               9.625F,   9.875F,   10.125F,  10.375F,  10.625F,  10.875F,
               11.125F,  11.375F,  11.625F,  11.875F,  12.125F,  12.375F,
               12.625F,  12.875F,  13.125F,  13.375F,  13.625F,  13.875F,
               36.375F,  81.125F,  125.875F, 170.625F, 174.375F, 137.125F,
               99.875F,  62.625F,  51.625F,  66.875F,  82.125F,  97.375F,
               115.75F,  137.25F,  158.75F,  180.25F,  180.375F, 159.125F,
               137.875F, 116.625F, 101.875F, 93.625F,  85.375F,  77.125F,
               82.375F,  101.125F, 119.875F, 138.625F, 131.125F, 97.375F,
               63.625F,  29.875F,  31.5F,    68.5F,    105.5F,   142.5F,
               155.625F, 144.875F, 134.125F, 123.375F, 105.875F, 81.625F,
               57.375F,  33.125F,  18.75F,   14.25F,   9.75F,    5.25F,
               6.875F,   14.625F,  22.375F,  30.125F,  34.75F,   36.25F,
               37.75F,   39.25F,   53.75F,   81.25F,   108.75F,  136.25F,
               146.25F,  138.75F,  131.25F,  123.75F,  113.5F,   100.5F,
               87.5F,    74.5F,    68.875F,  70.625F,  72.375F,  74.125F,
               67.375F,  52.125F,  36.875F,  21.625F,  16.125F,  20.375F,
               24.625F,  28.875F,  42.625F,  65.875F,  89.125F,  112.375F,
               136.125F, 160.375F, 184.625F, 208.875F, 220.125F, 218.375F,
               216.625F, 214.875F, 205.5F,   188.5F,   171.5F,   154.5F,
               146,      146},
              {4,        4,        4.125F,   4.375F,   4.625F,   4.875F,
               5.125F,   5.375F,   5.625F,   5.875F,   6.125F,   6.375F,
               6.625F,   6.875F,   7.125F,   7.375F,   7.625F,   7.875F,
               8.125F,   8.375F,   8.625F,   8.875F,   9.125F,   9.375F,
               9.625F,   9.875F,   10.125F,  10.375F,  10.625F,  10.875F,
               11.125F,  11.375F,  11.625F,  11.875F,  12.125F,  12.375F,
               12.625F,  12.875F,  13.125F,  13.375F,  13.625F,  13.875F,
               36.375F,  81.125F,  125.875F, 170.625F, 174.375F, 137.125F,
               99.875F,  62.625F,  51.625F,  66.875F,  82.125F,  97.375F,
               115.75F,  137.25F,  158.75F,  180.25F,  180.375F, 159.125F,
               137.875F, 116.625F, 101.875F, 93.625F,  85.375F,  77.125F,
               82.375F,  101.125F, 119.875F, 138.625F, 131.125F, 97.375F,
               63.625F,  29.875F,  31.5F,    68.5F,    105.5F,   142.5F,
               155.625F, 144.875F, 134.125F, 123.375F, 105.875F, 81.625F,
               57.375F,  33.125F,  18.75F,   14.25F,   9.75F,    5.25F,
               6.875F,   14.625F,  22.375F,  30.125F,  34.75F,   36.25F,
               37.75F,   39.25F,   53.75F,   81.25F,   108.75F,  136.25F,
               146.25F,  138.75F,  131.25F,  123.75F,  113.5F,   100.5F,
               87.5F,    74.5F,    68.875F,  70.625F,  72.375F,  74.125F,
               67.375F,  52.125F,  36.875F,  21.625F,  16.125F,  20.375F,
               24.625F,  28.875F,  42.625F,  65.875F,  89.125F,  112.375F,
               136.125F, 160.375F, 184.625F, 208.875F, 220.125F, 218.375F,
               216.625F, 214.875F, 205.5F,   188.5F,   171.5F,   154.5F,
               146,      146}}},
        // 2*2 -> 16*16
        Pf32{{{FLT_MAX, 1e38F}, {0, FLT_TRUE_MIN}},
             {
                 {3.402823466e+38F, 3.402823466e+38F, 3.402823466e+38F,
                  3.402823466e+38F, 3.252647029e+38F, 2.952294156e+38F,
                  2.651941079e+38F, 2.351588205e+38F, 2.051235331e+38F,
                  1.750882254e+38F, 1.45052938e+38F, 1.150176405e+38F,
                  9.99999968e+37F, 9.99999968e+37F, 9.99999968e+37F,
                  9.99999968e+37F},
                 {3.402823466e+38F, 3.402823466e+38F, 3.402823466e+38F,
                  3.402823466e+38F, 3.252647029e+38F, 2.952294156e+38F,
                  2.651941079e+38F, 2.351588205e+38F, 2.051235331e+38F,
                  1.750882254e+38F, 1.45052938e+38F, 1.150176405e+38F,
                  9.99999968e+37F, 9.99999968e+37F, 9.99999968e+37F,
                  9.99999968e+37F},
                 {3.402823466e+38F, 3.402823466e+38F, 3.402823466e+38F,
                  3.402823466e+38F, 3.252647029e+38F, 2.952294156e+38F,
                  2.651941079e+38F, 2.351588205e+38F, 2.051235331e+38F,
                  1.750882254e+38F, 1.45052938e+38F, 1.150176405e+38F,
                  9.99999968e+37F, 9.99999968e+37F, 9.99999968e+37F,
                  9.99999968e+37F},
                 {3.402823466e+38F, 3.402823466e+38F, 3.402823466e+38F,
                  3.402823466e+38F, 3.252647029e+38F, 2.952294156e+38F,
                  2.651941079e+38F, 2.351588205e+38F, 2.051235331e+38F,
                  1.750882254e+38F, 1.45052938e+38F, 1.150176405e+38F,
                  9.99999968e+37F, 9.99999968e+37F, 9.99999968e+37F,
                  9.99999968e+37F},
                 {3.190146987e+38F, 3.190146987e+38F, 3.190146987e+38F,
                  3.190146987e+38F, 3.049356641e+38F, 2.767775745e+38F,
                  2.48619485e+38F, 2.204613955e+38F, 1.923033059e+38F,
                  1.641452164e+38F, 1.359871269e+38F, 1.078290373e+38F,
                  9.374999257e+37F, 9.374999257e+37F, 9.374999257e+37F,
                  9.374999257e+37F},
                 {2.764794028e+38F, 2.764794028e+38F, 2.764794028e+38F,
                  2.764794028e+38F, 2.642775661e+38F, 2.398738925e+38F,
                  2.15470219e+38F, 1.910665454e+38F, 1.666628618e+38F,
                  1.422591882e+38F, 1.178555147e+38F, 9.3451831e+37F,
                  8.12499993e+37F, 8.12499993e+37F, 8.12499993e+37F,
                  8.12499993e+37F},
                 {2.33944107e+38F, 2.33944107e+38F, 2.33944107e+38F,
                  2.33944107e+38F, 2.236194883e+38F, 2.029702105e+38F,
                  1.82320953e+38F, 1.616716853e+38F, 1.410224277e+38F,
                  1.2037316e+38F, 9.972389236e+37F, 7.907462974e+37F,
                  6.87499959e+37F, 6.87499959e+37F, 6.87499959e+37F,
                  6.87499959e+37F},
                 {1.914088111e+38F, 1.914088111e+38F, 1.914088111e+38F,
                  1.914088111e+38F, 1.829613903e+38F, 1.660665386e+38F,
                  1.491716869e+38F, 1.322768353e+38F, 1.153819836e+38F,
                  9.848713187e+37F, 8.159228018e+37F, 6.469742341e+37F,
                  5.624999757e+37F, 5.624999757e+37F, 5.624999757e+37F,
                  5.624999757e+37F},
                 {1.488735254e+38F, 1.488735254e+38F, 1.488735254e+38F,
                  1.488735254e+38F, 1.423033025e+38F, 1.291628668e+38F,
                  1.160224209e+38F, 1.028819852e+38F, 8.974153939e+37F,
                  7.660109862e+37F, 6.346066292e+37F, 5.032021708e+37F,
                  4.374999924e+37F, 4.374999924e+37F, 4.374999924e+37F,
                  4.374999924e+37F},
                 {1.063382295e+38F, 1.063382295e+38F, 1.063382295e+38F,
                  1.063382295e+38F, 1.016452146e+38F, 9.225918475e+37F,
                  8.287315998e+37F, 7.348713013e+37F, 6.410110029e+37F,
                  5.471507044e+37F, 4.53290406e+37F, 3.594301329e+37F,
                  3.124999837e+37F, 3.124999837e+37F, 3.124999837e+37F,
                  3.124999837e+37F},
                 {6.380293873e+37F, 6.380293873e+37F, 6.380293873e+37F,
                  6.380293873e+37F, 6.09871318e+37F, 5.535551288e+37F,
                  4.972389396e+37F, 4.409228011e+37F, 3.846066119e+37F,
                  3.28290448e+37F, 2.719742588e+37F, 2.156580696e+37F,
                  1.875000003e+37F, 1.875000003e+37F, 1.875000003e+37F,
                  1.875000003e+37F},
                 {2.126764666e+37F, 2.126764666e+37F, 2.126764666e+37F,
                  2.126764666e+37F, 2.032904393e+37F, 1.845183847e+37F,
                  1.657463174e+37F, 1.469742628e+37F, 1.282022082e+37F,
                  1.094301409e+37F, 9.065808627e+36F, 7.188602531e+36F,
                  6.2499998e+36F, 6.2499998e+36F, 6.2499998e+36F,
                  6.2499998e+36F},
                 {0, 0, 0, 0, 0, 0, 0, 0, 1.401298464e-45F, 1.401298464e-45F,
                  1.401298464e-45F, 1.401298464e-45F, 1.401298464e-45F,
                  1.401298464e-45F, 1.401298464e-45F, 1.401298464e-45F},
                 {0, 0, 0, 0, 0, 0, 0, 0, 1.401298464e-45F, 1.401298464e-45F,
                  1.401298464e-45F, 1.401298464e-45F, 1.401298464e-45F,
                  1.401298464e-45F, 1.401298464e-45F, 1.401298464e-45F},
                 {0, 0, 0, 0, 0, 0, 0, 0, 1.401298464e-45F, 1.401298464e-45F,
                  1.401298464e-45F, 1.401298464e-45F, 1.401298464e-45F,
                  1.401298464e-45F, 1.401298464e-45F, 1.401298464e-45F},
                 {0, 0, 0, 0, 0, 0, 0, 0, 1.401298464e-45F, 1.401298464e-45F,
                  1.401298464e-45F, 1.401298464e-45F, 1.401298464e-45F,
                  1.401298464e-45F, 1.401298464e-45F, 1.401298464e-45F},
             }},
        // 35*2 -> 280*16
        Pf32{{{0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  82,
               155, 104, 108, 227, 46,  162, 21,  220, 235, 183, 113, 225,
               146, 196, 144, 104, 148, 19,  126, 172, 9,   12,  61},
              {4,  5,   6,   7,   8,  9,   10, 11,  12,  13,  14, 193,
               44, 105, 191, 106, 73, 148, 13, 161, 118, 21,  3,  34,
               40, 150, 120, 68,  75, 14,  31, 124, 221, 214, 146}},
             {{0,         0,         0,         0,         0.0625F,   0.1875F,
               0.3125F,   0.4375F,   0.5625F,   0.6875F,   0.8125F,   0.9375F,
               1.0625F,   1.1875F,   1.3125F,   1.4375F,   1.5625F,   1.6875F,
               1.8125F,   1.9375F,   2.0625F,   2.1875F,   2.3125F,   2.4375F,
               2.5625F,   2.6875F,   2.8125F,   2.9375F,   3.0625F,   3.1875F,
               3.3125F,   3.4375F,   3.5625F,   3.6875F,   3.8125F,   3.9375F,
               4.0625F,   4.1875F,   4.3125F,   4.4375F,   4.5625F,   4.6875F,
               4.8125F,   4.9375F,   5.0625F,   5.1875F,   5.3125F,   5.4375F,
               5.5625F,   5.6875F,   5.8125F,   5.9375F,   6.0625F,   6.1875F,
               6.3125F,   6.4375F,   6.5625F,   6.6875F,   6.8125F,   6.9375F,
               7.0625F,   7.1875F,   7.3125F,   7.4375F,   7.5625F,   7.6875F,
               7.8125F,   7.9375F,   8.0625F,   8.1875F,   8.3125F,   8.4375F,
               8.5625F,   8.6875F,   8.8125F,   8.9375F,   9.0625F,   9.1875F,
               9.3125F,   9.4375F,   9.5625F,   9.6875F,   9.8125F,   9.9375F,
               14.5F,     23.5F,     32.5F,     41.5F,     50.5F,     59.5F,
               68.5F,     77.5F,     86.5625F,  95.6875F,  104.8125F, 113.9375F,
               123.0625F, 132.1875F, 141.3125F, 150.4375F, 151.8125F, 145.4375F,
               139.0625F, 132.6875F, 126.3125F, 119.9375F, 113.5625F, 107.1875F,
               104.25F,   104.75F,   105.25F,   105.75F,   106.25F,   106.75F,
               107.25F,   107.75F,   115.4375F, 130.3125F, 145.1875F, 160.0625F,
               174.9375F, 189.8125F, 204.6875F, 219.5625F, 215.6875F, 193.0625F,
               170.4375F, 147.8125F, 125.1875F, 102.5625F, 79.9375F,  57.3125F,
               53.25F,    67.75F,    82.25F,    96.75F,    111.25F,   125.75F,
               140.25F,   154.75F,   153.1875F, 135.5625F, 117.9375F, 100.3125F,
               82.6875F,  65.0625F,  47.4375F,  29.8125F,  33.4375F,  58.3125F,
               83.1875F,  108.0625F, 132.9375F, 157.8125F, 182.6875F, 207.5625F,
               220.9375F, 222.8125F, 224.6875F, 226.5625F, 228.4375F, 230.3125F,
               232.1875F, 234.0625F, 231.75F,   225.25F,   218.75F,   212.25F,
               205.75F,   199.25F,   192.75F,   186.25F,   178.625F,  169.875F,
               161.125F,  152.375F,  143.625F,  134.875F,  126.125F,  117.375F,
               120.0F,    134.0F,    148.0F,    162.0F,    176.0F,    190.0F,
               204.0F,    218.0F,    220.0625F, 210.1875F, 200.3125F, 190.4375F,
               180.5625F, 170.6875F, 160.8125F, 150.9375F, 149.125F,  155.375F,
               161.625F,  167.875F,  174.125F,  180.375F,  186.625F,  192.875F,
               192.75F,   186.25F,   179.75F,   173.25F,   166.75F,   160.25F,
               153.75F,   147.25F,   141.5F,    136.5F,    131.5F,    126.5F,
               121.5F,    116.5F,    111.5F,    106.5F,    106.75F,   112.25F,
               117.75F,   123.25F,   128.75F,   134.25F,   139.75F,   145.25F,
               139.9375F, 123.8125F, 107.6875F, 91.5625F,  75.4375F,  59.3125F,
               43.1875F,  27.0625F,  25.6875F,  39.0625F,  52.4375F,  65.8125F,
               79.1875F,  92.5625F,  105.9375F, 119.3125F, 128.875F,  134.625F,
               140.375F,  146.125F,  151.875F,  157.625F,  163.375F,  169.125F,
               161.8125F, 141.4375F, 121.0625F, 100.6875F, 80.3125F,  59.9375F,
               39.5625F,  19.1875F,  9.1875F,   9.5625F,   9.9375F,   10.3125F,
               10.6875F,  11.0625F,  11.4375F,  11.8125F,  15.0625F,  21.1875F,
               27.3125F,  33.4375F,  39.5625F,  45.6875F,  51.8125F,  57.9375F,
               61.0F,     61.0F,     61.0F,     61.0F},
              {0,         0,         0,         0,         0.0625F,   0.1875F,
               0.3125F,   0.4375F,   0.5625F,   0.6875F,   0.8125F,   0.9375F,
               1.0625F,   1.1875F,   1.3125F,   1.4375F,   1.5625F,   1.6875F,
               1.8125F,   1.9375F,   2.0625F,   2.1875F,   2.3125F,   2.4375F,
               2.5625F,   2.6875F,   2.8125F,   2.9375F,   3.0625F,   3.1875F,
               3.3125F,   3.4375F,   3.5625F,   3.6875F,   3.8125F,   3.9375F,
               4.0625F,   4.1875F,   4.3125F,   4.4375F,   4.5625F,   4.6875F,
               4.8125F,   4.9375F,   5.0625F,   5.1875F,   5.3125F,   5.4375F,
               5.5625F,   5.6875F,   5.8125F,   5.9375F,   6.0625F,   6.1875F,
               6.3125F,   6.4375F,   6.5625F,   6.6875F,   6.8125F,   6.9375F,
               7.0625F,   7.1875F,   7.3125F,   7.4375F,   7.5625F,   7.6875F,
               7.8125F,   7.9375F,   8.0625F,   8.1875F,   8.3125F,   8.4375F,
               8.5625F,   8.6875F,   8.8125F,   8.9375F,   9.0625F,   9.1875F,
               9.3125F,   9.4375F,   9.5625F,   9.6875F,   9.8125F,   9.9375F,
               14.5F,     23.5F,     32.5F,     41.5F,     50.5F,     59.5F,
               68.5F,     77.5F,     86.5625F,  95.6875F,  104.8125F, 113.9375F,
               123.0625F, 132.1875F, 141.3125F, 150.4375F, 151.8125F, 145.4375F,
               139.0625F, 132.6875F, 126.3125F, 119.9375F, 113.5625F, 107.1875F,
               104.25F,   104.75F,   105.25F,   105.75F,   106.25F,   106.75F,
               107.25F,   107.75F,   115.4375F, 130.3125F, 145.1875F, 160.0625F,
               174.9375F, 189.8125F, 204.6875F, 219.5625F, 215.6875F, 193.0625F,
               170.4375F, 147.8125F, 125.1875F, 102.5625F, 79.9375F,  57.3125F,
               53.25F,    67.75F,    82.25F,    96.75F,    111.25F,   125.75F,
               140.25F,   154.75F,   153.1875F, 135.5625F, 117.9375F, 100.3125F,
               82.6875F,  65.0625F,  47.4375F,  29.8125F,  33.4375F,  58.3125F,
               83.1875F,  108.0625F, 132.9375F, 157.8125F, 182.6875F, 207.5625F,
               220.9375F, 222.8125F, 224.6875F, 226.5625F, 228.4375F, 230.3125F,
               232.1875F, 234.0625F, 231.75F,   225.25F,   218.75F,   212.25F,
               205.75F,   199.25F,   192.75F,   186.25F,   178.625F,  169.875F,
               161.125F,  152.375F,  143.625F,  134.875F,  126.125F,  117.375F,
               120.0F,    134.0F,    148.0F,    162.0F,    176.0F,    190.0F,
               204.0F,    218.0F,    220.0625F, 210.1875F, 200.3125F, 190.4375F,
               180.5625F, 170.6875F, 160.8125F, 150.9375F, 149.125F,  155.375F,
               161.625F,  167.875F,  174.125F,  180.375F,  186.625F,  192.875F,
               192.75F,   186.25F,   179.75F,   173.25F,   166.75F,   160.25F,
               153.75F,   147.25F,   141.5F,    136.5F,    131.5F,    126.5F,
               121.5F,    116.5F,    111.5F,    106.5F,    106.75F,   112.25F,
               117.75F,   123.25F,   128.75F,   134.25F,   139.75F,   145.25F,
               139.9375F, 123.8125F, 107.6875F, 91.5625F,  75.4375F,  59.3125F,
               43.1875F,  27.0625F,  25.6875F,  39.0625F,  52.4375F,  65.8125F,
               79.1875F,  92.5625F,  105.9375F, 119.3125F, 128.875F,  134.625F,
               140.375F,  146.125F,  151.875F,  157.625F,  163.375F,  169.125F,
               161.8125F, 141.4375F, 121.0625F, 100.6875F, 80.3125F,  59.9375F,
               39.5625F,  19.1875F,  9.1875F,   9.5625F,   9.9375F,   10.3125F,
               10.6875F,  11.0625F,  11.4375F,  11.8125F,  15.0625F,  21.1875F,
               27.3125F,  33.4375F,  39.5625F,  45.6875F,  51.8125F,  57.9375F,
               61.0F,     61.0F,     61.0F,     61.0F},
              {0,         0,         0,         0,         0.0625F,   0.1875F,
               0.3125F,   0.4375F,   0.5625F,   0.6875F,   0.8125F,   0.9375F,
               1.0625F,   1.1875F,   1.3125F,   1.4375F,   1.5625F,   1.6875F,
               1.8125F,   1.9375F,   2.0625F,   2.1875F,   2.3125F,   2.4375F,
               2.5625F,   2.6875F,   2.8125F,   2.9375F,   3.0625F,   3.1875F,
               3.3125F,   3.4375F,   3.5625F,   3.6875F,   3.8125F,   3.9375F,
               4.0625F,   4.1875F,   4.3125F,   4.4375F,   4.5625F,   4.6875F,
               4.8125F,   4.9375F,   5.0625F,   5.1875F,   5.3125F,   5.4375F,
               5.5625F,   5.6875F,   5.8125F,   5.9375F,   6.0625F,   6.1875F,
               6.3125F,   6.4375F,   6.5625F,   6.6875F,   6.8125F,   6.9375F,
               7.0625F,   7.1875F,   7.3125F,   7.4375F,   7.5625F,   7.6875F,
               7.8125F,   7.9375F,   8.0625F,   8.1875F,   8.3125F,   8.4375F,
               8.5625F,   8.6875F,   8.8125F,   8.9375F,   9.0625F,   9.1875F,
               9.3125F,   9.4375F,   9.5625F,   9.6875F,   9.8125F,   9.9375F,
               14.5F,     23.5F,     32.5F,     41.5F,     50.5F,     59.5F,
               68.5F,     77.5F,     86.5625F,  95.6875F,  104.8125F, 113.9375F,
               123.0625F, 132.1875F, 141.3125F, 150.4375F, 151.8125F, 145.4375F,
               139.0625F, 132.6875F, 126.3125F, 119.9375F, 113.5625F, 107.1875F,
               104.25F,   104.75F,   105.25F,   105.75F,   106.25F,   106.75F,
               107.25F,   107.75F,   115.4375F, 130.3125F, 145.1875F, 160.0625F,
               174.9375F, 189.8125F, 204.6875F, 219.5625F, 215.6875F, 193.0625F,
               170.4375F, 147.8125F, 125.1875F, 102.5625F, 79.9375F,  57.3125F,
               53.25F,    67.75F,    82.25F,    96.75F,    111.25F,   125.75F,
               140.25F,   154.75F,   153.1875F, 135.5625F, 117.9375F, 100.3125F,
               82.6875F,  65.0625F,  47.4375F,  29.8125F,  33.4375F,  58.3125F,
               83.1875F,  108.0625F, 132.9375F, 157.8125F, 182.6875F, 207.5625F,
               220.9375F, 222.8125F, 224.6875F, 226.5625F, 228.4375F, 230.3125F,
               232.1875F, 234.0625F, 231.75F,   225.25F,   218.75F,   212.25F,
               205.75F,   199.25F,   192.75F,   186.25F,   178.625F,  169.875F,
               161.125F,  152.375F,  143.625F,  134.875F,  126.125F,  117.375F,
               120.0F,    134.0F,    148.0F,    162.0F,    176.0F,    190,
               204.0F,    218.0F,    220.0625F, 210.1875F, 200.3125F, 190.4375F,
               180.5625F, 170.6875F, 160.8125F, 150.9375F, 149.125F,  155.375F,
               161.625F,  167.875F,  174.125F,  180.375F,  186.625F,  192.875F,
               192.75F,   186.25F,   179.75F,   173.25F,   166.75F,   160.25F,
               153.75F,   147.25F,   141.5F,    136.5F,    131.5F,    126.5F,
               121.5F,    116.5F,    111.5F,    106.5F,    106.75F,   112.25F,
               117.75F,   123.25F,   128.75F,   134.25F,   139.75F,   145.25F,
               139.9375F, 123.8125F, 107.6875F, 91.5625F,  75.4375F,  59.3125F,
               43.1875F,  27.0625F,  25.6875F,  39.0625F,  52.4375F,  65.8125F,
               79.1875F,  92.5625F,  105.9375F, 119.3125F, 128.875F,  134.625F,
               140.375F,  146.125F,  151.875F,  157.625F,  163.375F,  169.125F,
               161.8125F, 141.4375F, 121.0625F, 100.6875F, 80.3125F,  59.9375F,
               39.5625F,  19.1875F,  9.1875F,   9.5625F,   9.9375F,   10.3125F,
               10.6875F,  11.0625F,  11.4375F,  11.8125F,  15.0625F,  21.1875F,
               27.3125F,  33.4375F,  39.5625F,  45.6875F,  51.8125F,  57.9375F,
               61.0F,     61.0F,     61.0F,     61.0F},
              {0,         0,         0,         0,         0.0625F,   0.1875F,
               0.3125F,   0.4375F,   0.5625F,   0.6875F,   0.8125F,   0.9375F,
               1.0625F,   1.1875F,   1.3125F,   1.4375F,   1.5625F,   1.6875F,
               1.8125F,   1.9375F,   2.0625F,   2.1875F,   2.3125F,   2.4375F,
               2.5625F,   2.6875F,   2.8125F,   2.9375F,   3.0625F,   3.1875F,
               3.3125F,   3.4375F,   3.5625F,   3.6875F,   3.8125F,   3.9375F,
               4.0625F,   4.1875F,   4.3125F,   4.4375F,   4.5625F,   4.6875F,
               4.8125F,   4.9375F,   5.0625F,   5.1875F,   5.3125F,   5.4375F,
               5.5625F,   5.6875F,   5.8125F,   5.9375F,   6.0625F,   6.1875F,
               6.3125F,   6.4375F,   6.5625F,   6.6875F,   6.8125F,   6.9375F,
               7.0625F,   7.1875F,   7.3125F,   7.4375F,   7.5625F,   7.6875F,
               7.8125F,   7.9375F,   8.0625F,   8.1875F,   8.3125F,   8.4375F,
               8.5625F,   8.6875F,   8.8125F,   8.9375F,   9.0625F,   9.1875F,
               9.3125F,   9.4375F,   9.5625F,   9.6875F,   9.8125F,   9.9375F,
               14.5F,     23.5F,     32.5F,     41.5F,     50.5F,     59.5F,
               68.5F,     77.5F,     86.5625F,  95.6875F,  104.8125F, 113.9375F,
               123.0625F, 132.1875F, 141.3125F, 150.4375F, 151.8125F, 145.4375F,
               139.0625F, 132.6875F, 126.3125F, 119.9375F, 113.5625F, 107.1875F,
               104.25F,   104.75F,   105.25F,   105.75F,   106.25F,   106.75F,
               107.25F,   107.75F,   115.4375F, 130.3125F, 145.1875F, 160.0625F,
               174.9375F, 189.8125F, 204.6875F, 219.5625F, 215.6875F, 193.0625F,
               170.4375F, 147.8125F, 125.1875F, 102.5625F, 79.9375F,  57.3125F,
               53.25F,    67.75F,    82.25F,    96.75F,    111.25F,   125.75F,
               140.25F,   154.75F,   153.1875F, 135.5625F, 117.9375F, 100.3125F,
               82.6875F,  65.0625F,  47.4375F,  29.8125F,  33.4375F,  58.3125F,
               83.1875F,  108.0625F, 132.9375F, 157.8125F, 182.6875F, 207.5625F,
               220.9375F, 222.8125F, 224.6875F, 226.5625F, 228.4375F, 230.3125F,
               232.1875F, 234.0625F, 231.75F,   225.25F,   218.75F,   212.25F,
               205.75F,   199.25F,   192.75F,   186.25F,   178.625F,  169.875F,
               161.125F,  152.375F,  143.625F,  134.875F,  126.125F,  117.375F,
               120,       134.0F,    148.0F,    162.0F,    176.0F,    190,
               204.0F,    218.0F,    220.0625F, 210.1875F, 200.3125F, 190.4375F,
               180.5625F, 170.6875F, 160.8125F, 150.9375F, 149.125F,  155.375F,
               161.625F,  167.875F,  174.125F,  180.375F,  186.625F,  192.875F,
               192.75F,   186.25F,   179.75F,   173.25F,   166.75F,   160.25F,
               153.75F,   147.25F,   141.5F,    136.5F,    131.5F,    126.5F,
               121.5F,    116.5F,    111.5F,    106.5F,    106.75F,   112.25F,
               117.75F,   123.25F,   128.75F,   134.25F,   139.75F,   145.25F,
               139.9375F, 123.8125F, 107.6875F, 91.5625F,  75.4375F,  59.3125F,
               43.1875F,  27.0625F,  25.6875F,  39.0625F,  52.4375F,  65.8125F,
               79.1875F,  92.5625F,  105.9375F, 119.3125F, 128.875F,  134.625F,
               140.375F,  146.125F,  151.875F,  157.625F,  163.375F,  169.125F,
               161.8125F, 141.4375F, 121.0625F, 100.6875F, 80.3125F,  59.9375F,
               39.5625F,  19.1875F,  9.1875F,   9.5625F,   9.9375F,   10.3125F,
               10.6875F,  11.0625F,  11.4375F,  11.8125F,  15.0625F,  21.1875F,
               27.3125F,  33.4375F,  39.5625F,  45.6875F,  51.8125F,  57.9375F,
               61.0F,     61.0F,     61.0F,     61.0F},
              {0.25F,        0.25F,        0.25F,        0.25F,
               0.3125F,      0.4375F,      0.5625F,      0.6875F,
               0.8125F,      0.9375F,      1.0625F,      1.1875F,
               1.3125F,      1.4375F,      1.5625F,      1.6875F,
               1.8125F,      1.9375F,      2.0625F,      2.1875F,
               2.3125F,      2.4375F,      2.5625F,      2.6875F,
               2.8125F,      2.9375F,      3.0625F,      3.1875F,
               3.3125F,      3.4375F,      3.5625F,      3.6875F,
               3.8125F,      3.9375F,      4.0625F,      4.1875F,
               4.3125F,      4.4375F,      4.5625F,      4.6875F,
               4.8125F,      4.9375F,      5.0625F,      5.1875F,
               5.3125F,      5.4375F,      5.5625F,      5.6875F,
               5.8125F,      5.9375F,      6.0625F,      6.1875F,
               6.3125F,      6.4375F,      6.5625F,      6.6875F,
               6.8125F,      6.9375F,      7.0625F,      7.1875F,
               7.3125F,      7.4375F,      7.5625F,      7.6875F,
               7.8125F,      7.9375F,      8.0625F,      8.1875F,
               8.3125F,      8.4375F,      8.5625F,      8.6875F,
               8.8125F,      8.9375F,      9.0625F,      9.1875F,
               9.3125F,      9.4375F,      9.5625F,      9.6875F,
               9.8125F,      9.9375F,      10.0625F,     10.1875F,
               15.16796875F, 25.00390625F, 34.83984375F, 44.67578125F,
               54.51171875F, 64.34765625F, 74.18359375F, 84.01953125F,
               92.6328125F,  100.0234375F, 107.4140625F, 114.8046875F,
               122.1953125F, 129.5859375F, 136.9765625F, 144.3671875F,
               145.3125F,    139.8125F,    134.3125F,    128.8125F,
               123.3125F,    117.8125F,    112.3125F,    106.8125F,
               104.6328125F, 105.7734375F, 106.9140625F, 108.0546875F,
               109.1953125F, 110.3359375F, 111.4765625F, 112.6171875F,
               119.828125F,  133.109375F,  146.390625F,  159.671875F,
               172.953125F,  186.234375F,  199.515625F,  212.796875F,
               208.703125F,  187.234375F,  165.765625F,  144.296875F,
               122.828125F,  101.359375F,  79.890625F,   58.421875F,
               54.77734375F, 68.95703125F, 83.13671875F, 97.31640625F,
               111.4960938F, 125.6757812F, 139.8554688F, 154.0351562F,
               152.3359375F, 134.7578125F, 117.1796875F, 99.6015625F,
               82.0234375F,  64.4453125F,  46.8671875F,  29.2890625F,
               32.73828125F, 57.21484375F, 81.69140625F, 106.1679688F,
               130.6445312F, 155.1210938F, 179.5976562F, 204.0742188F,
               217.0234375F, 218.4453125F, 219.8671875F, 221.2890625F,
               222.7109375F, 224.1328125F, 225.5546875F, 226.9765625F,
               224.2617188F, 217.4101562F, 210.5585938F, 203.7070312F,
               196.8554688F, 190.0039062F, 183.1523438F, 176.3007812F,
               168.703125F,  160.359375F,  152.015625F,  143.671875F,
               135.328125F,  126.984375F,  118.640625F,  110.296875F,
               112.8085938F, 126.1757812F, 139.5429688F, 152.9101562F,
               166.2773438F, 179.6445312F, 193.0117188F, 206.3789062F,
               208.4570312F, 199.2460938F, 190.0351562F, 180.8242188F,
               171.6132812F, 162.4023438F, 153.1914062F, 143.9804688F,
               142.734375F,  149.453125F,  156.171875F,  162.890625F,
               169.609375F,  176.328125F,  183.046875F,  189.765625F,
               189.9609375F, 183.6328125F, 177.3046875F, 170.9765625F,
               164.6484375F, 158.3203125F, 151.9921875F, 145.6640625F,
               139.953125F,  134.859375F,  129.765625F,  124.671875F,
               119.578125F,  114.484375F,  109.390625F,  104.296875F,
               104.3554688F, 109.5664062F, 114.7773438F, 119.9882812F,
               125.1992188F, 130.4101562F, 135.6210938F, 140.8320312F,
               135.640625F,  120.046875F,  104.453125F,  88.859375F,
               73.265625F,   57.671875F,   42.078125F,   26.484375F,
               25.0234375F,  37.6953125F,  50.3671875F,  63.0390625F,
               75.7109375F,  88.3828125F,  101.0546875F, 113.7265625F,
               123.1210938F, 129.2382812F, 135.3554688F, 141.4726562F,
               147.5898438F, 153.7070312F, 159.8242188F, 165.9414062F,
               159.828125F,  141.484375F,  123.140625F,  104.796875F,
               86.453125F,   68.109375F,   49.765625F,   31.421875F,
               22.3984375F,  22.6953125F,  22.9921875F,  23.2890625F,
               23.5859375F,  23.8828125F,  24.1796875F,  24.4765625F,
               27.23046875F, 32.44140625F, 37.65234375F, 42.86328125F,
               48.07421875F, 53.28515625F, 58.49609375F, 63.70703125F,
               66.3125F,     66.3125F,     66.3125F,     66.3125F},
              {0.75F,        0.75F,        0.75F,        0.75F,
               0.8125F,      0.9375F,      1.0625F,      1.1875F,
               1.3125F,      1.4375F,      1.5625F,      1.6875F,
               1.8125F,      1.9375F,      2.0625F,      2.1875F,
               2.3125F,      2.4375F,      2.5625F,      2.6875F,
               2.8125F,      2.9375F,      3.0625F,      3.1875F,
               3.3125F,      3.4375F,      3.5625F,      3.6875F,
               3.8125F,      3.9375F,      4.0625F,      4.1875F,
               4.3125F,      4.4375F,      4.5625F,      4.6875F,
               4.8125F,      4.9375F,      5.0625F,      5.1875F,
               5.3125F,      5.4375F,      5.5625F,      5.6875F,
               5.8125F,      5.9375F,      6.0625F,      6.1875F,
               6.3125F,      6.4375F,      6.5625F,      6.6875F,
               6.8125F,      6.9375F,      7.0625F,      7.1875F,
               7.3125F,      7.4375F,      7.5625F,      7.6875F,
               7.8125F,      7.9375F,      8.0625F,      8.1875F,
               8.3125F,      8.4375F,      8.5625F,      8.6875F,
               8.8125F,      8.9375F,      9.0625F,      9.1875F,
               9.3125F,      9.4375F,      9.5625F,      9.6875F,
               9.8125F,      9.9375F,      10.0625F,     10.1875F,
               10.3125F,     10.4375F,     10.5625F,     10.6875F,
               16.50390625F, 28.01171875F, 39.51953125F, 51.02734375F,
               62.53515625F, 74.04296875F, 85.55078125F, 97.05859375F,
               104.7734375F, 108.6953125F, 112.6171875F, 116.5390625F,
               120.4609375F, 124.3828125F, 128.3046875F, 132.2265625F,
               132.3125F,    128.5625F,    124.8125F,    121.0625F,
               117.3125F,    113.5625F,    109.8125F,    106.0625F,
               105.3984375F, 107.8203125F, 110.2421875F, 112.6640625F,
               115.0859375F, 117.5078125F, 119.9296875F, 122.3515625F,
               128.609375F,  138.703125F,  148.796875F,  158.890625F,
               168.984375F,  179.078125F,  189.171875F,  199.265625F,
               194.734375F,  175.578125F,  156.421875F,  137.265625F,
               118.109375F,  98.953125F,   79.796875F,   60.640625F,
               57.83203125F, 71.37109375F, 84.91015625F, 98.44921875F,
               111.9882812F, 125.5273438F, 139.0664062F, 152.6054688F,
               150.6328125F, 133.1484375F, 115.6640625F, 98.1796875F,
               80.6953125F,  63.2109375F,  45.7265625F,  28.2421875F,
               31.33984375F, 55.01953125F, 78.69921875F, 102.3789062F,
               126.0585938F, 149.7382812F, 173.4179688F, 197.0976562F,
               209.1953125F, 209.7109375F, 210.2265625F, 210.7421875F,
               211.2578125F, 211.7734375F, 212.2890625F, 212.8046875F,
               209.2851562F, 201.7304688F, 194.1757812F, 186.6210938F,
               179.0664062F, 171.5117188F, 163.9570312F, 156.4023438F,
               148.859375F,  141.328125F,  133.796875F,  126.265625F,
               118.734375F,  111.203125F,  103.671875F,  96.140625F,
               98.42578125F, 110.5273438F, 122.6289062F, 134.7304688F,
               146.8320312F, 158.9335938F, 171.0351562F, 183.1367188F,
               185.2460938F, 177.3632812F, 169.4804688F, 161.5976562F,
               153.7148438F, 145.8320312F, 137.9492188F, 130.0664062F,
               129.953125F,  137.609375F,  145.265625F,  152.921875F,
               160.578125F,  168.234375F,  175.890625F,  183.546875F,
               184.3828125F, 178.3984375F, 172.4140625F, 166.4296875F,
               160.4453125F, 154.4609375F, 148.4765625F, 142.4921875F,
               136.859375F,  131.578125F,  126.296875F,  121.015625F,
               115.734375F,  110.453125F,  105.171875F,  99.890625F,
               99.56640625F, 104.1992188F, 108.8320312F, 113.4648438F,
               118.0976562F, 122.7304688F, 127.3632812F, 131.9960938F,
               127.046875F,  112.515625F,  97.984375F,   83.453125F,
               68.921875F,   54.390625F,   39.859375F,   25.328125F,
               23.6953125F,  34.9609375F,  46.2265625F,  57.4921875F,
               68.7578125F,  80.0234375F,  91.2890625F,  102.5546875F,
               111.6132812F, 118.4648438F, 125.3164062F, 132.1679688F,
               139.0195312F, 145.8710938F, 152.7226562F, 159.5742188F,
               155.859375F,  141.578125F,  127.296875F,  113.015625F,
               98.734375F,   84.453125F,   70.171875F,   55.890625F,
               48.8203125F,  48.9609375F,  49.1015625F,  49.2421875F,
               49.3828125F,  49.5234375F,  49.6640625F,  49.8046875F,
               51.56640625F, 54.94921875F, 58.33203125F, 61.71484375F,
               65.09765625F, 68.48046875F, 71.86328125F, 75.24609375F,
               76.9375F,     76.9375F,     76.9375F,     76.9375F},
              {1.25F,        1.25F,        1.25F,        1.25F,
               1.3125F,      1.4375F,      1.5625F,      1.6875F,
               1.8125F,      1.9375F,      2.0625F,      2.1875F,
               2.3125F,      2.4375F,      2.5625F,      2.6875F,
               2.8125F,      2.9375F,      3.0625F,      3.1875F,
               3.3125F,      3.4375F,      3.5625F,      3.6875F,
               3.8125F,      3.9375F,      4.0625F,      4.1875F,
               4.3125F,      4.4375F,      4.5625F,      4.6875F,
               4.8125F,      4.9375F,      5.0625F,      5.1875F,
               5.3125F,      5.4375F,      5.5625F,      5.6875F,
               5.8125F,      5.9375F,      6.0625F,      6.1875F,
               6.3125F,      6.4375F,      6.5625F,      6.6875F,
               6.8125F,      6.9375F,      7.0625F,      7.1875F,
               7.3125F,      7.4375F,      7.5625F,      7.6875F,
               7.8125F,      7.9375F,      8.0625F,      8.1875F,
               8.3125F,      8.4375F,      8.5625F,      8.6875F,
               8.8125F,      8.9375F,      9.0625F,      9.1875F,
               9.3125F,      9.4375F,      9.5625F,      9.6875F,
               9.8125F,      9.9375F,      10.0625F,     10.1875F,
               10.3125F,     10.4375F,     10.5625F,     10.6875F,
               10.8125F,     10.9375F,     11.0625F,     11.1875F,
               17.83984375F, 31.01953125F, 44.19921875F, 57.37890625F,
               70.55859375F, 83.73828125F, 96.91796875F, 110.0976562F,
               116.9140625F, 117.3671875F, 117.8203125F, 118.2734375F,
               118.7265625F, 119.1796875F, 119.6328125F, 120.0859375F,
               119.3125F,    117.3125F,    115.3125F,    113.3125F,
               111.3125F,    109.3125F,    107.3125F,    105.3125F,
               106.1640625F, 109.8671875F, 113.5703125F, 117.2734375F,
               120.9765625F, 124.6796875F, 128.3828125F, 132.0859375F,
               137.390625F,  144.296875F,  151.203125F,  158.109375F,
               165.015625F,  171.921875F,  178.828125F,  185.734375F,
               180.765625F,  163.921875F,  147.078125F,  130.234375F,
               113.390625F,  96.546875F,   79.703125F,   62.859375F,
               60.88671875F, 73.78515625F, 86.68359375F, 99.58203125F,
               112.4804688F, 125.3789062F, 138.2773438F, 151.1757812F,
               148.9296875F, 131.5390625F, 114.1484375F, 96.7578125F,
               79.3671875F,  61.9765625F,  44.5859375F,  27.1953125F,
               29.94140625F, 52.82421875F, 75.70703125F, 98.58984375F,
               121.4726562F, 144.3554688F, 167.2382812F, 190.1210938F,
               201.3671875F, 200.9765625F, 200.5859375F, 200.1953125F,
               199.8046875F, 199.4140625F, 199.0234375F, 198.6328125F,
               194.3085938F, 186.0507812F, 177.7929688F, 169.5351562F,
               161.2773438F, 153.0195312F, 144.7617188F, 136.5039062F,
               129.015625F,  122.296875F,  115.578125F,  108.859375F,
               102.140625F,  95.421875F,   88.703125F,   81.984375F,
               84.04296875F, 94.87890625F, 105.7148438F, 116.5507812F,
               127.3867188F, 138.2226562F, 149.0585938F, 159.8945312F,
               162.0351562F, 155.4804688F, 148.9257812F, 142.3710938F,
               135.8164062F, 129.2617188F, 122.7070312F, 116.1523438F,
               117.171875F,  125.765625F,  134.359375F,  142.953125F,
               151.546875F,  160.140625F,  168.734375F,  177.328125F,
               178.8046875F, 173.1640625F, 167.5234375F, 161.8828125F,
               156.2421875F, 150.6015625F, 144.9609375F, 139.3203125F,
               133.765625F,  128.296875F,  122.828125F,  117.359375F,
               111.890625F,  106.421875F,  100.953125F,  95.484375F,
               94.77734375F, 98.83203125F, 102.8867188F, 106.9414062F,
               110.9960938F, 115.0507812F, 119.1054688F, 123.1601562F,
               118.453125F,  104.984375F,  91.515625F,   78.046875F,
               64.578125F,   51.109375F,   37.640625F,   24.171875F,
               22.3671875F,  32.2265625F,  42.0859375F,  51.9453125F,
               61.8046875F,  71.6640625F,  81.5234375F,  91.3828125F,
               100.1054688F, 107.6914062F, 115.2773438F, 122.8632812F,
               130.4492188F, 138.0351562F, 145.6210938F, 153.2070312F,
               151.890625F,  141.671875F,  131.453125F,  121.234375F,
               111.015625F,  100.796875F,  90.578125F,   80.359375F,
               75.2421875F,  75.2265625F,  75.2109375F,  75.1953125F,
               75.1796875F,  75.1640625F,  75.1484375F,  75.1328125F,
               75.90234375F, 77.45703125F, 79.01171875F, 80.56640625F,
               82.12109375F, 83.67578125F, 85.23046875F, 86.78515625F,
               87.5625F,     87.5625F,     87.5625F,     87.5625F},
              {1.75F,        1.75F,        1.75F,        1.75F,
               1.8125F,      1.9375F,      2.0625F,      2.1875F,
               2.3125F,      2.4375F,      2.5625F,      2.6875F,
               2.8125F,      2.9375F,      3.0625F,      3.1875F,
               3.3125F,      3.4375F,      3.5625F,      3.6875F,
               3.8125F,      3.9375F,      4.0625F,      4.1875F,
               4.3125F,      4.4375F,      4.5625F,      4.6875F,
               4.8125F,      4.9375F,      5.0625F,      5.1875F,
               5.3125F,      5.4375F,      5.5625F,      5.6875F,
               5.8125F,      5.9375F,      6.0625F,      6.1875F,
               6.3125F,      6.4375F,      6.5625F,      6.6875F,
               6.8125F,      6.9375F,      7.0625F,      7.1875F,
               7.3125F,      7.4375F,      7.5625F,      7.6875F,
               7.8125F,      7.9375F,      8.0625F,      8.1875F,
               8.3125F,      8.4375F,      8.5625F,      8.6875F,
               8.8125F,      8.9375F,      9.0625F,      9.1875F,
               9.3125F,      9.4375F,      9.5625F,      9.6875F,
               9.8125F,      9.9375F,      10.0625F,     10.1875F,
               10.3125F,     10.4375F,     10.5625F,     10.6875F,
               10.8125F,     10.9375F,     11.0625F,     11.1875F,
               11.3125F,     11.4375F,     11.5625F,     11.6875F,
               19.17578125F, 34.02734375F, 48.87890625F, 63.73046875F,
               78.58203125F, 93.43359375F, 108.2851562F, 123.1367188F,
               129.0546875F, 126.0390625F, 123.0234375F, 120.0078125F,
               116.9921875F, 113.9765625F, 110.9609375F, 107.9453125F,
               106.3125F,    106.0625F,    105.8125F,    105.5625F,
               105.3125F,    105.0625F,    104.8125F,    104.5625F,
               106.9296875F, 111.9140625F, 116.8984375F, 121.8828125F,
               126.8671875F, 131.8515625F, 136.8359375F, 141.8203125F,
               146.171875F,  149.890625F,  153.609375F,  157.328125F,
               161.046875F,  164.765625F,  168.484375F,  172.203125F,
               166.796875F,  152.265625F,  137.734375F,  123.203125F,
               108.671875F,  94.140625F,   79.609375F,   65.078125F,
               63.94140625F, 76.19921875F, 88.45703125F, 100.7148438F,
               112.9726562F, 125.2304688F, 137.4882812F, 149.7460938F,
               147.2265625F, 129.9296875F, 112.6328125F, 95.3359375F,
               78.0390625F,  60.7421875F,  43.4453125F,  26.1484375F,
               28.54296875F, 50.62890625F, 72.71484375F, 94.80078125F,
               116.8867188F, 138.9726562F, 161.0585938F, 183.1445312F,
               193.5390625F, 192.2421875F, 190.9453125F, 189.6484375F,
               188.3515625F, 187.0546875F, 185.7578125F, 184.4609375F,
               179.3320312F, 170.3710938F, 161.4101562F, 152.4492188F,
               143.4882812F, 134.5273438F, 125.5664062F, 116.6054688F,
               109.171875F,  103.265625F,  97.359375F,   91.453125F,
               85.546875F,   79.640625F,   73.734375F,   67.828125F,
               69.66015625F, 79.23046875F, 88.80078125F, 98.37109375F,
               107.9414062F, 117.5117188F, 127.0820312F, 136.6523438F,
               138.8242188F, 133.5976562F, 128.3710938F, 123.1445312F,
               117.9179688F, 112.6914062F, 107.4648438F, 102.2382812F,
               104.390625F,  113.921875F,  123.453125F,  132.984375F,
               142.515625F,  152.046875F,  161.578125F,  171.109375F,
               173.2265625F, 167.9296875F, 162.6328125F, 157.3359375F,
               152.0390625F, 146.7421875F, 141.4453125F, 136.1484375F,
               130.671875F,  125.015625F,  119.359375F,  113.703125F,
               108.046875F,  102.390625F,  96.734375F,   91.078125F,
               89.98828125F, 93.46484375F, 96.94140625F, 100.4179688F,
               103.8945312F, 107.3710938F, 110.8476562F, 114.3242188F,
               109.859375F,  97.453125F,   85.046875F,   72.640625F,
               60.234375F,   47.828125F,   35.421875F,   23.015625F,
               21.0390625F,  29.4921875F,  37.9453125F,  46.3984375F,
               54.8515625F,  63.3046875F,  71.7578125F,  80.2109375F,
               88.59765625F, 96.91796875F, 105.2382812F, 113.5585938F,
               121.8789062F, 130.1992188F, 138.5195312F, 146.8398438F,
               147.921875F,  141.765625F,  135.609375F,  129.453125F,
               123.296875F,  117.140625F,  110.984375F,  104.828125F,
               101.6640625F, 101.4921875F, 101.3203125F, 101.1484375F,
               100.9765625F, 100.8046875F, 100.6328125F, 100.4609375F,
               100.2382812F, 99.96484375F, 99.69140625F, 99.41796875F,
               99.14453125F, 98.87109375F, 98.59765625F, 98.32421875F,
               98.1875F,     98.1875F,     98.1875F,     98.1875F},
              {2.25F,        2.25F,        2.25F,        2.25F,
               2.3125F,      2.4375F,      2.5625F,      2.6875F,
               2.8125F,      2.9375F,      3.0625F,      3.1875F,
               3.3125F,      3.4375F,      3.5625F,      3.6875F,
               3.8125F,      3.9375F,      4.0625F,      4.1875F,
               4.3125F,      4.4375F,      4.5625F,      4.6875F,
               4.8125F,      4.9375F,      5.0625F,      5.1875F,
               5.3125F,      5.4375F,      5.5625F,      5.6875F,
               5.8125F,      5.9375F,      6.0625F,      6.1875F,
               6.3125F,      6.4375F,      6.5625F,      6.6875F,
               6.8125F,      6.9375F,      7.0625F,      7.1875F,
               7.3125F,      7.4375F,      7.5625F,      7.6875F,
               7.8125F,      7.9375F,      8.0625F,      8.1875F,
               8.3125F,      8.4375F,      8.5625F,      8.6875F,
               8.8125F,      8.9375F,      9.0625F,      9.1875F,
               9.3125F,      9.4375F,      9.5625F,      9.6875F,
               9.8125F,      9.9375F,      10.0625F,     10.1875F,
               10.3125F,     10.4375F,     10.5625F,     10.6875F,
               10.8125F,     10.9375F,     11.0625F,     11.1875F,
               11.3125F,     11.4375F,     11.5625F,     11.6875F,
               11.8125F,     11.9375F,     12.0625F,     12.1875F,
               20.51171875F, 37.03515625F, 53.55859375F, 70.08203125F,
               86.60546875F, 103.1289062F, 119.6523438F, 136.1757812F,
               141.1953125F, 134.7109375F, 128.2265625F, 121.7421875F,
               115.2578125F, 108.7734375F, 102.2890625F, 95.8046875F,
               93.3125F,     94.8125F,     96.3125F,     97.8125F,
               99.3125F,     100.8125F,    102.3125F,    103.8125F,
               107.6953125F, 113.9609375F, 120.2265625F, 126.4921875F,
               132.7578125F, 139.0234375F, 145.2890625F, 151.5546875F,
               154.953125F,  155.484375F,  156.015625F,  156.546875F,
               157.078125F,  157.609375F,  158.140625F,  158.671875F,
               152.828125F,  140.609375F,  128.390625F,  116.171875F,
               103.953125F,  91.734375F,   79.515625F,   67.296875F,
               66.99609375F, 78.61328125F, 90.23046875F, 101.8476562F,
               113.4648438F, 125.0820312F, 136.6992188F, 148.3164062F,
               145.5234375F, 128.3203125F, 111.1171875F, 93.9140625F,
               76.7109375F,  59.5078125F,  42.3046875F,  25.1015625F,
               27.14453125F, 48.43359375F, 69.72265625F, 91.01171875F,
               112.3007812F, 133.5898438F, 154.8789062F, 176.1679688F,
               185.7109375F, 183.5078125F, 181.3046875F, 179.1015625F,
               176.8984375F, 174.6953125F, 172.4921875F, 170.2890625F,
               164.3554688F, 154.6914062F, 145.0273438F, 135.3632812F,
               125.6992188F, 116.0351562F, 106.3710938F, 96.70703125F,
               89.328125F,   84.234375F,   79.140625F,   74.046875F,
               68.953125F,   63.859375F,   58.765625F,   53.671875F,
               55.27734375F, 63.58203125F, 71.88671875F, 80.19140625F,
               88.49609375F, 96.80078125F, 105.1054688F, 113.4101562F,
               115.6132812F, 111.7148438F, 107.8164062F, 103.9179688F,
               100.0195312F, 96.12109375F, 92.22265625F, 88.32421875F,
               91.609375F,   102.078125F,  112.546875F,  123.015625F,
               133.484375F,  143.953125F,  154.421875F,  164.890625F,
               167.6484375F, 162.6953125F, 157.7421875F, 152.7890625F,
               147.8359375F, 142.8828125F, 137.9296875F, 132.9765625F,
               127.578125F,  121.734375F,  115.890625F,  110.046875F,
               104.203125F,  98.359375F,   92.515625F,   86.671875F,
               85.19921875F, 88.09765625F, 90.99609375F, 93.89453125F,
               96.79296875F, 99.69140625F, 102.5898438F, 105.4882812F,
               101.265625F,  89.921875F,   78.578125F,   67.234375F,
               55.890625F,   44.546875F,   33.203125F,   21.859375F,
               19.7109375F,  26.7578125F,  33.8046875F,  40.8515625F,
               47.8984375F,  54.9453125F,  61.9921875F,  69.0390625F,
               77.08984375F, 86.14453125F, 95.19921875F, 104.2539062F,
               113.3085938F, 122.3632812F, 131.4179688F, 140.4726562F,
               143.953125F,  141.859375F,  139.765625F,  137.671875F,
               135.578125F,  133.484375F,  131.390625F,  129.296875F,
               128.0859375F, 127.7578125F, 127.4296875F, 127.1015625F,
               126.7734375F, 126.4453125F, 126.1171875F, 125.7890625F,
               124.5742188F, 122.4726562F, 120.3710938F, 118.2695312F,
               116.1679688F, 114.0664062F, 111.9648438F, 109.8632812F,
               108.8125F,    108.8125F,    108.8125F,    108.8125F},
              {2.75F,        2.75F,        2.75F,        2.75F,
               2.8125F,      2.9375F,      3.0625F,      3.1875F,
               3.3125F,      3.4375F,      3.5625F,      3.6875F,
               3.8125F,      3.9375F,      4.0625F,      4.1875F,
               4.3125F,      4.4375F,      4.5625F,      4.6875F,
               4.8125F,      4.9375F,      5.0625F,      5.1875F,
               5.3125F,      5.4375F,      5.5625F,      5.6875F,
               5.8125F,      5.9375F,      6.0625F,      6.1875F,
               6.3125F,      6.4375F,      6.5625F,      6.6875F,
               6.8125F,      6.9375F,      7.0625F,      7.1875F,
               7.3125F,      7.4375F,      7.5625F,      7.6875F,
               7.8125F,      7.9375F,      8.0625F,      8.1875F,
               8.3125F,      8.4375F,      8.5625F,      8.6875F,
               8.8125F,      8.9375F,      9.0625F,      9.1875F,
               9.3125F,      9.4375F,      9.5625F,      9.6875F,
               9.8125F,      9.9375F,      10.0625F,     10.1875F,
               10.3125F,     10.4375F,     10.5625F,     10.6875F,
               10.8125F,     10.9375F,     11.0625F,     11.1875F,
               11.3125F,     11.4375F,     11.5625F,     11.6875F,
               11.8125F,     11.9375F,     12.0625F,     12.1875F,
               12.3125F,     12.4375F,     12.5625F,     12.6875F,
               21.84765625F, 40.04296875F, 58.23828125F, 76.43359375F,
               94.62890625F, 112.8242188F, 131.0195312F, 149.2148438F,
               153.3359375F, 143.3828125F, 133.4296875F, 123.4765625F,
               113.5234375F, 103.5703125F, 93.6171875F,  83.6640625F,
               80.3125F,     83.5625F,     86.8125F,     90.0625F,
               93.3125F,     96.5625F,     99.8125F,     103.0625F,
               108.4609375F, 116.0078125F, 123.5546875F, 131.1015625F,
               138.6484375F, 146.1953125F, 153.7421875F, 161.2890625F,
               163.734375F,  161.078125F,  158.421875F,  155.765625F,
               153.109375F,  150.453125F,  147.796875F,  145.140625F,
               138.859375F,  128.953125F,  119.046875F,  109.140625F,
               99.234375F,   89.328125F,   79.421875F,   69.515625F,
               70.05078125F, 81.02734375F, 92.00390625F, 102.9804688F,
               113.9570312F, 124.9335938F, 135.9101562F, 146.8867188F,
               143.8203125F, 126.7109375F, 109.6015625F, 92.4921875F,
               75.3828125F,  58.2734375F,  41.1640625F,  24.0546875F,
               25.74609375F, 46.23828125F, 66.73046875F, 87.22265625F,
               107.7148438F, 128.2070312F, 148.6992188F, 169.1914062F,
               177.8828125F, 174.7734375F, 171.6640625F, 168.5546875F,
               165.4453125F, 162.3359375F, 159.2265625F, 156.1171875F,
               149.3789062F, 139.0117188F, 128.6445312F, 118.2773438F,
               107.9101562F, 97.54296875F, 87.17578125F, 76.80859375F,
               69.484375F,   65.203125F,   60.921875F,   56.640625F,
               52.359375F,   48.078125F,   43.796875F,   39.515625F,
               40.89453125F, 47.93359375F, 54.97265625F, 62.01171875F,
               69.05078125F, 76.08984375F, 83.12890625F, 90.16796875F,
               92.40234375F, 89.83203125F, 87.26171875F, 84.69140625F,
               82.12109375F, 79.55078125F, 76.98046875F, 74.41015625F,
               78.828125F,   90.234375F,   101.640625F,  113.046875F,
               124.453125F,  135.859375F,  147.265625F,  158.671875F,
               162.0703125F, 157.4609375F, 152.8515625F, 148.2421875F,
               143.6328125F, 139.0234375F, 134.4140625F, 129.8046875F,
               124.484375F,  118.453125F,  112.421875F,  106.390625F,
               100.359375F,  94.328125F,   88.296875F,   82.265625F,
               80.41015625F, 82.73046875F, 85.05078125F, 87.37109375F,
               89.69140625F, 92.01171875F, 94.33203125F, 96.65234375F,
               92.671875F,   82.390625F,   72.109375F,   61.828125F,
               51.546875F,   41.265625F,   30.984375F,   20.703125F,
               18.3828125F,  24.0234375F,  29.6640625F,  35.3046875F,
               40.9453125F,  46.5859375F,  52.2265625F,  57.8671875F,
               65.58203125F, 75.37109375F, 85.16015625F, 94.94921875F,
               104.7382812F, 114.5273438F, 124.3164062F, 134.1054688F,
               139.984375F,  141.953125F,  143.921875F,  145.890625F,
               147.859375F,  149.828125F,  151.796875F,  153.765625F,
               154.5078125F, 154.0234375F, 153.5390625F, 153.0546875F,
               152.5703125F, 152.0859375F, 151.6015625F, 151.1171875F,
               148.9101562F, 144.9804688F, 141.0507812F, 137.1210938F,
               133.1914062F, 129.2617188F, 125.3320312F, 121.4023438F,
               119.4375F,    119.4375F,    119.4375F,    119.4375F},
              {3.25F,        3.25F,        3.25F,        3.25F,
               3.3125F,      3.4375F,      3.5625F,      3.6875F,
               3.8125F,      3.9375F,      4.0625F,      4.1875F,
               4.3125F,      4.4375F,      4.5625F,      4.6875F,
               4.8125F,      4.9375F,      5.0625F,      5.1875F,
               5.3125F,      5.4375F,      5.5625F,      5.6875F,
               5.8125F,      5.9375F,      6.0625F,      6.1875F,
               6.3125F,      6.4375F,      6.5625F,      6.6875F,
               6.8125F,      6.9375F,      7.0625F,      7.1875F,
               7.3125F,      7.4375F,      7.5625F,      7.6875F,
               7.8125F,      7.9375F,      8.0625F,      8.1875F,
               8.3125F,      8.4375F,      8.5625F,      8.6875F,
               8.8125F,      8.9375F,      9.0625F,      9.1875F,
               9.3125F,      9.4375F,      9.5625F,      9.6875F,
               9.8125F,      9.9375F,      10.0625F,     10.1875F,
               10.3125F,     10.4375F,     10.5625F,     10.6875F,
               10.8125F,     10.9375F,     11.0625F,     11.1875F,
               11.3125F,     11.4375F,     11.5625F,     11.6875F,
               11.8125F,     11.9375F,     12.0625F,     12.1875F,
               12.3125F,     12.4375F,     12.5625F,     12.6875F,
               12.8125F,     12.9375F,     13.0625F,     13.1875F,
               23.18359375F, 43.05078125F, 62.91796875F, 82.78515625F,
               102.6523438F, 122.5195312F, 142.3867188F, 162.2539062F,
               165.4765625F, 152.0546875F, 138.6328125F, 125.2109375F,
               111.7890625F, 98.3671875F,  84.9453125F,  71.5234375F,
               67.3125F,     72.3125F,     77.3125F,     82.3125F,
               87.3125F,     92.3125F,     97.3125F,     102.3125F,
               109.2265625F, 118.0546875F, 126.8828125F, 135.7109375F,
               144.5390625F, 153.3671875F, 162.1953125F, 171.0234375F,
               172.515625F,  166.671875F,  160.828125F,  154.984375F,
               149.140625F,  143.296875F,  137.453125F,  131.609375F,
               124.890625F,  117.296875F,  109.703125F,  102.109375F,
               94.515625F,   86.921875F,   79.328125F,   71.734375F,
               73.10546875F, 83.44140625F, 93.77734375F, 104.1132812F,
               114.4492188F, 124.7851562F, 135.1210938F, 145.4570312F,
               142.1171875F, 125.1015625F, 108.0859375F, 91.0703125F,
               74.0546875F,  57.0390625F,  40.0234375F,  23.0078125F,
               24.34765625F, 44.04296875F, 63.73828125F, 83.43359375F,
               103.1289062F, 122.8242188F, 142.5195312F, 162.2148438F,
               170.0546875F, 166.0390625F, 162.0234375F, 158.0078125F,
               153.9921875F, 149.9765625F, 145.9609375F, 141.9453125F,
               134.4023438F, 123.3320312F, 112.2617188F, 101.1914062F,
               90.12109375F, 79.05078125F, 67.98046875F, 56.91015625F,
               49.640625F,   46.171875F,   42.703125F,   39.234375F,
               35.765625F,   32.296875F,   28.828125F,   25.359375F,
               26.51171875F, 32.28515625F, 38.05859375F, 43.83203125F,
               49.60546875F, 55.37890625F, 61.15234375F, 66.92578125F,
               69.19140625F, 67.94921875F, 66.70703125F, 65.46484375F,
               64.22265625F, 62.98046875F, 61.73828125F, 60.49609375F,
               66.046875F,   78.390625F,   90.734375F,   103.078125F,
               115.421875F,  127.765625F,  140.109375F,  152.453125F,
               156.4921875F, 152.2265625F, 147.9609375F, 143.6953125F,
               139.4296875F, 135.1640625F, 130.8984375F, 126.6328125F,
               121.390625F,  115.171875F,  108.953125F,  102.734375F,
               96.515625F,   90.296875F,   84.078125F,   77.859375F,
               75.62109375F, 77.36328125F, 79.10546875F, 80.84765625F,
               82.58984375F, 84.33203125F, 86.07421875F, 87.81640625F,
               84.078125F,   74.859375F,   65.640625F,   56.421875F,
               47.203125F,   37.984375F,   28.765625F,   19.546875F,
               17.0546875F,  21.2890625F,  25.5234375F,  29.7578125F,
               33.9921875F,  38.2265625F,  42.4609375F,  46.6953125F,
               54.07421875F, 64.59765625F, 75.12109375F, 85.64453125F,
               96.16796875F, 106.6914062F, 117.2148438F, 127.7382812F,
               136.015625F,  142.046875F,  148.078125F,  154.109375F,
               160.140625F,  166.171875F,  172.203125F,  178.234375F,
               180.9296875F, 180.2890625F, 179.6484375F, 179.0078125F,
               178.3671875F, 177.7265625F, 177.0859375F, 176.4453125F,
               173.2460938F, 167.4882812F, 161.7304688F, 155.9726562F,
               150.2148438F, 144.4570312F, 138.6992188F, 132.9414062F,
               130.0625F,    130.0625F,    130.0625F,    130.0625F},
              {3.75F,        3.75F,        3.75F,        3.75F,
               3.8125F,      3.9375F,      4.0625F,      4.1875F,
               4.3125F,      4.4375F,      4.5625F,      4.6875F,
               4.8125F,      4.9375F,      5.0625F,      5.1875F,
               5.3125F,      5.4375F,      5.5625F,      5.6875F,
               5.8125F,      5.9375F,      6.0625F,      6.1875F,
               6.3125F,      6.4375F,      6.5625F,      6.6875F,
               6.8125F,      6.9375F,      7.0625F,      7.1875F,
               7.3125F,      7.4375F,      7.5625F,      7.6875F,
               7.8125F,      7.9375F,      8.0625F,      8.1875F,
               8.3125F,      8.4375F,      8.5625F,      8.6875F,
               8.8125F,      8.9375F,      9.0625F,      9.1875F,
               9.3125F,      9.4375F,      9.5625F,      9.6875F,
               9.8125F,      9.9375F,      10.0625F,     10.1875F,
               10.3125F,     10.4375F,     10.5625F,     10.6875F,
               10.8125F,     10.9375F,     11.0625F,     11.1875F,
               11.3125F,     11.4375F,     11.5625F,     11.6875F,
               11.8125F,     11.9375F,     12.0625F,     12.1875F,
               12.3125F,     12.4375F,     12.5625F,     12.6875F,
               12.8125F,     12.9375F,     13.0625F,     13.1875F,
               13.3125F,     13.4375F,     13.5625F,     13.6875F,
               24.51953125F, 46.05859375F, 67.59765625F, 89.13671875F,
               110.6757812F, 132.2148438F, 153.7539062F, 175.2929688F,
               177.6171875F, 160.7265625F, 143.8359375F, 126.9453125F,
               110.0546875F, 93.1640625F,  76.2734375F,  59.3828125F,
               54.3125F,     61.0625F,     67.8125F,     74.5625F,
               81.3125F,     88.0625F,     94.8125F,     101.5625F,
               109.9921875F, 120.1015625F, 130.2109375F, 140.3203125F,
               150.4296875F, 160.5390625F, 170.6484375F, 180.7578125F,
               181.296875F,  172.265625F,  163.234375F,  154.203125F,
               145.171875F,  136.140625F,  127.109375F,  118.078125F,
               110.921875F,  105.640625F,  100.359375F,  95.078125F,
               89.796875F,   84.515625F,   79.234375F,   73.953125F,
               76.16015625F, 85.85546875F, 95.55078125F, 105.2460938F,
               114.9414062F, 124.6367188F, 134.3320312F, 144.0273438F,
               140.4140625F, 123.4921875F, 106.5703125F, 89.6484375F,
               72.7265625F,  55.8046875F,  38.8828125F,  21.9609375F,
               22.94921875F, 41.84765625F, 60.74609375F, 79.64453125F,
               98.54296875F, 117.4414062F, 136.3398438F, 155.2382812F,
               162.2265625F, 157.3046875F, 152.3828125F, 147.4609375F,
               142.5390625F, 137.6171875F, 132.6953125F, 127.7734375F,
               119.4257812F, 107.6523438F, 95.87890625F, 84.10546875F,
               72.33203125F, 60.55859375F, 48.78515625F, 37.01171875F,
               29.796875F,   27.140625F,   24.484375F,   21.828125F,
               19.171875F,   16.515625F,   13.859375F,   11.203125F,
               12.12890625F, 16.63671875F, 21.14453125F, 25.65234375F,
               30.16015625F, 34.66796875F, 39.17578125F, 43.68359375F,
               45.98046875F, 46.06640625F, 46.15234375F, 46.23828125F,
               46.32421875F, 46.41015625F, 46.49609375F, 46.58203125F,
               53.265625F,   66.546875F,   79.828125F,   93.109375F,
               106.390625F,  119.671875F,  132.953125F,  146.234375F,
               150.9140625F, 146.9921875F, 143.0703125F, 139.1484375F,
               135.2265625F, 131.3046875F, 127.3828125F, 123.4609375F,
               118.296875F,  111.890625F,  105.484375F,  99.078125F,
               92.671875F,   86.265625F,   79.859375F,   73.453125F,
               70.83203125F, 71.99609375F, 73.16015625F, 74.32421875F,
               75.48828125F, 76.65234375F, 77.81640625F, 78.98046875F,
               75.484375F,   67.328125F,   59.171875F,   51.015625F,
               42.859375F,   34.703125F,   26.546875F,   18.390625F,
               15.7265625F,  18.5546875F,  21.3828125F,  24.2109375F,
               27.0390625F,  29.8671875F,  32.6953125F,  35.5234375F,
               42.56640625F, 53.82421875F, 65.08203125F, 76.33984375F,
               87.59765625F, 98.85546875F, 110.1132812F, 121.3710938F,
               132.046875F,  142.140625F,  152.234375F,  162.328125F,
               172.421875F,  182.515625F,  192.609375F,  202.703125F,
               207.3515625F, 206.5546875F, 205.7578125F, 204.9609375F,
               204.1640625F, 203.3671875F, 202.5703125F, 201.7734375F,
               197.5820312F, 189.9960938F, 182.4101562F, 174.8242188F,
               167.2382812F, 159.6523438F, 152.0664062F, 144.4804688F,
               140.6875F,    140.6875F,    140.6875F,    140.6875F},
              {4.0F,      4.0F,      4.0F,      4.0F,      4.0625F,   4.1875F,
               4.3125F,   4.4375F,   4.5625F,   4.6875F,   4.8125F,   4.9375F,
               5.0625F,   5.1875F,   5.3125F,   5.4375F,   5.5625F,   5.6875F,
               5.8125F,   5.9375F,   6.0625F,   6.1875F,   6.3125F,   6.4375F,
               6.5625F,   6.6875F,   6.8125F,   6.9375F,   7.0625F,   7.1875F,
               7.3125F,   7.4375F,   7.5625F,   7.6875F,   7.8125F,   7.9375F,
               8.0625F,   8.1875F,   8.3125F,   8.4375F,   8.5625F,   8.6875F,
               8.8125F,   8.9375F,   9.0625F,   9.1875F,   9.3125F,   9.4375F,
               9.5625F,   9.6875F,   9.8125F,   9.9375F,   10.0625F,  10.1875F,
               10.3125F,  10.4375F,  10.5625F,  10.6875F,  10.8125F,  10.9375F,
               11.0625F,  11.1875F,  11.3125F,  11.4375F,  11.5625F,  11.6875F,
               11.8125F,  11.9375F,  12.0625F,  12.1875F,  12.3125F,  12.4375F,
               12.5625F,  12.6875F,  12.8125F,  12.9375F,  13.0625F,  13.1875F,
               13.3125F,  13.4375F,  13.5625F,  13.6875F,  13.8125F,  13.9375F,
               25.1875F,  47.5625F,  69.9375F,  92.3125F,  114.6875F, 137.0625F,
               159.4375F, 181.8125F, 183.6875F, 165.0625F, 146.4375F, 127.8125F,
               109.1875F, 90.5625F,  71.9375F,  53.3125F,  47.8125F,  55.4375F,
               63.0625F,  70.6875F,  78.3125F,  85.9375F,  93.5625F,  101.1875F,
               110.375F,  121.125F,  131.875F,  142.625F,  153.375F,  164.125F,
               174.875F,  185.625F,  185.6875F, 175.0625F, 164.4375F, 153.8125F,
               143.1875F, 132.5625F, 121.9375F, 111.3125F, 103.9375F, 99.8125F,
               95.6875F,  91.5625F,  87.4375F,  83.3125F,  79.1875F,  75.0625F,
               77.6875F,  87.0625F,  96.4375F,  105.8125F, 115.1875F, 124.5625F,
               133.9375F, 143.3125F, 139.5625F, 122.6875F, 105.8125F, 88.9375F,
               72.0625F,  55.1875F,  38.3125F,  21.4375F,  22.25F,    40.75F,
               59.25F,    77.75F,    96.25F,    114.75F,   133.25F,   151.75F,
               158.3125F, 152.9375F, 147.5625F, 142.1875F, 136.8125F, 131.4375F,
               126.0625F, 120.6875F, 111.9375F, 99.8125F,  87.6875F,  75.5625F,
               63.4375F,  51.3125F,  39.1875F,  27.0625F,  19.875F,   17.625F,
               15.375F,   13.125F,   10.875F,   8.625F,    6.375F,    4.125F,
               4.9375F,   8.8125F,   12.6875F,  16.5625F,  20.4375F,  24.3125F,
               28.1875F,  32.0625F,  34.375F,   35.125F,   35.875F,   36.625F,
               37.375F,   38.125F,   38.875F,   39.625F,   46.875F,   60.625F,
               74.375F,   88.125F,   101.875F,  115.625F,  129.375F,  143.125F,
               148.125F,  144.375F,  140.625F,  136.875F,  133.125F,  129.375F,
               125.625F,  121.875F,  116.75F,   110.25F,   103.75F,   97.25F,
               90.75F,    84.25F,    77.75F,    71.25F,    68.4375F,  69.3125F,
               70.1875F,  71.0625F,  71.9375F,  72.8125F,  73.6875F,  74.5625F,
               71.1875F,  63.5625F,  55.9375F,  48.3125F,  40.6875F,  33.0625F,
               25.4375F,  17.8125F,  15.0625F,  17.1875F,  19.3125F,  21.4375F,
               23.5625F,  25.6875F,  27.8125F,  29.9375F,  36.8125F,  48.4375F,
               60.0625F,  71.6875F,  83.3125F,  94.9375F,  106.5625F, 118.1875F,
               130.0625F, 142.1875F, 154.3125F, 166.4375F, 178.5625F, 190.6875F,
               202.8125F, 214.9375F, 220.5625F, 219.6875F, 218.8125F, 217.9375F,
               217.0625F, 216.1875F, 215.3125F, 214.4375F, 209.75F,   201.25F,
               192.75F,   184.25F,   175.75F,   167.25F,   158.75F,   150.25F,
               146.0F,    146.0F,    146.0F,    146.0F},
              {4.0F,      4.0F,      4.0F,      4.0F,      4.0625F,   4.1875F,
               4.3125F,   4.4375F,   4.5625F,   4.6875F,   4.8125F,   4.9375F,
               5.0625F,   5.1875F,   5.3125F,   5.4375F,   5.5625F,   5.6875F,
               5.8125F,   5.9375F,   6.0625F,   6.1875F,   6.3125F,   6.4375F,
               6.5625F,   6.6875F,   6.8125F,   6.9375F,   7.0625F,   7.1875F,
               7.3125F,   7.4375F,   7.5625F,   7.6875F,   7.8125F,   7.9375F,
               8.0625F,   8.1875F,   8.3125F,   8.4375F,   8.5625F,   8.6875F,
               8.8125F,   8.9375F,   9.0625F,   9.1875F,   9.3125F,   9.4375F,
               9.5625F,   9.6875F,   9.8125F,   9.9375F,   10.0625F,  10.1875F,
               10.3125F,  10.4375F,  10.5625F,  10.6875F,  10.8125F,  10.9375F,
               11.0625F,  11.1875F,  11.3125F,  11.4375F,  11.5625F,  11.6875F,
               11.8125F,  11.9375F,  12.0625F,  12.1875F,  12.3125F,  12.4375F,
               12.5625F,  12.6875F,  12.8125F,  12.9375F,  13.0625F,  13.1875F,
               13.3125F,  13.4375F,  13.5625F,  13.6875F,  13.8125F,  13.9375F,
               25.1875F,  47.5625F,  69.9375F,  92.3125F,  114.6875F, 137.0625F,
               159.4375F, 181.8125F, 183.6875F, 165.0625F, 146.4375F, 127.8125F,
               109.1875F, 90.5625F,  71.9375F,  53.3125F,  47.8125F,  55.4375F,
               63.0625F,  70.6875F,  78.3125F,  85.9375F,  93.5625F,  101.1875F,
               110.375F,  121.125F,  131.875F,  142.625F,  153.375F,  164.125F,
               174.875F,  185.625F,  185.6875F, 175.0625F, 164.4375F, 153.8125F,
               143.1875F, 132.5625F, 121.9375F, 111.3125F, 103.9375F, 99.8125F,
               95.6875F,  91.5625F,  87.4375F,  83.3125F,  79.1875F,  75.0625F,
               77.6875F,  87.0625F,  96.4375F,  105.8125F, 115.1875F, 124.5625F,
               133.9375F, 143.3125F, 139.5625F, 122.6875F, 105.8125F, 88.9375F,
               72.0625F,  55.1875F,  38.3125F,  21.4375F,  22.25F,    40.75F,
               59.25F,    77.75F,    96.25F,    114.75F,   133.25F,   151.75F,
               158.3125F, 152.9375F, 147.5625F, 142.1875F, 136.8125F, 131.4375F,
               126.0625F, 120.6875F, 111.9375F, 99.8125F,  87.6875F,  75.5625F,
               63.4375F,  51.3125F,  39.1875F,  27.0625F,  19.875F,   17.625F,
               15.375F,   13.125F,   10.875F,   8.625F,    6.375F,    4.125F,
               4.9375F,   8.8125F,   12.6875F,  16.5625F,  20.4375F,  24.3125F,
               28.1875F,  32.0625F,  34.375F,   35.125F,   35.875F,   36.625F,
               37.375F,   38.125F,   38.875F,   39.625F,   46.875F,   60.625F,
               74.375F,   88.125F,   101.875F,  115.625F,  129.375F,  143.125F,
               148.125F,  144.375F,  140.625F,  136.875F,  133.125F,  129.375F,
               125.625F,  121.875F,  116.75F,   110.25F,   103.75F,   97.25F,
               90.75F,    84.25F,    77.75F,    71.25F,    68.4375F,  69.3125F,
               70.1875F,  71.0625F,  71.9375F,  72.8125F,  73.6875F,  74.5625F,
               71.1875F,  63.5625F,  55.9375F,  48.3125F,  40.6875F,  33.0625F,
               25.4375F,  17.8125F,  15.0625F,  17.1875F,  19.3125F,  21.4375F,
               23.5625F,  25.6875F,  27.8125F,  29.9375F,  36.8125F,  48.4375F,
               60.0625F,  71.6875F,  83.3125F,  94.9375F,  106.5625F, 118.1875F,
               130.0625F, 142.1875F, 154.3125F, 166.4375F, 178.5625F, 190.6875F,
               202.8125F, 214.9375F, 220.5625F, 219.6875F, 218.8125F, 217.9375F,
               217.0625F, 216.1875F, 215.3125F, 214.4375F, 209.75F,   201.25F,
               192.75F,   184.25F,   175.75F,   167.25F,   158.75F,   150.25F,
               146.0F,    146.0F,    146.0F,    146.0F},
              {4.0F,      4.0F,      4.0F,      4.0F,      4.0625F,   4.1875F,
               4.3125F,   4.4375F,   4.5625F,   4.6875F,   4.8125F,   4.9375F,
               5.0625F,   5.1875F,   5.3125F,   5.4375F,   5.5625F,   5.6875F,
               5.8125F,   5.9375F,   6.0625F,   6.1875F,   6.3125F,   6.4375F,
               6.5625F,   6.6875F,   6.8125F,   6.9375F,   7.0625F,   7.1875F,
               7.3125F,   7.4375F,   7.5625F,   7.6875F,   7.8125F,   7.9375F,
               8.0625F,   8.1875F,   8.3125F,   8.4375F,   8.5625F,   8.6875F,
               8.8125F,   8.9375F,   9.0625F,   9.1875F,   9.3125F,   9.4375F,
               9.5625F,   9.6875F,   9.8125F,   9.9375F,   10.0625F,  10.1875F,
               10.3125F,  10.4375F,  10.5625F,  10.6875F,  10.8125F,  10.9375F,
               11.0625F,  11.1875F,  11.3125F,  11.4375F,  11.5625F,  11.6875F,
               11.8125F,  11.9375F,  12.0625F,  12.1875F,  12.3125F,  12.4375F,
               12.5625F,  12.6875F,  12.8125F,  12.9375F,  13.0625F,  13.1875F,
               13.3125F,  13.4375F,  13.5625F,  13.6875F,  13.8125F,  13.9375F,
               25.1875F,  47.5625F,  69.9375F,  92.3125F,  114.6875F, 137.0625F,
               159.4375F, 181.8125F, 183.6875F, 165.0625F, 146.4375F, 127.8125F,
               109.1875F, 90.5625F,  71.9375F,  53.3125F,  47.8125F,  55.4375F,
               63.0625F,  70.6875F,  78.3125F,  85.9375F,  93.5625F,  101.1875F,
               110.375F,  121.125F,  131.875F,  142.625F,  153.375F,  164.125F,
               174.875F,  185.625F,  185.6875F, 175.0625F, 164.4375F, 153.8125F,
               143.1875F, 132.5625F, 121.9375F, 111.3125F, 103.9375F, 99.8125F,
               95.6875F,  91.5625F,  87.4375F,  83.3125F,  79.1875F,  75.0625F,
               77.6875F,  87.0625F,  96.4375F,  105.8125F, 115.1875F, 124.5625F,
               133.9375F, 143.3125F, 139.5625F, 122.6875F, 105.8125F, 88.9375F,
               72.0625F,  55.1875F,  38.3125F,  21.4375F,  22.25F,    40.75F,
               59.25F,    77.75F,    96.25F,    114.75F,   133.25F,   151.75F,
               158.3125F, 152.9375F, 147.5625F, 142.1875F, 136.8125F, 131.4375F,
               126.0625F, 120.6875F, 111.9375F, 99.8125F,  87.6875F,  75.5625F,
               63.4375F,  51.3125F,  39.1875F,  27.0625F,  19.875F,   17.625F,
               15.375F,   13.125F,   10.875F,   8.625F,    6.375F,    4.125F,
               4.9375F,   8.8125F,   12.6875F,  16.5625F,  20.4375F,  24.3125F,
               28.1875F,  32.0625F,  34.375F,   35.125F,   35.875F,   36.625F,
               37.375F,   38.125F,   38.875F,   39.625F,   46.875F,   60.625F,
               74.375F,   88.125F,   101.875F,  115.625F,  129.375F,  143.125F,
               148.125F,  144.375F,  140.625F,  136.875F,  133.125F,  129.375F,
               125.625F,  121.875F,  116.75F,   110.25F,   103.75F,   97.25F,
               90.75F,    84.25F,    77.75F,    71.25F,    68.4375F,  69.3125F,
               70.1875F,  71.0625F,  71.9375F,  72.8125F,  73.6875F,  74.5625F,
               71.1875F,  63.5625F,  55.9375F,  48.3125F,  40.6875F,  33.0625F,
               25.4375F,  17.8125F,  15.0625F,  17.1875F,  19.3125F,  21.4375F,
               23.5625F,  25.6875F,  27.8125F,  29.9375F,  36.8125F,  48.4375F,
               60.0625F,  71.6875F,  83.3125F,  94.9375F,  106.5625F, 118.1875F,
               130.0625F, 142.1875F, 154.3125F, 166.4375F, 178.5625F, 190.6875F,
               202.8125F, 214.9375F, 220.5625F, 219.6875F, 218.8125F, 217.9375F,
               217.0625F, 216.1875F, 215.3125F, 214.4375F, 209.75F,   201.25F,
               192.75F,   184.25F,   175.75F,   167.25F,   158.75F,   150.25F,
               146.0F,    146.0F,    146.0F,    146.0F},
              {4.0F,      4.0F,      4.0F,      4.0F,      4.0625F,   4.1875F,
               4.3125F,   4.4375F,   4.5625F,   4.6875F,   4.8125F,   4.9375F,
               5.0625F,   5.1875F,   5.3125F,   5.4375F,   5.5625F,   5.6875F,
               5.8125F,   5.9375F,   6.0625F,   6.1875F,   6.3125F,   6.4375F,
               6.5625F,   6.6875F,   6.8125F,   6.9375F,   7.0625F,   7.1875F,
               7.3125F,   7.4375F,   7.5625F,   7.6875F,   7.8125F,   7.9375F,
               8.0625F,   8.1875F,   8.3125F,   8.4375F,   8.5625F,   8.6875F,
               8.8125F,   8.9375F,   9.0625F,   9.1875F,   9.3125F,   9.4375F,
               9.5625F,   9.6875F,   9.8125F,   9.9375F,   10.0625F,  10.1875F,
               10.3125F,  10.4375F,  10.5625F,  10.6875F,  10.8125F,  10.9375F,
               11.0625F,  11.1875F,  11.3125F,  11.4375F,  11.5625F,  11.6875F,
               11.8125F,  11.9375F,  12.0625F,  12.1875F,  12.3125F,  12.4375F,
               12.5625F,  12.6875F,  12.8125F,  12.9375F,  13.0625F,  13.1875F,
               13.3125F,  13.4375F,  13.5625F,  13.6875F,  13.8125F,  13.9375F,
               25.1875F,  47.5625F,  69.9375F,  92.3125F,  114.6875F, 137.0625F,
               159.4375F, 181.8125F, 183.6875F, 165.0625F, 146.4375F, 127.8125F,
               109.1875F, 90.5625F,  71.9375F,  53.3125F,  47.8125F,  55.4375F,
               63.0625F,  70.6875F,  78.3125F,  85.9375F,  93.5625F,  101.1875F,
               110.375F,  121.125F,  131.875F,  142.625F,  153.375F,  164.125F,
               174.875F,  185.625F,  185.6875F, 175.0625F, 164.4375F, 153.8125F,
               143.1875F, 132.5625F, 121.9375F, 111.3125F, 103.9375F, 99.8125F,
               95.6875F,  91.5625F,  87.4375F,  83.3125F,  79.1875F,  75.0625F,
               77.6875F,  87.0625F,  96.4375F,  105.8125F, 115.1875F, 124.5625F,
               133.9375F, 143.3125F, 139.5625F, 122.6875F, 105.8125F, 88.9375F,
               72.0625F,  55.1875F,  38.3125F,  21.4375F,  22.25F,    40.75F,
               59.25F,    77.75F,    96.25F,    114.75F,   133.25F,   151.75F,
               158.3125F, 152.9375F, 147.5625F, 142.1875F, 136.8125F, 131.4375F,
               126.0625F, 120.6875F, 111.9375F, 99.8125F,  87.6875F,  75.5625F,
               63.4375F,  51.3125F,  39.1875F,  27.0625F,  19.875F,   17.625F,
               15.375F,   13.125F,   10.875F,   8.625F,    6.375F,    4.125F,
               4.9375F,   8.8125F,   12.6875F,  16.5625F,  20.4375F,  24.3125F,
               28.1875F,  32.0625F,  34.375F,   35.125F,   35.875F,   36.625F,
               37.375F,   38.125F,   38.875F,   39.625F,   46.875F,   60.625F,
               74.375F,   88.125F,   101.875F,  115.625F,  129.375F,  143.125F,
               148.125F,  144.375F,  140.625F,  136.875F,  133.125F,  129.375F,
               125.625F,  121.875F,  116.75F,   110.25F,   103.75F,   97.25F,
               90.75F,    84.25F,    77.75F,    71.25F,    68.4375F,  69.3125F,
               70.1875F,  71.0625F,  71.9375F,  72.8125F,  73.6875F,  74.5625F,
               71.1875F,  63.5625F,  55.9375F,  48.3125F,  40.6875F,  33.0625F,
               25.4375F,  17.8125F,  15.0625F,  17.1875F,  19.3125F,  21.4375F,
               23.5625F,  25.6875F,  27.8125F,  29.9375F,  36.8125F,  48.4375F,
               60.0625F,  71.6875F,  83.3125F,  94.9375F,  106.5625F, 118.1875F,
               130.0625F, 142.1875F, 154.3125F, 166.4375F, 178.5625F, 190.6875F,
               202.8125F, 214.9375F, 220.5625F, 219.6875F, 218.8125F, 217.9375F,
               217.0625F, 216.1875F, 215.3125F, 214.4375F, 209.75F,   201.25F,
               192.75F,   184.25F,   175.75F,   167.25F,   158.75F,   150.25F,
               146.0F,    146.0F,    146.0F,    146.0F}}}));
