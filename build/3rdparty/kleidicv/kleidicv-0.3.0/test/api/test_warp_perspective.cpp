// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/utils.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE

// clang-format off
static const float transform_identity[] = {
  1.0, 0, 0,
  0, 1.0, 0,
  0, 0, 1.0
};

static const float transform_transpose[] = {
  0, 1.0, 0,
  1.0, 0, 0,
  0, 0, 1.0
};

static const float transform_small[] = {
  0.8, 0.1, 2,
  0.1, 0.8, -2,
  0.001, 0.001, 1.7
};
// clang-format on

template <class ScalarType>
static void random_initializer(test::Array2D<ScalarType> &source) {
  test::PseudoRandomNumberGenerator<ScalarType> generator;
  source.fill(generator);
}

template <class ScalarType>
class WarpPerspective : public testing::Test {
 public:
  static void test(
      size_t src_w, size_t src_h, size_t dst_w, size_t dst_h,
      const float transform[9], size_t channels, size_t padding,
      void (*initializer)(test::Array2D<ScalarType> &) = random_initializer) {
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    test::Array2D<ScalarType> source{src_total_width, src_h, padding, channels};
    test::Array2D<ScalarType> actual{dst_total_width, dst_h, padding, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, padding,
                                       channels};

    initializer(source);
    actual.fill(42);

    calculate_expected(source, transform, expected);

    ASSERT_EQ(KLEIDICV_OK, kleidicv_warp_perspective_u8(
                               source.data(), source.stride(), source.width(),
                               source.height(), actual.data(), actual.stride(),
                               actual.width(), actual.height(), transform,
                               channels, KLEIDICV_INTERPOLATION_NEAREST,
                               KLEIDICV_BORDER_TYPE_REPLICATE, {}));

    EXPECT_EQ_ARRAY2D(actual, expected);
  }

  static void test_special_source(
      size_t src_w, size_t src_h, size_t src_stride, size_t dst_w, size_t dst_h,
      const float transform[9], size_t channels,
      void (*initializer)(test::Array2D<ScalarType> &) = random_initializer) {
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    // Use fake parameters to avoid large filling of padding
    size_t total_width_bytes = src_total_width * sizeof(ScalarType);
    size_t physical_stride = std::max(src_stride, total_width_bytes);
    size_t physical_width = physical_stride / sizeof(ScalarType);

    test::Array2D<ScalarType> source{physical_width, src_h, 0, channels};
    initializer(source);

    test::Array2D<ScalarType> actual{dst_total_width, dst_h, 0, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, 0, channels};

    actual.fill(42);

    calculate_expected(source, transform, expected);

    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_warp_perspective_u8(
                  source.data(), src_stride, src_w, src_h, actual.data(),
                  actual.stride(), actual.width(), actual.height(), transform,
                  channels, KLEIDICV_INTERPOLATION_NEAREST,
                  KLEIDICV_BORDER_TYPE_REPLICATE, {}));

    EXPECT_EQ_ARRAY2D(actual, expected);
  }

 private:
  static void calculate_expected(test::Array2D<ScalarType> &src,
                                 const float transform[9],
                                 test::Array2D<ScalarType> &expected) {
    for (size_t y = 0; y < expected.height(); ++y) {
      for (size_t x = 0; x < expected.width() / src.channels(); ++x) {
        float dx = static_cast<float>(x), dy = static_cast<float>(y);

        float dw = transform[6] * dx + transform[7] * dy + transform[8];
        float inv_w = 1.F / dw;
        float fx =
            inv_w * (transform[0] * dx + transform[1] * dy + transform[2]);
        float fy =
            inv_w * (transform[3] * dx + transform[4] * dy + transform[5]);
        ptrdiff_t ix = static_cast<ptrdiff_t>(
            std::max<double>(0, std::min(fx + 0.5, src.width() - 1.0)));
        ptrdiff_t iy = static_cast<ptrdiff_t>(
            std::max<double>(0, std::min(fy + 0.5, src.height() - 1.0)));
        for (size_t ch = 0; ch < src.channels(); ++ch) {
          *expected.at(y, x * src.channels() + ch) =
              *src.at(iy, ix * src.channels() + ch);
        }
      }
    }
  }
};

using WarpPerspectiveElementTypes = ::testing::Types<uint8_t>;
TYPED_TEST_SUITE(WarpPerspective, WarpPerspectiveElementTypes);

TYPED_TEST(WarpPerspective, IdentityNoPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test(src_w, src_h, dst_w, dst_h, transform_identity, 1, 0);
}

TYPED_TEST(WarpPerspective, TransposeNoPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test(src_w, src_h, dst_w, dst_h, transform_transpose, 1, 0);
}

TYPED_TEST(WarpPerspective, SmallPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test(src_w, src_h, dst_w, dst_h, transform_small, 1, 13);
}

TYPED_TEST(WarpPerspective, Upscale) {
  // clang-format off
  const float transform_upscale[] = {
    0.2, 0.05, 0.1,
    0.02, 0.2, -0.1,
    0.0001, 0.0001, 1.1
  };
  // clang-format on

  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w * 3;
  size_t dst_h = src_h * 2;
  TestFixture::test(src_w, src_h, dst_w, dst_h, transform_upscale, 1, 3);
}

TYPED_TEST(WarpPerspective, RandomTransform) {
  float transform[9];
  test::PseudoRandomNumberGenerator<float> generator;
  for (size_t i = 0; i < 9; ++i) {
    transform[i] = generator.next().value_or(1.0);
  }

  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test(src_w, src_h, dst_w, dst_h, transform, 1, 19);
}

TYPED_TEST(WarpPerspective, DivisionByZero) {
  // clang-format off
  const float transform_div_by_zero[] = {
    1, 0, 0,
    0, 1, 0,
    1, 1, -3
  };
  // clang-format on

  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test(src_w, src_h, dst_w, dst_h, transform_div_by_zero, 1, 3);
}

static const size_t big_width = 1ULL << 17, big_height = 1ULL << 17;
static const size_t part_width = 16, part_height = 16;

template <class ScalarType>
static void part_initializer(test::Array2D<ScalarType> &source) {
  ScalarType counter = 0;
  for (size_t y = big_height; y < big_height + part_height; ++y) {
    for (size_t x = big_width; x < big_width + part_width; ++x) {
      *source.at(y, x) = ++counter;
    }
  }
}

TYPED_TEST(WarpPerspective, BigSourceImage) {
  // clang-format off
  const float transform_cut[] = {
    1, 0, 1ULL<<17,
    0, 1, 1ULL<<17,
    0, 0, 1
  };
  // clang-format on

  // Let stride * height be bigger than 1 << 32
  size_t dst_w = part_width;
  size_t dst_h = part_height;
  size_t src_w = big_width + part_width;
  size_t src_h = big_height + part_height;

  TestFixture::test(src_w, src_h, dst_w, dst_h, transform_cut, 1, 0,
                    part_initializer);
}

static const size_t oneline_big_width = 1ULL << 23;
static const size_t oneline_part_width = 16, oneline_part_offset = 1ULL << 17,
                    oneline_dst_height = 16;

template <class ScalarType>
static void oneline_part_initializer(test::Array2D<ScalarType> &source) {
  ScalarType counter = 0;
  for (size_t x = oneline_part_offset;
       x < oneline_part_offset + oneline_part_width; ++x) {
    *source.at(0, x) = ++counter;
  }
}

TYPED_TEST(WarpPerspective, OneLineBigSourceImage) {
  // clang-format off
  const float tr[] = {
    1, 0, oneline_part_offset,
    0, 0, 0,
    0, 0, 1
  };
  // clang-format on

  size_t dst_w = oneline_part_width;
  size_t dst_h = oneline_dst_height;
  size_t src_w = oneline_big_width + oneline_part_width;
  size_t src_h = 1;

  TestFixture::test_special_source(src_w, src_h, 0, dst_w, dst_h, tr, 1,
                                   oneline_part_initializer);

  TestFixture::test_special_source(src_w, src_h, (1ULL << 32) - 1, dst_w, dst_h,
                                   tr, 1, oneline_part_initializer);
}

TYPED_TEST(WarpPerspective, OneLineSmallSourceImage) {
  // clang-format off
  const float tr[] = {
    -1, 0, 1,
    0, 0, 0,
    0, 0, 1
  };
  // clang-format on

  size_t dst_w = oneline_part_width;
  size_t dst_h = oneline_dst_height;
  size_t src_w = 2;
  size_t src_h = 1;

  TestFixture::test_special_source(src_w, src_h, src_h, dst_w, dst_h, tr, 1);
}

TYPED_TEST(WarpPerspective, NullPointer) {
  const TypeParam src[4] = {};
  TypeParam dst[8];
  test::test_null_args(kleidicv_warp_perspective_u8, src, 2, 2, 2, dst, 8, 8, 1,
                       transform_identity, 1, KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_REPLICATE,
                       kleidicv_border_values_t{});
}

TYPED_TEST(WarpPerspective, ZeroImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_warp_perspective_u8(
                src, 1, 0, 1, dst, 1, 0, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                kleidicv_border_values_t{}));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 0, dst, 1, 1, 0, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                kleidicv_border_values_t{}));
}

TYPED_TEST(WarpPerspective, InvalidImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, dst, 8, 8, 1,
                transform_identity, 1, KLEIDICV_INTERPOLATION_NEAREST,
                KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{}));

  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_warp_perspective_u8(
          src, 1, KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, dst, 8,
          8, 1, transform_identity, 1, KLEIDICV_INTERPOLATION_NEAREST,
          KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{}));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 1, KLEIDICV_MAX_IMAGE_PIXELS + 1, 1,
                transform_identity, 1, KLEIDICV_INTERPOLATION_NEAREST,
                KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{}));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 1, KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                kleidicv_border_values_t{}));
}

TYPED_TEST(WarpPerspective, UnsupportedTwoChannels) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 8, 8, 1, transform_identity, 2,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                kleidicv_border_values_t{}));
}

TYPED_TEST(WarpPerspective, UnsupportedBorderTypeConst) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 8, 8, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_CONSTANT,
                kleidicv_border_values_t{}));
}

TYPED_TEST(WarpPerspective, UnsupportedTooSmallImage) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 8, 7, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                kleidicv_border_values_t{}));
}

TYPED_TEST(WarpPerspective, UnsupportedInterpolation) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1, dst, 8, 8, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_LINEAR, KLEIDICV_BORDER_TYPE_REPLICATE,
                kleidicv_border_values_t{}));
}

TYPED_TEST(WarpPerspective, UnsupportedBigStride) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1UL << 32, 1, 1, dst, 8, 8, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                kleidicv_border_values_t{}));
}

TYPED_TEST(WarpPerspective, UnsupportedBigHeight) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, 1, 1UL << 24, dst, 8, 8, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                kleidicv_border_values_t{}));
}

TYPED_TEST(WarpPerspective, UnsupportedBigWidth) {
  const TypeParam src[1] = {};
  TypeParam dst[8];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_warp_perspective_u8(
                src, 1, 1UL << 24, 1, dst, 8, 8, 1, transform_identity, 1,
                KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
                kleidicv_border_values_t{}));
}
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
