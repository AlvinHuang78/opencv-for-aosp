// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_REMAP

template <class ScalarType>
class RemapS16 : public testing::Test {
 public:
  static void test_random(size_t src_w, size_t src_h, size_t dst_w,
                          size_t dst_h, size_t channels, size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    test::PseudoRandomNumberGenerator<int16_t> coord_generator;
    mapxy.fill(coord_generator);
    execute_test(mapxy, src_w, src_h, dst_w, dst_h, channels, padding);
  }

  static void test_outside_random(size_t src_w, size_t src_h, size_t dst_w,
                                  size_t dst_h, size_t channels,
                                  size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    test::PseudoRandomNumberGeneratorIntRange<int16_t> coord_generator{
        static_cast<int16_t>(-src_w), static_cast<int16_t>(2 * src_w)};
    mapxy.fill(coord_generator);
    execute_test(mapxy, src_w, src_h, dst_w, dst_h, channels, padding);
  }

  static void test_blend(size_t src_w, size_t src_h, size_t dst_w, size_t dst_h,
                         size_t channels, size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    for (size_t row = 0; row < dst_h; ++row) {
      for (size_t column = 0; column < dst_w; ++column) {
        // Use a second degree function to add a nonlinear blend to the image
        *mapxy.at(row, column * 2) = std::max<int16_t>(
            0, std::min(
                   static_cast<int16_t>(src_w - 1),
                   static_cast<int16_t>(column * 2 - column * column / dst_w)));
        *mapxy.at(row, column * 2 + 1) = std::max<int16_t>(
            0, std::min(static_cast<int16_t>(src_h - 1),
                        static_cast<int16_t>(row * (dst_w - column) / dst_w +
                                             4 * row / dst_h)));
      }
    }
    execute_test(mapxy, src_w, src_h, dst_w, dst_h, channels, padding);
  }

  // Test coordinates with edge values that may easily overflow
  static void test_corner_cases(size_t src_w, size_t src_h, size_t dst_w,
                                size_t dst_h, size_t channels, size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    // One more y than x so we'll see many combinations
    const int16_t corner_x_values[] = {-32768,
                                       -1,
                                       0,
                                       static_cast<int16_t>(src_w - 2),
                                       static_cast<int16_t>(src_w - 1),
                                       static_cast<int16_t>(src_w),
                                       32767};
    const int16_t corner_y_values[] = {-32768,
                                       -1,
                                       0,
                                       1,
                                       static_cast<int16_t>(src_h - 2),
                                       static_cast<int16_t>(src_h - 1),
                                       static_cast<int16_t>(src_h),
                                       32767};
    static const size_t nx = sizeof(corner_x_values) / sizeof(int16_t);
    static const size_t ny = sizeof(corner_y_values) / sizeof(int16_t);
    size_t counter = 0;
    for (size_t row = 0; row < dst_h; ++row) {
      for (size_t column = 0; column < dst_w; ++column) {
        *mapxy.at(row, column * 2) = corner_x_values[counter % nx];
        *mapxy.at(row, column * 2 + 1) = corner_y_values[counter % ny];
      }
    }
    execute_test(mapxy, src_w, src_h, dst_w, dst_h, channels, padding);
  }

 private:
  static void execute_test(test::Array2D<int16_t> &mapxy, size_t src_w,
                           size_t src_h, size_t dst_w, size_t dst_h,
                           size_t channels, size_t padding) {
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    test::Array2D<ScalarType> source{src_total_width, src_h, padding, channels};
    test::Array2D<ScalarType> actual{dst_total_width, dst_h, padding, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, padding,
                                       channels};

    test::PseudoRandomNumberGenerator<ScalarType> generator;
    source.fill(generator);
    actual.fill(42);

    calculate_expected(source, mapxy, expected);

    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_remap_s16_u8(
                  source.data(), source.stride(), source.width(),
                  source.height(), actual.data(), actual.stride(),
                  actual.width(), actual.height(), channels, mapxy.data(),
                  mapxy.stride(), KLEIDICV_BORDER_TYPE_REPLICATE, {}));

    EXPECT_EQ_ARRAY2D(actual, expected);
  }
  static void calculate_expected(test::Array2D<ScalarType> &src,
                                 test::Array2D<int16_t> &mapxy,
                                 test::Array2D<ScalarType> &expected) {
    for (size_t row = 0; row < expected.height(); row++) {
      for (size_t column = 0; column < expected.width() / src.channels();
           ++column) {
        for (size_t ch = 0; ch < src.channels(); ++ch) {
          int16_t y = std::max<int16_t>(
              0, std::min<int16_t>(src.height() - 1,
                                   *mapxy.at(row, column * 2 + 1)));
          int16_t x = std::max<int16_t>(
              0,
              std::min<int16_t>(src.width() - 1, *mapxy.at(row, column * 2)));
          *expected.at(row, column * src.channels() + ch) =
              *src.at(y, x * src.channels() + ch);
        }
      }
    }
  }
};

using RemapElementTypes = ::testing::Types<uint8_t>;
TYPED_TEST_SUITE(RemapS16, RemapElementTypes);

TYPED_TEST(RemapS16, RandomNoPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test_random(src_w, src_h, dst_w, dst_h, 1, 0);
}

TYPED_TEST(RemapS16, OutsideRandomPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test_outside_random(src_w, src_h, dst_w, dst_h, 1, 13);
}

TYPED_TEST(RemapS16, BlendPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test_blend(src_w, src_h, dst_w, dst_h, 1, 13);
}

TYPED_TEST(RemapS16, CornerCases) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test_corner_cases(src_w, src_h, dst_w, dst_h, 1, 17);
}

TYPED_TEST(RemapS16, NullPointer) {
  const TypeParam src[4] = {};
  TypeParam dst[1];
  int16_t mapxy[2] = {};
  test::test_null_args(kleidicv_remap_s16_u8, src, 2, 2, 2, dst, 1, 1, 1, 1,
                       mapxy, 4, KLEIDICV_BORDER_TYPE_REPLICATE,
                       kleidicv_border_values_t{});
}

TYPED_TEST(RemapS16, ZeroImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[1];
  int16_t mapxy[2] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_remap_s16_u8(src, 1, 0, 1, dst, 1, 0, 1, 1, mapxy, 4,
                                  KLEIDICV_BORDER_TYPE_REPLICATE,
                                  kleidicv_border_values_t{}));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_remap_s16_u8(src, 1, 1, 0, dst, 1, 1, 0, 1, mapxy, 4,
                                  KLEIDICV_BORDER_TYPE_REPLICATE,
                                  kleidicv_border_values_t{}));
}

TYPED_TEST(RemapS16, InvalidImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};

  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_remap_s16_u8(src, 1, KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, dst, 8, 8,
                            1, 1, mapxy, 4, KLEIDICV_BORDER_TYPE_REPLICATE,
                            kleidicv_border_values_t{}));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_remap_s16_u8(src, 1, KLEIDICV_MAX_IMAGE_PIXELS,
                                  KLEIDICV_MAX_IMAGE_PIXELS, dst, 8, 8, 1, 1,
                                  mapxy, 4, KLEIDICV_BORDER_TYPE_REPLICATE,
                                  kleidicv_border_values_t{}));

  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_remap_s16_u8(src, 1, 1, 1, dst, 1, KLEIDICV_MAX_IMAGE_PIXELS + 1,
                            1, 1, mapxy, 4, KLEIDICV_BORDER_TYPE_REPLICATE,
                            kleidicv_border_values_t{}));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_remap_s16_u8(
                src, 1, 1, 1, dst, 1, KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS, 1, mapxy, 4,
                KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{}));
}

TYPED_TEST(RemapS16, UnsupportedTwoChannels) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_remap_s16_u8(src, 1, 1, 1, dst, 8, 8, 1, 2, mapxy, 4,
                                  KLEIDICV_BORDER_TYPE_REPLICATE,
                                  kleidicv_border_values_t{}));
}

TYPED_TEST(RemapS16, UnsupportedBorderTypeConst) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_remap_s16_u8(src, 1, 1, 1, dst, 8, 8, 1, 1, mapxy, 4,
                                  KLEIDICV_BORDER_TYPE_CONSTANT,
                                  kleidicv_border_values_t{}));
}

TYPED_TEST(RemapS16, UnsupportedTooSmallImage) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_remap_s16_u8(src, 1, 1, 1, dst, 8, 7, 1, 1, mapxy, 4,
                                  KLEIDICV_BORDER_TYPE_REPLICATE,
                                  kleidicv_border_values_t{}));
}

template <class ScalarType>
class RemapS16Point5 : public testing::Test {
 public:
  static const uint16_t FRAC_BITS = 5;
  static const uint16_t FRAC_MAX = 1 << FRAC_BITS;
  static const uint16_t FRAC_MAX_SQUARE = FRAC_MAX * FRAC_MAX;

  static void test_random(size_t src_w, size_t src_h, size_t dst_w,
                          size_t dst_h, size_t channels, size_t padding) {
    test::Array2D<int16_t> mapxy(2 * dst_w, dst_h, padding, 2);
    test::PseudoRandomNumberGenerator<int16_t> coord_generator;
    mapxy.fill(coord_generator);
    test::Array2D<uint16_t> mapfrac(dst_w, dst_h, padding);
    test::PseudoRandomNumberGenerator<uint16_t> frac_generator;
    mapfrac.fill(frac_generator);
    execute_test(mapxy, mapfrac, src_w, src_h, dst_w, dst_h, channels, padding);
  }

  static void test_outside_random(size_t src_w, size_t src_h, size_t dst_w,
                                  size_t dst_h, size_t channels,
                                  size_t padding) {
    test::Array2D<int16_t> mapxy(2 * dst_w, dst_h, padding, 2);
    test::PseudoRandomNumberGeneratorIntRange<int16_t> coord_generator{
        static_cast<int16_t>(-src_w), static_cast<int16_t>(2 * src_w)};
    mapxy.fill(coord_generator);
    test::Array2D<uint16_t> mapfrac(dst_w, dst_h, padding);
    test::PseudoRandomNumberGeneratorIntRange<uint16_t> frac_generator(
        0, static_cast<uint16_t>(FRAC_MAX_SQUARE * 3 / 2));
    mapfrac.fill(frac_generator);
    execute_test(mapxy, mapfrac, src_w, src_h, dst_w, dst_h, channels, padding);
  }

  static void test_blend(size_t src_w, size_t src_h, size_t dst_w, size_t dst_h,
                         size_t channels, size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    test::Array2D<uint16_t> mapfrac(dst_w, dst_h, padding);
    for (size_t row = 0; row < dst_h; ++row) {
      for (size_t column = 0; column < dst_w; ++column) {
        // Use a second degree function to add a nonlinear blend to the image
        auto r = static_cast<double>(row), c = static_cast<double>(column),
             w = static_cast<double>(dst_w), h = static_cast<double>(dst_h);
        double x = c * 2 - c * c / w;
        double y = r * (w - c) / w + 4 * r / h;
        *mapxy.at(row, column * 2) = std::max<int16_t>(
            0,
            std::min(static_cast<int16_t>(src_w - 1), static_cast<int16_t>(x)));
        *mapxy.at(row, column * 2 + 1) = std::max<int16_t>(
            0,
            std::min(static_cast<int16_t>(src_h - 1), static_cast<int16_t>(y)));
        *mapfrac.at(row, column) =
            static_cast<uint16_t>(FRAC_MAX * (x - static_cast<int16_t>(x))) |
            (static_cast<uint16_t>(FRAC_MAX * (y - static_cast<int16_t>(y)))
             << FRAC_BITS);
      }
    }
    execute_test(mapxy, mapfrac, src_w, src_h, dst_w, dst_h, channels, padding);
  }

  // Test coordinates with edge values that may easily overflow
  static void test_corner_cases(size_t src_w, size_t src_h, size_t dst_w,
                                size_t dst_h, size_t channels, size_t padding) {
    test::Array2D<int16_t> mapxy{2 * dst_w, dst_h, padding, 2};
    test::Array2D<uint16_t> mapfrac(dst_w, dst_h, padding);
    // One more y than x so we'll see many combinations
    const int16_t corner_x_values[] = {-32768,
                                       -1,
                                       0,
                                       static_cast<int16_t>(src_w - 2),
                                       static_cast<int16_t>(src_w - 1),
                                       static_cast<int16_t>(src_w),
                                       32767};
    const int16_t corner_y_values[] = {-32768,
                                       -1,
                                       0,
                                       1,
                                       static_cast<int16_t>(src_h - 2),
                                       static_cast<int16_t>(src_h - 1),
                                       static_cast<int16_t>(src_h),
                                       32767};
    static const size_t nx = sizeof(corner_x_values) / sizeof(int16_t);
    static const size_t ny = sizeof(corner_y_values) / sizeof(int16_t);
    size_t counter = 0;
    for (size_t row = 0; row < dst_h; ++row) {
      for (size_t column = 0; column < dst_w; ++column) {
        *mapxy.at(row, column * 2) = corner_x_values[counter % nx];
        *mapxy.at(row, column * 2 + 1) = corner_y_values[counter % ny];
        *mapfrac.at(row, column) =
            static_cast<uint16_t>((counter * 3) % FRAC_MAX) |
            (static_cast<uint16_t>((counter * 5) % FRAC_MAX) << FRAC_BITS);
      }
    }
    execute_test(mapxy, mapfrac, src_w, src_h, dst_w, dst_h, channels, padding);
  }

 private:
  static void execute_test(test::Array2D<int16_t> &mapxy,
                           test::Array2D<uint16_t> &mapfrac, size_t src_w,
                           size_t src_h, size_t dst_w, size_t dst_h,
                           size_t channels, size_t padding) {
    size_t src_total_width = channels * src_w;
    size_t dst_total_width = channels * dst_w;

    test::Array2D<ScalarType> source{src_total_width, src_h, padding, channels};
    test::Array2D<ScalarType> actual{dst_total_width, dst_h, padding, channels};
    test::Array2D<ScalarType> expected{dst_total_width, dst_h, padding,
                                       channels};
    test::PseudoRandomNumberGenerator<ScalarType> generator;
    source.fill(generator);
    actual.fill(42);

    calculate_expected(source, mapxy, mapfrac, expected);

    ASSERT_EQ(
        KLEIDICV_OK,
        kleidicv_remap_s16point5_u8(
            source.data(), source.stride(), source.width(), source.height(),
            actual.data(), actual.stride(), actual.width(), actual.height(),
            channels, mapxy.data(), mapxy.stride(), mapfrac.data(),
            mapfrac.stride(), KLEIDICV_BORDER_TYPE_REPLICATE, {}));

    EXPECT_EQ_ARRAY2D(actual, expected);
  }

  static ScalarType lerp2d(size_t cx, size_t cy, ScalarType a, ScalarType b,
                           ScalarType c, ScalarType d) {
    size_t inv_cx = FRAC_MAX - cx, inv_cy = FRAC_MAX - cy;
    ScalarType r = static_cast<ScalarType>((inv_cx * inv_cy * a +
                                            cx * inv_cy * b + inv_cx * cy * c +
                                            cx * cy * d + FRAC_MAX_SQUARE / 2) /
                                           FRAC_MAX_SQUARE);
    return r;
  }

  static void calculate_expected(test::Array2D<ScalarType> &src,
                                 test::Array2D<int16_t> &mapxy,
                                 test::Array2D<uint16_t> &mapfrac,
                                 test::Array2D<ScalarType> &expected) {
    for (size_t row = 0; row < expected.height(); row++) {
      for (size_t column = 0; column < expected.width() / src.channels();
           ++column) {
        for (size_t ch = 0; ch < src.channels(); ++ch) {
          // Clang-tidy thinks mapfrac may contain garbage, but it is fully
          // initialized at all code paths and the map size always equals dst
          // map pixel size
          // NOLINTBEGIN(clang-analyzer-core.UndefinedBinaryOperatorResult)
          uint8_t x_frac = *mapfrac.at(row, column) & (FRAC_MAX - 1);
          uint8_t y_frac =
              (*mapfrac.at(row, column) >> FRAC_BITS) & (FRAC_MAX - 1);
          // NOLINTEND(clang-analyzer-core.UndefinedBinaryOperatorResult)
          int16_t x0 =
              std::min<int16_t>(src.width() - 1, *mapxy.at(row, column * 2));
          int16_t y0 = std::min<int16_t>(src.height() - 1,
                                         *mapxy.at(row, column * 2 + 1));
          if (x0 < 0) {
            x0 = 0, x_frac = 0;
          }
          if (y0 < 0) {
            y0 = 0, y_frac = 0;
          }
          int16_t x1 = static_cast<size_t>(x0) >= src.width() - 1 ? x0 : x0 + 1;
          int16_t y1 =
              static_cast<size_t>(y0) >= src.height() - 1 ? y0 : y0 + 1;
          *expected.at(row, column * src.channels() + ch) =
              lerp2d(x_frac, y_frac, *src.at(y0, x0 * src.channels() + ch),
                     *src.at(y0, x1 * src.channels() + ch),
                     *src.at(y1, x0 * src.channels() + ch),
                     *src.at(y1, x1 * src.channels() + ch));
        }
      }
    }
  }
};

using RemapS16Point5ElementTypes = ::testing::Types<uint8_t>;
TYPED_TEST_SUITE(RemapS16Point5, RemapS16Point5ElementTypes);

TYPED_TEST(RemapS16Point5, RandomNoPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test_random(src_w, src_h, dst_w, dst_h, 1, 0);
}

TYPED_TEST(RemapS16Point5, BlendPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test_blend(src_w, src_h, dst_w, dst_h, 1, 13);
}

TYPED_TEST(RemapS16Point5, OutsideRandomPadding) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test_outside_random(src_w, src_h, dst_w, dst_h, 1, 13);
}

TYPED_TEST(RemapS16Point5, CornerCases) {
  size_t src_w = 3 * test::Options::vector_lanes<TypeParam>() - 1;
  size_t src_h = 4;
  size_t dst_w = src_w;
  size_t dst_h = src_h;
  TestFixture::test_corner_cases(src_w, src_h, dst_w, dst_h, 1, 17);
}

TYPED_TEST(RemapS16Point5, NullPointer) {
  const TypeParam src[4] = {};
  TypeParam dst[1];
  int16_t mapxy[2] = {};
  uint16_t mapfrac[1] = {};
  test::test_null_args(kleidicv_remap_s16point5_u8, src, 2, 2, 2, dst, 1, 1, 1,
                       1, mapxy, 4, mapfrac, 2, KLEIDICV_BORDER_TYPE_REPLICATE,
                       kleidicv_border_values_t{});
}

TYPED_TEST(RemapS16Point5, ZeroImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[1];
  int16_t mapxy[2] = {};
  uint16_t mapfrac[1] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_remap_s16point5_u8(
                src, 1, 0, 1, dst, 1, 0, 1, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{}));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_remap_s16point5_u8(
                src, 1, 1, 0, dst, 1, 1, 0, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{}));
}

TYPED_TEST(RemapS16Point5, InvalidImageSize) {
  const TypeParam src[1] = {};
  TypeParam dst[1];
  int16_t mapxy[2] = {};
  uint16_t mapfrac[1] = {};

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_remap_s16point5_u8(src, 1, KLEIDICV_MAX_IMAGE_PIXELS + 1,
                                        1, dst, 1, 1, 1, 1, mapxy, 4, mapfrac,
                                        2, KLEIDICV_BORDER_TYPE_REPLICATE,
                                        kleidicv_border_values_t{}));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_remap_s16point5_u8(
                src, 1, KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS,
                dst, 1, 1, 1, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{}));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_remap_s16point5_u8(
                src, 1, 1, 1, dst, 1, KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 1,
                mapxy, 4, mapfrac, 2, KLEIDICV_BORDER_TYPE_REPLICATE,
                kleidicv_border_values_t{}));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_remap_s16point5_u8(
                src, 1, 1, 1, dst, 1, KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{}));
}

TYPED_TEST(RemapS16Point5, UnsupportedTwoChannels) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};
  uint16_t mapfrac[8] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_remap_s16point5_u8(
                src, 1, 1, 1, dst, 8, 8, 1, 2, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{}));
}

TYPED_TEST(RemapS16Point5, UnsupportedBorderTypeConst) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};
  uint16_t mapfrac[8] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_remap_s16point5_u8(
                src, 1, 1, 1, dst, 8, 8, 1, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_CONSTANT, kleidicv_border_values_t{}));
}

TYPED_TEST(RemapS16Point5, UnsupportedTooSmallImage) {
  const TypeParam src[1] = {};
  TypeParam dst[8];
  int16_t mapxy[16] = {};
  uint16_t mapfrac[8] = {};

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_remap_s16point5_u8(
                src, 1, 1, 1, dst, 8, 7, 1, 1, mapxy, 4, mapfrac, 2,
                KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{}));
}
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
