// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

class GrayTest final {
 public:
  explicit GrayTest(bool hasAlpha)
      : hasAlpha_(hasAlpha), outChannels_{hasAlpha ? 4U : 3U} {}

  // Sets the number of padding bytes at the end of rows.
  GrayTest &with_padding(size_t padding) {
    padding_ = padding;
    return *this;
  }

  // Sets the number of elementsin a row.
  GrayTest &with_width(size_t width) {
    width_ = width;
    return *this;
  }

  template <typename F>
  void execute_test(F impl) {
    // Set width to be more than vector length, but not quite a multiple of
    // vector length to force both vector and scalar paths
    test::Array2D<uint8_t> source{width_, 3, padding_};
    test::Array2D<uint8_t> actual{width_ * outChannels_, 3, padding_};
    test::Array2D<uint8_t> expected{width_ * outChannels_, 3, padding_};

    source.set(0, 0, {1});
    source.set(1, 0, {0xFF});
    source.set(2, 0, {10, 15, 1});
    actual.fill(42);
    calculate_expected(source, expected);

    auto err = impl(source.data(), source.stride(), actual.data(),
                    actual.stride(), width_, actual.height());

    ASSERT_EQ(KLEIDICV_OK, err);
    EXPECT_EQ_ARRAY2D(actual, expected);

    test::test_null_args(impl, source.data(), source.stride(), actual.data(),
                         actual.stride(), width_, actual.height());

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(source.data(), source.stride(), actual.data(),
                   actual.stride(), KLEIDICV_MAX_IMAGE_PIXELS + 1, 1));
    EXPECT_EQ(
        KLEIDICV_ERROR_RANGE,
        impl(source.data(), source.stride(), actual.data(), actual.stride(),
             KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS));
  }

 private:
  void calculate_expected(test::Array2D<uint8_t> &src,
                          test::Array2D<uint8_t> &expected) const {
    for (size_t vindex = 0; vindex < expected.height(); vindex++) {
      for (size_t hindex = 0; hindex < expected.width() / outChannels_;
           hindex++) {
        // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
        uint8_t y = *src.at(vindex, hindex);
        // NOLINTEND(clang-analyzer-core.uninitialized.Assign)

        if (hasAlpha_) {
          expected.set(vindex, hindex * outChannels_, {y, y, y, 0xff});
        } else {
          expected.set(vindex, hindex * outChannels_, {y, y, y});
        }
      }
    }
  }

  size_t width_{3 * test::Options::vector_lanes<uint8_t>() - 1};
  size_t padding_{0};
  bool hasAlpha_;
  size_t outChannels_;
};

class ColorTest final {
 public:
  ColorTest(size_t src_channels, size_t dst_channels, bool swapBlue)
      : inChannels_(src_channels),
        outChannels_(dst_channels),
        swapBlue_(swapBlue) {}

  template <typename F>
  void execute_test(F impl) {
    // Set width to be more than vector length, but not quite a multiple of
    // vector length to force both vector and scalar paths
    size_t logical_width = 3 * test::Options::vector_lanes<uint8_t>() - 1;
    test::Array2D<uint8_t> source{logical_width * inChannels_, 3};
    test::Array2D<uint8_t> actual{logical_width * outChannels_, 3};
    test::Array2D<uint8_t> expected{logical_width * outChannels_, 3};

    source.set(0, 0, {123, 230, 11, 203});
    source.set(1, 0, {0xFF, 0xFF, 0xFF, 0});
    source.set(2, 0, {0, 0, 0, 0xFF});

    calculate_expected(source, expected);

    auto err = impl(source.data(), source.stride(), actual.data(),
                    actual.stride(), logical_width, actual.height());

    ASSERT_EQ(KLEIDICV_OK, err);
    EXPECT_EQ_ARRAY2D(actual, expected);

    test::test_null_args(impl, source.data(), source.stride(), actual.data(),
                         actual.stride(), logical_width, actual.height());

    EXPECT_EQ(KLEIDICV_OK, impl(source.data(), source.stride(), actual.data(),
                                actual.stride(), 0, 1));
    EXPECT_EQ(KLEIDICV_OK, impl(source.data(), source.stride(), actual.data(),
                                actual.stride(), 1, 0));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(source.data(), source.stride(), actual.data(),
                   actual.stride(), KLEIDICV_MAX_IMAGE_PIXELS + 1, 1));
    EXPECT_EQ(
        KLEIDICV_ERROR_RANGE,
        impl(source.data(), source.stride(), actual.data(), actual.stride(),
             KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS));
  }

 private:
  void calculate_expected(test::Array2D<uint8_t> &src,
                          test::Array2D<uint8_t> &expected) const {
    for (size_t vindex = 0; vindex < expected.height(); vindex++) {
      for (size_t hindex = 0; hindex < expected.width() / outChannels_;
           hindex++) {
        // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
        uint8_t r = *src.at(vindex, hindex * inChannels_);
        uint8_t g = *src.at(vindex, hindex * inChannels_ + 1);
        uint8_t b = *src.at(vindex, hindex * inChannels_ + 2);
        // NOLINTEND(clang-analyzer-core.uninitialized.Assign)
        uint8_t a = 0xff;
        if (inChannels_ == 4) {
          a = *src.at(vindex, hindex * inChannels_ + 3);
        }

        if (outChannels_ == 4) {
          if (swapBlue_) {
            expected.set(vindex, hindex * outChannels_, {b, g, r, a});
          } else {
            expected.set(vindex, hindex * outChannels_, {r, g, b, a});
          }
        } else {
          if (swapBlue_) {
            expected.set(vindex, hindex * outChannels_, {b, g, r});
          } else {
            expected.set(vindex, hindex * outChannels_, {r, g, b});
          }
        }
      }
    }
  }

  size_t inChannels_;
  size_t outChannels_;
  bool swapBlue_;
};

TEST(GRAY2, RGB) {
  GrayTest{false}.execute_test(kleidicv_gray_to_rgb_u8);
  GrayTest{false}.with_padding(1).execute_test(kleidicv_gray_to_rgb_u8);
  GrayTest{false}
      .with_width(2 * test::Options::vector_lanes<uint8_t>())
      .execute_test(kleidicv_gray_to_rgb_u8);
}

TEST(GRAY2, RGBA) {
  GrayTest{true}.execute_test(kleidicv_gray_to_rgba_u8);
  GrayTest{true}.with_padding(1).execute_test(kleidicv_gray_to_rgba_u8);
  GrayTest{true}
      .with_width(2 * test::Options::vector_lanes<uint8_t>())
      .execute_test(kleidicv_gray_to_rgba_u8);
}

TEST(RGB2, RGB) {
  ColorTest color_test(3, 3, false);
  color_test.execute_test(kleidicv_rgb_to_rgb_u8);
}

TEST(RGBA2, RGBA) {
  ColorTest color_test(4, 4, false);
  color_test.execute_test(kleidicv_rgba_to_rgba_u8);
}

TEST(RGB2, BGR) {
  ColorTest color_test(3, 3, true);
  color_test.execute_test(kleidicv_rgb_to_bgr_u8);
}

TEST(RGBA2, BGRA) {
  ColorTest color_test(4, 4, true);
  color_test.execute_test(kleidicv_rgba_to_bgra_u8);
}

TEST(RGB2, BGRA) {
  ColorTest color_test(3, 4, true);
  color_test.execute_test(kleidicv_rgb_to_bgra_u8);
}

TEST(RGB2, RGBA) {
  ColorTest color_test(3, 4, false);
  color_test.execute_test(kleidicv_rgb_to_rgba_u8);
}

TEST(RGBA2, BGR) {
  ColorTest color_test(4, 3, true);
  color_test.execute_test(kleidicv_rgba_to_bgr_u8);
}

TEST(RGBA2, RGB) {
  ColorTest color_test(4, 3, false);
  color_test.execute_test(kleidicv_rgba_to_rgb_u8);
}
