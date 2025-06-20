// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

class ResizeToQuarterTest final {
 public:
  explicit ResizeToQuarterTest(size_t channels, size_t padding)
      : channels_(channels), padding_(padding) {}

  void execute_test(size_t src_width, size_t src_height, size_t dst_width,
                    size_t dst_height) const {
    size_t src_logical_width = channels_ * src_width;
    size_t out_logical_width = channels_ * dst_width;

    test::Array2D<uint8_t> source{src_logical_width, src_height, padding_,
                                  channels_};
    test::Array2D<uint8_t> actual{out_logical_width, dst_height, padding_,
                                  channels_};
    actual.fill(42);
    test::Array2D<uint8_t> expected{out_logical_width, dst_height, padding_,
                                    channels_};

    // Set bottom-row and right-column to 0xFF, the rest to element position in
    // buffer
    for (size_t h = 0; h < src_height; ++h) {
      for (size_t w = 0; w < src_logical_width; ++w) {
        if (h >= src_height - 1 || w >= src_logical_width - channels_) {
          *source.at(h, w) = 0xFF;
        } else {
          *source.at(h, w) = (h * src_logical_width + w) % 0xFF;
        }
      }
    }

    calculate_expected(source, expected);

    ASSERT_EQ(KLEIDICV_OK, kleidicv_resize_to_quarter_u8(
                               source.data(), source.stride(), source.width(),
                               source.height(), actual.data(), actual.stride(),
                               actual.width(), actual.height()));

    EXPECT_EQ_ARRAY2D(actual, expected);
  }

 private:
  // Calculates average value of 2x2 window with respect to array boundaries.
  void fill_from_window(test::Array2D<uint8_t> &src,
                        test::Array2D<uint8_t> &dst, size_t src_row,
                        size_t src_col, size_t dst_row, size_t dst_col) const {
    for (size_t channel = 0; channel < channels_; ++channel) {
      uint16_t value = 0;
      uint8_t divider = 1;

      // Top-left corner, always present
      value = *src.at(src_row, src_col + channel);
      // Top-right corner
      if (src_col + channels_ < src.width()) {
        value += *src.at(src_row, src_col + channels_ + channel);
        divider++;
      }
      // Bottom-left corner
      if (src_row + 1 < src.height()) {
        value += *src.at(src_row + 1, src_col + channel);
        divider++;
      }
      // Bottom-right corner
      if (src_row + 1 < src.height() && src_col + channels_ < src.width()) {
        value += *src.at(src_row + 1, src_col + channels_ + channel);
        divider++;
      }

      // Rounding division
      *dst.at(dst_row, dst_col + channel) =
          static_cast<uint8_t>((value + (divider / 2)) / divider);
    }
  }

  void calculate_expected(test::Array2D<uint8_t> &src,
                          test::Array2D<uint8_t> &expected) const {
    size_t src_vindex = 0;
    size_t src_hindex = 0;

    for (size_t exp_vindex = 0; exp_vindex < expected.height(); exp_vindex++) {
      for (size_t exp_hindex = 0; exp_hindex < expected.width();
           exp_hindex += channels_) {
        fill_from_window(src, expected, src_vindex, src_hindex, exp_vindex,
                         exp_hindex);
        src_hindex += 2 * channels_;
      }
      src_vindex += 2;
      src_hindex = 0;
    }
  }

  size_t channels_;
  size_t padding_;
};

// Test basic dimension cases without padding
TEST(ResizeToQuarter, EvenDimsNoPadding) {
  // For now resize only reasonably supports single-channel data
  ResizeToQuarterTest resize_test(1, 0);

  size_t src_width = 2 * test::Options::vector_lanes<uint8_t>();
  // Set height to at least double 2x2 window height
  size_t src_height = 4;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, OddDimsFullNoPadding) {
  // For now resize only reasonably supports single-channel data
  ResizeToQuarterTest resize_test(1, 0);

  size_t src_width = 2 * test::Options::vector_lanes<uint8_t>() - 1;
  // Set height to at least double 2x2 window height
  size_t src_height = 5;
  size_t dst_width = src_width / 2 + 1;
  size_t dst_height = src_height / 2 + 1;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

// The rest of tests are done with padding since this is a more elaborate case
TEST(ResizeToQuarter, EvenDims) {
  // For now resize only reasonably supports single-channel data
  ResizeToQuarterTest resize_test(1, 1);

  size_t src_width = 2 * test::Options::vector_lanes<uint8_t>();
  // Set height to at least double 2x2 window height
  size_t src_height = 4;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, OddDimsFull) {
  // For now resize only reasonably supports single-channel data
  ResizeToQuarterTest resize_test(1, 1);

  size_t src_width = 2 * test::Options::vector_lanes<uint8_t>() - 1;
  // Set height to at least double 2x2 window height
  size_t src_height = 5;
  size_t dst_width = src_width / 2 + 1;
  size_t dst_height = src_height / 2 + 1;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, EvenHeightOddWidthFull) {
  // For now resize only reasonably supports single-channel data
  ResizeToQuarterTest resize_test(1, 1);

  size_t src_width = 2 * test::Options::vector_lanes<uint8_t>() - 1;
  // Set height to at least double 2x2 window height
  size_t src_height = 4;
  size_t dst_width = src_width / 2 + 1;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, OddHeightEvenWidthFull) {
  // For now resize only reasonably supports single-channel data
  ResizeToQuarterTest resize_test(1, 1);

  size_t src_width = 2 * test::Options::vector_lanes<uint8_t>();
  // Set height to at least double 2x2 window height
  size_t src_height = 5;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2 + 1;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, OddDimsFullHeight) {
  // For now resize only reasonably supports single-channel data
  ResizeToQuarterTest resize_test(1, 1);

  size_t src_width = 2 * test::Options::vector_lanes<uint8_t>() - 1;
  // Set height to at least double 2x2 window height
  size_t src_height = 5;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2 + 1;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, OddDimsFullWidth) {
  // For now resize only reasonably supports single-channel data
  ResizeToQuarterTest resize_test(1, 1);

  size_t src_width = 2 * test::Options::vector_lanes<uint8_t>() - 1;
  // Set height to at least double 2x2 window height
  size_t src_height = 5;
  size_t dst_width = src_width / 2 + 1;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, OddDimsTruncated) {
  // For now resize only reasonably supports single-channel data
  ResizeToQuarterTest resize_test(1, 1);

  size_t src_width = 2 * test::Options::vector_lanes<uint8_t>() - 1;
  // Set height to at least double 2x2 window height
  size_t src_height = 5;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, NullPointer) {
  const uint8_t src[4] = {};
  uint8_t dst[1];
  test::test_null_args(kleidicv_resize_to_quarter_u8, src, 2, 2, 2, dst, 1, 1,
                       1);
}

TEST(ResizeToQuarter, ZeroImageSize) {
  const uint8_t src[1] = {};
  uint8_t dst[1];

  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_resize_to_quarter_u8(src, 1, 0, 1, dst, 1, 0, 1));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_resize_to_quarter_u8(src, 1, 1, 0, dst, 1, 1, 0));
}

TEST(ResizeToQuarter, InvalidImageSize) {
  const uint8_t src[1] = {};
  uint8_t dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_resize_to_quarter_u8(src, 1, KLEIDICV_MAX_IMAGE_PIXELS + 1,
                                          1, dst, 1, 1, 1));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE, kleidicv_resize_to_quarter_u8(
                                      src, 1, KLEIDICV_MAX_IMAGE_PIXELS,
                                      KLEIDICV_MAX_IMAGE_PIXELS, dst, 1, 1, 1));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_resize_to_quarter_u8(src, 1, 8, 8, dst, 1, 4, 3));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_resize_to_quarter_u8(src, 1, 8, 8, dst, 1, 3, 4));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_resize_to_quarter_u8(src, 1, 9, 9, dst, 1, 4, 1));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_resize_to_quarter_u8(src, 1, 9, 9, dst, 1, 7, 5));
}
