// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/kernel.h"
#include "framework/types.h"
#include "framework/utils.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

// Test for Scharr operator with interleaved output.
class ScharrInterleavedTest {
  using InputType = uint8_t;
  using IntermediateType = int16_t;
  using OutputType = int16_t;

 public:
  ScharrInterleavedTest()
      : vertical_mask_{create_vertical_mask()},
        horizontal_mask_{create_horizontal_mask()} {}

  void test() {
    // Use the default array layouts with one channel for testing.
    for (auto layout : test::default_1channel_array_layouts(3, 3)) {
      test(layout);
    }
  }

 private:
  static test::Array2D<IntermediateType> create_vertical_mask() {
    test::Array2D<IntermediateType> mask{3, 3};

    mask.set(0, 0, {-3, -10, -3});
    mask.set(1, 0, {0, 0, 0});
    mask.set(2, 0, {3, 10, 3});

    return mask;
  }

  static test::Array2D<IntermediateType> create_horizontal_mask() {
    test::Array2D<IntermediateType> mask{3, 3};

    mask.set(0, 0, {-3, 0, 3});
    mask.set(1, 0, {-10, 0, 10});
    mask.set(2, 0, {-3, 0, 3});

    return mask;
  }

  void test(const test::ArrayLayout& layout) {
    // Input
    input_ = test::Array2D<InputType>{layout};
    ASSERT_TRUE(input_.valid());
    test::PseudoRandomNumberGenerator<InputType> element_generator;
    input_.fill(element_generator);

    // Output has less rows and columns as borders are not handled in the same
    // way as in case of most of the other filter operations.
    test::ArrayLayout output_layout{(layout.width - 2) * 2, layout.height - 2,
                                    layout.padding, layout.channels * 2};

    // Expected
    expected_ = test::Array2D<OutputType>{output_layout};
    ASSERT_TRUE(expected_.valid());
    calculate_expected();

    // Actual
    actual_ = test::Array2D<OutputType>{output_layout};
    ASSERT_TRUE(actual_.valid());
    EXPECT_EQ(KLEIDICV_OK,
              kleidicv_scharr_interleaved_s16_u8(
                  input_.data(), input_.stride(),
                  input_.width() / input_.channels(), input_.height(),
                  input_.channels(), actual_.data(), actual_.stride()));

    // Check results
    EXPECT_EQ_ARRAY2D(expected_, actual_);
  }

  void calculate_expected() {
    for (size_t row = 0; row < expected_.height(); ++row) {
      for (size_t column = 0; column < expected_.width(); column += 2) {
        IntermediateType horizontal_result =
            calculate_expected_at(horizontal_mask_, input_, row, column / 2);
        expected_.at(row, column)[0] =
            static_cast<OutputType>(horizontal_result);

        IntermediateType vertical_result =
            calculate_expected_at(vertical_mask_, input_, row, column / 2);
        expected_.at(row, column)[1] = static_cast<OutputType>(vertical_result);
      }
    }
  }

  IntermediateType calculate_expected_at(
      const test::Kernel<IntermediateType>& kernel,
      const test::TwoDimensional<InputType>& source, size_t row,
      size_t column) {
    IntermediateType result{0};
    for (size_t height = 0; height < kernel.height(); ++height) {
      for (size_t width = 0; width < kernel.width(); ++width) {
        IntermediateType coefficient = kernel.at(height, width)[0];
        InputType value =
            source.at(row + height, column + width * source.channels())[0];
        result = test::saturating_add(
            result, test::saturating_mul(coefficient,
                                         static_cast<IntermediateType>(value)));
      }
    }

    return result;
  }

  const test::Kernel<IntermediateType> vertical_mask_;
  const test::Kernel<IntermediateType> horizontal_mask_;

  test::Array2D<InputType> input_;
  test::Array2D<OutputType> expected_;
  test::Array2D<OutputType> actual_;
};  // end of class ScharrInterleavedTest

// Tests kleidicv_scharr_interleaved_s16_u8 API.
TEST(ScharrInterleaved, API) { ScharrInterleavedTest{}.test(); }

TEST(ScharrInterleaved, NullPointer) {
  uint8_t src[1] = {};
  int16_t dst[1];
  test::test_null_args(kleidicv_scharr_interleaved_s16_u8, src, sizeof(uint8_t),
                       3, 3, 1, dst, sizeof(int16_t));
}

TEST(ScharrInterleaved, Misalignment) {
  uint8_t src[1] = {};
  int16_t dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            kleidicv_scharr_interleaved_s16_u8(src, sizeof(uint8_t), 3, 4, 1,
                                               dst, 1));
}

TEST(ScharrInterleaved, UndersizedImage) {
  uint8_t src[1] = {};
  int16_t dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_scharr_interleaved_s16_u8(src, sizeof(uint8_t), 1, 1, 1,
                                               dst, sizeof(int16_t)));

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_scharr_interleaved_s16_u8(src, sizeof(uint8_t), 3, 1, 1,
                                               dst, sizeof(int16_t)));
}

TEST(ScharrInterleaved, OversizedImage) {
  uint8_t src[1] = {};
  int16_t dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_scharr_interleaved_s16_u8(src, sizeof(uint8_t),
                                               KLEIDICV_MAX_IMAGE_PIXELS + 1, 3,
                                               1, dst, sizeof(int16_t)));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_scharr_interleaved_s16_u8(
                src, sizeof(uint8_t), KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS, 1, dst, sizeof(int16_t)));
}

TEST(ScharrInterleaved, ChannelNumber) {
  uint8_t src[1] = {};
  int16_t dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_scharr_interleaved_s16_u8(src, sizeof(uint8_t), 1, 1, 2,
                                               dst, sizeof(int16_t)));
}

#ifdef KLEIDICV_ALLOCATION_TESTS
TEST(ScharrInterleaved, Allocation) {
  uint8_t src[1] = {};
  int16_t dst[1];
  MockMallocToFail::enable();
  kleidicv_error_t ret = kleidicv_scharr_interleaved_s16_u8(
      src, sizeof(uint8_t), 3, 3, 1, dst, sizeof(int16_t));
  MockMallocToFail::disable();

  EXPECT_EQ(KLEIDICV_ERROR_ALLOCATION, ret);
}
#endif
