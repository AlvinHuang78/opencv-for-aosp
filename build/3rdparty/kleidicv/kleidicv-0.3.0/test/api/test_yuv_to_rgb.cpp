// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/utils.h"
#include "test_config.h"

class YuvToRgbTest final {
 public:
  explicit YuvToRgbTest(bool switch_blue) : switch_blue_(switch_blue) {}

  template <typename F>
  void execute_scalar_test(F impl) {
    size_t scalar_path_width = 5;
    execute_test(impl, scalar_path_width, 0);
    // Padding version
    execute_test(impl, scalar_path_width, test::Options::vector_length());
  }

  template <typename F>
  void execute_vector_test(F impl) {
    size_t vector_path_width = (2 * test::Options::vector_lanes<uint8_t>()) - 3;
    execute_test(impl, vector_path_width, 0);
    // Padding version
    execute_test(impl, vector_path_width, test::Options::vector_length());
  }

 private:
  static const size_t kDstChannels = 3;
  bool switch_blue_;

  template <typename F>
  void execute_test(F impl, size_t logical_width, size_t padding) {
    test::Array2D<uint8_t> src{logical_width * 3, 5, padding};
    src.fill(0);
    src.set(0, 0, {10, 100, 130, 20, 100, 130, 255, 255, 255, 199, 255, 255});
    src.set(1, 0, {1, 100, 130, 120, 100, 130, 0, 255, 255, 17, 255, 255});
    src.set(2, 0, {2, 0, 1, 3, 0, 1, 240, 3, 4, 228, 3, 4});
    src.set(4, 0, {7, 7, 8, 11, 7, 8, 128, 9, 10, 129, 9, 10});

    test::Array2D<uint8_t> expected{logical_width * kDstChannels, src.height(),
                                    padding};
    expected.fill(0);
    calculate_expected(src, expected);

    test::Array2D<uint8_t> actual{logical_width * kDstChannels, src.height(),
                                  padding};
    actual.fill(42);
    auto err = impl(src.data(), src.stride(), actual.data(), actual.stride(),
                    logical_width, expected.height());

    ASSERT_EQ(KLEIDICV_OK, err);
    EXPECT_EQ_ARRAY2D(expected, actual);

    test::test_null_args(impl, src.data(), src.stride(), actual.data(),
                         actual.stride(), logical_width, expected.height());

    EXPECT_EQ(KLEIDICV_OK, impl(src.data(), src.stride(), actual.data(),
                                actual.stride(), 0, 1));
    EXPECT_EQ(KLEIDICV_OK, impl(src.data(), src.stride(), actual.data(),
                                actual.stride(), 1, 0));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), actual.data(), actual.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS + 1, 1));
    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(src.data(), src.stride(), actual.data(), actual.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS));
  }

  static uint8_t saturate_cast_s32_to_u8(int32_t rhs) {
    return static_cast<uint8_t>(
        std::min(std::max(0, rhs),
                 static_cast<int32_t>(std::numeric_limits<uint8_t>::max())));
  }

  void calculate_expected(test::Array2D<uint8_t> &src_arr,
                          test::Array2D<uint8_t> &exp_arr) const {
    for (size_t vindex = 0; vindex < exp_arr.height(); vindex++) {
      for (size_t hindex = 0; hindex < exp_arr.width() / kDstChannels;
           hindex++) {
        // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
        int32_t y = *src_arr.at(vindex, hindex * 3);
        int32_t u = *src_arr.at(vindex, hindex * 3 + 1);
        int32_t v = *src_arr.at(vindex, hindex * 3 + 2);
        // NOLINTEND(clang-analyzer-core.uninitialized.Assign)

        static const int32_t U2B = 33292, U2G = -6472, V2G = -9519, V2R = 18678;

        int32_t b = y + (((u - 128) * U2B + (1 << 13)) >> 14);
        int32_t g = y + (((u - 128) * U2G + (v - 128) * V2G + (1 << 13)) >> 14);
        int32_t r = y + (((v - 128) * V2R + (1 << 13)) >> 14);

        uint8_t c0_u8 = saturate_cast_s32_to_u8(switch_blue_ ? b : r);
        uint8_t c1_u8 = saturate_cast_s32_to_u8(g);
        uint8_t c2_u8 = saturate_cast_s32_to_u8(switch_blue_ ? r : b);

        exp_arr.set(vindex, hindex * kDstChannels, {c0_u8, c1_u8, c2_u8});
      }
    }
  }
};

TEST(YuvToRgb, YuvRgbScalar) {
  YuvToRgbTest yuv2rgb_test(false);
  yuv2rgb_test.execute_scalar_test(kleidicv_yuv_to_rgb_u8);
}

TEST(YuvToRgb, YuvRgbVector) {
  YuvToRgbTest yuv2rgb_test(false);
  yuv2rgb_test.execute_vector_test(kleidicv_yuv_to_rgb_u8);
}

TEST(YuvToRgb, YuvBgrScalar) {
  YuvToRgbTest yuv2rgb_test(true);
  yuv2rgb_test.execute_scalar_test(kleidicv_yuv_to_bgr_u8);
}

TEST(YuvToRgb, YuvBgrVector) {
  YuvToRgbTest yuv2rgb_test(true);
  yuv2rgb_test.execute_vector_test(kleidicv_yuv_to_bgr_u8);
}
