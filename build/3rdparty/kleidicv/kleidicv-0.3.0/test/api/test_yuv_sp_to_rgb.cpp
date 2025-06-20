// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/utils.h"
#include "test_config.h"

class YuvSpTest final {
 public:
  YuvSpTest(size_t channel_number, bool switch_blue)
      : channel_number_(channel_number), switch_blue_(switch_blue) {}

  template <typename F>
  void execute_scalar_test(F impl, bool is_nv21) {
    size_t scalar_path_width = 5;
    execute_test(impl, scalar_path_width, is_nv21, 0);
    // Padding version
    execute_test(impl, scalar_path_width, is_nv21,
                 test::Options::vector_length());
  }

  template <typename F>
  void execute_vector_test(F impl, bool is_nv21) {
    size_t vector_path_width = (2 * test::Options::vector_lanes<uint8_t>()) - 3;
    execute_test(impl, vector_path_width, is_nv21, 0);
    // Padding version
    execute_test(impl, vector_path_width, is_nv21,
                 test::Options::vector_length());
  }

 private:
  template <typename F>
  void execute_test(F impl, size_t logical_width, bool is_nv21,
                    size_t padding) {
    test::Array2D<uint8_t> input_y{logical_width, 5, padding};
    input_y.fill(0);
    input_y.set(0, 0, {10, 20, 255, 199});
    input_y.set(1, 0, {1, 120, 0, 17});
    input_y.set(2, 0, {2, 3, 240, 228});
    input_y.set(4, 0, {7, 11, 128, 129});

    // the width of the UV input must be even
    test::Array2D<uint8_t> input_uv{
        KLEIDICV_TARGET_NAMESPACE::align_up(logical_width, 2), 3, padding};
    input_uv.fill(0);
    input_uv.set(0, 0, {100, 130, 255, 255});
    input_uv.set(1, 0, {0, 1, 3, 4});
    input_uv.set(2, 0, {7, 8, 9, 10});

    test::Array2D<uint8_t> expected{logical_width * channel_number_,
                                    input_y.height(), padding};
    expected.fill(0);
    calculate_expected(input_y, input_uv, expected, is_nv21);

    test::Array2D<uint8_t> actual{logical_width * channel_number_,
                                  input_y.height(), padding};
    actual.fill(42);
    auto err =
        impl(input_y.data(), input_y.stride(), input_uv.data(),
             input_uv.stride(), actual.data(), actual.stride(),
             expected.width() / channel_number_, expected.height(), is_nv21);

    ASSERT_EQ(KLEIDICV_OK, err);
    EXPECT_EQ_ARRAY2D(expected, actual);

    test::test_null_args(impl, input_y.data(), input_y.stride(),
                         input_uv.data(), input_uv.stride(), actual.data(),
                         actual.stride(), expected.width() / channel_number_,
                         expected.height(), is_nv21);

    EXPECT_EQ(KLEIDICV_OK, impl(input_y.data(), input_y.stride(),
                                input_uv.data(), input_uv.stride(),
                                actual.data(), actual.stride(), 0, 1, is_nv21));
    EXPECT_EQ(KLEIDICV_OK, impl(input_y.data(), input_y.stride(),
                                input_uv.data(), input_uv.stride(),
                                actual.data(), actual.stride(), 1, 0, is_nv21));

    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              impl(input_y.data(), input_y.stride(), input_uv.data(),
                   input_uv.stride(), actual.data(), actual.stride(),
                   KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, is_nv21));
    EXPECT_EQ(
        KLEIDICV_ERROR_RANGE,
        impl(input_y.data(), input_y.stride(), input_uv.data(),
             input_uv.stride(), actual.data(), actual.stride(),
             KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, is_nv21));
  }

  void calculate_expected(test::Array2D<uint8_t> &y_arr,
                          test::Array2D<uint8_t> &uv_arr,
                          test::Array2D<uint8_t> &exp_arr, bool is_nv21) const {
    for (size_t vindex = 0; vindex < exp_arr.height(); vindex++) {
      for (size_t hindex = 0; hindex < exp_arr.width() / channel_number_;
           hindex++) {
        // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
        int32_t y = *y_arr.at(vindex, hindex);
        // NOLINTEND(clang-analyzer-core.uninitialized.Assign)
        y = std::max(0, y - 16);
        int32_t u =
            *uv_arr.at(vindex / 2, (hindex - hindex % 2) + (is_nv21 ? 1 : 0));
        u -= 128;
        int32_t v =
            *uv_arr.at(vindex / 2, (hindex - hindex % 2) + (is_nv21 ? 0 : 1));
        v -= 128;

        int32_t r = ((1220542 * y) + (1673527 * v) + (1 << 19)) >> 20;
        int32_t g =
            ((1220542 * y) - (409993 * u) - (852492 * v) + (1 << 19)) >> 20;
        int32_t b = ((1220542 * y) + (2116026 * u) + (1 << 19)) >> 20;

        uint8_t r_u8 = saturate_cast_s32_to_u8(r);
        uint8_t g_u8 = saturate_cast_s32_to_u8(g);
        uint8_t b_u8 = saturate_cast_s32_to_u8(b);

        if (switch_blue_) {
          std::swap(b_u8, r_u8);
        }

        if (channel_number_ == 4) {
          exp_arr.set(vindex, hindex * channel_number_,
                      {r_u8, g_u8, b_u8, 0xff});
        } else {
          exp_arr.set(vindex, hindex * channel_number_, {r_u8, g_u8, b_u8});
        }
      }
    }
  }

  static uint8_t saturate_cast_s32_to_u8(int32_t rhs) {
    return static_cast<uint8_t>(
        std::min(std::max(0, rhs),
                 static_cast<int32_t>(std::numeric_limits<uint8_t>::max())));
  }

  size_t channel_number_;
  bool switch_blue_;
};

TEST(YuvSp, RgbNv12Scalar) {
  YuvSpTest yuv_test(3, false);
  yuv_test.execute_scalar_test(kleidicv_yuv_sp_to_rgb_u8, false);
}

TEST(YuvSp, RgbNv12Vector) {
  YuvSpTest yuv_test(3, false);
  yuv_test.execute_vector_test(kleidicv_yuv_sp_to_rgb_u8, false);
}

TEST(YuvSp, RgbNv21Scalar) {
  YuvSpTest yuv_test(3, false);
  yuv_test.execute_scalar_test(kleidicv_yuv_sp_to_rgb_u8, true);
}

TEST(YuvSp, RgbNv21Vector) {
  YuvSpTest yuv_test(3, false);
  yuv_test.execute_vector_test(kleidicv_yuv_sp_to_rgb_u8, true);
}

TEST(YuvSp, BgrNv12Scalar) {
  YuvSpTest yuv_test(3, true);
  yuv_test.execute_scalar_test(kleidicv_yuv_sp_to_bgr_u8, false);
}

TEST(YuvSp, BgrNv12Vector) {
  YuvSpTest yuv_test(3, true);
  yuv_test.execute_vector_test(kleidicv_yuv_sp_to_bgr_u8, false);
}

TEST(YuvSp, BgrNv21Scalar) {
  YuvSpTest yuv_test(3, true);
  yuv_test.execute_scalar_test(kleidicv_yuv_sp_to_bgr_u8, true);
}

TEST(YuvSp, BgrNv21Vector) {
  YuvSpTest yuv_test(3, true);
  yuv_test.execute_vector_test(kleidicv_yuv_sp_to_bgr_u8, true);
}

TEST(YuvSp, RgbaNv12Scalar) {
  YuvSpTest yuv_test(4, false);
  yuv_test.execute_scalar_test(kleidicv_yuv_sp_to_rgba_u8, false);
}

TEST(YuvSp, RgbaNv12Vector) {
  YuvSpTest yuv_test(4, false);
  yuv_test.execute_vector_test(kleidicv_yuv_sp_to_rgba_u8, false);
}

TEST(YuvSp, RgbaNv21Scalar) {
  YuvSpTest yuv_test(4, false);
  yuv_test.execute_scalar_test(kleidicv_yuv_sp_to_rgba_u8, true);
}

TEST(YuvSp, RgbaNv21Vector) {
  YuvSpTest yuv_test(4, false);
  yuv_test.execute_vector_test(kleidicv_yuv_sp_to_rgba_u8, true);
}

TEST(YuvSp, BgraNv12Scalar) {
  YuvSpTest yuv_test(4, true);
  yuv_test.execute_scalar_test(kleidicv_yuv_sp_to_bgra_u8, false);
}

TEST(YuvSp, BgraNv12Vector) {
  YuvSpTest yuv_test(4, true);
  yuv_test.execute_vector_test(kleidicv_yuv_sp_to_bgra_u8, false);
}

TEST(YuvSp, BgraNv21Scalar) {
  YuvSpTest yuv_test(4, true);
  yuv_test.execute_scalar_test(kleidicv_yuv_sp_to_bgra_u8, true);
}

TEST(YuvSp, BgraNv21Vector) {
  YuvSpTest yuv_test(4, true);
  yuv_test.execute_vector_test(kleidicv_yuv_sp_to_bgra_u8, true);
}
