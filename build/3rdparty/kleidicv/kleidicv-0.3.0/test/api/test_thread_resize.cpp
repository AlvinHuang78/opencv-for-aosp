// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <thread>

#include "framework/array.h"
#include "framework/generator.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv_thread/kleidicv_thread.h"
#include "multithreading_fake.h"

// Tuple of width, height, thread count.
typedef std::tuple<size_t, size_t, size_t> P;

class ResizeThread : public testing::TestWithParam<P> {
 public:
  void test_resize_to_quarter() {
    size_t src_width = 0, src_height = 0, thread_count = 0;
    std::tie(src_width, src_height, thread_count) = GetParam();
    check<uint8_t>(kleidicv_resize_to_quarter_u8,
                   kleidicv_thread_resize_to_quarter_u8, thread_count,
                   src_width, src_height, src_width / 2, src_height / 2);
    check<uint8_t>(kleidicv_resize_to_quarter_u8,
                   kleidicv_thread_resize_to_quarter_u8, thread_count,
                   src_width, src_height, (src_width + 1) / 2,
                   (src_height + 1) / 2);
  }
  void test_resize_u8_2x2() {
    size_t src_width = 0, src_height = 0, thread_count = 0;
    std::tie(src_width, src_height, thread_count) = GetParam();
    check<uint8_t>(kleidicv_resize_linear_u8, kleidicv_thread_resize_linear_u8,
                   thread_count, src_width, src_height, 2 * src_width,
                   2 * src_height);
  }
  void test_resize_u8_4x4() {
    size_t src_width = 0, src_height = 0, thread_count = 0;
    std::tie(src_width, src_height, thread_count) = GetParam();
    check<uint8_t>(kleidicv_resize_linear_u8, kleidicv_thread_resize_linear_u8,
                   thread_count, src_width, src_height, 4 * src_width,
                   4 * src_height);
  }
  void test_resize_f32_2x2() {
    size_t src_width = 0, src_height = 0, thread_count = 0;
    std::tie(src_width, src_height, thread_count) = GetParam();
    check<float>(kleidicv_resize_linear_f32, kleidicv_thread_resize_linear_f32,
                 thread_count, src_width, src_height, 2 * src_width,
                 2 * src_height);
  }
  void test_resize_f32_4x4() {
    size_t src_width = 0, src_height = 0, thread_count = 0;
    std::tie(src_width, src_height, thread_count) = GetParam();
    check<float>(kleidicv_resize_linear_f32, kleidicv_thread_resize_linear_f32,
                 thread_count, src_width, src_height, 4 * src_width,
                 4 * src_height);
  }
  void test_resize_f32_8x8() {
    size_t src_width = 0, src_height = 0, thread_count = 0;
    std::tie(src_width, src_height, thread_count) = GetParam();
    check<float>(kleidicv_resize_linear_f32, kleidicv_thread_resize_linear_f32,
                 thread_count, src_width, src_height, 8 * src_width,
                 8 * src_height);
  }

 private:
  template <typename ScalarType, typename SingleThreadedFunc,
            typename MultithreadedFunc>
  static void check(SingleThreadedFunc single_threaded_func,
                    MultithreadedFunc multithreaded_func, size_t thread_count,
                    size_t src_width, size_t src_height, size_t dst_width,
                    size_t dst_height) {
    test::Array2D<ScalarType> src(src_width, src_height),
        dst_single(dst_width, dst_height), dst_multi(dst_width, dst_height);
    test::PseudoRandomNumberGenerator<ScalarType> generator;
    src.fill(generator);

    kleidicv_error_t single_result = single_threaded_func(
        src.data(), src.stride(), src_width, src_height, dst_single.data(),
        dst_single.stride(), dst_width, dst_height);

    kleidicv_error_t multi_result =
        multithreaded_func(src.data(), src.stride(), src_width, src_height,
                           dst_multi.data(), dst_multi.stride(), dst_width,
                           dst_height, get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_OK, single_result);
    EXPECT_EQ(KLEIDICV_OK, multi_result);
    EXPECT_EQ_ARRAY2D(dst_single, dst_multi);
  }
};

TEST_P(ResizeThread, ResizeToQuarter) { test_resize_to_quarter(); }

TEST_P(ResizeThread, ResizeUint2x2) { test_resize_u8_2x2(); }

TEST_P(ResizeThread, ResizeUint4x4) { test_resize_u8_4x4(); }

TEST_P(ResizeThread, ResizeFloat2x2) { test_resize_f32_2x2(); }

TEST_P(ResizeThread, ResizeFloat4x4) { test_resize_f32_4x4(); }

TEST_P(ResizeThread, ResizeFloat8x8) { test_resize_f32_8x8(); }

INSTANTIATE_TEST_SUITE_P(, ResizeThread,
                         testing::Values(P{1, 1, 1}, P{1, 2, 1}, P{1, 2, 2},
                                         P{2, 1, 2}, P{2, 2, 1}, P{1, 3, 2},
                                         P{2, 3, 1}, P{6, 4, 1}, P{4, 5, 2},
                                         P{2, 6, 3}, P{1, 7, 4}, P{12, 34, 5}));

TEST(ResizeThreadTest, NotImplemented) {
  {
    uint8_t src[1] = {}, dst[1] = {};
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              kleidicv_thread_resize_linear_u8(src, sizeof(src), 2, 2, dst,
                                               sizeof(dst), 3, 7,
                                               get_multithreading_fake(2)));
  }
  {
    float src[1] = {}, dst[1] = {};
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              kleidicv_thread_resize_linear_f32(src, sizeof(src), 2, 2, dst,
                                                sizeof(dst), 3, 7,
                                                get_multithreading_fake(2)));
  }
}
