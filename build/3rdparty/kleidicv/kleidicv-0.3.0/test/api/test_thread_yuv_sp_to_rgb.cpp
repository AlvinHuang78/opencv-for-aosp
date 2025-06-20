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
typedef std::tuple<unsigned, unsigned, unsigned> P;

class YuvSpThread : public testing::TestWithParam<P> {
 public:
  template <typename SingleThreadedFunc, typename MultithreadedFunc>
  void check(SingleThreadedFunc single_threaded_func,
             MultithreadedFunc multithreaded_func, size_t channels) {
    unsigned width = 0, height = 0, thread_count = 0;
    std::tie(width, height, thread_count) = GetParam();
    test::Array2D<uint8_t> src_y(width, height),
        src_uv((width + 1) & ~1, (height + 1) / 2),
        dst_single(size_t{width} * channels, height),
        dst_multi(size_t{width} * channels, height);

    test::PseudoRandomNumberGenerator<uint8_t> generator;
    src_y.fill(generator);
    src_uv.fill(generator);

    kleidicv_error_t single_result = single_threaded_func(
        src_y.data(), src_y.stride(), src_uv.data(), src_uv.stride(),
        dst_single.data(), dst_single.stride(), width, height, false);

    kleidicv_error_t multi_result = multithreaded_func(
        src_y.data(), src_y.stride(), src_uv.data(), src_uv.stride(),
        dst_multi.data(), dst_multi.stride(), width, height, false,
        get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_OK, single_result);
    EXPECT_EQ(KLEIDICV_OK, multi_result);
    EXPECT_EQ_ARRAY2D(dst_multi, dst_single);
  }
};

TEST_P(YuvSpThread, ToBGR) {
  check(kleidicv_yuv_sp_to_bgr_u8, kleidicv_thread_yuv_sp_to_bgr_u8, 3);
}
TEST_P(YuvSpThread, ToBGRA) {
  check(kleidicv_yuv_sp_to_bgra_u8, kleidicv_thread_yuv_sp_to_bgra_u8, 4);
}
TEST_P(YuvSpThread, ToRGB) {
  check(kleidicv_yuv_sp_to_rgb_u8, kleidicv_thread_yuv_sp_to_rgb_u8, 3);
}
TEST_P(YuvSpThread, ToRGBA) {
  check(kleidicv_yuv_sp_to_rgba_u8, kleidicv_thread_yuv_sp_to_rgba_u8, 4);
}

INSTANTIATE_TEST_SUITE_P(, YuvSpThread,
                         testing::Values(P{1, 1, 1}, P{1, 2, 1}, P{1, 2, 2},
                                         P{2, 1, 2}, P{2, 2, 1}, P{1, 3, 2},
                                         P{2, 3, 1}, P{6, 4, 1}, P{4, 5, 2},
                                         P{2, 6, 3}, P{1, 7, 4}, P{12, 34, 5}));
