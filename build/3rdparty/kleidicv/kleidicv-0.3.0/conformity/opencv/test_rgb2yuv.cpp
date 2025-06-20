// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "tests.h"

template <bool SwitchBlue>
cv::Mat exec_rgb2yuv(cv::Mat& input) {
  cv::Mat result;
  if constexpr (SwitchBlue) {
    cv::cvtColor(input, result, cv::COLOR_BGR2YUV);
  } else {
    cv::cvtColor(input, result, cv::COLOR_RGB2YUV);
  }
  return result;
}

#if MANAGER
template <bool SwitchBlue, size_t Channels>
bool test_rgb2yuv(int index, RecreatedMessageQueue& request_queue,
                  RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t x = 5; x <= 16; ++x) {
    for (size_t y = 5; y <= 16; ++y) {
      cv::Mat input(x, y, CV_8UC(Channels));
      rng.fill(input, cv::RNG::UNIFORM, 0, 255);

      cv::Mat actual = exec_rgb2yuv<SwitchBlue>(input);
      cv::Mat expected = get_expected_from_subordinate(index, request_queue,
                                                       reply_queue, input);

      if (are_matrices_different<uint8_t>(0, actual, expected)) {
        fail_print_matrices(x, y, input, actual, expected);
        return true;
      }
    }
  }

  return false;
}
#endif

std::vector<test>& rgb2yuv_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("RGB2YUV", (test_rgb2yuv<false, 3>), exec_rgb2yuv<false>),
    TEST("RGBA2YUV", (test_rgb2yuv<false, 4>), exec_rgb2yuv<false>),
    TEST("BGR2YUV", (test_rgb2yuv<true, 3>), exec_rgb2yuv<true>),
    TEST("BGRA2YUV", (test_rgb2yuv<true, 4>), exec_rgb2yuv<true>),
  };
  // clang-format on
  return tests;
}
