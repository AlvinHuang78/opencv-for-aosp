// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "tests.h"

template <bool Vertical>
cv::Mat exec_sobel(cv::Mat& input) {
  cv::Mat result;
  if constexpr (Vertical) {
    cv::Sobel(input, result, CV_16S, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
  } else {
    cv::Sobel(input, result, CV_16S, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
  }
  return result;
}

#if MANAGER
template <bool Vertical, size_t Channels>
bool test_sobel(int index, RecreatedMessageQueue& request_queue,
                RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t x = 5; x <= 16; ++x) {
    for (size_t y = 5; y <= 16; ++y) {
      cv::Mat input(x, y, CV_8UC(Channels));
      rng.fill(input, cv::RNG::UNIFORM, 0, 255);

      cv::Mat actual = exec_sobel<Vertical>(input);
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

std::vector<test>& sobel_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Sobel Vertical, 1 channel", (test_sobel<true, 1>), exec_sobel<true>),
    TEST("Sobel Vertical, 2 channel", (test_sobel<true, 2>), exec_sobel<true>),
    TEST("Sobel Vertical, 3 channel", (test_sobel<true, 3>), exec_sobel<true>),
    TEST("Sobel Vertical, 4 channel", (test_sobel<true, 4>), exec_sobel<true>),

    TEST("Sobel Horizontal, 1 channel", (test_sobel<false, 1>), exec_sobel<false>),
    TEST("Sobel Horizontal, 2 channel", (test_sobel<false, 2>), exec_sobel<false>),
    TEST("Sobel Horizontal, 3 channel", (test_sobel<false, 3>), exec_sobel<false>),
    TEST("Sobel Horizontal, 4 channel", (test_sobel<false, 4>), exec_sobel<false>),
  };
  // clang-format on
  return tests;
}
