// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "tests.h"

cv::Mat exec_scharr_interleaved(cv::Mat& input) {
  std::vector<cv::Mat> pyramid;
  // `winSize` is not take into account when `maxLevel` is set to 0.
  cv::buildOpticalFlowPyramid(input, pyramid, cv::Size(3, 3), 0, true);
  return pyramid[1].clone();
}

#if MANAGER
bool test_scharr_interleaved(int index, RecreatedMessageQueue& request_queue,
                             RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t x = 1; x <= 16; ++x) {
    for (size_t y = 1; y <= 16; ++y) {
      cv::Mat input(x, y, CV_8UC1);
      rng.fill(input, cv::RNG::UNIFORM, 0, 255);

      cv::Mat actual = exec_scharr_interleaved(input);
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

std::vector<test>& scharr_interleaved_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Scharr Interleaved, 1 channel", test_scharr_interleaved, exec_scharr_interleaved)
  };
  // clang-format on
  return tests;
}
