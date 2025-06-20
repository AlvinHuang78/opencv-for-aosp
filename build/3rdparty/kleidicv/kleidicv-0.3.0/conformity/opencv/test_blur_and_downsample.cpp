// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "tests.h"

template <size_t BorderType>
cv::Mat exec_blur_and_downsample(cv::Mat& input) {
  cv::Mat result;
  cv::pyrDown(input, result, cv::Size(), BorderType);
  return result;
}

#if MANAGER
template <size_t BorderType>
bool test_blur_and_downsample(int index, RecreatedMessageQueue& request_queue,
                              RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t x = 5; x <= 16; ++x) {
    for (size_t y = 5; y <= 16; ++y) {
      cv::Mat input(x, y, CV_8UC1);
      rng.fill(input, cv::RNG::UNIFORM, 0, 255);

      cv::Mat actual = exec_blur_and_downsample<BorderType>(input);
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

std::vector<test>& blur_and_downsample_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Blur and Downsample, BORDER_REFLECT_101", (test_blur_and_downsample<cv::BORDER_REFLECT_101>), exec_blur_and_downsample<cv::BORDER_REFLECT_101>),
    TEST("Blur and Downsample, BORDER_REFLECT", (test_blur_and_downsample<cv::BORDER_REFLECT>), exec_blur_and_downsample<cv::BORDER_REFLECT>),
    TEST("Blur and Downsample, BORDER_WRAP", (test_blur_and_downsample<cv::BORDER_WRAP>), exec_blur_and_downsample<cv::BORDER_WRAP>),
    TEST("Blur and Downsample, BORDER_REPLICATE", (test_blur_and_downsample<cv::BORDER_REPLICATE>), exec_blur_and_downsample<cv::BORDER_REPLICATE>),
  };
  // clang-format on
  return tests;
}
