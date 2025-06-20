// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <type_traits>
#include <vector>

#include "opencv2/core/hal/interface.h"
#include "tests.h"

template <int Scale, int Shift>
cv::Mat exec_scale(cv::Mat& input_mat) {
  cv::Mat result;
  input_mat.convertTo(result, -1, Scale / 1000.0, Shift / 1000.0);
  return result;
}

#if MANAGER
template <int Scale, int Shift, size_t Format>
bool test_scale(int index, RecreatedMessageQueue& request_queue,
                RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t x = 5; x <= 16; ++x) {
    for (size_t y = 5; y <= 16; ++y) {
      cv::Mat input_mat(x, y, Format);
      rng.fill(input_mat, cv::RNG::NORMAL, 0.0, 1.0e10);
      cv::Mat actual_mat = exec_scale<Scale, Shift>(input_mat);
      cv::Mat expected_mat = get_expected_from_subordinate(
          index, request_queue, reply_queue, input_mat);

      bool success =
          (CV_MAT_DEPTH(Format) == CV_32F &&
           !are_float_matrices_different<float>(1e-5, actual_mat,
                                                expected_mat)) ||
          (CV_MAT_DEPTH(Format) == CV_8U &&
           !are_matrices_different<uint8_t>(0, actual_mat, expected_mat));
      if (!success) {
        fail_print_matrices(x, y, input_mat, actual_mat, expected_mat);
        return true;
      }
    }
  }

  return false;
}
#endif

std::vector<test>& scale_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Scale float32, scale=1.0, shift=2.0", (test_scale<1000, 2000, CV_32FC4>), (exec_scale<1000, 2000>)),
    TEST("Scale float32, scale=-10.0, shift=0.0", (test_scale<(-10000), 0, CV_32FC1>), (exec_scale<(-10000), 0>)),
    TEST("Scale float32, scale=3.14, shift=2.72", (test_scale<3140, 2720, CV_32FC2>), (exec_scale<3140, 2720>)),
    TEST("Scale float32, scale=7e5, shift=8e-3", (test_scale<700000000, 8, CV_32FC3>), (exec_scale<700000000, 8>)),
    TEST("Scale float32, scale=1e-3, shift=8e-3", (test_scale<1, 8, CV_32FC4>), (exec_scale<1, 8>)),

    TEST("Scale uint8, scale=1.0, shift=2.0", (test_scale<1000, 2000, CV_8UC4>), (exec_scale<1000, 2000>)),
    TEST("Scale uint8, scale=-10.0, shift=0.0", (test_scale<(-10000), 0, CV_8UC1>), (exec_scale<(-10000), 0>)),
    TEST("Scale uint8, scale=3.14, shift=-2.72", (test_scale<3140, -2720, CV_8UC2>), (exec_scale<3140, -2720>)),
    TEST("Scale uint8, scale=17.17, shift=3.9", (test_scale<17170, 3900, CV_8UC3>), (exec_scale<17170, 3900>)),
    TEST("Scale uint8, scale=0.13, shift=230", (test_scale<130, 230000, CV_8UC4>), (exec_scale<130, 230000>)),
  };
  // clang-format on
  return tests;
}
