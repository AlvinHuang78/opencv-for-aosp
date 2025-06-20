// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <type_traits>
#include <vector>

#include "opencv2/core/hal/interface.h"
#include "tests.h"

static cv::Mat exec_exp(cv::Mat& input_mat) {
  cv::Mat result;
  cv::exp(input_mat, result);
  return result;
}

#if MANAGER
template <size_t Channels>
bool test_exp(int index, RecreatedMessageQueue& request_queue,
              RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t x = 5; x <= 16; ++x) {
    for (size_t y = 5; y <= 16; ++y) {
      cv::Mat input_mat(x, y, CV_32FC(Channels));
      // Use inputs where results are not flowing over or under
      rng.fill(input_mat, cv::RNG::UNIFORM, -80.0, 80.0);
      cv::Mat actual_mat = exec_exp(input_mat);
      cv::Mat expected_mat = get_expected_from_subordinate(
          index, request_queue, reply_queue, input_mat);

      // OpenCV works with less precision, so a relatively big expected error
      // range is defined
      if (are_float_matrices_different<float>(1.5, expected_mat, actual_mat)) {
        fail_print_matrices(x, y, input_mat, actual_mat, expected_mat);
        return true;
      }
    }
  }

  return false;
}
#endif

std::vector<test>& exp_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Exp float, 1 channel", (test_exp<1>), exec_exp),
    TEST("Exp float, 2 channel", (test_exp<2>), exec_exp),
    TEST("Exp float, 3 channel", (test_exp<3>), exec_exp),
    TEST("Exp float, 4 channel", (test_exp<4>), exec_exp),
  };
  // clang-format on
  return tests;
}
