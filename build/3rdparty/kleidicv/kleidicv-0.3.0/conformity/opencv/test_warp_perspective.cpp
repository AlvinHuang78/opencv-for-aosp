// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <type_traits>
#include <vector>

#include "opencv2/core/hal/interface.h"
#include "opencv2/imgproc/hal/interface.h"
#include "utils.h"

// clang-format off
double transform[] = {
  0.8, 0.1, 2,
  0.1, 0.8, -2,
  0.001, 0.001, 1.7
};
// clang-format on

// BorderValue is interpreted as 1/1000, i.e. 500 for 0.5
template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
cv::Mat exec_warp_perspective(cv::Mat& source_mat) {
  cv::Mat result(source_mat.rows, source_mat.cols, Format);
  cv::Mat M(3, 3, CV_64FC1, reinterpret_cast<uint8_t*>(transform));
  cv::warpPerspective(source_mat, result, M, result.size(), Interpolation,
                      BorderMode, BorderValue / 1000.0);
  return result;
}

#if MANAGER
const int kMaxHeight = 42, kMaxWidth = 42;

template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
bool test_warp_perspective(int index, RecreatedMessageQueue& request_queue,
                           RecreatedMessageQueue& reply_queue) {
  for (size_t w = 8; w <= kMaxWidth; w += 3) {
    for (size_t h = 8; h <= kMaxHeight; h += 4) {
      cv::Mat source_mat(h, w, Format);
      for (size_t row = 0; row < h; ++row) {
        for (size_t column = 0; column < w; ++column) {
          source_mat.at<ScalarType>(row, column) =
              (row * (w + 12) + column) %
              std::numeric_limits<ScalarType>::max();
        }
      }

      cv::Mat actual_mat =
          exec_warp_perspective<ScalarType, Format, Interpolation, BorderMode,
                                BorderValue>(source_mat);
      cv::Mat expected_mat = get_expected_from_subordinate(
          index, request_queue, reply_queue, source_mat);

      bool success =
          !are_matrices_different<ScalarType>(1, actual_mat, expected_mat);
      if (!success) {
        fail_print_matrices(w, h, source_mat, actual_mat, expected_mat);
        return true;
      }
    }
  }
  return false;
}
#endif

std::vector<test>& warp_perspective_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("WarpPerspective uint8", (test_warp_perspective<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>), (exec_warp_perspective<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>)),
  };
  // clang-format on
  return tests;
}
