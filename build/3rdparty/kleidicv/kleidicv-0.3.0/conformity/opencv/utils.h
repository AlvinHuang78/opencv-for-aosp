// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_OPENCV_CONFORMITY_UTILS_H_
#define KLEIDICV_OPENCV_CONFORMITY_UTILS_H_

#include <iostream>
#include <string>
#include <utility>

#include "common.h"

template <typename T, size_t Channels>
constexpr int get_opencv_matrix_type() {
  if constexpr (std::is_same_v<T, uint8_t>) {
    return CV_8UC(Channels);
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return CV_16UC(Channels);
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return CV_16SC(Channels);
  }
}

#if MANAGER
template <typename T>
static auto abs_diff(T a, T b) {
  return a > b ? a - b : b - a;
}

static inline bool check_matrix_size_and_type(cv::Mat& A, cv::Mat& B) {
  if (A.rows != B.rows || A.cols != B.cols || A.type() != B.type()) {
    std::cout << "Matrix size/type mismatch" << std::endl;
    return true;
  }

  return false;
}

// Expected matrix should not contain zeros
template <typename T>
bool are_float_matrices_different(T threshold_percent, cv::Mat& exp,
                                  cv::Mat& act) {
  if (check_matrix_size_and_type(exp, act)) {
    return true;
  }

  for (int i = 0; i < exp.rows; ++i) {
    for (int j = 0; j < (exp.cols * CV_MAT_CN(exp.type())); ++j) {
      T diff = abs_diff<T>(exp.at<T>(i, j), act.at<T>(i, j));
      T diff_percentage = (diff / std::abs(exp.at<T>(i, j))) * 100;
      if (diff_percentage > threshold_percent) {
        std::cout << "=== Mismatch at: " << i << " " << j << std::endl
                  << "Expected: " << exp.at<T>(i, j)
                  << "   Actual: " << act.at<T>(i, j) << std::endl
                  << "Relative diff: " << diff_percentage << std::endl
                  << std::endl;
        return true;
      }
    }
  }

  return false;
}

template <typename T>
bool are_matrices_different(T threshold, cv::Mat& A, cv::Mat& B) {
  if (check_matrix_size_and_type(A, B)) {
    return true;
  }

  for (int i = 0; i < A.rows; ++i) {
    for (int j = 0; j < (A.cols * CV_MAT_CN(A.type())); ++j) {
      if (abs_diff<T>(A.at<T>(i, j), B.at<T>(i, j)) > threshold) {
        std::cout << "=== Mismatch at: " << i << " " << j << std::endl
                  << std::endl;
        return true;
      }
    }
  }

  return false;
}

void fail_print_matrices(size_t height, size_t width, cv::Mat& input,
                         cv::Mat& manager_result, cv::Mat& subord_result);

cv::Mat get_expected_from_subordinate(int index,
                                      RecreatedMessageQueue& request_queue,
                                      RecreatedMessageQueue& reply_queue,
                                      cv::Mat& input);

int run_tests(RecreatedMessageQueue& request_queue,
              RecreatedMessageQueue& reply_queue);

typedef bool (*test_function)(int index, RecreatedMessageQueue& request_queue,
                              RecreatedMessageQueue& reply_queue);
using test = std::pair<std::string, test_function>;
#define TEST(name, test_func, x) {name, test_func}
#else
void wait_for_requests(OpenedMessageQueue& request_queue,
                       OpenedMessageQueue& reply_queue);

typedef cv::Mat (*exec_function)(cv::Mat& input);
using test = std::pair<std::string, exec_function>;
#define TEST(name, x, exec_func) {name, exec_func}
#endif

#endif  // KLEIDICV_OPENCV_CONFORMITY_UTILS_H_
