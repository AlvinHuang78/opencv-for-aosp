// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "tests.h"

#include <initializer_list>
#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.h"

static std::vector<test> merge_tests(
    std::initializer_list<std::vector<test>& (*)()> test_groups) {
  std::vector<test> all_tests;
  for (auto getter : test_groups) {
    std::vector<test>& group = getter();
    all_tests.insert(all_tests.cend(), group.cbegin(), group.cend());
  }
  return all_tests;
}

std::vector<test> all_tests = merge_tests({
    // clang-format off
    binary_op_tests_get,
    cvtcolor_tests_get,
#if KLEIDICV_ENABLE_ALL_OPENCV_HAL
    separable_filter_2d_tests_get,
#endif
    gaussian_blur_tests_get,
    rgb2yuv_tests_get,
    yuv2rgb_tests_get,
    sobel_tests_get,
    exp_tests_get,
    float_conversion_tests_get,
    resize_tests_get,
    scale_tests_get,
    min_max_tests_get,
    in_range_tests_get,
    remap_tests_get,
    warp_perspective_tests_get,
    blur_and_downsample_tests_get,
    scharr_interleaved_tests_get,
    // clang-format on
});

#if MANAGER
void fail_print_matrices(size_t height, size_t width, cv::Mat& input,
                         cv::Mat& manager_result, cv::Mat& subord_result) {
  std::cout << "[FAIL]" << std::endl;
  std::cout << "height=" << height << std::endl;
  std::cout << "width=" << width << std::endl;
  std::cout << "=== Input Matrix:" << std::endl;
  std::cout << input << std::endl << std::endl;
  std::cout << "=== Manager result (actual):" << std::endl;
  std::cout << manager_result << std::endl << std::endl;
  std::cout << "=== Subordinate result (expected):" << std::endl;
  std::cout << subord_result << std::endl << std::endl;
}

cv::Mat get_expected_from_subordinate(int index,
                                      RecreatedMessageQueue& request_queue,
                                      RecreatedMessageQueue& reply_queue,
                                      cv::Mat& input) {
  request_queue.request_operation(index, input);
  reply_queue.wait();
  if (reply_queue.last_cmd() != index) {
    throw std::runtime_error("Invalid reply from subordinate");
  }

  return reply_queue.cv_mat_from_last_msg();
}
#endif

#if MANAGER
int run_tests(RecreatedMessageQueue& request_queue,
              RecreatedMessageQueue& reply_queue) {
  int ret_val = 0;
  for (int i = 0; i < static_cast<int>(all_tests.size()); ++i) {
    std::cout << "Testing " + all_tests[i].first << std::endl;
    if (all_tests[i].second(i, request_queue, reply_queue)) {
      ret_val = 1;
    }
  }
  request_queue.request_exit();

  return ret_val;
}
#else
void wait_for_requests(OpenedMessageQueue& request_queue,
                       OpenedMessageQueue& reply_queue) {
  while (true) {
    request_queue.wait();
    int cmd = request_queue.last_cmd();

    if (cmd < 0) {
      // Exit requested
      break;
    }

    if (cmd > static_cast<int>(all_tests.size())) {
      throw std::runtime_error("Invalid operation requested in subordinate");
    }

    cv::Mat input = request_queue.cv_mat_from_last_msg();
    cv::Mat result = all_tests[cmd].second(input);
    reply_queue.reply_operation(cmd, result);
  }
}
#endif
