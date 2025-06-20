// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "opencv2/core/hal/interface.h"
#include "opencv2/imgproc/hal/interface.h"
#include "tests.h"

template <int LowerBound, int UpperBound>
cv::Mat exec_in_range(cv::Mat& input) {
  cv::Mat result;
  cv::inRange(input, LowerBound / 1000.0, UpperBound / 1000.0, result);
  return result;
}

#if MANAGER
template <int LowerBound, int UpperBound, size_t Format>
bool test_in_range(int index, RecreatedMessageQueue& request_queue,
                   RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t x = 5; x <= 16; ++x) {
    for (size_t y = 5; y <= 16; ++y) {
      cv::Mat input(x, y, Format);
      rng.fill(input, cv::RNG::UNIFORM, 0, 255);
      cv::Mat actual = exec_in_range<LowerBound, UpperBound>(input);
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

template <int LowerBound, int UpperBound, size_t Format>
bool test_in_range_custom(int index, RecreatedMessageQueue& request_queue,
                          RecreatedMessageQueue& reply_queue) {
  cv::Mat input(custom_data_float_height, custom_data_float_width, CV_32FC1,
                custom_data_float);

  cv::Mat actual = exec_in_range<LowerBound, UpperBound>(input);
  cv::Mat expected =
      get_expected_from_subordinate(index, request_queue, reply_queue, input);

  if (are_matrices_different<uint8_t>(0, actual, expected)) {
    fail_print_matrices(custom_data_float_height, custom_data_float_width,
                        input, actual, expected);
    return true;
  }

  return false;
}

#endif

std::vector<test>& in_range_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("InRange uint8,  lower_bound =           1, upper_bound =           2", (test_in_range<1000, 2000, CV_8UC1>), (exec_in_range<1000, 2000>)),
    TEST("InRange uint8,  lower_bound =          10, upper_bound =          11", (test_in_range<10000, 11000, CV_8UC1>), (exec_in_range<10000, 11000>)),
    TEST("InRange uint8,  lower_bound =           3, upper_bound =           4", (test_in_range<3000, 4000, CV_8UC1>), (exec_in_range<3000, 4000>)),
    TEST("InRange uint8,  lower_bound =         254, upper_bound =         255", (test_in_range<254000, 255000, CV_8UC1>), (exec_in_range<254000, 255000>)),
    TEST("InRange uint8,  lower_bound =          13, upper_bound =          12", (test_in_range<13000, 12000, CV_8UC1>), (exec_in_range<13000, 12000>)),
    TEST("InRange float,  lower_bound = -999999.999, upper_bound = -999999.998", (test_in_range<(-999999999), (-999999998), CV_32FC1>), (exec_in_range<(-999999999), (-999999998)>)),
    TEST("InRange float,  lower_bound = -999999.999, upper_bound = -999989.999", (test_in_range<(-999999999), (-999989999), CV_32FC1>), (exec_in_range<(-999999999), (-999989999)>)),
    TEST("InRange float,  lower_bound =           1, upper_bound =           2", (test_in_range<1000, 2000, CV_32FC1>), (exec_in_range<1000, 2000>)),
    TEST("InRange float,  lower_bound =       1.111, upper_bound =       1.112", (test_in_range<1111, 1112, CV_32FC1>), (exec_in_range<1111, 1112>)),
    TEST("InRange float,  lower_bound =           3, upper_bound =           3", (test_in_range<3000, 3000, CV_32FC1>), (exec_in_range<3000, 3000>)),
    TEST("InRange float,  lower_bound =      10.001, upper_bound =     10.0011", (test_in_range<10001, 100011, CV_32FC1>), (exec_in_range<10001, 100011>)),
    TEST("InRange float,  lower_bound =      13.999, upper_bound =      13.998", (test_in_range<13999, 13998, CV_32FC1>), (exec_in_range<13999, 13998>)),
    TEST("InRange float,  lower_bound =      14.999, upper_bound =      20.998", (test_in_range<14999, 20998, CV_32FC1>), (exec_in_range<14999, 20998>)),
    TEST("InRange float,  lower_bound =  999999.998, upper_bound =  999999.999", (test_in_range<(999999998), (999999999), CV_32FC1>), (exec_in_range<(999999998), (999999999)>)),
    TEST("InRange float,  lower_bound =  999989.999, upper_bound =  999999.999", (test_in_range<(999989999), (999999999), CV_32FC1>), (exec_in_range<(999989999), (999999999)>)),
    TEST("InRange float,  lower_bound =           0, upper_bound =           1", (test_in_range_custom<(0), (1), CV_32FC1>), (exec_in_range<(0), (1)>)),
    TEST("InRange float,  lower_bound =       -11.4, upper_bound =      1111.1", (test_in_range_custom<(-11400), (1111100), CV_32FC1>), (exec_in_range<(-11400), (1111100)>)),
  };
  // clang-format on
  return tests;
}
