// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>

#include "tests.h"

// This function tries running the Separable Filter 2D API via OpenCV with the
// specified InputType and KernelType. An exception should be thrown in case the
// constraint in the HAL has not been met.
// Returns a 1x1-sized boolean matrix.
template <size_t KernelSize, int InputType, int KernelType>
cv::Mat exec_separable_filter_2d_channel_check(cv::Mat& input) {
  cv::Mat kernel(KernelSize, 1, KernelType);
  cv::Mat result;
  try {
    cv::sepFilter2D(input, result, -1, kernel, kernel, cv::Point(-1, -1), 0,
                    cv::BORDER_REPLICATE);
  } catch (const cv::Exception&) {
    return cv::Mat(1, 1, CV_8UC1, cv::Scalar(1));
  }
  return cv::Mat(1, 1, CV_8UC1, cv::Scalar(0));
}

template <typename TypeParam, size_t KernelSize, size_t BorderType>
cv::Mat exec_separable_filter_2d(cv::Mat& input) {
  uint32_t kernel_seed =
      *reinterpret_cast<uint32_t*>(&input.at<TypeParam>(input.rows - 1, 0));
  // clone is required, otherwise the result matrix is treated as part of a
  // bigger image, and it would have impact on what border types are supported
  cv::Mat input_mat = input.rowRange(0, input.rows - 1).clone();

  cv::RNG rng(kernel_seed);
  cv::Mat kernel_x(KernelSize, 1, get_opencv_matrix_type<TypeParam, 1>());
  // use the minimum value 1 for the kernels in order to properly work around
  // the potential OpenCV bug (mentioned lower in "test_separable_filter_2d")
  rng.fill(kernel_x, cv::RNG::UNIFORM, 1,
           std::numeric_limits<TypeParam>::max());
  cv::Mat kernel_y(KernelSize, 1, get_opencv_matrix_type<TypeParam, 1>());
  rng.fill(kernel_y, cv::RNG::UNIFORM, 1,
           std::numeric_limits<TypeParam>::max());

  cv::Mat result;
  cv::sepFilter2D(input_mat, result, -1, kernel_x, kernel_y, cv::Point(-1, -1),
                  0, BorderType);
  return result;
}

#if MANAGER
// The purpose of this test is to check one of the initial constraints of the
// Separable Filter 2D HAL, that the kernel can only have one channel.
template <size_t KernelSize, int InputType, int KernelType>
bool test_separable_filter_2d_channel_check(
    int index, RecreatedMessageQueue& request_queue,
    RecreatedMessageQueue& reply_queue) {
  cv::Mat input(10, 10, InputType, cv::Scalar(0));

  cv::Mat actual =
      exec_separable_filter_2d_channel_check<KernelSize, InputType, KernelType>(
          input);
  cv::Mat expected =
      get_expected_from_subordinate(index, request_queue, reply_queue, input);

  bool actual_exception_caught = actual.at<uint8_t>(0, 0);
  bool expected_exception_caught = expected.at<uint8_t>(0, 0);

  if (actual_exception_caught != expected_exception_caught) {
    std::cout << "[FAIL]" << std::endl
              << "Actual: "
              << (actual_exception_caught ? "exception" : "no exception")
              << std::endl
              << "Expected: "
              << (expected_exception_caught ? "exception" : "no exception")
              << std::endl
              << std::endl;
    return true;
  }

  return false;
}

template <typename TypeParam, size_t KernelSize, size_t BorderType,
          size_t Channels>
bool test_separable_filter_2d(int index, RecreatedMessageQueue& request_queue,
                              RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t y = 5; y <= 16; ++y) {
    for (size_t x = 5; x <= 16; ++x) {
      // One extra line allocated to be sure the kernel seed can be placed next
      // to the real input
      cv::Mat input(y + 1, x, get_opencv_matrix_type<TypeParam, Channels>());
      // use the minimum value 1 for the input in order to properly work around
      // the potential OpenCV bug (mentioned lower)
      rng.fill(input, cv::RNG::UNIFORM, 1,
               std::numeric_limits<TypeParam>::max());

      uint32_t kernel_seed = rng.next();

      // kernel seed is embedded into the input matrix
      *reinterpret_cast<uint32_t*>(&input.at<TypeParam>(input.rows - 1, 0)) =
          kernel_seed;

      cv::Mat actual =
          exec_separable_filter_2d<TypeParam, KernelSize, BorderType>(input);
      cv::Mat expected = get_expected_from_subordinate(index, request_queue,
                                                       reply_queue, input);

      // bypass OpenCV bug/inconsistency where the output matrix contains 0's
      // (or also the smallest negative value for signed types), whereas this
      // should not be possible mathematically
      for (size_t i = 0; i < (y * Channels); ++i) {
        for (size_t j = 0; j < (x * Channels); ++j) {
          if (expected.at<TypeParam>(i, j) ==
              std::numeric_limits<TypeParam>::lowest()) {
            expected.at<TypeParam>(i, j) =
                std::numeric_limits<TypeParam>::max();
          } else if (expected.at<TypeParam>(i, j) == 0) {
            expected.at<TypeParam>(i, j) =
                std::numeric_limits<TypeParam>::max();
          }
        }
      }

      if (are_matrices_different<TypeParam>(0, actual, expected)) {
        fail_print_matrices(y, x, input, actual, expected);
        return true;
      }
    }
  }

  return false;
}
#endif

std::vector<test>& separable_filter_2d_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Separable Filter 2D 5x5 channels: CV_8UC1 input, CV_8UC1 kernel", (test_separable_filter_2d_channel_check<5, CV_8UC1, CV_8UC1>), (exec_separable_filter_2d_channel_check<5, CV_8UC1, CV_8UC1>)),
    TEST("Separable Filter 2D 5x5 channels: CV_8UC1 input, CV_8UC2 kernel", (test_separable_filter_2d_channel_check<5, CV_8UC1, CV_8UC2>), (exec_separable_filter_2d_channel_check<5, CV_8UC1, CV_8UC2>)),
    TEST("Separable Filter 2D 5x5 channels: CV_8UC2 input, CV_8UC1 kernel", (test_separable_filter_2d_channel_check<5, CV_8UC2, CV_8UC1>), (exec_separable_filter_2d_channel_check<5, CV_8UC2, CV_8UC1>)),
    TEST("Separable Filter 2D 5x5 channels: CV_8UC2 input, CV_8UC2 kernel", (test_separable_filter_2d_channel_check<5, CV_8UC2, CV_8UC2>), (exec_separable_filter_2d_channel_check<5, CV_8UC2, CV_8UC2>)),

    TEST("Separable Filter 2D 5x5 (u8), BORDER_REFLECT_101, 1 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT_101, 1>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT_101>)),
    TEST("Separable Filter 2D 5x5 (u8), BORDER_REFLECT_101, 2 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT_101, 2>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT_101>)),
    TEST("Separable Filter 2D 5x5 (u8), BORDER_REFLECT_101, 3 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT_101, 3>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT_101>)),
    TEST("Separable Filter 2D 5x5 (u8), BORDER_REFLECT_101, 4 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT_101, 4>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT_101>)),

    TEST("Separable Filter 2D 5x5 (u8), BORDER_REFLECT, 1 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT, 1>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT>)),
    TEST("Separable Filter 2D 5x5 (u8), BORDER_REFLECT, 2 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT, 2>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT>)),
    TEST("Separable Filter 2D 5x5 (u8), BORDER_REFLECT, 3 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT, 3>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT>)),
    TEST("Separable Filter 2D 5x5 (u8), BORDER_REFLECT, 4 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT, 4>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REFLECT>)),

    TEST("Separable Filter 2D 5x5 (u8), BORDER_REPLICATE, 1 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REPLICATE, 1>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REPLICATE>)),
    TEST("Separable Filter 2D 5x5 (u8), BORDER_REPLICATE, 2 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REPLICATE, 2>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REPLICATE>)),
    TEST("Separable Filter 2D 5x5 (u8), BORDER_REPLICATE, 3 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REPLICATE, 3>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REPLICATE>)),
    TEST("Separable Filter 2D 5x5 (u8), BORDER_REPLICATE, 4 channel", (test_separable_filter_2d<uint8_t, 5, cv::BORDER_REPLICATE, 4>), (exec_separable_filter_2d<uint8_t, 5, cv::BORDER_REPLICATE>)),

    TEST("Separable Filter 2D 5x5 (u16), BORDER_REFLECT_101, 1 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT_101, 1>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT_101>)),
    TEST("Separable Filter 2D 5x5 (u16), BORDER_REFLECT_101, 2 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT_101, 2>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT_101>)),
    TEST("Separable Filter 2D 5x5 (u16), BORDER_REFLECT_101, 3 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT_101, 3>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT_101>)),
    TEST("Separable Filter 2D 5x5 (u16), BORDER_REFLECT_101, 4 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT_101, 4>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT_101>)),

    TEST("Separable Filter 2D 5x5 (u16), BORDER_REFLECT, 1 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT, 1>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT>)),
    TEST("Separable Filter 2D 5x5 (u16), BORDER_REFLECT, 2 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT, 2>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT>)),
    TEST("Separable Filter 2D 5x5 (u16), BORDER_REFLECT, 3 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT, 3>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT>)),
    TEST("Separable Filter 2D 5x5 (u16), BORDER_REFLECT, 4 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT, 4>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REFLECT>)),

    TEST("Separable Filter 2D 5x5 (u16), BORDER_REPLICATE, 1 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REPLICATE, 1>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REPLICATE>)),
    TEST("Separable Filter 2D 5x5 (u16), BORDER_REPLICATE, 2 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REPLICATE, 2>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REPLICATE>)),
    TEST("Separable Filter 2D 5x5 (u16), BORDER_REPLICATE, 3 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REPLICATE, 3>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REPLICATE>)),
    TEST("Separable Filter 2D 5x5 (u16), BORDER_REPLICATE, 4 channel", (test_separable_filter_2d<uint16_t, 5, cv::BORDER_REPLICATE, 4>), (exec_separable_filter_2d<uint16_t, 5, cv::BORDER_REPLICATE>)),

    TEST("Separable Filter 2D 5x5 (s16), BORDER_REFLECT_101, 1 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT_101, 1>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT_101>)),
    TEST("Separable Filter 2D 5x5 (s16), BORDER_REFLECT_101, 2 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT_101, 2>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT_101>)),
    TEST("Separable Filter 2D 5x5 (s16), BORDER_REFLECT_101, 3 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT_101, 3>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT_101>)),
    TEST("Separable Filter 2D 5x5 (s16), BORDER_REFLECT_101, 4 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT_101, 4>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT_101>)),

    TEST("Separable Filter 2D 5x5 (s16), BORDER_REFLECT, 1 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT, 1>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT>)),
    TEST("Separable Filter 2D 5x5 (s16), BORDER_REFLECT, 2 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT, 2>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT>)),
    TEST("Separable Filter 2D 5x5 (s16), BORDER_REFLECT, 3 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT, 3>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT>)),
    TEST("Separable Filter 2D 5x5 (s16), BORDER_REFLECT, 4 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT, 4>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REFLECT>)),

    TEST("Separable Filter 2D 5x5 (s16), BORDER_REPLICATE, 1 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REPLICATE, 1>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REPLICATE>)),
    TEST("Separable Filter 2D 5x5 (s16), BORDER_REPLICATE, 2 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REPLICATE, 2>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REPLICATE>)),
    TEST("Separable Filter 2D 5x5 (s16), BORDER_REPLICATE, 3 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REPLICATE, 3>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REPLICATE>)),
    TEST("Separable Filter 2D 5x5 (s16), BORDER_REPLICATE, 4 channel", (test_separable_filter_2d<int16_t, 5, cv::BORDER_REPLICATE, 4>), (exec_separable_filter_2d<int16_t, 5, cv::BORDER_REPLICATE>)),
  };
  // clang-format on
  return tests;
}
