// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <vector>

#include "tests.h"

template <bool Signed>
cv::Mat exec_float32_to_int8(cv::Mat& input) {
  cv::Mat result;
  input.convertTo(result, Signed ? CV_8SC1 : CV_8UC1);
  return result;
}

cv::Mat exec_int8_to_float32(cv::Mat& input) {
  cv::Mat result;
  input.convertTo(result, CV_32FC1);
  return result;
}

#if MANAGER
template <bool Signed, size_t Channels>
bool test_float32_to_int8_random(int index,
                                 RecreatedMessageQueue& request_queue,
                                 RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t x = 5; x <= 16; ++x) {
    for (size_t y = 5; y <= 16; ++y) {
      cv::Mat input(x, y, CV_32FC(Channels));
      rng.fill(input, cv::RNG::UNIFORM, Signed ? -1000 : 0, 1000);

      cv::Mat actual = exec_float32_to_int8<Signed>(input);
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

template <bool Signed, size_t Channels>
bool test_int8_to_float32_random(int index,
                                 RecreatedMessageQueue& request_queue,
                                 RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t x = 5; x <= 16; ++x) {
    for (size_t y = 5; y <= 16; ++y) {
      cv::Mat input(x, y, Signed ? CV_8SC(Channels) : CV_8UC(Channels));
      rng.fill(input, cv::RNG::UNIFORM, Signed ? -1000 : 0, 1000);

      cv::Mat actual = exec_int8_to_float32(input);
      cv::Mat expected = get_expected_from_subordinate(index, request_queue,
                                                       reply_queue, input);

      if (are_matrices_different<float>(0, actual, expected)) {
        fail_print_matrices(x, y, input, actual, expected);
        return true;
      }
    }
  }

  return false;
}

static constexpr int custom_data_int8_height = 1;
static constexpr int custom_data_int8_width = 7;

static int8_t
    custom_data_int8[custom_data_int8_height * custom_data_int8_width] = {
        // clang-format off
        -128, -127, -1, 0, 1, 126, 127
        // clang-format on
};

static uint8_t
    custom_data_uint8[custom_data_int8_height * custom_data_int8_width] = {
        // clang-format off
        0, 1, 126, 127, 128, 254, 255
        // clang-format on
};

template <bool Signed>
bool test_float32_to_int8_custom(int index,
                                 RecreatedMessageQueue& request_queue,
                                 RecreatedMessageQueue& reply_queue) {
  cv::Mat input(custom_data_float_height, custom_data_float_width, CV_32FC1,
                custom_data_float);

  cv::Mat actual = exec_float32_to_int8<Signed>(input);
  cv::Mat expected =
      get_expected_from_subordinate(index, request_queue, reply_queue, input);

  if (are_matrices_different<uint8_t>(0, actual, expected)) {
    fail_print_matrices(custom_data_float_height, custom_data_float_width,
                        input, actual, expected);
    return true;
  }

  return false;
}

template <bool Signed>
bool test_int8_to_float32_custom(int index,
                                 RecreatedMessageQueue& request_queue,
                                 RecreatedMessageQueue& reply_queue) {
  cv::Mat input(custom_data_int8_height, custom_data_int8_width,
                Signed ? CV_8SC1 : CV_8UC1,
                Signed ? static_cast<void*>(custom_data_int8)
                       : static_cast<void*>(custom_data_uint8));

  cv::Mat actual = exec_int8_to_float32(input);
  cv::Mat expected =
      get_expected_from_subordinate(index, request_queue, reply_queue, input);

  if (are_matrices_different<float>(0, actual, expected)) {
    fail_print_matrices(custom_data_int8_height, custom_data_int8_width, input,
                        actual, expected);
    return true;
  }

  return false;
}
#endif

std::vector<test>& float_conversion_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Float32 to Signed Int8, random, 1 channel", (test_float32_to_int8_random<true, 1>), exec_float32_to_int8<true>),
    TEST("Float32 to Signed Int8, random, 2 channel", (test_float32_to_int8_random<true, 2>), exec_float32_to_int8<true>),
    TEST("Float32 to Signed Int8, random, 3 channel", (test_float32_to_int8_random<true, 3>), exec_float32_to_int8<true>),
    TEST("Float32 to Signed Int8, random, 4 channel", (test_float32_to_int8_random<true, 4>), exec_float32_to_int8<true>),

    TEST("Float32 to Unsigned Int8, random, 1 channel", (test_float32_to_int8_random<false, 1>), exec_float32_to_int8<false>),
    TEST("Float32 to Unsigned Int8, random, 2 channel", (test_float32_to_int8_random<false, 2>), exec_float32_to_int8<false>),
    TEST("Float32 to Unsigned Int8, random, 3 channel", (test_float32_to_int8_random<false, 3>), exec_float32_to_int8<false>),
    TEST("Float32 to Unsigned Int8, random, 4 channel", (test_float32_to_int8_random<false, 4>), exec_float32_to_int8<false>),

    TEST("Float32 to Signed Int8, custom (special)", test_float32_to_int8_custom<true>, exec_float32_to_int8<true>),
    TEST("Float32 to Unsigned Int8, custom (special)", test_float32_to_int8_custom<false>, exec_float32_to_int8<false>),

    TEST("Signed Int8 to Float32, random, 1 channel", (test_int8_to_float32_random<true, 1>), exec_int8_to_float32),
    TEST("Signed Int8 to Float32, random, 2 channel", (test_int8_to_float32_random<true, 2>), exec_int8_to_float32),
    TEST("Signed Int8 to Float32, random, 3 channel", (test_int8_to_float32_random<true, 3>), exec_int8_to_float32),
    TEST("Signed Int8 to Float32, random, 4 channel", (test_int8_to_float32_random<true, 4>), exec_int8_to_float32),

    TEST("Unsigned Int8 to Float32, random, 1 channel", (test_int8_to_float32_random<false, 1>), exec_int8_to_float32),
    TEST("Unsigned Int8 to Float32, random, 2 channel", (test_int8_to_float32_random<false, 2>), exec_int8_to_float32),
    TEST("Unsigned Int8 to Float32, random, 3 channel", (test_int8_to_float32_random<false, 3>), exec_int8_to_float32),
    TEST("Unsigned Int8 to Float32, random, 4 channel", (test_int8_to_float32_random<false, 4>), exec_int8_to_float32),

    TEST("Signed Int8 to Float32, custom", test_int8_to_float32_custom<true>, exec_int8_to_float32),
    TEST("Unsigned Int8 to Float32, custom", test_int8_to_float32_custom<false>, exec_int8_to_float32),
  };
  // clang-format on
  return tests;
}
