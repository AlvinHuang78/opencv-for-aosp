// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <type_traits>
#include <vector>

#include "opencv2/core/hal/interface.h"
#include "opencv2/imgproc/hal/interface.h"
#include "tests.h"

const int kMaxHeight = 32, kMaxWidth = 32;

template <class ScalarType>
static cv::Mat get_source_mat(int format) {
  auto generate_source = [&]() {
    cv::Mat m(kMaxHeight, kMaxWidth, format);
    for (size_t row = 0; row < kMaxHeight; ++row) {
      for (size_t column = 0; column < kMaxWidth; ++column) {
        m.at<ScalarType>(row, column) =
            (row * kMaxWidth + column) % std::numeric_limits<ScalarType>::max();
      }
    }
    return m;
  };
  static cv::Mat source = generate_source();
  return source;
}

// BorderValue is interpreted as 1/1000, i.e. 500 for 0.5
template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
cv::Mat exec_remap_s16(cv::Mat& mapxy_mat) {
  cv::Mat source_mat = get_source_mat<ScalarType>(Format);
  cv::Mat result(mapxy_mat.rows, mapxy_mat.cols, Format);
  cv::Mat empty;
  remap(source_mat, result, mapxy_mat, empty, Interpolation, BorderMode,
        BorderValue / 1000.0);
  return result;
}

#if MANAGER
template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
bool test_remap_s16(int index, RecreatedMessageQueue& request_queue,
                    RecreatedMessageQueue& reply_queue) {
  cv::Mat source_mat = get_source_mat<ScalarType>(Format);
  cv::RNG rng(0);

  for (size_t w = 5; w <= kMaxWidth; w += 3) {
    for (size_t h = 5; h <= kMaxHeight; h += 2) {
      cv::Mat source_mat = get_source_mat<ScalarType>(Format);
      cv::Mat mapxy_mat(w, h, CV_16SC2);
      rng.fill(mapxy_mat, cv::RNG::UNIFORM, -3, kMaxWidth + 3);

      cv::Mat actual_mat = exec_remap_s16<ScalarType, Format, Interpolation,
                                          BorderMode, BorderValue>(mapxy_mat);
      cv::Mat expected_mat = get_expected_from_subordinate(
          index, request_queue, reply_queue, mapxy_mat);

      bool success =
          (CV_MAT_DEPTH(Format) == CV_8U &&
           !are_matrices_different<uint8_t>(1, actual_mat, expected_mat)) ||
          (CV_MAT_DEPTH(Format) == CV_16U &&
           !are_matrices_different<uint16_t>(1, actual_mat, expected_mat));
      if (!success) {
        fail_print_matrices(w, h, source_mat, actual_mat, expected_mat);
        return true;
      }
    }
  }
  return false;
}
#endif

// BorderValue is interpreted as 1/1000, i.e. 500 for 0.5
template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
cv::Mat exec_remap_s16point5(cv::Mat& map_mat) {
  cv::Mat empty;
  // integer part is 16SC2, that is twice as much data as the fractional part,
  // 16UC1
  int height = map_mat.rows * 2 / 3;
  cv::Mat mapxy_mat = map_mat.rowRange(0, height);
  ushort* p_frac = map_mat.rowRange(height, map_mat.rows).ptr<ushort>();
  cv::Mat mapfrac_mat{height, map_mat.cols, CV_16UC1, p_frac};
  cv::Mat result(mapxy_mat.rows, mapxy_mat.cols, Format);
  cv::Mat source_mat = get_source_mat<ScalarType>(Format);
  remap(source_mat, result, mapxy_mat, mapfrac_mat, Interpolation, BorderMode,
        BorderValue / 1000.0);
  return result;
}

#if MANAGER
template <class ScalarType, int Format, int Interpolation, int BorderMode,
          int BorderValue>
bool test_remap_s16point5(int index, RecreatedMessageQueue& request_queue,
                          RecreatedMessageQueue& reply_queue) {
  cv::Mat source_mat = get_source_mat<ScalarType>(Format);
  cv::RNG rng(0);

  for (int w = 5; w <= kMaxWidth; ++w) {
    for (int h = 6; h <= kMaxHeight; h += 2) {  // h must be even
      cv::Mat map_mat(h * 3 / 2, w, CV_16SC2);
      cv::Mat mapxy_mat = map_mat.rowRange(0, h);
      ushort* p_frac = map_mat.rowRange(h, map_mat.rows).ptr<ushort>();
      cv::Mat mapfrac_mat{h, map_mat.cols, CV_16UC1, p_frac};
      rng.fill(mapxy_mat, cv::RNG::UNIFORM, -3, kMaxWidth + 3);
      // Test out of range fractional part too
      rng.fill(mapfrac_mat, cv::RNG::UNIFORM, 0, cv::INTER_TAB_SIZE2 * 3 / 2);

      cv::Mat actual_mat =
          exec_remap_s16point5<ScalarType, Format, Interpolation, BorderMode,
                               BorderValue>(map_mat);
      cv::Mat expected_mat = get_expected_from_subordinate(
          index, request_queue, reply_queue, map_mat);

      bool success =
          (CV_MAT_DEPTH(Format) == CV_8U &&
           !are_matrices_different<uint8_t>(1, actual_mat, expected_mat)) ||
          (CV_MAT_DEPTH(Format) == CV_16U &&
           !are_matrices_different<uint16_t>(1, actual_mat, expected_mat));
      if (!success) {
        fail_print_matrices(w, h, source_mat, actual_mat, expected_mat);
        return true;
      }
    }
  }
  return false;
}
#endif

std::vector<test>& remap_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("RemapS16 uint8", (test_remap_s16<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>), (exec_remap_s16<uint8_t, CV_8UC1, cv::INTER_NEAREST, cv::BORDER_REPLICATE, 0>)),
    TEST("RemapS16Point5 uint8", (test_remap_s16point5<uint8_t, CV_8UC1, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>), (exec_remap_s16point5<uint8_t, CV_8UC1, cv::INTER_LINEAR, cv::BORDER_REPLICATE, 0>)),
  };
  // clang-format on
  return tests;
}
