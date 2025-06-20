// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_OPENCV_CONFORMITY_TESTS_H_
#define KLEIDICV_OPENCV_CONFORMITY_TESTS_H_

#include <vector>

#include "utils.h"

std::vector<test>& binary_op_tests_get();
std::vector<test>& cvtcolor_tests_get();
std::vector<test>& separable_filter_2d_tests_get();
std::vector<test>& gaussian_blur_tests_get();
std::vector<test>& rgb2yuv_tests_get();
std::vector<test>& yuv2rgb_tests_get();
std::vector<test>& sobel_tests_get();
std::vector<test>& exp_tests_get();
std::vector<test>& float_conversion_tests_get();
std::vector<test>& resize_tests_get();
std::vector<test>& scale_tests_get();
std::vector<test>& min_max_tests_get();
std::vector<test>& in_range_tests_get();
std::vector<test>& remap_tests_get();
std::vector<test>& warp_perspective_tests_get();
std::vector<test>& blur_and_downsample_tests_get();
std::vector<test>& scharr_interleaved_tests_get();

#endif  // KLEIDICV_OPENCV_CONFORMITY_TESTS_H_
