// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "framework/utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <ios>
#include <iostream>
#include <limits>
#include <type_traits>

bool MockMallocToFail::enabled = false;

bool test::Options::are_long_running_tests_skipped_ = true;

namespace test {

template <typename ElementType>
void dump(const TwoDimensional<ElementType> *elements) {
  if (!elements) {
    return;
  }

  for (size_t row = 0; row < elements->height(); ++row) {
    for (size_t column = 0; column < elements->width(); ++column) {
      if constexpr (std::is_integral_v<ElementType>) {
        auto mask =
            std::numeric_limits<std::make_unsigned_t<ElementType>>::max();
        ElementType value = elements->at(row, column)[0];
        // Unary + is used to ensure values are printed as integers, not chars
        std::cout << std::setw(2 * sizeof(ElementType)) << std::setfill('0')
                  << std::hex << +(value & mask) << " ";
      } else {
        std::cout << elements->at(row, column)[0] << " ";
      }
    }

    std::cout << std::endl;
  }

  if ((elements->height() > 0) && (elements->width() > 0)) {
    std::cout << std::endl << std::flush;
  }
}

template void dump<int8_t>(const TwoDimensional<int8_t> *);
template void dump<uint8_t>(const TwoDimensional<uint8_t> *);
template void dump<int16_t>(const TwoDimensional<int16_t> *);
template void dump<uint16_t>(const TwoDimensional<uint16_t> *);
template void dump<int32_t>(const TwoDimensional<int32_t> *);
template void dump<uint32_t>(const TwoDimensional<uint32_t> *);
template void dump<int64_t>(const TwoDimensional<int64_t> *);
template void dump<uint64_t>(const TwoDimensional<uint64_t> *);
template void dump<float>(const TwoDimensional<float> *);
template void dump<double>(const TwoDimensional<double> *);

const std::array<kleidicv_border_values_t, 1> &default_border_values() {
  static const std::array<kleidicv_border_values_t, 1> kDefaultBorderValues{
      {{0, 0, 0, 0}}};
  return kDefaultBorderValues;
}

std::array<test::ArrayLayout, 7> small_array_layouts(size_t min_width,
                                                     size_t min_height) {
  size_t vl = test::Options::vector_length();
  size_t width = std::max(min_width, vl);

  return {{
      // clang-format off
      //          width,         height,  padding, channels
      {       min_width,     min_height,        0,        1},
      {   min_width * 2,     min_height,        0,        2},
      {   min_width * 3,     min_height,       vl,        3},
      {   min_width * 3,     min_height,        0,        1},
      {       width + 1,     min_height,        0,        1},
      {       2 * width,     min_height,       vl,        1},
      {       4 * width,     min_height,       vl,        1},
      // clang-format on
  }};
}

std::array<test::ArrayLayout, 14> default_array_layouts(size_t min_width,
                                                        size_t min_height) {
  size_t vl = test::Options::vector_length();
  size_t width = std::max(min_width, vl);
  size_t height = std::max(min_height, vl);

  return {{
      // clang-format off
      //         width,         height,  padding, channels
      {      min_width,         height,        0,        1},
      {  min_width * 2,     min_height,        0,        2},
      {  min_width * 3,     min_height,        0,        3},
      {  min_width * 4,     min_height,        0,        4},
      {      min_width,         height,       vl,        1},
      {  min_width * 2,     min_height,       vl,        2},
      {  min_width * 3,     min_height,       vl,        3},
      {  min_width * 4,     min_height,       vl,        4},
      {  4 * width + 1, min_height + 1,        0,        1},
      {2 * (width - 1), min_height + 1,        0,        2},
      {3 * (width - 1), min_height + 2,        0,        3},
      {  4 * width + 1, min_height + 1,       vl,        1},
      {2 * (width - 1), min_height + 1,       vl,        2},
      {3 * (width - 1), min_height + 2,       vl,        3},
      // clang-format on
  }};
}

std::array<test::ArrayLayout, 6> default_1channel_array_layouts(
    size_t min_width, size_t min_height) {
  size_t vl = test::Options::vector_length();
  size_t width = std::max(min_width, vl);
  size_t height = std::max(min_height, vl);

  return {{
      // clang-format off
      //         width,         height,  padding, channels
      {      min_width,         height,        0,        1},
      {      min_width,         height,       vl,        1},
      {       width + 1,     min_height,        0,        1},
      {       2 * width,     min_height,       vl,        1},
      {  2 * width + 1, min_height + 1,        0,        1},
      {  4 * width + 1, min_height + 1,       vl,        1},
      // clang-format on
  }};
}

}  // namespace test
