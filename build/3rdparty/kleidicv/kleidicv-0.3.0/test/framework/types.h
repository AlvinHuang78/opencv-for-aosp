// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TEST_FRAMEWORK_TYPES_H_
#define KLEIDICV_TEST_FRAMEWORK_TYPES_H_

#include <cstddef>

namespace test {

// Represents a point with two non-negative coordinates where 'x' is the
// horizontal and 'y' is the vertical component.
struct Point {
  size_t x;
  size_t y;
};  // end of struct Point

// Describes the layout of a two-dimensional array.
struct ArrayLayout {
  size_t width;
  size_t height;
  size_t padding;
  size_t channels;
};  // end of struct ArrayLayout

}  // namespace test

#endif  // KLEIDICV_TEST_FRAMEWORK_TYPES_H_
