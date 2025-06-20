// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_WORKSPACE_BORDER_15X15_H
#define KLEIDICV_WORKSPACE_BORDER_15X15_H

#include "border_types.h"
#include "kleidicv/kleidicv.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Border offsets for fixed-size filters.
template <typename T, const size_t S>
class FixedBorderInfo;

// Border offsets for 15x15 filters.
template <typename T>
class FixedBorderInfo<T, 15UL> final {
 public:
  // Simple object holding read-only constant offsets.
  class Offsets final {
   public:
    // NOLINTBEGIN(hicpp-member-init)
    Offsets() = default;
    // NOLINTEND(hicpp-member-init)

    Offsets(ptrdiff_t o0, ptrdiff_t o1, ptrdiff_t o2, ptrdiff_t o3,
            ptrdiff_t o4, ptrdiff_t o5, ptrdiff_t o6, ptrdiff_t o7,
            ptrdiff_t o8, ptrdiff_t o9, ptrdiff_t o10, ptrdiff_t o11,
            ptrdiff_t o12, ptrdiff_t o13, ptrdiff_t o14)
        : offsets_{o0, o1, o2,  o3,  o4,  o5,  o6, o7,
                   o8, o9, o10, o11, o12, o13, o14} {}

    ptrdiff_t c0() const { return offsets_[0]; }
    ptrdiff_t c1() const { return offsets_[1]; }
    ptrdiff_t c2() const { return offsets_[2]; }
    ptrdiff_t c3() const { return offsets_[3]; }
    ptrdiff_t c4() const { return offsets_[4]; }
    ptrdiff_t c5() const { return offsets_[5]; }
    ptrdiff_t c6() const { return offsets_[6]; }
    ptrdiff_t c7() const { return offsets_[7]; }
    ptrdiff_t c8() const { return offsets_[8]; }
    ptrdiff_t c9() const { return offsets_[9]; }
    ptrdiff_t c10() const { return offsets_[10]; }
    ptrdiff_t c11() const { return offsets_[11]; }
    ptrdiff_t c12() const { return offsets_[12]; }
    ptrdiff_t c13() const { return offsets_[13]; }
    ptrdiff_t c14() const { return offsets_[14]; }

   private:
    ptrdiff_t offsets_[15];
  };

  FixedBorderInfo(size_t height, FixedBorderType border_type)
      : height_(height), border_type_(border_type) {}

  // Returns offsets without the influence of any border.
  Offsets offsets_without_border() const KLEIDICV_STREAMING_COMPATIBLE {
    return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
  }

  // NOLINTBEGIN(readability-function-cognitive-complexity)
  // Returns offsets for columns affected by left border.
  Offsets offsets_with_left_border(size_t column_index) const
      KLEIDICV_STREAMING_COMPATIBLE {
    switch (border_type_) {
      case FixedBorderType::REPLICATE:
        if (column_index == 0) {
          return get(0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 1) {
          return get(-1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 2) {
          return get(-2, -2, -2, -2, -2, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 3) {
          return get(-3, -3, -3, -3, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 4) {
          return get(-4, -4, -4, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 5) {
          return get(-5, -5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else {
          return get(-6, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        }
        break;

      case FixedBorderType::REFLECT:
        if (column_index == 0) {
          return get(6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 1) {
          return get(4, 3, 2, 1, 0, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 2) {
          return get(2, 1, 0, -1, -2, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 3) {
          return get(0, -1, -2, -3, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 4) {
          return get(-2, -3, -4, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 5) {
          return get(-4, -5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else {
          return get(-6, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        }
        break;

      case FixedBorderType::WRAP:
        if (column_index == 0) {
          return get(height_ - 7, height_ - 6, height_ - 5, height_ - 4,
                     height_ - 3, height_ - 2, height_ - 1, 0, 1, 2, 3, 4, 5, 6,
                     7);
        } else if (column_index == 1) {
          return get(height_ - 7, height_ - 6, height_ - 5, height_ - 4,
                     height_ - 3, height_ - 2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 2) {
          return get(height_ - 7, height_ - 6, height_ - 5, height_ - 4,
                     height_ - 3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 3) {
          return get(height_ - 7, height_ - 6, height_ - 5, height_ - 4, -3, -2,
                     -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 4) {
          return get(height_ - 7, height_ - 6, height_ - 5, -4, -3, -2, -1, 0,
                     1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 5) {
          return get(height_ - 7, height_ - 6, -5, -4, -3, -2, -1, 0, 1, 2, 3,
                     4, 5, 6, 7);
        } else {
          return get(height_ - 7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
                     7);
        }
        break;

      case FixedBorderType::REVERSE:
        if (column_index == 0) {
          return get(7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 1) {
          return get(5, 4, 3, 2, 1, 0, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 2) {
          return get(3, 2, 1, 0, -1, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 3) {
          return get(1, 0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 4) {
          return get(-1, -2, -3, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else if (column_index == 5) {
          return get(-3, -4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        } else {
          return get(-5, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
        }
        break;
    }
    // Unreachable. Compiler should emit a warning-as-error if any cases are
    // uncovered above.
    return Offsets{};  // GCOVR_EXCL_LINE
  }

  // Returns offsets for columns affected by right border.
  Offsets offsets_with_right_border(size_t column_index) const
      KLEIDICV_STREAMING_COMPATIBLE {
    switch (border_type_) {
      case FixedBorderType::REPLICATE:
        if (column_index == (height_ - 7)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6);
        } else if (column_index == (height_ - 6)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5, 5);
        } else if (column_index == (height_ - 5)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4, 4, 4);
        } else if (column_index == (height_ - 4)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3, 3, 3, 3);
        } else if (column_index == (height_ - 3)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2, 2, 2, 2, 2);
        } else if (column_index == (height_ - 2)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 1, 1, 1, 1, 1, 1);
        } else {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0);
        }
        break;

      case FixedBorderType::REFLECT:
        if (column_index == (height_ - 7)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6);
        } else if (column_index == (height_ - 6)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5, 4);
        } else if (column_index == (height_ - 5)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4, 3, 2);
        } else if (column_index == (height_ - 4)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 3, 2, 1, 0);
        } else if (column_index == (height_ - 3)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2, 1, 0, -1, -2);
        } else if (column_index == (height_ - 2)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 1, 0, -1, -2, -3, -4);
        } else {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 0, -1, -2, -3, -4, -5, -6);
        }
        break;

      case FixedBorderType::WRAP:
        if (column_index == (height_ - 7)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
                     7 - height_);
        } else if (column_index == (height_ - 6)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6 - height_,
                     7 - height_);
        } else if (column_index == (height_ - 5)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 - height_,
                     6 - height_, 7 - height_);
        } else if (column_index == (height_ - 4)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4 - height_,
                     5 - height_, 6 - height_, 7 - height_);
        } else if (column_index == (height_ - 3)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3 - height_,
                     4 - height_, 5 - height_, 6 - height_, 7 - height_);
        } else if (column_index == (height_ - 2)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2 - height_, 3 - height_,
                     4 - height_, 5 - height_, 6 - height_, 7 - height_);
        } else {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1 - height_, 2 - height_,
                     3 - height_, 4 - height_, 5 - height_, 6 - height_,
                     7 - height_);
        }
        break;

      case FixedBorderType::REVERSE:
        if (column_index == (height_ - 7)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 5);
        } else if (column_index == (height_ - 6)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 4, 3);
        } else if (column_index == (height_ - 5)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 3, 2, 1);
        } else if (column_index == (height_ - 4)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0, -1);
        } else if (column_index == (height_ - 3)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 1, 0, -1, -2, -3);
        } else if (column_index == (height_ - 2)) {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, 1, 0, -1, -2, -3, -4, -5);
        } else {
          return get(-7, -6, -5, -4, -3, -2, -1, 0, -1, -2, -3, -4, -5, -6, -7);
        }
        break;
    }
    // Unreachable. Compiler should emit a warning-as-error if any cases are
    // uncovered above.
    return Offsets{};  // GCOVR_EXCL_LINE
  }
  // NOLINTEND(readability-function-cognitive-complexity)

  // Returns offsets for rows or columns affected by any border.
  Offsets offsets_with_border(size_t row_or_column_index) const
      KLEIDICV_STREAMING_COMPATIBLE {
    if (row_or_column_index <= 6U) {
      // Rows and columns have the same offsets.
      return offsets_with_left_border(row_or_column_index);
    }
    if (row_or_column_index >= (height_ - 7U)) {
      // Rows and columns have the same offsets.
      return offsets_with_right_border(row_or_column_index);
    }
    return offsets_without_border();
  }

 private:
  // Takes care of static signed to unsigned casts.
  Offsets get(ptrdiff_t o0, ptrdiff_t o1, ptrdiff_t o2, ptrdiff_t o3,
              ptrdiff_t o4, ptrdiff_t o5, ptrdiff_t o6, ptrdiff_t o7,
              ptrdiff_t o8, ptrdiff_t o9, ptrdiff_t o10, ptrdiff_t o11,
              ptrdiff_t o12, ptrdiff_t o13,
              ptrdiff_t o14) const KLEIDICV_STREAMING_COMPATIBLE {
    return Offsets{o0, o1, o2,  o3,  o4,  o5,  o6, o7,
                   o8, o9, o10, o11, o12, o13, o14};
  }

  size_t height_;
  FixedBorderType border_type_;
};  // end of class FixedBorderInfo<T, 15UL>

// Shorthand for 15x15 filter border type.
template <typename T>
using FixedBorderInfo15x15 = FixedBorderInfo<T, 15UL>;

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_WORKSPACE_BORDER_15X15_H
