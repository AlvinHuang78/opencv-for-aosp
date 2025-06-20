// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_WORKSPACE_BORDER_5X5_H
#define KLEIDICV_WORKSPACE_BORDER_5X5_H

#include "border_types.h"
#include "kleidicv/kleidicv.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Border offsets for fixed-size filters.
template <typename T, const size_t S>
class FixedBorderInfo;

// Border offsets for 5x5 filters.
template <typename T>
class FixedBorderInfo<T, 5UL> final {
 public:
  // Simple object holding read-only constant offsets.
  class Offsets final {
   public:
    // NOLINTBEGIN(hicpp-member-init)
    Offsets() = default;
    // NOLINTEND(hicpp-member-init)

    Offsets(ptrdiff_t o0, ptrdiff_t o1, ptrdiff_t o2, ptrdiff_t o3,
            ptrdiff_t o4)
        : offsets_{o0, o1, o2, o3, o4} {}

    ptrdiff_t c0() const { return offsets_[0]; }
    ptrdiff_t c1() const { return offsets_[1]; }
    ptrdiff_t c2() const { return offsets_[2]; }
    ptrdiff_t c3() const { return offsets_[3]; }
    ptrdiff_t c4() const { return offsets_[4]; }

   private:
    ptrdiff_t offsets_[5];
  };

  FixedBorderInfo(size_t height, FixedBorderType border_type)
      : height_(height), border_type_(border_type) {}

  // Returns offsets without the influence of any border.
  Offsets offsets_without_border() const KLEIDICV_STREAMING_COMPATIBLE {
    return get(-2, -1, 0, 1, 2);
  }

  // Returns offsets for columns affected by left border.
  Offsets offsets_with_left_border(size_t column_index) const
      KLEIDICV_STREAMING_COMPATIBLE {
    switch (border_type_) {
      case FixedBorderType::REPLICATE:
        if (column_index == 0) {
          return get(0, 0, 0, 1, 2);
        } else {
          return get(-1, -1, 0, 1, 2);
        }
        break;

      case FixedBorderType::REFLECT:
        if (column_index == 0) {
          return get(1, 0, 0, 1, 2);
        } else {
          return get(-1, -1, 0, 1, 2);
        }
        break;

      case FixedBorderType::WRAP:
        if (column_index == 0) {
          return get(height_ - 2, height_ - 1, 0, 1, 2);
        } else {
          return get(height_ - 2, -1, 0, 1, 2);
        }
        break;

      case FixedBorderType::REVERSE:
        if (column_index == 0) {
          return get(2, 1, 0, 1, 2);
        } else {
          return get(0, -1, 0, 1, 2);
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
        if (column_index == (height_ - 2)) {
          return get(-2, -1, 0, 1, 1);
        } else {
          return get(-2, -1, 0, 0, 0);
        }
        break;

      case FixedBorderType::REFLECT:
        if (column_index == (height_ - 2)) {
          return get(-2, -1, 0, 1, 1);
        } else {
          return get(-2, -1, 0, 0, -1);
        }
        break;

      case FixedBorderType::WRAP:
        if (column_index == (height_ - 2)) {
          return get(-2, -1, 0, 1, 2 - height_);
        } else {
          return get(-2, -1, 0, 1 - height_, 2 - height_);
        }
        break;

      case FixedBorderType::REVERSE:
        if (column_index == (height_ - 2)) {
          return get(-2, -1, 0, 1, 0);
        } else {
          return get(-2, -1, 0, -1, -2);
        }
        break;
    }
    // Unreachable. Compiler should emit a warning-as-error if any cases are
    // uncovered above.
    return Offsets{};  // GCOVR_EXCL_LINE
  }

  // Returns offsets for rows or columns affected by any border.
  Offsets offsets_with_border(size_t row_or_column_index) const
      KLEIDICV_STREAMING_COMPATIBLE {
    if (row_or_column_index <= 1U) {
      // Rows and columns have the same offsets.
      return offsets_with_left_border(row_or_column_index);
    }
    if (row_or_column_index >= (height_ - 2U)) {
      // Rows and columns have the same offsets.
      return offsets_with_right_border(row_or_column_index);
    }
    return offsets_without_border();
  }

 private:
  // Takes care of static signed to unsigned casts.
  Offsets get(ptrdiff_t o0, ptrdiff_t o1, ptrdiff_t o2, ptrdiff_t o3,
              ptrdiff_t o4) const KLEIDICV_STREAMING_COMPATIBLE {
    return Offsets{o0, o1, o2, o3, o4};
  }

  size_t height_;
  FixedBorderType border_type_;
};  // end of class FixedBorderInfo<T, 5UL>

// Shorthand for 5x5 filter border type.
template <typename T>
using FixedBorderInfo5x5 = FixedBorderInfo<T, 5UL>;

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_WORKSPACE_BORDER_5X5_H
