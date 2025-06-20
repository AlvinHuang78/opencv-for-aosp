// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_WORKSPACE_BORDER_3X3_H
#define KLEIDICV_WORKSPACE_BORDER_3X3_H

#include "border_types.h"
#include "kleidicv/kleidicv.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Border offsets for fixed-size filters.
template <typename T, const size_t S>
class FixedBorderInfo;

// Border offsets for 3x3 filters.
template <typename T>
class FixedBorderInfo<T, 3UL> final {
 public:
  // Simple object holding read-only constant offsets.
  class Offsets final {
   public:
    Offsets() = default;

    Offsets(ptrdiff_t o0, ptrdiff_t o1, ptrdiff_t o2) : offsets_{o0, o1, o2} {}

    ptrdiff_t c0() const { return offsets_[0]; }
    ptrdiff_t c1() const { return offsets_[1]; }
    ptrdiff_t c2() const { return offsets_[2]; }

   private:
    ptrdiff_t offsets_[3];
  };

  FixedBorderInfo(size_t height, FixedBorderType border_type)
      : height_(height), border_type_(border_type) {}

  // Returns offsets without the influence of any border.
  Offsets offsets_without_border() const { return get(-1, 0, 1); }

  // Returns offsets for columns affected by left border.
  Offsets offsets_with_left_border(size_t /* column_index */) const
      KLEIDICV_STREAMING_COMPATIBLE {
    switch (border_type_) {
      case FixedBorderType::REPLICATE:
      case FixedBorderType::REFLECT:
        return get(0, 0, 1);
        break;

      case FixedBorderType::WRAP:
        return get(height_ - 1, 0, 1);
        break;

      case FixedBorderType::REVERSE:
        return get(1, 0, 1);
        break;
    }
    // Unreachable. Compiler should emit a warning-as-error if any cases are
    // uncovered above.
    return Offsets{};  // GCOVR_EXCL_LINE
  }

  // Returns offsets for columns affected by right border.
  Offsets offsets_with_right_border(size_t /* column_index */) const
      KLEIDICV_STREAMING_COMPATIBLE {
    switch (border_type_) {
      case FixedBorderType::REPLICATE:
      case FixedBorderType::REFLECT:
        return get(-1, 0, 0);
        break;

      case FixedBorderType::WRAP:
        return get(-1, 0, 1 - height_);
        break;

      case FixedBorderType::REVERSE:
        return get(-1, 0, -1);
        break;
    }
    // Unreachable. Compiler should emit a warning-as-error if any cases are
    // uncovered above.
    return Offsets{};  // GCOVR_EXCL_LINE
  }

  // Returns offsets for rows or columns affected by any border.
  Offsets offsets_with_border(size_t row_or_column_index) const
      KLEIDICV_STREAMING_COMPATIBLE {
    if (row_or_column_index == 0U) {
      // Rows and columns have the same offsets.
      return offsets_with_left_border(row_or_column_index);
    }
    if (row_or_column_index == (height_ - 1U)) {
      // Rows and columns have the same offsets.
      return offsets_with_right_border(row_or_column_index);
    }
    return offsets_without_border();
  }

 private:
  // Takes care of static signed to unsigned casts.
  Offsets get(ptrdiff_t o0, ptrdiff_t o1, ptrdiff_t o2) const {
    return Offsets{o0, o1, o2};
  }

  size_t height_;
  FixedBorderType border_type_;
};  // end of class FixedBorderInfo<T, 3UL>

// Shorthand for 3x3 filter border type.
template <typename T>
using FixedBorderInfo3x3 = FixedBorderInfo<T, 3UL>;

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_WORKSPACE_BORDER_3X3_H
