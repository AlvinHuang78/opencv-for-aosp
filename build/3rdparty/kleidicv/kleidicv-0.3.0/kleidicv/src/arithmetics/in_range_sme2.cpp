// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "in_range_sc.h"

namespace kleidicv::sme2 {

template <typename T>
KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
in_range(const T *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
         size_t width, size_t height, T lower_bound, T upper_bound) {
  return in_range_sc<T>(src, src_stride, dst, dst_stride, width, height,
                        lower_bound, upper_bound);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t in_range<type>(       \
      const type *src, size_t src_stride, uint8_t *dst, size_t dst_stride, \
      size_t width, size_t height, type lower_bound, type upper_bound)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::sme2
