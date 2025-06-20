// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "add_abs_with_threshold_sc.h"

namespace kleidicv::sme2 {

template <typename T>
KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
saturating_add_abs_with_threshold(const T *src_a, size_t src_a_stride,
                                  const T *src_b, size_t src_b_stride, T *dst,
                                  size_t dst_stride, size_t width,
                                  size_t height, T threshold) {
  return saturating_add_abs_with_threshold_sc(src_a, src_a_stride, src_b,
                                              src_b_stride, dst, dst_stride,
                                              width, height, threshold);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                   \
  saturating_add_abs_with_threshold<type>(                             \
      const type *src_a, size_t src_a_stride, const type *src_b,       \
      size_t src_b_stride, type *dst, size_t dst_stride, size_t width, \
      size_t height, type threshold)

KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);

}  // namespace kleidicv::sme2
