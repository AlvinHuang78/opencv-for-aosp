// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "compare_sc.h"

namespace kleidicv::sve2 {

template <typename ScalarType>
kleidicv_error_t compare_equal(const ScalarType *src_a, size_t src_a_stride,
                               const ScalarType *src_b, size_t src_b_stride,
                               ScalarType *dst, size_t dst_stride, size_t width,
                               size_t height) {
  return compare_sc<ComparatorEqual<ScalarType>>(
      src_a, src_a_stride, src_b, src_b_stride, dst, dst_stride, width, height);
}

template <typename ScalarType>
kleidicv_error_t compare_greater(const ScalarType *src_a, size_t src_a_stride,
                                 const ScalarType *src_b, size_t src_b_stride,
                                 ScalarType *dst, size_t dst_stride,
                                 size_t width, size_t height) {
  return compare_sc<ComparatorGreater<ScalarType>>(
      src_a, src_a_stride, src_b, src_b_stride, dst, dst_stride, width, height);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(name, type)                      \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t name<type>(       \
      const type *src_a, size_t src_a_stride, const type *src_b,       \
      size_t src_b_stride, type *dst, size_t dst_stride, size_t width, \
      size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(compare_equal, uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(compare_greater, uint8_t);

}  // namespace kleidicv::sve2
