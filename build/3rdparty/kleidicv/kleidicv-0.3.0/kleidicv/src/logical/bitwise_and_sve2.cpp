// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "bitwise_and_sc.h"

namespace kleidicv::sve2 {

template <typename T>
kleidicv_error_t bitwise_and(const T *src_a, size_t src_a_stride,
                             const T *src_b, size_t src_b_stride, T *dst,
                             size_t dst_stride, size_t width, size_t height) {
  return bitwise_and_sc(src_a, src_a_stride, src_b, src_b_stride, dst,
                        dst_stride, width, height);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                             \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t bitwise_and<type>( \
      const type *src_a, size_t src_a_stride, const type *src_b,        \
      size_t src_b_stride, type *dst, size_t dst_stride, size_t width,  \
      size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);

}  // namespace kleidicv::sve2
