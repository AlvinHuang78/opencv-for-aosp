// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "threshold_sc.h"

namespace kleidicv::sve2 {

template <typename T>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
threshold_binary(const T *src, size_t src_stride, T *dst, size_t dst_stride,
                 size_t width, size_t height, T threshold, T value) {
  return threshold_binary_sc(src, src_stride, dst, dst_stride, width, height,
                             threshold, value);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                  \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t threshold_binary<type>( \
      const type *src, size_t src_stride, type *dst, size_t dst_stride,      \
      size_t width, size_t height, type threshold, type value)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);

}  // namespace kleidicv::sve2
