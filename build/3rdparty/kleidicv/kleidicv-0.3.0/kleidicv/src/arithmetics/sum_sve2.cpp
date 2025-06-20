// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "sum_sc.h"

namespace kleidicv::sve2 {

template <typename T>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t sum(const T *src, size_t src_stride,
                                              size_t width, size_t height,
                                              T *sum) {
  return sum_sc(src, src_stride, width, height, sum);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t sum<type>(        \
      const type *src, size_t src_stride, size_t width, size_t height, \
      type *sum)

KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::sve2
