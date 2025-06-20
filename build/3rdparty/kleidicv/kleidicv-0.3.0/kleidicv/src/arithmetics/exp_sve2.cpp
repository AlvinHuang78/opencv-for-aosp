// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "exp_sc.h"

namespace kleidicv::sve2 {

template <typename T>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t exp(const T* src, size_t src_stride,
                                              T* dst, size_t dst_stride,
                                              size_t width, size_t height) {
  return exp_sc<T, ExpTryShortPath<T>>(src, src_stride, dst, dst_stride, width,
                                       height);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                             \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t exp<type>(         \
      const type* src, size_t src_stride, type* dst, size_t dst_stride, \
      size_t width, size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::sve2
