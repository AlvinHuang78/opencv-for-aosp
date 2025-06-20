// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "scale_sc.h"

namespace kleidicv::sme2 {

template <typename T>
KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
scale(const T* src, size_t src_stride, T* dst, size_t dst_stride, size_t width,
      size_t height, float scale, float shift) {
  return scale_sc<T>(src, src_stride, dst, dst_stride, width, height, scale,
                     shift);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                             \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t scale<type>(       \
      const type* src, size_t src_stride, type* dst, size_t dst_stride, \
      size_t width, size_t height, float scale, float shift)

KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::sme2
