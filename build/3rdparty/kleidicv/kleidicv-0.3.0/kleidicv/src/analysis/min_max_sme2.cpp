// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "min_max_sc.h"

namespace kleidicv::sme2 {

template <typename T>
KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
min_max(const T *src, size_t src_stride, size_t width, size_t height,
        T *min_value, T *max_value) {
  return min_max_sc<T>(src, src_stride, width, height, min_value, max_value);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t min_max<type>(    \
      const type *src, size_t src_stride, size_t width, size_t height, \
      type *min_value, type *max_value)

KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::sme2
