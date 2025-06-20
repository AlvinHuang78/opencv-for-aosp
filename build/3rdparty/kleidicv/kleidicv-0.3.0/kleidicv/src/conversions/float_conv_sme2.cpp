// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "float_conv_sc.h"

namespace kleidicv::sme2 {

template <typename InputType, typename OutputType>
KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
float_conversion(const InputType* src, size_t src_stride, OutputType* dst,
                 size_t dst_stride, size_t width, size_t height) {
  return float_conversion_sc<InputType, OutputType>(src, src_stride, dst,
                                                    dst_stride, width, height);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(itype, otype)                           \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                          \
  float_conversion<itype, otype>(const itype* src, size_t src_stride,         \
                                 otype* dst, size_t dst_stride, size_t width, \
                                 size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(float, int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(float, uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int8_t, float);
KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t, float);

}  // namespace kleidicv::sme2
