// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/resize/resize_linear.h"
#include "resize_linear_sc.h"

namespace kleidicv::sme2 {
KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
resize_linear_stripe_u8(const uint8_t *src, size_t src_stride, size_t src_width,
                        size_t src_height, size_t y_begin, size_t y_end,
                        uint8_t *dst, size_t dst_stride, size_t dst_width,
                        size_t dst_height) {
  return resize_linear_stripe_u8_sc(src, src_stride, src_width, src_height,
                                    y_begin, y_end, dst, dst_stride, dst_width,
                                    dst_height);
}

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
resize_linear_stripe_f32(const float *src, size_t src_stride, size_t src_width,
                         size_t src_height, size_t y_begin, size_t y_end,
                         float *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height) {
  return resize_linear_stripe_f32_sc(src, src_stride, src_width, src_height,
                                     y_begin, y_end, dst, dst_stride, dst_width,
                                     dst_height);
}

}  // namespace kleidicv::sme2
