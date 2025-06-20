// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "sobel_sc.h"

namespace kleidicv::sve2 {

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t sobel_3x3_horizontal_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end,
    size_t channels) {
  return sobel_3x3_horizontal_stripe_s16_u8_sc(src, src_stride, dst, dst_stride,
                                               width, height, y_begin, y_end,
                                               channels);
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t sobel_3x3_vertical_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end,
    size_t channels) {
  return sobel_3x3_vertical_stripe_s16_u8_sc(src, src_stride, dst, dst_stride,
                                             width, height, y_begin, y_end,
                                             channels);
}

}  // namespace kleidicv::sve2
