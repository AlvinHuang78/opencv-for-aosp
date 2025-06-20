// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "rgb_to_yuv_sc.h"

namespace kleidicv::sve2 {

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgb_to_yuv_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  return rgb_to_yuv_u8_sc(src, src_stride, dst, dst_stride, width, height);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_yuv_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  return rgba_to_yuv_u8_sc(src, src_stride, dst, dst_stride, width, height);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t bgr_to_yuv_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  return bgr_to_yuv_u8_sc(src, src_stride, dst, dst_stride, width, height);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t bgra_to_yuv_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  return bgra_to_yuv_u8_sc(src, src_stride, dst, dst_stride, width, height);
}

}  // namespace kleidicv::sve2
