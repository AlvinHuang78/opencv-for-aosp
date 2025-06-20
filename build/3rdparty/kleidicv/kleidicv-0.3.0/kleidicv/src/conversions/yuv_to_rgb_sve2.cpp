// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "yuv_to_rgb_sc.h"
namespace kleidicv::sve2 {

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv_to_rgb_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  return yuv_to_rgb_u8_sc(src, src_stride, dst, dst_stride, width, height);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  return yuv_to_bgr_u8_sc(src, src_stride, dst, dst_stride, width, height);
}

}  // namespace kleidicv::sve2
