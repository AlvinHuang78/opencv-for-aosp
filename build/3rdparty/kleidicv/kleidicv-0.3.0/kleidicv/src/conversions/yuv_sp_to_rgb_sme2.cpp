// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "yuv_sp_to_rgb_sc.h"

namespace kleidicv::sme2 {

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
yuv_sp_to_rgb_u8(const uint8_t *src_y, size_t src_y_stride,
                 const uint8_t *src_uv, size_t src_uv_stride, uint8_t *dst,
                 size_t dst_stride, size_t width, size_t height, bool is_nv21) {
  return yuv_sp_to_rgb_u8_sc(src_y, src_y_stride, src_uv, src_uv_stride, dst,
                             dst_stride, width, height, is_nv21);
}

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
yuv_sp_to_rgba_u8(const uint8_t *src_y, size_t src_y_stride,
                  const uint8_t *src_uv, size_t src_uv_stride, uint8_t *dst,
                  size_t dst_stride, size_t width, size_t height,
                  bool is_nv21) {
  return yuv_sp_to_rgba_u8_sc(src_y, src_y_stride, src_uv, src_uv_stride, dst,
                              dst_stride, width, height, is_nv21);
}

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
yuv_sp_to_bgr_u8(const uint8_t *src_y, size_t src_y_stride,
                 const uint8_t *src_uv, size_t src_uv_stride, uint8_t *dst,
                 size_t dst_stride, size_t width, size_t height, bool is_nv21) {
  return yuv_sp_to_bgr_u8_sc(src_y, src_y_stride, src_uv, src_uv_stride, dst,
                             dst_stride, width, height, is_nv21);
}

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
yuv_sp_to_bgra_u8(const uint8_t *src_y, size_t src_y_stride,
                  const uint8_t *src_uv, size_t src_uv_stride, uint8_t *dst,
                  size_t dst_stride, size_t width, size_t height,
                  bool is_nv21) {
  return yuv_sp_to_bgra_u8_sc(src_y, src_y_stride, src_uv, src_uv_stride, dst,
                              dst_stride, width, height, is_nv21);
}

}  // namespace kleidicv::sme2
