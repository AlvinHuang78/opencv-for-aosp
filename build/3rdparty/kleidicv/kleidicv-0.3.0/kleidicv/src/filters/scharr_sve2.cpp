// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/filters/scharr.h"
#include "scharr_sc.h"

namespace kleidicv::sve2 {

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t kleidicv_scharr_interleaved_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride, size_t y_begin,
    size_t y_end) {
  return kleidicv_scharr_interleaved_stripe_s16_u8_sc(
      src, src_stride, src_width, src_height, src_channels, dst, dst_stride,
      y_begin, y_end);
}

}  // namespace kleidicv::sve2
