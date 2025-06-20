// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "blur_and_downsample_sc.h"
#include "kleidicv/filters/blur_and_downsample.h"

namespace kleidicv::sme2 {

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
kleidicv_blur_and_downsample_stripe_u8(const uint8_t *src, size_t src_stride,
                                       size_t src_width, size_t src_height,
                                       uint8_t *dst, size_t dst_stride,
                                       size_t y_begin, size_t y_end,
                                       size_t channels,
                                       FixedBorderType fixed_border_type,
                                       kleidicv_filter_context_t *context) {
  return blur_and_downsample_stripe_u8_sc(
      src, src_stride, src_width, src_height, dst, dst_stride, y_begin, y_end,
      channels, fixed_border_type, context);
}

}  // namespace kleidicv::sme2
