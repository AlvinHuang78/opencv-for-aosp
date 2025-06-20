// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "gaussian_blur_sc.h"
#include "kleidicv/filters/gaussian_blur.h"

namespace kleidicv::sme2 {

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
gaussian_blur_stripe_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
                        size_t dst_stride, size_t width, size_t height,
                        size_t y_begin, size_t y_end, size_t channels,
                        size_t kernel_width, size_t kernel_height,
                        float sigma_x, float sigma_y,
                        FixedBorderType border_type,
                        kleidicv_filter_context_t *context) {
  return gaussian_blur_stripe_u8_sc(
      src, src_stride, dst, dst_stride, width, height, y_begin, y_end, channels,
      kernel_width, kernel_height, sigma_x, sigma_y, border_type, context);
}

}  // namespace kleidicv::sme2
