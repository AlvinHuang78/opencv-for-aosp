// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/filters/gaussian_blur.h"
#include "kleidicv/kleidicv.h"

KLEIDICV_MULTIVERSION_C_API(
    kleidicv_gaussian_blur_stripe_u8, &kleidicv::neon::gaussian_blur_stripe_u8,
    KLEIDICV_SVE2_IMPL_IF(kleidicv::sve2::gaussian_blur_stripe_u8),
    &kleidicv::sme2::gaussian_blur_stripe_u8);

extern "C" {

kleidicv_error_t kleidicv_gaussian_blur_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, float sigma_x, float sigma_y,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context) {
  if (!kleidicv::gaussian_blur_is_implemented(
          width, height, kernel_width, kernel_height, sigma_x, sigma_y)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  auto fixed_border_type = kleidicv::get_fixed_border_type(border_type);
  if (!fixed_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  return kleidicv_gaussian_blur_stripe_u8(src, src_stride, dst, dst_stride,
                                          width, height, 0, height, channels,
                                          kernel_width, kernel_height, sigma_x,
                                          sigma_y, *fixed_border_type, context);
}

}  // extern "C"
