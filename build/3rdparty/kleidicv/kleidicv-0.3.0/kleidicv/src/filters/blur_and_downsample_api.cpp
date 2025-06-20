// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/filters/blur_and_downsample.h"
#include "kleidicv/kleidicv.h"

KLEIDICV_MULTIVERSION_C_API(
    kleidicv_blur_and_downsample_stripe_u8,
    &kleidicv::neon::kleidicv_blur_and_downsample_stripe_u8,
    &kleidicv::sve2::kleidicv_blur_and_downsample_stripe_u8,
    &kleidicv::sme2::kleidicv_blur_and_downsample_stripe_u8);

extern "C" {

kleidicv_error_t kleidicv_blur_and_downsample_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t channels,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context) {
  if (!kleidicv::blur_and_downsample_is_implemented(src_width, src_height,
                                                    channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  auto fixed_border_type = kleidicv::get_fixed_border_type(border_type);
  if (!fixed_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  return kleidicv_blur_and_downsample_stripe_u8(
      src, src_stride, src_width, src_height, dst, dst_stride, 0, src_height,
      channels, *fixed_border_type, context);
}

}  // extern "C"
