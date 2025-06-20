// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/transform/warp_perspective.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE

KLEIDICV_MULTIVERSION_C_API(kleidicv_warp_perspective_stripe_u8,
                            &kleidicv::neon::warp_perspective_stripe<uint8_t>,
                            &kleidicv::sve2::warp_perspective_stripe<uint8_t>,
                            nullptr);

extern "C" {

kleidicv_error_t kleidicv_warp_perspective_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    const float transformation[9], size_t channels,
    kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type,
    kleidicv_border_values_t border_values) {
  if (!kleidicv::warp_perspective_is_implemented<uint8_t>(
          dst_width, interpolation, border_type, channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  return kleidicv_warp_perspective_stripe_u8(
      src, src_stride, src_width, src_height, dst, dst_stride, dst_width,
      dst_height, 0, dst_height, transformation, 1, interpolation, border_type,
      border_values);
}

}  // extern "C"
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
