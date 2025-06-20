// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/resize/resize_linear.h"

KLEIDICV_MULTIVERSION_C_API(
    kleidicv_resize_linear_stripe_u8, &kleidicv::neon::resize_linear_stripe_u8,
    KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::resize_linear_stripe_u8),
    &kleidicv::sme2::resize_linear_stripe_u8);

KLEIDICV_MULTIVERSION_C_API(
    kleidicv_resize_linear_stripe_f32,
    &kleidicv::neon::resize_linear_stripe_f32,
    KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::resize_linear_stripe_f32),
    &kleidicv::sme2::resize_linear_stripe_f32);

namespace kleidicv {

static kleidicv_error_t resize_linear_u8(const uint8_t *src, size_t src_stride,
                                         size_t src_width, size_t src_height,
                                         uint8_t *dst, size_t dst_stride,
                                         size_t dst_width, size_t dst_height) {
  if (!resize_linear_u8_is_implemented(src_width, src_height, dst_width,
                                       dst_height)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  return kleidicv_resize_linear_stripe_u8(src, src_stride, src_width,
                                          src_height, 0, src_height, dst,
                                          dst_stride, dst_width, dst_height);
}

static kleidicv_error_t resize_linear_f32(const float *src, size_t src_stride,
                                          size_t src_width, size_t src_height,
                                          float *dst, size_t dst_stride,
                                          size_t dst_width, size_t dst_height) {
  if (!resize_linear_f32_is_implemented(src_width, src_height, dst_width,
                                        dst_height)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  return kleidicv_resize_linear_stripe_f32(src, src_stride, src_width,
                                           src_height, 0, src_height, dst,
                                           dst_stride, dst_width, dst_height);
}

}  // namespace kleidicv

KLEIDICV_MULTIVERSION_C_API(kleidicv_resize_linear_u8,
                            &kleidicv::resize_linear_u8, nullptr, nullptr);

KLEIDICV_MULTIVERSION_C_API(kleidicv_resize_linear_f32,
                            &kleidicv::resize_linear_f32, nullptr, nullptr);
