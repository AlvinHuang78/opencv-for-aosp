// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "warp_perspective_sc.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
namespace kleidicv::sve2 {

template <typename T>
kleidicv_error_t warp_perspective_stripe(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t y_begin, size_t y_end, const float transformation[9],
    size_t channels, kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type,
    kleidicv_border_values_t border_values) {
  return warp_perspective_stripe_sc<T>(
      src, src_stride, src_width, src_height, dst, dst_stride, dst_width,
      dst_height, y_begin, y_end, transformation, channels, interpolation,
      border_type, border_values);
}

#define KLEIDICV_INSTANTIATE_WARP_PERSPECTIVE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                           \
  warp_perspective_stripe<type>(                                               \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t y_begin, size_t y_end, const float transformation[9],             \
      size_t channels, kleidicv_interpolation_type_t interpolation,            \
      kleidicv_border_type_t border_type,                                      \
      kleidicv_border_values_t border_values)

KLEIDICV_INSTANTIATE_WARP_PERSPECTIVE(uint8_t);

}  // namespace kleidicv::sve2
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
