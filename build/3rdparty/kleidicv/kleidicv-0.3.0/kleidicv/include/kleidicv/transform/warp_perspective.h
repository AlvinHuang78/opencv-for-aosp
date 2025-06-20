// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TRANSFORM_WARP_PERSPECTIVE_H
#define KLEIDICV_TRANSFORM_WARP_PERSPECTIVE_H

#include <limits>
#include <type_traits>

#include "kleidicv/ctypes.h"

extern "C" {
// For internal use only. See kleidicv_warp_perspective_u8 instead.
// Calculates a stripe of the `dst` image that is a transformed part of `src`.
// The stripe is defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_warp_perspective_stripe_u8,
                         const uint8_t *src, size_t src_stride,
                         size_t src_width, size_t src_height, uint8_t *dst,
                         size_t dst_stride, size_t dst_width, size_t dst_height,
                         size_t y_begin, size_t y_end,
                         const float transformation[9], size_t channels,
                         kleidicv_interpolation_type_t interpolation,
                         kleidicv_border_type_t border_type,
                         kleidicv_border_values_t border_values);
}

namespace kleidicv {

template <typename T>
inline bool warp_perspective_is_implemented(
    size_t dst_width, kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type,
    size_t channels) KLEIDICV_STREAMING_COMPATIBLE {
  if constexpr (std::is_same<T, uint8_t>::value) {
    return (dst_width >= 8 &&
            interpolation ==
                kleidicv_interpolation_type_t::KLEIDICV_INTERPOLATION_NEAREST &&
            border_type ==
                kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_REPLICATE &&
            channels == 1);
  } else {
    return false;
  }
}

namespace neon {

template <typename T>
kleidicv_error_t warp_perspective_stripe(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t y_begin, size_t y_end, const float transformation[9],
    size_t channels, kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type, kleidicv_border_values_t border_values);
}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t warp_perspective_stripe(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t y_begin, size_t y_end, const float transformation[9],
    size_t channels, kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type, kleidicv_border_values_t border_values);
}  // namespace sve2

}  // namespace kleidicv

#endif  // KLEIDICV_TRANSFORM_WARP_PERSPECTIVE_H
