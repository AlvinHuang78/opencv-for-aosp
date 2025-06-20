// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_REMAP_REMAP_H
#define KLEIDICV_REMAP_REMAP_H

#include <type_traits>

#include "kleidicv/ctypes.h"

namespace kleidicv {

template <typename T>
inline bool remap_s16_is_implemented(
    size_t dst_width, kleidicv_border_type_t border_type,
    size_t channels) KLEIDICV_STREAMING_COMPATIBLE {
  if constexpr (std::is_same<T, uint8_t>::value) {
    return (dst_width >= 8 &&
            border_type ==
                kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_REPLICATE &&
            channels == 1);
  } else {
    return false;
  }
}

template <typename T>
inline bool remap_s16point5_is_implemented(
    size_t dst_width, kleidicv_border_type_t border_type,
    size_t channels) KLEIDICV_STREAMING_COMPATIBLE {
  if constexpr (std::is_same<T, uint8_t>::value) {
    return (dst_width >= 8 &&
            border_type ==
                kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_REPLICATE &&
            channels == 1);
  } else {
    return false;
  }
}

// Constants for Remap16Point5
static const uint16_t REMAP16POINT5_FRAC_BITS = 5;
static const uint16_t REMAP16POINT5_FRAC_MAX = 1 << REMAP16POINT5_FRAC_BITS;
static const uint16_t REMAP16POINT5_FRAC_MAX_SQUARE =
    REMAP16POINT5_FRAC_MAX * REMAP16POINT5_FRAC_MAX;

namespace neon {

template <typename T>
kleidicv_error_t remap_s16(const T *src, size_t src_stride, size_t src_width,
                           size_t src_height, T *dst, size_t dst_stride,
                           size_t dst_width, size_t dst_height, size_t channels,
                           const int16_t *mapxy, size_t mapxy_stride,
                           kleidicv_border_type_t border_type,
                           kleidicv_border_values_t border_values);

template <typename T>
kleidicv_error_t remap_s16point5(const T *src, size_t src_stride,
                                 size_t src_width, size_t src_height, T *dst,
                                 size_t dst_stride, size_t dst_width,
                                 size_t dst_height, size_t channels,
                                 const int16_t *mapxy, size_t mapxy_stride,
                                 const uint16_t *mapfrac, size_t mapfrac_stride,
                                 kleidicv_border_type_t border_type,
                                 kleidicv_border_values_t border_values);

}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t remap_s16(const T *src, size_t src_stride, size_t src_width,
                           size_t src_height, T *dst, size_t dst_stride,
                           size_t dst_width, size_t dst_height, size_t channels,
                           const int16_t *mapxy, size_t mapxy_stride,
                           kleidicv_border_type_t border_type,
                           kleidicv_border_values_t border_values);

template <typename T>
kleidicv_error_t remap_s16point5(const T *src, size_t src_stride,
                                 size_t src_width, size_t src_height, T *dst,
                                 size_t dst_stride, size_t dst_width,
                                 size_t dst_height, size_t channels,
                                 const int16_t *mapxy, size_t mapxy_stride,
                                 const uint16_t *mapfrac, size_t mapfrac_stride,
                                 kleidicv_border_type_t border_type,
                                 kleidicv_border_values_t border_values);

}  // namespace sve2

namespace sme2 {

template <typename T>
kleidicv_error_t remap_s16point5(const T *src, size_t src_stride,
                                 size_t src_width, size_t src_height, T *dst,
                                 size_t dst_stride, size_t dst_width,
                                 size_t dst_height, size_t channels,
                                 const int16_t *mapxy, size_t mapxy_stride,
                                 const uint16_t *mapfrac, size_t mapfrac_stride,
                                 kleidicv_border_type_t border_type,
                                 kleidicv_border_values_t border_values);

}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_REMAP_REMAP_H
