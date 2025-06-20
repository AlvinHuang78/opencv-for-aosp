// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "remap_sc.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
namespace kleidicv::sve2 {

template <typename T>
kleidicv_error_t remap_s16(const T *src, size_t src_stride, size_t src_width,
                           size_t src_height, T *dst, size_t dst_stride,
                           size_t dst_width, size_t dst_height, size_t channels,
                           const int16_t *mapxy, size_t mapxy_stride,
                           kleidicv_border_type_t border_type,
                           kleidicv_border_values_t border_values) {
  return remap_s16_sc<uint8_t>(src, src_stride, src_width, src_height, dst,
                               dst_stride, dst_width, dst_height, channels,
                               mapxy, mapxy_stride, border_type, border_values);
}

template <typename T>
kleidicv_error_t remap_s16point5(const T *src, size_t src_stride,
                                 size_t src_width, size_t src_height, T *dst,
                                 size_t dst_stride, size_t dst_width,
                                 size_t dst_height, size_t channels,
                                 const int16_t *mapxy, size_t mapxy_stride,
                                 const uint16_t *mapfrac, size_t mapfrac_stride,
                                 kleidicv_border_type_t border_type,
                                 kleidicv_border_values_t border_values) {
  return remap_s16point5_sc<uint8_t>(
      src, src_stride, src_width, src_height, dst, dst_stride, dst_width,
      dst_height, channels, mapxy, mapxy_stride, mapfrac, mapfrac_stride,
      border_type, border_values);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(type)                          \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_s16<type>(          \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const int16_t *mapxy, size_t mapxy_stride,              \
      kleidicv_border_type_t border_type,                                      \
      kleidicv_border_values_t border_values)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(uint8_t);

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(type)                    \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_s16point5<type>(    \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const int16_t *mapxy, size_t mapxy_stride,              \
      const uint16_t *mapfrac, size_t mapfrac_stride,                          \
      kleidicv_border_type_t border_type,                                      \
      kleidicv_border_values_t border_values)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(uint8_t);

}  // namespace kleidicv::sve2
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
