// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/filters/separable_filter_2d.h"
#include "separable_filter_2d_sc.h"

namespace kleidicv::sve2 {

template <typename T>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t separable_filter_2d_stripe(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    const T *kernel_x, size_t kernel_width, const T *kernel_y,
    size_t kernel_height, FixedBorderType border_type,
    kleidicv_filter_context_t *context) {
  return separable_filter_2d_stripe_sc<T>(
      src, src_stride, dst, dst_stride, width, height, y_begin, y_end, channels,
      kernel_x, kernel_width, kernel_y, kernel_height, border_type, context);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                    \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                           \
  separable_filter_2d_stripe<type>(                                            \
      const type *src, size_t src_stride, type *dst, size_t dst_stride,        \
      size_t width, size_t height, size_t y_begin, size_t y_end,               \
      size_t channels, const type *kernel_x, size_t kernel_width,              \
      const type *kernel_y, size_t kernel_height, FixedBorderType border_type, \
      kleidicv_filter_context_t *context)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);

}  // namespace kleidicv::sve2
