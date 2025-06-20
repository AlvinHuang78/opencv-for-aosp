// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_FILTERS_GAUSSIAN_BLUR_H
#define KLEIDICV_FILTERS_GAUSSIAN_BLUR_H

#include "kleidicv/config.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"
#include "kleidicv/workspace/border_types.h"

extern "C" {
// For internal use only. See instead kleidicv_gaussian_blur_u8.
// Blur a horizontal stripe across an image. The stripe is defined by the
// range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_gaussian_blur_stripe_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, size_t y_begin,
                         size_t y_end, size_t channels, size_t kernel_width,
                         size_t kernel_height, float sigma_x, float sigma_y,
                         kleidicv::FixedBorderType border_type,
                         kleidicv_filter_context_t *context);
}

namespace kleidicv {

inline bool gaussian_blur_is_implemented(size_t width, size_t height,
                                         size_t kernel_width,
                                         size_t kernel_height, float sigma_x,
                                         float sigma_y) {
  if (kernel_width != kernel_height) {
    return false;
  }

  if (sigma_x != sigma_y) {
    return false;
  }

  if (width < kernel_width - 1 || height < kernel_width - 1) {
    return false;
  }

  switch (kernel_width) {
    case 3:
    case 5:
    case 7:
    case 15:
      break;
    default:
      return false;
  }

  return true;
}

namespace neon {

kleidicv_error_t gaussian_blur_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, float sigma_x, float sigma_y,
    FixedBorderType border_type, kleidicv_filter_context_t *context);

}  // namespace neon

namespace sve2 {

kleidicv_error_t gaussian_blur_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, float sigma_x, float sigma_y,
    FixedBorderType border_type, kleidicv_filter_context_t *context);

}  // namespace sve2

namespace sme2 {

kleidicv_error_t gaussian_blur_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, float sigma_x, float sigma_y,
    FixedBorderType border_type, kleidicv_filter_context_t *context);

}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_FILTERS_GAUSSIAN_BLUR_H
