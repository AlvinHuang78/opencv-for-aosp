// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_FILTERS_SOBEL_H
#define KLEIDICV_FILTERS_SOBEL_H

#include "kleidicv/kleidicv.h"

extern "C" {
// For internal use only. See instead kleidicv_sobel_3x3_horizontal_s16_u8.
// Filter a horizontal stripe across an image. The stripe is defined by the
// range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_sobel_3x3_horizontal_stripe_s16_u8,
                         const uint8_t *src, size_t src_stride, int16_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels);
// For internal use only. See instead kleidicv_sobel_3x3_vertical_s16_u8.
// Filter a horizontal stripe across an image. The stripe is defined by the
// range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_sobel_3x3_vertical_stripe_s16_u8,
                         const uint8_t *src, size_t src_stride, int16_t *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t y_begin, size_t y_end, size_t channels);
}

namespace kleidicv {

inline bool sobel_is_implemented(size_t width, size_t height,
                                 size_t kernel_size) {
  return width >= kernel_size - 1 && height >= kernel_size - 1;
}

namespace neon {
kleidicv_error_t sobel_3x3_horizontal_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels);
kleidicv_error_t sobel_3x3_vertical_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels);
}  // namespace neon

namespace sve2 {
kleidicv_error_t sobel_3x3_horizontal_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels);
kleidicv_error_t sobel_3x3_vertical_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels);
}  // namespace sve2

namespace sme2 {
kleidicv_error_t sobel_3x3_horizontal_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels);
kleidicv_error_t sobel_3x3_vertical_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels);
}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_FILTERS_SOBEL_H
