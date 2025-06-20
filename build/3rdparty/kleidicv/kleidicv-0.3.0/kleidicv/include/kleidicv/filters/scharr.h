// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_FILTERS_SCHARR_H
#define KLEIDICV_FILTERS_SCHARR_H

#include "kleidicv/config.h"
#include "kleidicv/kleidicv.h"

extern "C" {
// For internal use only. See instead kleidicv_scharr_interleaved_s16_u8.
// Filter a horizontal stripe across an image. The stripe is defined by the
// range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_scharr_interleaved_stripe_s16_u8,
                         const uint8_t *src, size_t src_stride,
                         size_t src_width, size_t src_height,
                         size_t src_channels, int16_t *dst, size_t dst_stride,
                         size_t y_begin, size_t y_end);
}

namespace kleidicv {

inline bool scharr_interleaved_is_implemented(size_t src_width,
                                              size_t src_height,
                                              size_t src_channels) {
  return src_width > 2 && src_height > 2 && src_channels == 1;
}

namespace neon {

kleidicv_error_t kleidicv_scharr_interleaved_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride, size_t y_begin,
    size_t y_end);

}  // namespace neon

namespace sve2 {

kleidicv_error_t kleidicv_scharr_interleaved_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride, size_t y_begin,
    size_t y_end);
}  // namespace sve2

namespace sme2 {

kleidicv_error_t kleidicv_scharr_interleaved_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride, size_t y_begin,
    size_t y_end);
}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_FILTERS_SCHARR_H
