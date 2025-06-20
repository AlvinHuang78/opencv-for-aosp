// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_FILTERS_BLUR_AND_DOWNSAMPLE_H
#define KLEIDICV_FILTERS_BLUR_AND_DOWNSAMPLE_H

#include "kleidicv/config.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"
#include "kleidicv/workspace/border_types.h"

extern "C" {
// For internal use only. See instead kleidicv_blur_and_downsample_u8.
// Blurs and downsamples a horizontal stripe across an image. The stripe is
// defined by the range (y_begin, y_end].
KLEIDICV_API_DECLARATION(kleidicv_blur_and_downsample_stripe_u8,
                         const uint8_t *src, size_t src_stride,
                         size_t src_width, size_t src_height, uint8_t *dst,
                         size_t dst_stride, size_t y_begin, size_t y_end,
                         size_t channels,
                         kleidicv::FixedBorderType fixed_border_type,
                         kleidicv_filter_context_t *context);
}

namespace kleidicv {

inline bool blur_and_downsample_is_implemented(size_t src_width,
                                               size_t src_height,
                                               size_t channels) {
  return (src_width >= 4 && src_height >= 4) && (channels == 1);
}

namespace neon {

kleidicv_error_t kleidicv_blur_and_downsample_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t y_begin, size_t y_end,
    size_t channels, FixedBorderType fixed_border_type,
    kleidicv_filter_context_t *context);

}  // namespace neon

namespace sve2 {

kleidicv_error_t kleidicv_blur_and_downsample_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t y_begin, size_t y_end,
    size_t channels, FixedBorderType fixed_border_type,
    kleidicv_filter_context_t *context);

}  // namespace sve2

namespace sme2 {

kleidicv_error_t kleidicv_blur_and_downsample_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t y_begin, size_t y_end,
    size_t channels, FixedBorderType fixed_border_type,
    kleidicv_filter_context_t *context);

}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_FILTERS_BLUR_AND_DOWNSAMPLE_H
