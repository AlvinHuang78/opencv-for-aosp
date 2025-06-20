// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_CONVERSIONS_RGB_TO_YUV_H
#define KLEIDICV_CONVERSIONS_RGB_TO_YUV_H

#include "kleidicv/kleidicv.h"

namespace kleidicv {

/*
Analog YUV to RGB conversion according to ITU-R BT.601-7:

  Y = 0.299 * R + 0.587 * G + 0.114 * B;
  U = (B - Y) / 1.772 = (B - Y) * 0,5643340858
  V = (R - Y) / 1.402 = (R - Y) * 0,7132667618

With 14-bit scaling and rounding, the integer constants are:

  Y = 4899 * R + 9617 * G + 1868 * B
  U = (B - Y) * 8061
  V = (R - Y) * 14369

The final results are calculated using rounding shift right and saturating
to 8-bit unsigned values:

  X = saturating_cast<uint8_t>((X' + (1 << 13)) >> 14)

Sources:
  [1] https://www.itu.int/rec/R-REC-BT.601
*/

// Weights according to the calculation at the top of this file.
static constexpr size_t kWeightScale = 14;
static constexpr int16_t kRYWeight = 4899;
static constexpr int16_t kGYWeight = 9617;
static constexpr int16_t kBYWeight = 1868;
static constexpr int16_t kRVWeight = 14369;
static constexpr int16_t kBUWeight = 8061;

namespace neon {

kleidicv_error_t bgr_to_yuv_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

kleidicv_error_t rgb_to_yuv_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

kleidicv_error_t bgra_to_yuv_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);

kleidicv_error_t rgba_to_yuv_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);
}  // namespace neon

namespace sve2 {

kleidicv_error_t bgr_to_yuv_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

kleidicv_error_t rgb_to_yuv_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

kleidicv_error_t bgra_to_yuv_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);

kleidicv_error_t rgba_to_yuv_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);
}  // namespace sve2

namespace sme2 {

kleidicv_error_t bgr_to_yuv_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

kleidicv_error_t rgb_to_yuv_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

kleidicv_error_t bgra_to_yuv_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);

kleidicv_error_t rgba_to_yuv_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);
}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_CONVERSIONS_RGB_TO_YUV_H
