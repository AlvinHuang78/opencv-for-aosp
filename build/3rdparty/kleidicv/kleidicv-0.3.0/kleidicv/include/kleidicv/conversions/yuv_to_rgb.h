// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_CONVERSIONS_YUV_TO_RGB_H
#define KLEIDICV_CONVERSIONS_YUV_TO_RGB_H

#include "kleidicv/kleidicv.h"

namespace kleidicv {

/* Analog YUV to RGB conversion according to ITU-R BT.601-7 in matrix form:

  [ Ya ] = [  0.299  0.587  0.114 ] [ Ra ]
  [ Ua ] = [ -0.299 -0.587  0.886 ] [ Ga ]
  [ Va ] = [  0.701 -0.587 -0.114 ] [ Ba ]

After re-normalization of the analog signal:

  Yan = Ya
  Uan = Ua / 1.772
  Van = Va / 1.402

  [ Yan ] = [  0.299000  0.587000  0.114000 ] [ Ra ]
  [ Uan ] = [ -0.168736 -0.331264  0.500000 ] [ Ga ]
  [ Van ] = [  0.500000 -0.418688 -0.081312 ] [ Ba ]

Inverse transformation:
   R = Y + 1.14V
   G = Y - 0.395U - 0.581V
   B = Y + 2.033U

With 14-bit scaling and rounding, the integer constants are:

  [ R ] = [ (not scaled)      0  18678 ] [ Y' ]
  [ G ] = [ (not scaled)  -6472  -9519 ] [ U' ] + (1 << 13) >> 14
  [ B ] = [ (not scaled)  33292      0 ] [ V' ]

The final results are calculated using rounding shift right and saturating
to 8-bit unsigned values:

  X = saturating_cast<uint8_t>((X' + (1 << 13)) >> 14)

Sources:
  [1] https://www.itu.int/rec/R-REC-BT.601
*/

static constexpr size_t kWeightScale = 14;
static constexpr size_t kPreShift = 16 - kWeightScale;
// UBWeight does not fit into signed int16, but it fits into unsigned uint16
static constexpr int32_t kUBWeight = 33292;
static constexpr uint16_t kUnsignedUBWeight = 33292;
static constexpr int16_t kUGWeight = -6472;
static constexpr int16_t kVGWeight = -9519;
static constexpr int16_t kVRWeight = 18678;
static constexpr int32_t kBDelta4 =
    (-128 * kUBWeight + (1 << (kWeightScale - 1))) * (1 << kPreShift);
static constexpr int32_t kGDelta4 =
    (-128 * kUGWeight - 128 * kVGWeight + (1 << (kWeightScale - 1))) *
    (1 << kPreShift);
static constexpr int32_t kRDelta4 =
    (-128 * kVRWeight + (1 << (kWeightScale - 1))) * (1 << kPreShift);

namespace neon {
kleidicv_error_t yuv_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

kleidicv_error_t yuv_to_rgb_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

}  // namespace neon

namespace sve2 {
kleidicv_error_t yuv_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

kleidicv_error_t yuv_to_rgb_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

}  // namespace sve2

namespace sme2 {
kleidicv_error_t yuv_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

kleidicv_error_t yuv_to_rgb_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_CONVERSIONS_YUV_TO_RGB_H
