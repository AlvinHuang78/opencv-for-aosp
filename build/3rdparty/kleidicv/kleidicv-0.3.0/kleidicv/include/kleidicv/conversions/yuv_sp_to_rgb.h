// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_CONVERSIONS_YUV_SP_TO_RGB_H
#define KLEIDICV_CONVERSIONS_YUV_SP_TO_RGB_H

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

  [ Ra ] = [ 1.000000  0.000000  1.402000 ] [ Yan ]
  [ Ga ] = [ 1.000000 -0.344136 -0.714136 ] [ Uan ]
  [ Ba ] = [ 1.000000  1.772000  0.000000 ] [ Van ]

Inverse transformation after quantization:

  Y' = saturating_subtract(Yan - 16)
  U' = Uan - 128
  V' = Van - 128

  R = 255 / 219 * Ra
  G = 255 / 224 * Ga
  B = 255 / 224 * Ba

  [ R ] = [ 1.164384  0.000000  1.596027 ] [ Y' ]
  [ G ] = [ 1.164384 -0.391762 -0.812967 ] [ U' ]
  [ B ] = [ 1.164384  2.017232  0.000000 ] [ V' ]

The values used in this implementation are the following:

  [ R ] = [ 1.164000  0.000000  1.596000 ] [ Y' ]
  [ G ] = [ 1.164000 -0.391000 -0.813000 ] [ U' ]
  [ B ] = [ 1.164000  2.018000  0.000000 ] [ V' ]

With 20-bit scaling and rounding, the integer constants are:

  [ R ] = [ 1,220,542          0   1,673,527 ] [ Y' ]
  [ G ] = [ 1,220,542  - 409,993    -852,492 ] [ U' ] + (1 << 19) >> 20
  [ B ] = [ 1,220,542  2,116,026           0 ] [ V' ]

The final results are calculated using rounding shift right and saturating
to 8-bit unsigned values:

  X = saturating_cast<uint8_t>((X' + (1 << 19)) >> 20)

The estimated error is then:

  [ R ] = [ 0.000000  0.000000 -0.000000 ] [ Y' ]
  [ G ] = [ 0.000000 -0.000000 -0.000000 ] [ U' ]
  [ B ] = [ 0.000000 -0.000000  0.000000 ] [ V' ]

Sources:
  [1] https://www.itu.int/rec/R-REC-BT.601
*/

// Weights according to the calculation at the top of this file.
static constexpr size_t kWeightScale = 20;
static constexpr int32_t kYWeight = /* Weight(Y) */ 1220542;
static constexpr size_t kRVWeightIndex = 0;
static constexpr size_t kGUWeightIndex = 1;
static constexpr size_t kGVWeightIndex = 2;
static constexpr size_t kBUWeightIndex = 3;
static constexpr int32_t kUVWeights[4] = {
    /* Weight(RV) */ 1673527,
    /* Weight(GU) */ -409993,
    /* Weight(GV) */ -852492,
    /* Weight(BU) */ 2116026,
};

namespace neon {
kleidicv_error_t yuv_sp_to_rgb_u8(const uint8_t *src_y, size_t src_y_stride,
                                  const uint8_t *src_uv, size_t src_uv_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height, bool is_nv21);

kleidicv_error_t yuv_sp_to_rgba_u8(const uint8_t *src_y, size_t src_y_stride,
                                   const uint8_t *src_uv, size_t src_uv_stride,
                                   uint8_t *dst, size_t dst_stride,
                                   size_t width, size_t height, bool is_nv21);

kleidicv_error_t yuv_sp_to_bgr_u8(const uint8_t *src_y, size_t src_y_stride,
                                  const uint8_t *src_uv, size_t src_uv_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height, bool is_nv21);

kleidicv_error_t yuv_sp_to_bgra_u8(const uint8_t *src_y, size_t src_y_stride,
                                   const uint8_t *src_uv, size_t src_uv_stride,
                                   uint8_t *dst, size_t dst_stride,
                                   size_t width, size_t height, bool is_nv21);
}  // namespace neon

namespace sve2 {
kleidicv_error_t yuv_sp_to_rgb_u8(const uint8_t *src_y, size_t src_y_stride,
                                  const uint8_t *src_uv, size_t src_uv_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height, bool is_nv21);

kleidicv_error_t yuv_sp_to_rgba_u8(const uint8_t *src_y, size_t src_y_stride,
                                   const uint8_t *src_uv, size_t src_uv_stride,
                                   uint8_t *dst, size_t dst_stride,
                                   size_t width, size_t height, bool is_nv21);

kleidicv_error_t yuv_sp_to_bgr_u8(const uint8_t *src_y, size_t src_y_stride,
                                  const uint8_t *src_uv, size_t src_uv_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height, bool is_nv21);

kleidicv_error_t yuv_sp_to_bgra_u8(const uint8_t *src_y, size_t src_y_stride,
                                   const uint8_t *src_uv, size_t src_uv_stride,
                                   uint8_t *dst, size_t dst_stride,
                                   size_t width, size_t height, bool is_nv21);
}  // namespace sve2

namespace sme2 {
kleidicv_error_t yuv_sp_to_rgb_u8(const uint8_t *src_y, size_t src_y_stride,
                                  const uint8_t *src_uv, size_t src_uv_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height, bool is_nv21);

kleidicv_error_t yuv_sp_to_rgba_u8(const uint8_t *src_y, size_t src_y_stride,
                                   const uint8_t *src_uv, size_t src_uv_stride,
                                   uint8_t *dst, size_t dst_stride,
                                   size_t width, size_t height, bool is_nv21);

kleidicv_error_t yuv_sp_to_bgr_u8(const uint8_t *src_y, size_t src_y_stride,
                                  const uint8_t *src_uv, size_t src_uv_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height, bool is_nv21);

kleidicv_error_t yuv_sp_to_bgra_u8(const uint8_t *src_y, size_t src_y_stride,
                                   const uint8_t *src_uv, size_t src_uv_stride,
                                   uint8_t *dst, size_t dst_stride,
                                   size_t width, size_t height, bool is_nv21);
}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_CONVERSIONS_YUV_SP_TO_RGB_H
