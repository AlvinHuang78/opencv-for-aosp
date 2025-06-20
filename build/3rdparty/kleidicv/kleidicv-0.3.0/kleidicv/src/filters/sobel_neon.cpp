// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/filters/separable_filter_3x3_neon.h"
#include "kleidicv/filters/sobel.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/workspace/separable.h"

namespace kleidicv::neon {

// Template for 3x3 Sobel filters which calculate horizontal derivative
// approximations, often denoted as Gx.
//
// The applied weights, as the kernel is mirrored both vertically and
// horizontally during the convolution:
//      [ -1, 0, 1 ]   [ 1 ]
//  F = [ -2, 0, 2 ] = [ 2 ] * [ -1,  0, 1 ]
//      [ -1, 0, 1 ]   [ 1 ]
template <typename T>
class HorizontalSobel3x3;

// 3x3 Sobel filter for uint8_t types which calculates horizontal derivative
// approximations, often denoted as Gx.
template <>
class HorizontalSobel3x3<uint8_t> {
 public:
  using SourceType = uint8_t;
  using BufferType = int16_t;
  using DestinationType = int16_t;

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void vertical_vector_path(uint8x16_t src[3], BufferType *dst) const {
    int16x8_t acc_l = vaddl_u8(vget_low_u8(src[0]), vget_low_u8(src[2]));
    int16x8_t acc_h = vaddl_u8(vget_high_u8(src[0]), vget_high_u8(src[2]));
    uint8x16_t shift_l = vshll_n_u8(vget_low_u8(src[1]), 1);
    uint8x16_t shift_h = vshll_n_u8(vget_high_u8(src[1]), 1);
    acc_l = vaddq_u16(acc_l, shift_l);
    acc_h = vaddq_u16(acc_h, shift_h);
    vst1q(&dst[0], acc_l);
    vst1q(&dst[VecTraits<BufferType>::num_lanes()], acc_h);
  }

  // Applies vertical filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void vertical_scalar_path(const SourceType src[3], BufferType *dst) const {
    // Explicitly narrow. Overflow is permitted.
    dst[0] = static_cast<DestinationType>(src[0] + 2 * src[1] + src[2]);
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ -1, 0, 1 ]T
  void horizontal_vector_path(int16x8_t src[3], DestinationType *dst) const {
    vst1q(&dst[0], vsubq_s16(src[2], src[0]));
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ -1, 0, 1 ]T
  void horizontal_scalar_path(const BufferType src[3],
                              DestinationType *dst) const {
    // Explicitly narrow. Overflow is permitted.
    dst[0] = static_cast<DestinationType>(src[2] - src[0]);
  }
};  // end of class HorizontalSobel3x3<uint8_t>

// Template for 3x3 Sobel filters which calculate vertical derivative
// approximations, often denoted as Gy.
//
// The applied weights, as the kernel is mirrored both vertically and
// horizontally during the convolution:
//      [ -1, -2, -1 ]   [ -1 ]
//  F = [  0,  0,  0 ] = [  0 ] * [ 1,  2, 1 ]
//      [  1,  2,  1 ]   [  1 ]
template <typename T>
class VerticalSobel3x3;

// 3x3 Sobel filter for uint8_t types which calculates vertical derivative
// approximations, often denoted as Gy.
template <>
class VerticalSobel3x3<uint8_t> {
 public:
  using SourceType = uint8_t;
  using BufferType = int16_t;
  using DestinationType = int16_t;

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ -1, 0, 1 ]T
  void vertical_vector_path(uint8x16_t src[3], BufferType *dst) const {
    uint16x8_t acc_l = vsubl_u8(vget_low_u8(src[2]), vget_low_u8(src[0]));
    uint16x8_t acc_h = vsubl_u8(vget_high_u8(src[2]), vget_high_u8(src[0]));
    vst1q(&dst[0], vreinterpretq_s16_u16(acc_l));
    vst1q(&dst[VecTraits<BufferType>::num_lanes()],
          vreinterpretq_s16_u16(acc_h));
  }

  // Applies vertical filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ -1, 0, 1 ]T
  void vertical_scalar_path(const SourceType src[3], BufferType *dst) const {
    // Explicitly narrow. Overflow is permitted.
    dst[0] = static_cast<DestinationType>(src[2] - src[0]);
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void horizontal_vector_path(int16x8_t src[3], DestinationType *dst) const {
    int16x8_t acc = vaddq_s16(src[0], src[2]);
    acc = vaddq_s16(acc, vshlq_n_s16(src[1], 1));
    vst1q(&dst[0], acc);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void horizontal_scalar_path(const BufferType src[3],
                              DestinationType *dst) const {
    // Explicitly narrow. Overflow is permitted.
    dst[0] = static_cast<DestinationType>(src[0] + 2 * src[1] + src[2]);
  }
};  // end of class VerticalSobel3x3<uint8_t>

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t sobel_3x3_horizontal_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end,
    size_t channels) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return KLEIDICV_ERROR_RANGE;
  }

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, channels};
  Rows<int16_t> dst_rows{dst, dst_stride, channels};

  auto workspace =
      SeparableFilterWorkspace::create(rect, channels, sizeof(int16_t));
  if (!workspace) {
    return KLEIDICV_ERROR_ALLOCATION;
  }

  HorizontalSobel3x3<uint8_t> horizontal_sobel;
  SeparableFilter3x3<HorizontalSobel3x3<uint8_t>> filter{horizontal_sobel};
  workspace->process(rect, y_begin, y_end, src_rows, dst_rows, channels,
                     FixedBorderType::REPLICATE, filter);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t sobel_3x3_vertical_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end,
    size_t channels) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return KLEIDICV_ERROR_RANGE;
  }

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, channels};
  Rows<int16_t> dst_rows{dst, dst_stride, channels};

  auto workspace =
      SeparableFilterWorkspace::create(rect, channels, sizeof(int16_t));
  if (!workspace) {
    return KLEIDICV_ERROR_ALLOCATION;
  }

  VerticalSobel3x3<uint8_t> vertical_sobel;
  SeparableFilter3x3<VerticalSobel3x3<uint8_t>> filter{vertical_sobel};
  workspace->process(rect, y_begin, y_end, src_rows, dst_rows, channels,
                     FixedBorderType::REPLICATE, filter);
  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
