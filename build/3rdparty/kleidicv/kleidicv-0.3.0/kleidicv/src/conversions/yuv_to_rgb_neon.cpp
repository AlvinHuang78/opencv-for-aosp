// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "kleidicv/conversions/yuv_to_rgb.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <bool BGR>
class YUVToRGBAll final : public UnrollOnce, public TryToAvoidTailLoop {
 public:
  using VecTraits = neon::VecTraits<uint8_t>;
  using ScalarType = VecTraits::ScalarType;
  using VectorType = VecTraits::VectorType;
  using Vector3Type = VecTraits::Vector3Type;

  explicit YUVToRGBAll()
      : b_delta4_(vdupq_n_u32(kBDelta4)),
        g_delta4_(vdupq_n_u32(kGDelta4)),
        r_delta4_(vdupq_n_u32(kRDelta4)) {}

  void vector_path(const ScalarType *src, ScalarType *dst) {
    // Load deinterleaved
    Vector3Type vsrc = vld3q_u8(src);
    int16x8_t y_l = vreinterpretq_s16_u8(vzip1q_u8(vsrc.val[0], vdupq_n_u8(0)));
    int16x8_t y_h = vreinterpretq_s16_u8(vzip2q_u8(vsrc.val[0], vdupq_n_u8(0)));
    int16x8_t u4_l =
        vreinterpretq_s16_u16(vshll_n_u8(vget_low_u8(vsrc.val[1]), kPreShift));
    int16x8_t u4_h =
        vreinterpretq_s16_u16(vshll_high_n_u8(vsrc.val[1], kPreShift));
    int16x8_t v4_l =
        vreinterpretq_s16_u16(vshll_n_u8(vget_low_u8(vsrc.val[2]), kPreShift));
    int16x8_t v4_h =
        vreinterpretq_s16_u16(vshll_high_n_u8(vsrc.val[2], kPreShift));
    uint8x16_t r, g, b;

    // Compute B value in 32-bit precision
    {
      // Multiplication is done with uint16_t because UBWeight only fits in
      // unsigned 16-bit
      int32x4_t b_ll = vreinterpretq_s32_u32(vmull_n_u16(
          vget_low_u16(vreinterpretq_u16_s16(u4_l)), kUnsignedUBWeight));
      int32x4_t b_hl = vreinterpretq_s32_u32(vmull_n_u16(
          vget_low_u16(vreinterpretq_u16_s16(u4_h)), kUnsignedUBWeight));
      int32x4_t b_lh = vreinterpretq_s32_u32(
          vmull_high_n_u16(vreinterpretq_u16_s16(u4_l), kUnsignedUBWeight));
      int32x4_t b_hh = vreinterpretq_s32_u32(
          vmull_high_n_u16(vreinterpretq_u16_s16(u4_h), kUnsignedUBWeight));

      b_ll = vaddq(b_ll, b_delta4_);
      b_hl = vaddq(b_hl, b_delta4_);
      b_lh = vaddq(b_lh, b_delta4_);
      b_hh = vaddq(b_hh, b_delta4_);

      int16x8_t b_l = vaddq(y_l, vuzp2q_s16(vreinterpretq_s16_s32(b_ll),
                                            vreinterpretq_s16_s32(b_lh)));
      int16x8_t b_h = vaddq(y_h, vuzp2q_s16(vreinterpretq_s16_s32(b_hl),
                                            vreinterpretq_s16_s32(b_hh)));

      b = vcombine_u8(vqmovun_s16(b_l), vqmovun_s16(b_h));
    }

    // Compute G value in 32-bit precision
    {
      int32x4_t g_ll = vmlal_n_s16(g_delta4_, vget_low_s16(u4_l), kUGWeight);
      int32x4_t g_hl = vmlal_n_s16(g_delta4_, vget_low_s16(u4_h), kUGWeight);
      int32x4_t g_lh = vmlal_high_n_s16(g_delta4_, u4_l, kUGWeight);
      int32x4_t g_hh = vmlal_high_n_s16(g_delta4_, u4_h, kUGWeight);

      g_ll = vmlal_n_s16(g_ll, vget_low_s16(v4_l), kVGWeight);
      g_hl = vmlal_n_s16(g_hl, vget_low_s16(v4_h), kVGWeight);
      g_lh = vmlal_high_n_s16(g_lh, v4_l, kVGWeight);
      g_hh = vmlal_high_n_s16(g_hh, v4_h, kVGWeight);

      int16x8_t g_l = vaddq(y_l, vuzp2q_s16(vreinterpretq_s16_s32(g_ll),
                                            vreinterpretq_s16_s32(g_lh)));
      int16x8_t g_h = vaddq(y_h, vuzp2q_s16(vreinterpretq_s16_s32(g_hl),
                                            vreinterpretq_s16_s32(g_hh)));

      g = vcombine_u8(vqmovun_s16(g_l), vqmovun_s16(g_h));
    }

    // Compute R value in 32-bit precision
    {
      int32x4_t r_ll = vmlal_n_s16(r_delta4_, vget_low_s16(v4_l), kVRWeight);
      int32x4_t r_hl = vmlal_n_s16(r_delta4_, vget_low_s16(v4_h), kVRWeight);
      int32x4_t r_lh = vmlal_high_n_s16(r_delta4_, v4_l, kVRWeight);
      int32x4_t r_hh = vmlal_high_n_s16(r_delta4_, v4_h, kVRWeight);

      int16x8_t r_l = vaddq(y_l, vuzp2q_s16(vreinterpretq_s16_s32(r_ll),
                                            vreinterpretq_s16_s32(r_lh)));
      int16x8_t r_h = vaddq(y_h, vuzp2q_s16(vreinterpretq_s16_s32(r_hl),
                                            vreinterpretq_s16_s32(r_hh)));

      r = vcombine_u8(vqmovun_s16(r_l), vqmovun_s16(r_h));
    }

    uint8x16x3_t rgb;
    rgb.val[r_index_] = r;
    rgb.val[g_index_] = g;
    rgb.val[b_index_] = b;

    // Store interleaved RGB pixels to memory.
    vst3q_u8(dst, rgb);
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    int32_t y = static_cast<int32_t>(src[0]);
    int32_t u = static_cast<int32_t>(src[1]);
    int32_t v = static_cast<int32_t>(src[2]);
    int32_t b = y + rounding_shift_right((u - 128) * kUBWeight, kWeightScale);
    int32_t g =
        y + rounding_shift_right((u - 128) * kUGWeight + (v - 128) * kVGWeight,
                                 kWeightScale);
    int32_t r = y + rounding_shift_right((v - 128) * kVRWeight, kWeightScale);
    dst[r_index_] = saturating_cast<int32_t, uint8_t>(r);
    dst[g_index_] = saturating_cast<int32_t, uint8_t>(g);
    dst[b_index_] = saturating_cast<int32_t, uint8_t>(b);
  }

 private:
  static constexpr size_t r_index_ = BGR ? 2 : 0;
  static constexpr size_t g_index_ = 1;
  static constexpr size_t b_index_ = BGR ? 0 : 2;
  int32x4_t b_delta4_, g_delta4_, r_delta4_;
};  // end of class YUVToRGBAll<bool BGR>

template <typename OperationType, typename ScalarType>
kleidicv_error_t yuv2rgb_operation(OperationType &operation,
                                   const ScalarType *src, size_t src_stride,
                                   ScalarType *dst, size_t dst_stride,
                                   size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows src_rows{src, src_stride, 3};
  Rows dst_rows{dst, dst_stride, 3};

  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

using YUVToRGB = YUVToRGBAll<false>;
using YUVToBGR = YUVToRGBAll<true>;

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv_to_rgb_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  YUVToRGB operation;
  return yuv2rgb_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  YUVToBGR operation;
  return yuv2rgb_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

}  // namespace kleidicv::neon
