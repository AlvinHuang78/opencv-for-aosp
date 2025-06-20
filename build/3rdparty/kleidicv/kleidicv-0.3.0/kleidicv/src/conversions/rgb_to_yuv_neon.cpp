// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/rgb_to_yuv.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <bool BGR, bool ALPHA>
class RGBToYUVAll final : public UnrollOnce, public TryToAvoidTailLoop {
 public:
  using VecTraits = neon::VecTraits<uint8_t>;
  using ScalarType = VecTraits::ScalarType;
  using VectorType = VecTraits::VectorType;
  using RawSourceVectorType =
      typename std::conditional<ALPHA, uint8x16x4_t, uint8x16x3_t>::type;

  explicit RGBToYUVAll() = default;

  // Returns the number of channels in the output image.
  static constexpr size_t input_channels() {
    return ALPHA ? /* RGBA */ 4 : /* RGB */ 3;
  }

  void vector_path(const ScalarType *src, ScalarType *dst) {
    RawSourceVectorType vsrc;
    int16x8_t r_l, r_h, g_l, g_h, b_l, b_h;
    if constexpr (ALPHA) {
      vsrc = vld1q_u8_x4(src);
      uint16x8_t rb_l = vuzp1q_u8(vsrc.val[0], vsrc.val[1]);
      uint16x8_t rb_h = vuzp1q_u8(vsrc.val[2], vsrc.val[3]);
      if constexpr (BGR) {
        b_l = vreinterpretq_s16_u8(vtrn1q_u8(rb_l, vdupq_n_u8(0)));
        b_h = vreinterpretq_s16_u8(vtrn1q_u8(rb_h, vdupq_n_u8(0)));
        r_l = vreinterpretq_s16_u8(vtrn2q_u8(rb_l, vdupq_n_u8(0)));
        r_h = vreinterpretq_s16_u8(vtrn2q_u8(rb_h, vdupq_n_u8(0)));
      } else {
        r_l = vreinterpretq_s16_u8(vtrn1q_u8(rb_l, vdupq_n_u8(0)));
        r_h = vreinterpretq_s16_u8(vtrn1q_u8(rb_h, vdupq_n_u8(0)));
        b_l = vreinterpretq_s16_u8(vtrn2q_u8(rb_l, vdupq_n_u8(0)));
        b_h = vreinterpretq_s16_u8(vtrn2q_u8(rb_h, vdupq_n_u8(0)));
      }
      uint16x8_t ga_l = vuzp2q_u8(vsrc.val[0], vsrc.val[1]);
      g_l = vreinterpretq_s16_u8(vtrn1q_u8(ga_l, vdupq_n_u8(0)));
      uint16x8_t ga_h = vuzp2q_u8(vsrc.val[2], vsrc.val[3]);
      g_h = vreinterpretq_s16_u8(vtrn1q_u8(ga_h, vdupq_n_u8(0)));
    } else {
      // Load deinterleaved
      vsrc = vld3q_u8(src);
      r_l = vreinterpretq_s16_u8(vzip1q_u8(vsrc.val[r_index_], vdupq_n_u8(0)));
      r_h = vreinterpretq_s16_u8(vzip2q_u8(vsrc.val[r_index_], vdupq_n_u8(0)));
      g_l = vreinterpretq_s16_u8(vzip1q_u8(vsrc.val[g_index_], vdupq_n_u8(0)));
      g_h = vreinterpretq_s16_u8(vzip2q_u8(vsrc.val[g_index_], vdupq_n_u8(0)));
      b_l = vreinterpretq_s16_u8(vzip1q_u8(vsrc.val[b_index_], vdupq_n_u8(0)));
      b_h = vreinterpretq_s16_u8(vzip2q_u8(vsrc.val[b_index_], vdupq_n_u8(0)));
    }
    // Compute Y value in 32-bit precision
    int16x8_t y_l, y_h;
    {
      int32x4_t y_ll = vmull_n_s16(vget_low_s16(r_l), kRYWeight);
      int32x4_t y_hl = vmull_n_s16(vget_low_s16(r_h), kRYWeight);
      int32x4_t y_lh = vmull_high_n_s16(r_l, kRYWeight);
      int32x4_t y_hh = vmull_high_n_s16(r_h, kRYWeight);

      y_ll = vmlal_n_s16(y_ll, vget_low_s16(g_l), kGYWeight);
      y_hl = vmlal_n_s16(y_hl, vget_low_s16(g_h), kGYWeight);
      y_lh = vmlal_high_n_s16(y_lh, g_l, kGYWeight);
      y_hh = vmlal_high_n_s16(y_hh, g_h, kGYWeight);

      y_ll = vmlal_n_s16(y_ll, vget_low_s16(b_l), kBYWeight);
      y_hl = vmlal_n_s16(y_hl, vget_low_s16(b_h), kBYWeight);
      y_lh = vmlal_high_n_s16(y_lh, b_l, kBYWeight);
      y_hh = vmlal_high_n_s16(y_hh, b_h, kBYWeight);

      y_l = combine_scaled_s16(y_ll, y_lh);
      y_h = combine_scaled_s16(y_hl, y_hh);
    }

    // Using the 16-bit Y value, calculate U
    int16x8_t u_l, u_h;
    {
      int16x8_t uy_l = vqsubq(b_l, y_l);
      int16x8_t uy_h = vqsubq(b_h, y_h);

      int32x4_t u_ll = vdupq_n_s32(half_);
      int32x4_t u_lh = u_ll;
      int32x4_t u_hl = u_ll;
      int32x4_t u_hh = u_ll;

      u_ll = vmlal_n_s16(u_ll, vget_low_s16(uy_l), kBUWeight);
      u_hl = vmlal_n_s16(u_hl, vget_low_s16(uy_h), kBUWeight);
      u_lh = vmlal_high_n_s16(u_lh, uy_l, kBUWeight);
      u_hh = vmlal_high_n_s16(u_hh, uy_h, kBUWeight);

      u_l = combine_scaled_s16(u_ll, u_lh);
      u_h = combine_scaled_s16(u_hl, u_hh);
    }

    // Using the 16-bit Y value, calculate V
    int16x8_t v_l, v_h;
    {
      int16x8_t vy_l = vqsubq(r_l, y_l);
      int16x8_t vy_h = vqsubq(r_h, y_h);

      int32x4_t v_ll = vdupq_n_s32(half_);
      int32x4_t v_lh = v_ll;
      int32x4_t v_hl = v_ll;
      int32x4_t v_hh = v_ll;

      v_ll = vmlal_n_s16(v_ll, vget_low_s16(vy_l), kRVWeight);
      v_hl = vmlal_n_s16(v_hl, vget_low_s16(vy_h), kRVWeight);
      v_lh = vmlal_high_n_s16(v_lh, vy_l, kRVWeight);
      v_hh = vmlal_high_n_s16(v_hh, vy_h, kRVWeight);

      v_l = combine_scaled_s16(v_ll, v_lh);
      v_h = combine_scaled_s16(v_hl, v_hh);
    }

    // Narrow the results to 8 bits
    uint8x16x3_t yuv;
    yuv.val[0] = vcombine_u8(vqmovun_s16(y_l), vqmovun_s16(y_h));
    yuv.val[1] = vcombine_u8(vqmovun_s16(u_l), vqmovun_s16(u_h));
    yuv.val[2] = vcombine_u8(vqmovun_s16(v_l), vqmovun_s16(v_h));

    // Store interleaved YUV pixels to memory.
    vst3q_u8(dst, yuv);
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    int32_t y = src[r_index_] * kRYWeight + src[g_index_] * kGYWeight +
                src[b_index_] * kBYWeight;
    y = rounding_shift_right(y, kWeightScale);
    int32_t u = (src[b_index_] - y) * kBUWeight + half_;
    u = rounding_shift_right(u, kWeightScale);
    int32_t v = (src[r_index_] - y) * kRVWeight + half_;
    v = rounding_shift_right(v, kWeightScale);
    dst[0] = saturating_cast<int32_t, uint8_t>(y);
    dst[1] = saturating_cast<int32_t, uint8_t>(u);
    dst[2] = saturating_cast<int32_t, uint8_t>(v);
  }

 private:
  static constexpr size_t r_index_ = BGR ? 2 : 0;
  static constexpr size_t g_index_ = 1;
  static constexpr size_t b_index_ = BGR ? 0 : 2;
  static constexpr size_t step_ = ALPHA ? 4 : 3;
  static constexpr uint32_t half_ =
      (std::numeric_limits<uint8_t>::max() / 2 + 1U) << kWeightScale;

  static int16x8_t combine_scaled_s16(int32x4_t a, int32x4_t b) {
    return vrshrn_high_n_s32(vrshrn_n_s32(a, kWeightScale), b, kWeightScale);
  }
};  // end of class RGBToYUVAll<bool BGR, bool ALPHA>

template <typename OperationType, typename ScalarType>
kleidicv_error_t rgb2yuv_operation(OperationType &operation,
                                   const ScalarType *src, size_t src_stride,
                                   ScalarType *dst, size_t dst_stride,
                                   size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows src_rows{src, src_stride, operation.input_channels()};
  Rows dst_rows{dst, dst_stride, 3};

  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

using RGBToYUV = RGBToYUVAll<false, false>;
using RGBAToYUV = RGBToYUVAll<false, true>;
using BGRToYUV = RGBToYUVAll<true, false>;
using BGRAToYUV = RGBToYUVAll<true, true>;

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgb_to_yuv_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  RGBToYUV operation;
  return rgb2yuv_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_yuv_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  RGBAToYUV operation;
  return rgb2yuv_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t bgr_to_yuv_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  BGRToYUV operation;
  return rgb2yuv_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t bgra_to_yuv_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  BGRAToYUV operation;
  return rgb2yuv_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

}  // namespace kleidicv::neon
