// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "kleidicv/conversions/yuv_sp_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <bool BGR, bool ALPHA>
class YUVSpToRGBxOrBGRx final : public UnrollOnce, public TryToAvoidTailLoop {
 public:
  using VecTraits = neon::VecTraits<uint8_t>;
  using ScalarType = VecTraits::ScalarType;
  using VectorType = VecTraits::VectorType;

  explicit YUVSpToRGBxOrBGRx(bool is_nv21)
      : y_weight_{vdupq_n_s32(kYWeight)},
        uv_weights_{vld2_s32(kUVWeights)},
        // Both the rounding shift right constant and the -128 value are
        // included.
        r_base_{vdupq_n_s32(static_cast<int32_t>(1 << (kWeightScale - 1)) -
                            128 * kUVWeights[kRVWeightIndex])},
        g_base_{vdupq_n_s32(static_cast<int32_t>(1 << (kWeightScale - 1)) -
                            128 * (kUVWeights[1] + kUVWeights[2]))},
        b_base_{vdupq_n_s32(static_cast<int32_t>(1 << (kWeightScale - 1)) -
                            128 * kUVWeights[3])},
        de_interleave_indices_{vld1q_s8_x4(kDeInterleaveTableIndices)},
        is_nv21_(is_nv21) {}

  // Returns the number of channels in the output image.
  static constexpr size_t output_channels() {
    return ALPHA ? /* RGBA */ 4 : /* RGB */ 3;
  }

  // Processes 2 * 16 bytes (even and odd rows) of the input YUV data, and
  // outputs 2 * 3 (or 4) * 16 bytes of RGB (or RGBA) data per loop iteration.
  void vector_path(VectorType y0, VectorType y1, VectorType uv,
                   ScalarType *rgbx_row_0, ScalarType *rgbx_row_1) {
    // Y' = saturating(Ya - 16) and widen to 32-bits.
    uint8x16_t y0_m16 = vqsubq_u8(y0, vdupq_n_u8(16));
    uint8x16_t y1_m16 = vqsubq_u8(y1, vdupq_n_u8(16));

    uint32x4_t y0_m16_even_l =
        vreinterpretq_u32_u8(vqtbl1q_s8(y0_m16, de_interleave_indices_.val[0]));
    uint32x4_t y0_m16_even_h =
        vreinterpretq_u32_u8(vqtbl1q_s8(y0_m16, de_interleave_indices_.val[1]));
    uint32x4_t y0_m16_odd_l =
        vreinterpretq_u32_u8(vqtbl1q_s8(y0_m16, de_interleave_indices_.val[2]));
    uint32x4_t y0_m16_odd_h =
        vreinterpretq_u32_u8(vqtbl1q_s8(y0_m16, de_interleave_indices_.val[3]));

    uint32x4_t y1_m16_even_l =
        vreinterpretq_u32_u8(vqtbl1q_s8(y1_m16, de_interleave_indices_.val[0]));
    uint32x4_t y1_m16_even_h =
        vreinterpretq_u32_u8(vqtbl1q_s8(y1_m16, de_interleave_indices_.val[1]));
    uint32x4_t y1_m16_odd_l =
        vreinterpretq_u32_u8(vqtbl1q_s8(y1_m16, de_interleave_indices_.val[2]));
    uint32x4_t y1_m16_odd_h =
        vreinterpretq_u32_u8(vqtbl1q_s8(y1_m16, de_interleave_indices_.val[3]));

    // Y = Weight(Y) * Y'
    y0_m16_even_l = vmulq_s32(vreinterpretq_u32_s32(y0_m16_even_l), y_weight_);
    y0_m16_even_h = vmulq_s32(vreinterpretq_u32_s32(y0_m16_even_h), y_weight_);
    y0_m16_odd_l = vmulq_s32(vreinterpretq_u32_s32(y0_m16_odd_l), y_weight_);
    y0_m16_odd_h = vmulq_s32(vreinterpretq_u32_s32(y0_m16_odd_h), y_weight_);

    y1_m16_even_l = vmulq_s32(vreinterpretq_u32_s32(y1_m16_even_l), y_weight_);
    y1_m16_even_h = vmulq_s32(vreinterpretq_u32_s32(y1_m16_even_h), y_weight_);
    y1_m16_odd_l = vmulq_s32(vreinterpretq_u32_s32(y1_m16_odd_l), y_weight_);
    y1_m16_odd_h = vmulq_s32(vreinterpretq_u32_s32(y1_m16_odd_h), y_weight_);

    // Widen U and V to 32 bits.
    int32x4_t u_l = vqtbl1q_s8(uv, de_interleave_indices_.val[0]);
    int32x4_t u_h = vqtbl1q_s8(uv, de_interleave_indices_.val[1]);

    int32x4_t v_l = vqtbl1q_s8(uv, de_interleave_indices_.val[2]);
    int32x4_t v_h = vqtbl1q_s8(uv, de_interleave_indices_.val[3]);

    // Swap U and V channels for NV21 (order is V, U).
    if (is_nv21_) {
      std::swap(u_l, v_l);
      std::swap(u_h, v_h);
    }

    // R - Y = Rbase + Weight(RV) * V =
    //         Weight(RV) * ((1 << (SCALE - 1)) - 128) + Weight(RV) * V
    int32x4_t r_sub_y_l = vmlaq_lane_s32(r_base_, v_l, uv_weights_.val[0], 0);
    int32x4_t r_sub_y_h = vmlaq_lane_s32(r_base_, v_h, uv_weights_.val[0], 0);

    // G - Y = Gbase + Weight(GU) * U + Weight(GV) * V =
    //         Weight(GU) * ((1 << (SCALE - 1)) - 128) +
    //         Weight(GV) * ((1 << (SCALE - 1)) - 128) +
    //         Weight(GU) * U + Weight(GV) * V
    int32x4_t g_sub_y_l = vmlaq_lane_s32(g_base_, u_l, uv_weights_.val[1], 0);
    int32x4_t g_sub_y_h = vmlaq_lane_s32(g_base_, u_h, uv_weights_.val[1], 0);
    g_sub_y_l = vmlaq_lane_s32(g_sub_y_l, v_l, uv_weights_.val[0], 1);
    g_sub_y_h = vmlaq_lane_s32(g_sub_y_h, v_h, uv_weights_.val[0], 1);

    // B - Y = Bbase + Weight(BU) * U =
    //         Weight(BU) * ((1 << (SCALE - 1)) - 128) + Weight(BU) * U
    int32x4_t b_sub_y_l = vmlaq_lane_s32(b_base_, u_l, uv_weights_.val[1], 1);
    int32x4_t b_sub_y_h = vmlaq_lane_s32(b_base_, u_h, uv_weights_.val[1], 1);

    // R = (R - Y) + Y
    int32x4_t r0_even_l = vaddq_s32(r_sub_y_l, y0_m16_even_l);
    int32x4_t r0_even_h = vaddq_s32(r_sub_y_h, y0_m16_even_h);
    int32x4_t r0_odd_l = vaddq_s32(r_sub_y_l, y0_m16_odd_l);
    int32x4_t r0_odd_h = vaddq_s32(r_sub_y_h, y0_m16_odd_h);
    int16x8_t r0_even = combine_scaled_s16(r0_even_l, r0_even_h);
    int16x8_t r0_odd = combine_scaled_s16(r0_odd_l, r0_odd_h);

    int32x4_t r1_even_l = vaddq_s32(r_sub_y_l, y1_m16_even_l);
    int32x4_t r1_even_h = vaddq_s32(r_sub_y_h, y1_m16_even_h);
    int32x4_t r1_odd_l = vaddq_s32(r_sub_y_l, y1_m16_odd_l);
    int32x4_t r1_odd_h = vaddq_s32(r_sub_y_h, y1_m16_odd_h);
    int16x8_t r1_even = combine_scaled_s16(r1_even_l, r1_even_h);
    int16x8_t r1_odd = combine_scaled_s16(r1_odd_l, r1_odd_h);

    // G = (G - Y) + Y
    int32x4_t g0_even_l = vaddq_s32(g_sub_y_l, y0_m16_even_l);
    int32x4_t g0_even_h = vaddq_s32(g_sub_y_h, y0_m16_even_h);
    int32x4_t g0_odd_l = vaddq_s32(g_sub_y_l, y0_m16_odd_l);
    int32x4_t g0_odd_h = vaddq_s32(g_sub_y_h, y0_m16_odd_h);
    int16x8_t g0_even = combine_scaled_s16(g0_even_l, g0_even_h);
    int16x8_t g0_odd = combine_scaled_s16(g0_odd_l, g0_odd_h);

    int32x4_t g1_even_l = vaddq_s32(g_sub_y_l, y1_m16_even_l);
    int32x4_t g1_even_h = vaddq_s32(g_sub_y_h, y1_m16_even_h);
    int32x4_t g1_odd_l = vaddq_s32(g_sub_y_l, y1_m16_odd_l);
    int32x4_t g1_odd_h = vaddq_s32(g_sub_y_h, y1_m16_odd_h);
    int16x8_t g1_even = combine_scaled_s16(g1_even_l, g1_even_h);
    int16x8_t g1_odd = combine_scaled_s16(g1_odd_l, g1_odd_h);

    // B = (B - Y) + Y
    int32x4_t b0_even_l = vaddq_s32(b_sub_y_l, y0_m16_even_l);
    int32x4_t b0_even_h = vaddq_s32(b_sub_y_h, y0_m16_even_h);
    int32x4_t b0_odd_l = vaddq_s32(b_sub_y_l, y0_m16_odd_l);
    int32x4_t b0_odd_h = vaddq_s32(b_sub_y_h, y0_m16_odd_h);
    int16x8_t b0_even = combine_scaled_s16(b0_even_l, b0_even_h);
    int16x8_t b0_odd = combine_scaled_s16(b0_odd_l, b0_odd_h);

    int32x4_t b1_even_l = vaddq_s32(b_sub_y_l, y1_m16_even_l);
    int32x4_t b1_even_h = vaddq_s32(b_sub_y_h, y1_m16_even_h);
    int32x4_t b1_odd_l = vaddq_s32(b_sub_y_l, y1_m16_odd_l);
    int32x4_t b1_odd_h = vaddq_s32(b_sub_y_h, y1_m16_odd_h);
    int16x8_t b1_even = combine_scaled_s16(b1_even_l, b1_even_h);
    int16x8_t b1_odd = combine_scaled_s16(b1_odd_l, b1_odd_h);

    // Zip even and odd RGB pixels.
    uint8x8x2_t r0 = vzip_u8(vqmovun_s16(r0_even), vqmovun_s16(r0_odd));
    uint8x8x2_t r1 = vzip_u8(vqmovun_s16(r1_even), vqmovun_s16(r1_odd));
    uint8x8x2_t g0 = vzip_u8(vqmovun_s16(g0_even), vqmovun_s16(g0_odd));
    uint8x8x2_t g1 = vzip_u8(vqmovun_s16(g1_even), vqmovun_s16(g1_odd));
    uint8x8x2_t b0 = vzip_u8(vqmovun_s16(b0_even), vqmovun_s16(b0_odd));
    uint8x8x2_t b1 = vzip_u8(vqmovun_s16(b1_even), vqmovun_s16(b1_odd));

    if constexpr (ALPHA) {
      uint8x16x4_t rgba0, rgba1;
      // Red channel
      rgba0.val[0] = vcombine_u8(r0.val[0], r0.val[1]);
      rgba1.val[0] = vcombine_u8(r1.val[0], r1.val[1]);
      // Green channel
      rgba0.val[1] = vcombine_u8(g0.val[0], g0.val[1]);
      rgba1.val[1] = vcombine_u8(g1.val[0], g1.val[1]);
      // Blue channel
      rgba0.val[2] = vcombine_u8(b0.val[0], b0.val[1]);
      rgba1.val[2] = vcombine_u8(b1.val[0], b1.val[1]);
      // Alpha channel
      rgba0.val[3] = vdupq_n_u8(0xFF);
      rgba1.val[3] = vdupq_n_u8(0xFF);

      if constexpr (BGR) {
        std::swap(rgba0.val[0], rgba0.val[2]);
        std::swap(rgba1.val[0], rgba1.val[2]);
      }

      // Store RGB pixels to memory.
      vst4q_u8(rgbx_row_0, rgba0);
      vst4q_u8(rgbx_row_1, rgba1);
    } else {
      uint8x16x3_t rgb0, rgb1;
      // Red channel
      rgb0.val[0] = vcombine_u8(r0.val[0], r0.val[1]);
      rgb1.val[0] = vcombine_u8(r1.val[0], r1.val[1]);
      // Green channel
      rgb0.val[1] = vcombine_u8(g0.val[0], g0.val[1]);
      rgb1.val[1] = vcombine_u8(g1.val[0], g1.val[1]);
      // Blue channel
      rgb0.val[2] = vcombine_u8(b0.val[0], b0.val[1]);
      rgb1.val[2] = vcombine_u8(b1.val[0], b1.val[1]);

      if constexpr (BGR) {
        std::swap(rgb0.val[0], rgb0.val[2]);
        std::swap(rgb1.val[0], rgb1.val[2]);
      }

      // Store RGB pixels to memory.
      vst3q_u8(rgbx_row_0, rgb0);
      vst3q_u8(rgbx_row_1, rgb1);
    }
  }

  // Processes inputs which are not long enough to fit a vector.
  void scalar_path(size_t length, const ScalarType *y_row_0,
                   const ScalarType *y_row_1, const ScalarType *uv_row,
                   ScalarType *rgbx_row_0, ScalarType *rgbx_row_1) {
    const uint8_t *y_rows[2] = {y_row_0, y_row_1};
    uint8_t *rgbx_rows[2] = {rgbx_row_0, rgbx_row_1};

    int32_t u_m128 = 0, v_m128 = 0;
    for (size_t index = 0; index < length; ++index) {
      disable_loop_vectorization();

      // There is one {U, V} pair for 4 Y values.
      if ((index % 2) == 0) {
        u_m128 = uv_row[0] - 128;
        v_m128 = uv_row[1] - 128;
        uv_row += 2;
        if (is_nv21_) {
          std::swap(u_m128, v_m128);
        }
      }

      for (size_t selector = 0; selector < 2; ++selector) {
        int32_t y = kYWeight * std::max(y_rows[selector][index] - 16, 0);
        int32_t r = y + kUVWeights[kRVWeightIndex] * v_m128;
        int32_t g = y + kUVWeights[kGUWeightIndex] * u_m128 +
                    kUVWeights[kGVWeightIndex] * v_m128;
        int32_t b = y + kUVWeights[kBUWeightIndex] * u_m128;

        r = rounding_shift_right(r, kWeightScale);
        g = rounding_shift_right(g, kWeightScale);
        b = rounding_shift_right(b, kWeightScale);

        if constexpr (BGR) {
          std::swap(r, b);
        }

        rgbx_rows[selector][0] = saturating_cast<int32_t, uint8_t>(r);
        rgbx_rows[selector][1] = saturating_cast<int32_t, uint8_t>(g);
        rgbx_rows[selector][2] = saturating_cast<int32_t, uint8_t>(b);
        if constexpr (ALPHA) {
          rgbx_rows[selector][3] = 0xFF;
        }

        rgbx_rows[selector] += ALPHA ? /* RGBA */ 4 : /* RGB */ 3;
      }
    }
  }

 private:
  static int16x8_t combine_scaled_s16(int32x4_t a, int32x4_t b) {
    return vcombine_s16(vmovn_s32(vshrq_n_s32(a, kWeightScale)),
                        vmovn_s32(vshrq_n_s32(b, kWeightScale)));
  }

  int32x4_t y_weight_;
  int32x2x2_t uv_weights_;
  int32x4_t r_base_;
  int32x4_t g_base_;
  int32x4_t b_base_;
  int8x16x4_t de_interleave_indices_;

  const bool is_nv21_;

  // clang-format off
  static constexpr int8_t kDeInterleaveTableIndices[64] = {
      /* low and even */
      0, -1, -1, -1,  2, -1, -1, -1,  4, -1, -1, -1,  6, -1, -1, -1,
      /* high and even */
      8, -1, -1, -1, 10, -1, -1, -1, 12, -1, -1, -1, 14, -1, -1, -1,
      /* low and odd */
      1, -1, -1, -1,  3, -1, -1, -1,  5, -1, -1, -1,  7, -1, -1, -1,
      /* high and odd */
      9, -1, -1, -1, 11, -1, -1, -1, 13, -1, -1, -1, 15, -1, -1, -1,
  };
  // clang-format on
};  // end of class YUVSpToRGBxOrBGRx<bool, bool>

using YUVSpToRGB = YUVSpToRGBxOrBGRx<false, false>;
using YUVSpToRGBA = YUVSpToRGBxOrBGRx<false, true>;
using YUVSpToBGR = YUVSpToRGBxOrBGRx<true, false>;
using YUVSpToBGRA = YUVSpToRGBxOrBGRx<true, true>;

template <typename OperationType, typename ScalarType>
kleidicv_error_t yuv2rgbx_operation(
    OperationType &operation, const ScalarType *src_y, size_t src_y_stride,
    const ScalarType *src_uv, size_t src_uv_stride, ScalarType *dst,
    size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src_y, src_y_stride, height);
  CHECK_POINTER_AND_STRIDE(src_uv, src_uv_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  ParallelRows y_rows{src_y, src_y_stride};
  Rows uv_rows{src_uv, src_uv_stride};
  ParallelRows rgbx_rows{dst, dst_stride, operation.output_channels()};

  RemoveContextAdapter remove_context_adapter{operation};
  OperationAdapter operation_adapter{remove_context_adapter};
  RemainingPathToScalarPathAdapter remaining_path_adapter{operation_adapter};
  OperationContextAdapter context_adapter{remaining_path_adapter};
  ParallelRowsAdapter parallel_rows_adapter{context_adapter};
  RowBasedOperation row_based_operation{parallel_rows_adapter};
  zip_parallel_rows(row_based_operation, rect, y_rows, uv_rows, rgbx_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv_sp_to_rgb_u8(const uint8_t *src_y, size_t src_y_stride,
                                  const uint8_t *src_uv, size_t src_uv_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height, bool is_nv21) {
  YUVSpToRGB operation{is_nv21};
  return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                            src_uv_stride, dst, dst_stride, width, height);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv_sp_to_rgba_u8(const uint8_t *src_y, size_t src_y_stride,
                                   const uint8_t *src_uv, size_t src_uv_stride,
                                   uint8_t *dst, size_t dst_stride,
                                   size_t width, size_t height, bool is_nv21) {
  YUVSpToRGBA operation{is_nv21};
  return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                            src_uv_stride, dst, dst_stride, width, height);
}
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
yuv_sp_to_bgr_u8(const uint8_t *src_y, size_t src_y_stride,
                 const uint8_t *src_uv, size_t src_uv_stride, uint8_t *dst,
                 size_t dst_stride, size_t width, size_t height, bool is_nv21) {
  YUVSpToBGR operation{is_nv21};
  return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                            src_uv_stride, dst, dst_stride, width, height);
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t yuv_sp_to_bgra_u8(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, bool is_nv21) {
  YUVSpToBGRA operation{is_nv21};
  return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                            src_uv_stride, dst, dst_stride, width, height);
}

}  // namespace kleidicv::neon
