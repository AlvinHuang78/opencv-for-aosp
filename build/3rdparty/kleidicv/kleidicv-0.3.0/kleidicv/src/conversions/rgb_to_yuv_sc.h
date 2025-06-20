// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RGB_TO_YUV_SC_H
#define KLEIDICV_RGB_TO_YUV_SC_H

#include <limits>
#include <memory>

#include "kleidicv/conversions/rgb_to_yuv.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <bool BGR>
class RGBToYUVBase : public UnrollOnce {
 public:
  using ContextType = Context;
  using ScalarType = uint8_t;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;

 protected:
  void vector_calculation_path(svbool_t pg, svint16_t r_0, svint16_t r_1,
                               svint16_t g_0, svint16_t g_1, svint16_t b_0,
                               svint16_t b_1,
                               ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    // Compute Y value in 32-bit precision
    svint16_t y_0, y_1;
    {
      svint32_t y_00 = svmullb(r_0, kRYWeight);
      svint32_t y_01 = svmullb(r_1, kRYWeight);
      svint32_t y_10 = svmullt(r_0, kRYWeight);
      svint32_t y_11 = svmullt(r_1, kRYWeight);

      y_00 = svmlalb(y_00, g_0, kGYWeight);
      y_01 = svmlalb(y_01, g_1, kGYWeight);
      y_10 = svmlalt(y_10, g_0, kGYWeight);
      y_11 = svmlalt(y_11, g_1, kGYWeight);

      y_00 = svmlalb(y_00, b_0, kBYWeight);
      y_01 = svmlalb(y_01, b_1, kBYWeight);
      y_10 = svmlalt(y_10, b_0, kBYWeight);
      y_11 = svmlalt(y_11, b_1, kBYWeight);

      y_0 = combine_scaled_s16(y_00, y_10);
      y_1 = combine_scaled_s16(y_01, y_11);
    }

    // Using the 16-bit Y value, calculate U
    svint16_t u_0, u_1;
    {
      svint16_t uy_0 = svsub_x(VecTraits::svptrue(), b_0, y_0);
      svint16_t uy_1 = svsub_x(VecTraits::svptrue(), b_1, y_1);

      svint32_t u_00 = svdup_n_s32(half_);
      svint32_t u_01 = u_00;
      svint32_t u_10 = u_00;
      svint32_t u_11 = u_00;

      u_00 = svmlalb(u_00, uy_0, kBUWeight);
      u_01 = svmlalb(u_01, uy_1, kBUWeight);
      u_10 = svmlalt(u_10, uy_0, kBUWeight);
      u_11 = svmlalt(u_11, uy_1, kBUWeight);

      u_0 = combine_scaled_s16(u_00, u_10);
      u_1 = combine_scaled_s16(u_01, u_11);
    }

    // Using the 16-bit Y value, calculate V
    svint16_t v_0, v_1;
    {
      svint16_t vy_0 = svsub_x(VecTraits::svptrue(), r_0, y_0);
      svint16_t vy_1 = svsub_x(VecTraits::svptrue(), r_1, y_1);

      svint32_t v_00 = svdup_n_s32(half_);
      svint32_t v_10 = v_00;
      svint32_t v_01 = v_00;
      svint32_t v_11 = v_00;

      v_00 = svmlalb(v_00, vy_0, kRVWeight);
      v_01 = svmlalb(v_01, vy_1, kRVWeight);
      v_10 = svmlalt(v_10, vy_0, kRVWeight);
      v_11 = svmlalt(v_11, vy_1, kRVWeight);

      v_0 = combine_scaled_s16(v_00, v_10);
      v_1 = combine_scaled_s16(v_01, v_11);
    }

    // Narrow the results to 8 bits
    svuint8x3_t yuv =
        svcreate3(svqxtunt(svqxtunb(y_0), y_1), svqxtunt(svqxtunb(u_0), u_1),
                  svqxtunt(svqxtunb(v_0), v_1));

    // Store interleaved YUV pixels to memory.
    svst3_u8(pg, dst, yuv);
  }

  static constexpr size_t r_index_ = BGR ? 2 : 0;
  static constexpr size_t g_index_ = 1;
  static constexpr size_t b_index_ = BGR ? 0 : 2;
  static constexpr uint32_t half_ =
      (std::numeric_limits<uint8_t>::max() / 2 + 1U) << kWeightScale;
  static svint16_t combine_scaled_s16(svint32_t even, svint32_t odd)
      KLEIDICV_STREAMING_COMPATIBLE {
    return svqrshrnt(svqrshrnb(even, kWeightScale), odd, kWeightScale);
  }
};  // end of class RGBToYUVBase<bool BGR>

// 3-channel input
template <bool BGR>
class RGBToYUV final : public RGBToYUVBase<BGR> {
 public:
  using ContextType = Context;
  using ScalarType = uint8_t;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;

  // Returns the number of channels in the output image.
  static constexpr size_t input_channels() KLEIDICV_STREAMING_COMPATIBLE {
    return 3;
  }

  void vector_path(ContextType ctx, const ScalarType *src,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    svuint8x3_t svsrc = svld3(pg, src);
    svint16_t r_0 = svreinterpret_s16_u16(svmovlb(svget3(svsrc, r_index_)));
    svint16_t r_1 = svreinterpret_s16_u16(svmovlt(svget3(svsrc, r_index_)));
    svint16_t g_0 = svreinterpret_s16_u16(svmovlb(svget3(svsrc, g_index_)));
    svint16_t g_1 = svreinterpret_s16_u16(svmovlt(svget3(svsrc, g_index_)));
    svint16_t b_0 = svreinterpret_s16_u16(svmovlb(svget3(svsrc, b_index_)));
    svint16_t b_1 = svreinterpret_s16_u16(svmovlt(svget3(svsrc, b_index_)));
    RGBToYUVBase<BGR>::vector_calculation_path(pg, r_0, r_1, g_0, g_1, b_0, b_1,
                                               dst);
  }

 private:
  static constexpr size_t r_index_ = RGBToYUVBase<BGR>::r_index_;
  static constexpr size_t g_index_ = RGBToYUVBase<BGR>::g_index_;
  static constexpr size_t b_index_ = RGBToYUVBase<BGR>::b_index_;
};  // end of class RGBToYUV<bool BGR>

// 4-channel input
template <bool BGR>
class RGBAToYUV final : public RGBToYUVBase<BGR>, public UsesTailPath {
 public:
  using ContextType = Context;
  using ScalarType = uint8_t;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;

  explicit RGBAToYUV(svuint8x4_t &sv4) KLEIDICV_STREAMING_COMPATIBLE
      : deinterleave16_indices_(sv4) {
    // clang-format off
    // From the unzipped RGBA -> RBRBRBRB..., take it apart to even and odd
    // pixels, and widen it to 16bits. For that, we need these tables:
    // 0, FF, 4, FF,  8, FF, 12. ...       red0
    // 1, FF, 5, FF,  9, FF, 13, ...        blue0
    // 2, FF, 6, FF, 10, FF, 14, ...       red1
    // 3, FF, 7, FF, 11, FF, 15, ...        blue1
    // clang-format on
    deinterleave16_indices_ =
        svcreate4(svreinterpret_u8_u16(svindex_u16(0xFF00, 0x0004)),
                  svreinterpret_u8_u16(svindex_u16(0xFF01, 0x0004)),
                  svreinterpret_u8_u16(svindex_u16(0xFF02, 0x0004)),
                  svreinterpret_u8_u16(svindex_u16(0xFF03, 0x0004)));
  }

  // Returns the number of channels in the output image.
  static constexpr size_t input_channels() KLEIDICV_STREAMING_COMPATIBLE {
    return 4;
  }

  void vector_path(ContextType ctx, const ScalarType *src,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    common_vector_path(pg, pg, pg, pg, pg, src, dst);
  }

  void tail_path(ContextType ctx, const ScalarType *src,
                 ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    svbool_t pg_0, pg_1, pg_2, pg_3;
    VecTraits::make_consecutive_predicates(pg, pg_0, pg_1, pg_2, pg_3);
    common_vector_path(pg, pg_0, pg_1, pg_2, pg_3, src, dst);
  }

 private:
  void common_vector_path(svbool_t pg, svbool_t pg_0, svbool_t pg_1,
                          svbool_t pg_2, svbool_t pg_3, const ScalarType *src,
                          ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    svint16_t r_0, r_1, g_0, g_1, b_0, b_1;

    svuint8_t src0 = svld1(pg_0, src);
    svuint8_t src1 = svld1_vnum(pg_1, src, 1);
    svuint8_t src2 = svld1_vnum(pg_2, src, 2);
    svuint8_t src3 = svld1_vnum(pg_3, src, 3);

    svuint8_t rb_l = svuzp1_u8(src0, src1);
    svuint8_t rb_h = svuzp1_u8(src2, src3);
    svuint8_t ga_l = svuzp2_u8(src0, src1);
    svuint8_t ga_h = svuzp2_u8(src2, src3);
    if (KLEIDICV_UNLIKELY(svcntb() >= 256)) {
      svuint8_t r, g, b;
      if constexpr (BGR) {
        b = svuzp1_u8(rb_l, rb_h);
        r = svuzp2_u8(rb_l, rb_h);
      } else {
        r = svuzp1_u8(rb_l, rb_h);
        b = svuzp2_u8(rb_l, rb_h);
      }
      g = svuzp1_u8(ga_l, ga_h);
      r_0 = svreinterpret_s16_u16(svmovlb(r));
      r_1 = svreinterpret_s16_u16(svmovlt(r));
      g_0 = svreinterpret_s16_u16(svmovlb(g));
      g_1 = svreinterpret_s16_u16(svmovlt(g));
      b_0 = svreinterpret_s16_u16(svmovlb(b));
      b_1 = svreinterpret_s16_u16(svmovlt(b));
    } else {
      b_0 = svreinterpret_s16_u8(
          svtbl2(svcreate2(rb_l, rb_h),
                 svget4(deinterleave16_indices_, b_index_ / 2)));
      b_1 = svreinterpret_s16_u8(
          svtbl2(svcreate2(rb_l, rb_h),
                 svget4(deinterleave16_indices_, b_index_ / 2 + 2)));
      r_0 = svreinterpret_s16_u8(
          svtbl2(svcreate2(rb_l, rb_h),
                 svget4(deinterleave16_indices_, r_index_ / 2)));
      r_1 = svreinterpret_s16_u8(
          svtbl2(svcreate2(rb_l, rb_h),
                 svget4(deinterleave16_indices_, r_index_ / 2 + 2)));

      g_0 = svreinterpret_s16_u8(
          svtbl2(svcreate2(ga_l, ga_h), svget4(deinterleave16_indices_, 0)));
      g_1 = svreinterpret_s16_u8(
          svtbl2(svcreate2(ga_l, ga_h), svget4(deinterleave16_indices_, 2)));
    }
    RGBToYUVBase<BGR>::vector_calculation_path(pg, r_0, r_1, g_0, g_1, b_0, b_1,
                                               dst);
  }

  static constexpr size_t r_index_ = RGBToYUVBase<BGR>::r_index_;
  static constexpr size_t g_index_ = RGBToYUVBase<BGR>::g_index_;
  static constexpr size_t b_index_ = RGBToYUVBase<BGR>::b_index_;

  svuint8x4_t &deinterleave16_indices_;
};  // end of class RGBAToYUV<bool BGR>

template <typename OperationType, typename ScalarType>
kleidicv_error_t rgb2yuv_operation(
    OperationType operation, const ScalarType *src, size_t src_stride,
    ScalarType *dst, size_t dst_stride, size_t width,
    size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows src_rows{src, src_stride, operation.input_channels()};
  Rows dst_rows{dst, dst_stride, 3};

  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t rgb_to_yuv_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  RGBToYUV<false> operation;
  return rgb2yuv_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t rgba_to_yuv_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  svuint8x4_t indices;
  RGBAToYUV<false> operation(indices);
  return rgb2yuv_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t bgr_to_yuv_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  RGBToYUV<true> operation;
  return rgb2yuv_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t bgra_to_yuv_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  svuint8x4_t indices;
  RGBAToYUV<true> operation(indices);
  return rgb2yuv_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_RGB_TO_YUV_SC_H
