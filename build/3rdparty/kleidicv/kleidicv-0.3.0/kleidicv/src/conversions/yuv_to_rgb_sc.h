// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_YUV_TO_RGB_SC_H
#define KLEIDICV_YUV_TO_RGB_SC_H

#include "kleidicv/conversions/yuv_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <bool BGR>
class YUVToRGB : public UnrollOnce {
 public:
  using ContextType = Context;
  using ScalarType = uint8_t;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = VecTraits::VectorType;
  using Vector3Type = VecTraits::Vector3Type;

  void vector_path(ContextType ctx, const ScalarType *src,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    Vector3Type svsrc = svld3(pg, src);
    svint16_t y_0 = svreinterpret_s16_u16(svshllb_n_u16(svget3(svsrc, 0), 0));
    svint16_t y_1 = svreinterpret_s16_u16(svshllt_n_u16(svget3(svsrc, 0), 0));
    svint16_t u4_0 =
        svreinterpret_s16_u16(svshllb_n_u16(svget3(svsrc, 1), kPreShift));
    svint16_t u4_1 =
        svreinterpret_s16_u16(svshllt_n_u16(svget3(svsrc, 1), kPreShift));
    svint16_t v4_0 =
        svreinterpret_s16_u16(svshllb_n_u16(svget3(svsrc, 2), kPreShift));
    svint16_t v4_1 =
        svreinterpret_s16_u16(svshllt_n_u16(svget3(svsrc, 2), kPreShift));
    svuint8_t r, g, b;

    // Compute B value in 32-bit precision
    {
      // Multiplication is done with uint16_t because UBWeight only fits in
      // unsigned 16-bit
      svint32_t b_00 = svreinterpret_s32_u32(
          svmullb(svreinterpret_u16_s16(u4_0), kUnsignedUBWeight));
      svint32_t b_01 = svreinterpret_s32_u32(
          svmullt(svreinterpret_u16_s16(u4_0), kUnsignedUBWeight));
      svint32_t b_10 = svreinterpret_s32_u32(
          svmullb(svreinterpret_u16_s16(u4_1), kUnsignedUBWeight));
      svint32_t b_11 = svreinterpret_s32_u32(
          svmullt(svreinterpret_u16_s16(u4_1), kUnsignedUBWeight));

      b_00 = svadd_n_s32_x(svptrue_b32(), b_00, kBDelta4);
      b_01 = svadd_n_s32_x(svptrue_b32(), b_01, kBDelta4);
      b_10 = svadd_n_s32_x(svptrue_b32(), b_10, kBDelta4);
      b_11 = svadd_n_s32_x(svptrue_b32(), b_11, kBDelta4);

      svint16_t b_0 = svadd_x(
          svptrue_b16(), y_0,
          svtrn2_s16(svreinterpret_s16_s32(b_00), svreinterpret_s16_s32(b_01)));
      svint16_t b_1 = svadd_x(
          svptrue_b16(), y_1,
          svtrn2_s16(svreinterpret_s16_s32(b_10), svreinterpret_s16_s32(b_11)));

      b = svqxtunt(svqxtunb(b_0), b_1);
    }

    // Compute G value in 32-bit precision
    {
      svint32_t svg_delta4 = svdup_n_s32(kGDelta4);
      svint32_t g_00 = svmlalb(svg_delta4, u4_0, kUGWeight);
      svint32_t g_01 = svmlalt(svg_delta4, u4_0, kUGWeight);
      svint32_t g_10 = svmlalb(svg_delta4, u4_1, kUGWeight);
      svint32_t g_11 = svmlalt(svg_delta4, u4_1, kUGWeight);

      g_00 = svmlalb(g_00, v4_0, kVGWeight);
      g_01 = svmlalt(g_01, v4_0, kVGWeight);
      g_10 = svmlalb(g_10, v4_1, kVGWeight);
      g_11 = svmlalt(g_11, v4_1, kVGWeight);

      svint16_t g_0 = svadd_x(
          svptrue_b16(), y_0,
          svtrn2_s16(svreinterpret_s16_s32(g_00), svreinterpret_s16_s32(g_01)));
      svint16_t g_1 = svadd_x(
          svptrue_b16(), y_1,
          svtrn2_s16(svreinterpret_s16_s32(g_10), svreinterpret_s16_s32(g_11)));

      g = svqxtunt(svqxtunb(g_0), g_1);
    }

    // Compute R value in 32-bit precision
    {
      svint32_t svr_delta4 = svdup_n_s32(kRDelta4);
      svint32_t r_00 = svmlalb(svr_delta4, v4_0, kVRWeight);
      svint32_t r_01 = svmlalt(svr_delta4, v4_0, kVRWeight);
      svint32_t r_10 = svmlalb(svr_delta4, v4_1, kVRWeight);
      svint32_t r_11 = svmlalt(svr_delta4, v4_1, kVRWeight);

      svint16_t r_0 = svadd_x(
          svptrue_b16(), y_0,
          svtrn2_s16(svreinterpret_s16_s32(r_00), svreinterpret_s16_s32(r_01)));
      svint16_t r_1 = svadd_x(
          svptrue_b16(), y_1,
          svtrn2_s16(svreinterpret_s16_s32(r_10), svreinterpret_s16_s32(r_11)));

      r = svqxtunt(svqxtunb(r_0), r_1);
    }

    // Narrow the results to 8 bits
    svuint8x3_t rgb;
    if constexpr (BGR) {
      rgb = svcreate3(b, g, r);
    } else {
      rgb = svcreate3(r, g, b);
    }

    // Store interleaved RGB pixels to memory.
    svst3_u8(pg, dst, rgb);
  }
};  // end of class YUVToRGB<bool BGR>

template <typename OperationType, typename ScalarType>
kleidicv_error_t yuv2rgb_operation(
    OperationType operation, const ScalarType *src, size_t src_stride,
    ScalarType *dst, size_t dst_stride, size_t width,
    size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows src_rows{src, src_stride, 3};
  Rows dst_rows{dst, dst_stride, 3};

  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t yuv_to_rgb_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  YUVToRGB<false> operation;
  return yuv2rgb_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t yuv_to_bgr_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  YUVToRGB<true> operation;
  return yuv2rgb_operation(operation, src, src_stride, dst, dst_stride, width,
                           height);
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_YUV_TO_RGB_SC_H
