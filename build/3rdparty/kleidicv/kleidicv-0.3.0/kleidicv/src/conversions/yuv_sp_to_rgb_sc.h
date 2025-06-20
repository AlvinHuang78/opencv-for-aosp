// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_YUV_SP_TO_RGB_SC_H
#define KLEIDICV_YUV_SP_TO_RGB_SC_H

#include "kleidicv/conversions/yuv_sp_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <bool BGR, bool ALPHA>
class YUVSpToRGBxOrBGRx final {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<uint8_t>;

  explicit YUVSpToRGBxOrBGRx(bool is_nv21) KLEIDICV_STREAMING_COMPATIBLE
      : is_nv21_(is_nv21) {}

  // Returns the number of channels in the output image.
  static constexpr size_t output_channels() KLEIDICV_STREAMING_COMPATIBLE {
    return ALPHA ? /* RGBA */ 4 : /* RGB */ 3;
  }

  // Processes 2 * 16 bytes (even and odd rows) of the input YUV data, and
  // outputs 2 * 3 (or 4) * 16 bytes of RGB (or RGBA) data per loop iteration.
  void vector_path(ContextType ctx, const uint8_t *y_row_0,
                   const uint8_t *y_row_1, const uint8_t *uv_row,
                   uint8_t *rgbx_row_0,
                   uint8_t *rgbx_row_1) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();

    // Both the rounding shift right constant and the -128 value are included.
    constexpr int32_t kOffset = 1 << (kWeightScale - 1);
    svint32_t r_base = svdup_s32(kOffset - 128 * kUVWeights[kRVWeightIndex]);
    svint32_t g_base =
        svdup_s32(kOffset - 128 * (kUVWeights[1] + kUVWeights[2]));
    svint32_t b_base = svdup_s32(kOffset - 128 * kUVWeights[3]);

    // Load channels: y0 and y1 are two adjacent rows.
    svuint8_t y0 = svld1(pg, y_row_0);
    svuint8_t y1 = svld1(pg, y_row_1);
    svuint8_t uv = svld1(pg, uv_row);

    // Y' = saturating(Ya - 16) and widen to signed 32-bits.
    svuint8_t y0_m16 = svqsub(y0, static_cast<uint8_t>(16));
    svuint16_t y0_m16_b = svmovlb(y0_m16);  // 'b' means bottom
    svuint16_t y0_m16_t = svmovlt(y0_m16);  // 't' means top
    svint32_t y0_m16_bb = svreinterpret_s32(svmovlb(y0_m16_b));
    svint32_t y0_m16_bt = svreinterpret_s32(svmovlt(y0_m16_b));
    svint32_t y0_m16_tb = svreinterpret_s32(svmovlb(y0_m16_t));
    svint32_t y0_m16_tt = svreinterpret_s32(svmovlt(y0_m16_t));

    svuint8_t y1_m16 = svqsub(y1, static_cast<uint8_t>(16));
    svuint16_t y1_m16_b = svmovlb(y1_m16);
    svuint16_t y1_m16_t = svmovlt(y1_m16);
    svint32_t y1_m16_bb = svreinterpret_s32(svmovlb(y1_m16_b));
    svint32_t y1_m16_bt = svreinterpret_s32(svmovlt(y1_m16_b));
    svint32_t y1_m16_tb = svreinterpret_s32(svmovlb(y1_m16_t));
    svint32_t y1_m16_tt = svreinterpret_s32(svmovlt(y1_m16_t));

    // Y = Weight(Y) * Y'
    y0_m16_bb = svmul_x(pg, y0_m16_bb, kYWeight);
    y0_m16_bt = svmul_x(pg, y0_m16_bt, kYWeight);
    y0_m16_tb = svmul_x(pg, y0_m16_tb, kYWeight);
    y0_m16_tt = svmul_x(pg, y0_m16_tt, kYWeight);

    y1_m16_bb = svmul_x(pg, y1_m16_bb, kYWeight);
    y1_m16_bt = svmul_x(pg, y1_m16_bt, kYWeight);
    y1_m16_tb = svmul_x(pg, y1_m16_tb, kYWeight);
    y1_m16_tt = svmul_x(pg, y1_m16_tt, kYWeight);

    // Widen U and V to 32 bits.
    svint16_t u = svreinterpret_s16(svmovlb(uv));
    svint16_t v = svreinterpret_s16(svmovlt(uv));

    if (is_nv21_) {
      // Swap U and V channels for NV21 (order is V, U).
      swap_scalable(u, v);
    }

    svint32_t u_b = svmovlb(u);
    svint32_t u_t = svmovlt(u);
    svint32_t v_b = svmovlb(v);
    svint32_t v_t = svmovlt(v);

    // R - Y = Rbase + Weight(RV) * V =
    //         Weight(RV) * ((1 << (SCALE - 1)) - 128) + Weight(RV) * V
    svint32_t r_sub_y_b = svmla_x(pg, r_base, v_b, kUVWeights[kRVWeightIndex]);
    svint32_t r_sub_y_t = svmla_x(pg, r_base, v_t, kUVWeights[kRVWeightIndex]);

    // G - Y = Gbase + Weight(GU) * U + Weight(GV) * V =
    //         Weight(GU) * ((1 << (SCALE - 1)) - 128) +
    //         Weight(GV) * ((1 << (SCALE - 1)) - 128) +
    //         Weight(GU) * U + Weight(GV) * V
    svint32_t g_sub_y_b = svmla_x(pg, g_base, u_b, kUVWeights[kGUWeightIndex]);
    svint32_t g_sub_y_t = svmla_x(pg, g_base, u_t, kUVWeights[kGUWeightIndex]);
    g_sub_y_b = svmla_x(pg, g_sub_y_b, v_b, kUVWeights[kGVWeightIndex]);
    g_sub_y_t = svmla_x(pg, g_sub_y_t, v_t, kUVWeights[kGVWeightIndex]);

    // B - Y = Bbase + Weight(BU) * U =
    //         Weight(BU) * ((1 << (SCALE - 1)) - 128) + Weight(BU) * U
    svint32_t b_sub_y_b = svmla_x(pg, b_base, u_b, kUVWeights[kBUWeightIndex]);
    svint32_t b_sub_y_t = svmla_x(pg, b_base, u_t, kUVWeights[kBUWeightIndex]);

    // R = (R - Y) + Y
    // FIXME: There are too many instructions here.
    // Is there a better way to do this?
    svint16_t r0_b = svaddhnb(r_sub_y_b, y0_m16_bb);
    r0_b = svaddhnt(r0_b, r_sub_y_t, y0_m16_bt);
    r0_b = svsra(svdup_n_s16(0), r0_b, kWeightScale - 16);
    svint16_t r0_t = svaddhnb(r_sub_y_b, y0_m16_tb);
    r0_t = svaddhnt(r0_t, r_sub_y_t, y0_m16_tt);
    r0_t = svsra(svdup_n_s16(0), r0_t, kWeightScale - 16);
    svuint8_t r0 = svqxtunt(svqxtunb(r0_b), r0_t);

    svint16_t r1_b = svaddhnb(r_sub_y_b, y1_m16_bb);
    r1_b = svaddhnt(r1_b, r_sub_y_t, y1_m16_bt);
    r1_b = svsra(svdup_n_s16(0), r1_b, kWeightScale - 16);
    svint16_t r1_t = svaddhnb(r_sub_y_b, y1_m16_tb);
    r1_t = svaddhnt(r1_t, r_sub_y_t, y1_m16_tt);
    r1_t = svsra(svdup_n_s16(0), r1_t, kWeightScale - 16);
    svuint8_t r1 = svqxtunt(svqxtunb(r1_b), r1_t);

    // G = (G - Y) + Y
    svint16_t g0_b = svaddhnb(g_sub_y_b, y0_m16_bb);
    g0_b = svaddhnt(g0_b, g_sub_y_t, y0_m16_bt);
    g0_b = svsra(svdup_n_s16(0), g0_b, kWeightScale - 16);
    svint16_t g0_t = svaddhnb(g_sub_y_b, y0_m16_tb);
    g0_t = svaddhnt(g0_t, g_sub_y_t, y0_m16_tt);
    g0_t = svsra(svdup_n_s16(0), g0_t, kWeightScale - 16);
    svuint8_t g0 = svqxtunt(svqxtunb(g0_b), g0_t);

    svint16_t g1_b = svaddhnb(g_sub_y_b, y1_m16_bb);
    g1_b = svaddhnt(g1_b, g_sub_y_t, y1_m16_bt);
    g1_b = svsra(svdup_n_s16(0), g1_b, kWeightScale - 16);
    svint16_t g1_t = svaddhnb(g_sub_y_b, y1_m16_tb);
    g1_t = svaddhnt(g1_t, g_sub_y_t, y1_m16_tt);
    g1_t = svsra(svdup_n_s16(0), g1_t, kWeightScale - 16);
    svuint8_t g1 = svqxtunt(svqxtunb(g1_b), g1_t);

    // B = (B - Y) + Y
    svint16_t b0_b = svaddhnb(b_sub_y_b, y0_m16_bb);
    b0_b = svaddhnt(b0_b, b_sub_y_t, y0_m16_bt);
    b0_b = svsra(svdup_n_s16(0), b0_b, kWeightScale - 16);
    svint16_t b0_t = svaddhnb(b_sub_y_b, y0_m16_tb);
    b0_t = svaddhnt(b0_t, b_sub_y_t, y0_m16_tt);
    b0_t = svsra(svdup_n_s16(0), b0_t, kWeightScale - 16);
    svuint8_t b0 = svqxtunt(svqxtunb(b0_b), b0_t);

    svint16_t b1_b = svaddhnb(b_sub_y_b, y1_m16_bb);
    b1_b = svaddhnt(b1_b, b_sub_y_t, y1_m16_bt);
    b1_b = svsra(svdup_n_s16(0), b1_b, kWeightScale - 16);
    svint16_t b1_t = svaddhnb(b_sub_y_b, y1_m16_tb);
    b1_t = svaddhnt(b1_t, b_sub_y_t, y1_m16_tt);
    b1_t = svsra(svdup_n_s16(0), b1_t, kWeightScale - 16);
    svuint8_t b1 = svqxtunt(svqxtunb(b1_b), b1_t);

    if constexpr (ALPHA) {
      svuint8x4_t rgba0 =
          svcreate4(BGR ? b0 : r0, g0, BGR ? r0 : b0, svdup_n_u8(0xFF));
      svuint8x4_t rgba1 =
          svcreate4(BGR ? b1 : r1, g1, BGR ? r1 : b1, svdup_n_u8(0xFF));
      // Store RGBA pixels to memory.
      svst4_u8(pg, rgbx_row_0, rgba0);
      svst4_u8(pg, rgbx_row_1, rgba1);
    } else {
      svuint8x3_t rgb0 = svcreate3(BGR ? b0 : r0, g0, BGR ? r0 : b0);
      svuint8x3_t rgb1 = svcreate3(BGR ? b1 : r1, g1, BGR ? r1 : b1);
      // Store RGB pixels to memory.
      svst3(pg, rgbx_row_0, rgb0);
      svst3(pg, rgbx_row_1, rgb1);
    }
  }

 private:
  const bool is_nv21_;
};  // end of class YUVSpToRGBxOrBGRx<bool, bool>

using YUVSpToRGB = YUVSpToRGBxOrBGRx<false, false>;
using YUVSpToRGBA = YUVSpToRGBxOrBGRx<false, true>;
using YUVSpToBGR = YUVSpToRGBxOrBGRx<true, false>;
using YUVSpToBGRA = YUVSpToRGBxOrBGRx<true, true>;

template <typename OperationType, typename ScalarType>
kleidicv_error_t yuv2rgbx_operation(
    OperationType &operation, const ScalarType *src_y, size_t src_y_stride,
    const ScalarType *src_uv, size_t src_uv_stride, ScalarType *dst,
    size_t dst_stride, size_t width,
    size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src_y, src_y_stride, height);
  CHECK_POINTER_AND_STRIDE(src_uv, src_uv_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  ParallelRows y_rows{src_y, src_y_stride};
  Rows uv_rows{src_uv, src_uv_stride};
  ParallelRows rgbx_rows{dst, dst_stride, operation.output_channels()};

  ForwardingOperation forwarding_operation{operation};
  OperationAdapter operation_adapter{forwarding_operation};
  RemainingPathAdapter remaining_path_adapter{operation_adapter};
  OperationContextAdapter context_adapter{remaining_path_adapter};
  ParallelRowsAdapter parallel_rows_adapter{context_adapter};
  RowBasedOperation row_based_operation{parallel_rows_adapter};
  zip_parallel_rows(row_based_operation, rect, y_rows, uv_rows, rgbx_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t yuv_sp_to_rgb_u8_sc(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, bool is_nv21) KLEIDICV_STREAMING_COMPATIBLE {
  YUVSpToRGB operation{is_nv21};
  return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                            src_uv_stride, dst, dst_stride, width, height);
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t yuv_sp_to_rgba_u8_sc(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, bool is_nv21) KLEIDICV_STREAMING_COMPATIBLE {
  YUVSpToRGBA operation{is_nv21};
  return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                            src_uv_stride, dst, dst_stride, width, height);
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t yuv_sp_to_bgr_u8_sc(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, bool is_nv21) KLEIDICV_STREAMING_COMPATIBLE {
  YUVSpToBGR operation{is_nv21};
  return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                            src_uv_stride, dst, dst_stride, width, height);
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t yuv_sp_to_bgra_u8_sc(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, bool is_nv21) KLEIDICV_STREAMING_COMPATIBLE {
  YUVSpToBGRA operation{is_nv21};
  return yuv2rgbx_operation(operation, src_y, src_y_stride, src_uv,
                            src_uv_stride, dst, dst_stride, width, height);
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_YUV_SP_TO_RGB_SC_H
