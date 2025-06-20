// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/ctypes.h"
#include "kleidicv/filters/blur_and_downsample.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"
#include "kleidicv/utils.h"
#include "kleidicv/workspace/blur_and_downsample_ws.h"
#include "kleidicv/workspace/border_5x5.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Applies Gaussian Blur binomial filter to even rows and columns
//
//              [ 1,  4,  6,  4, 1 ]           [ 1 ]
//              [ 4, 16, 24, 16, 4 ]           [ 4 ]
//  F = 1/256 * [ 6, 24, 36, 24, 6 ] = 1/256 * [ 6 ] * [ 1,  4,  6,  4, 1 ]
//              [ 4, 16, 24, 16, 4 ]           [ 4 ]
//              [ 1,  4,  6,  4, 1 ]           [ 1 ]
class BlurAndDownsample {
 public:
  using SourceType = uint8_t;
  using BufferType = uint16_t;
  using DestinationType = uint8_t;
  using SourceVecTraits =
      typename ::KLEIDICV_TARGET_NAMESPACE::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using SourceVector2Type = typename SourceVecTraits::Vector2Type;
  using BufferVecTraits =
      typename ::KLEIDICV_TARGET_NAMESPACE::VecTraits<BufferType>;
  using BufferVectorType = typename BufferVecTraits::VectorType;
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo5x5<SourceType>;
  using BorderType = FixedBorderType;
  using BorderOffsets = typename BorderInfoType::Offsets;

  static constexpr size_t margin = 2UL;

  void process_vertical(
      size_t width, Rows<const SourceType> src_rows, Rows<BufferType> dst_rows,
      BorderOffsets border_offsets) const KLEIDICV_STREAMING_COMPATIBLE {
    LoopUnroll2 loop{width * src_rows.channels(), SourceVecTraits::num_lanes()};

    loop.unroll_twice([&](ptrdiff_t index) KLEIDICV_STREAMING_COMPATIBLE {
      svbool_t pg_all = SourceVecTraits::svptrue();
      vertical_vector_path_2x(pg_all, src_rows, dst_rows, border_offsets,
                              index);
    });

    loop.unroll_once([&](ptrdiff_t index) KLEIDICV_STREAMING_COMPATIBLE {
      svbool_t pg_all = SourceVecTraits::svptrue();
      vertical_vector_path_1x(pg_all, src_rows, dst_rows, border_offsets,
                              index);
    });

    loop.remaining([&](ptrdiff_t index,
                       ptrdiff_t length) KLEIDICV_STREAMING_COMPATIBLE {
      svbool_t pg = SourceVecTraits::svwhilelt(index, length);
      vertical_vector_path_1x(pg, src_rows, dst_rows, border_offsets, index);
    });
  }

  void process_horizontal(size_t width, Rows<const BufferType> src_rows,
                          Rows<DestinationType> dst_rows,
                          BorderOffsets border_offsets) const
      KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t pg_all = BufferVecTraits::svptrue();
    LoopUnroll2 loop{width * src_rows.channels(), BufferVecTraits::num_lanes()};

    loop.unroll_twice([&](size_t index) KLEIDICV_STREAMING_COMPATIBLE {
      horizontal_vector_path_2x(pg_all, pg_all, src_rows, pg_all, dst_rows,
                                border_offsets, static_cast<ptrdiff_t>(index));
    });

    loop.remaining([&](size_t index,
                       size_t length) KLEIDICV_STREAMING_COMPATIBLE {
      svbool_t pg_src_0 = BufferVecTraits::svwhilelt(index, length);
      svbool_t pg_src_1 = BufferVecTraits::svwhilelt(
          index + BufferVecTraits::num_lanes(), length);
      svbool_t pg_dst =
          BufferVecTraits::svwhilelt((index + 1) / 2, (length + 1) / 2);
      horizontal_vector_path_2x(pg_src_0, pg_src_1, src_rows, pg_dst, dst_rows,
                                border_offsets, static_cast<ptrdiff_t>(index));
    });
  }

  void process_horizontal_borders(
      Rows<const BufferType> src_rows, Rows<DestinationType> dst_rows,
      BorderOffsets border_offsets) const KLEIDICV_STREAMING_COMPATIBLE {
    for (ptrdiff_t index = 0;
         index < static_cast<ptrdiff_t>(src_rows.channels()); ++index) {
      disable_loop_vectorization();
      svbool_t pg = svptrue_pat_b8(SV_VL1);
      horizontal_border_path(pg, src_rows, dst_rows, border_offsets, index);
    }
  }

 private:
  void vertical_vector_path_2x(
      svbool_t pg, Rows<const SourceType> src_rows, Rows<BufferType> dst_rows,
      BorderOffsets border_offsets,
      ptrdiff_t index) const KLEIDICV_STREAMING_COMPATIBLE {
    const auto *src_row_0 = &src_rows.at(border_offsets.c0())[index];
    const auto *src_row_1 = &src_rows.at(border_offsets.c1())[index];
    const auto *src_row_2 = &src_rows.at(border_offsets.c2())[index];
    const auto *src_row_3 = &src_rows.at(border_offsets.c3())[index];
    const auto *src_row_4 = &src_rows.at(border_offsets.c4())[index];

    SourceVector2Type src_0;
    SourceVector2Type src_1;
    SourceVector2Type src_2;
    SourceVector2Type src_3;
    SourceVector2Type src_4;

    src_0 =
        svcreate2(svld1(pg, &src_row_0[0]), svld1_vnum(pg, &src_row_0[0], 1));
    src_1 =
        svcreate2(svld1(pg, &src_row_1[0]), svld1_vnum(pg, &src_row_1[0], 1));
    src_2 =
        svcreate2(svld1(pg, &src_row_2[0]), svld1_vnum(pg, &src_row_2[0], 1));
    src_3 =
        svcreate2(svld1(pg, &src_row_3[0]), svld1_vnum(pg, &src_row_3[0], 1));
    src_4 =
        svcreate2(svld1(pg, &src_row_4[0]), svld1_vnum(pg, &src_row_4[0], 1));

    vertical_vector_path(pg, svget2(src_0, 0), svget2(src_1, 0),
                         svget2(src_2, 0), svget2(src_3, 0), svget2(src_4, 0),
                         &dst_rows[index]);
    vertical_vector_path(pg, svget2(src_0, 1), svget2(src_1, 1),
                         svget2(src_2, 1), svget2(src_3, 1), svget2(src_4, 1),
                         &dst_rows[index + static_cast<ptrdiff_t>(
                                               SourceVecTraits::num_lanes())]);
  }

  void vertical_vector_path_1x(
      svbool_t pg, Rows<const SourceType> src_rows, Rows<BufferType> dst_rows,
      BorderOffsets border_offsets,
      ptrdiff_t index) const KLEIDICV_STREAMING_COMPATIBLE {
    SourceVectorType src_0 =
        svld1(pg, &src_rows.at(border_offsets.c0())[index]);
    SourceVectorType src_1 =
        svld1(pg, &src_rows.at(border_offsets.c1())[index]);
    SourceVectorType src_2 =
        svld1(pg, &src_rows.at(border_offsets.c2())[index]);
    SourceVectorType src_3 =
        svld1(pg, &src_rows.at(border_offsets.c3())[index]);
    SourceVectorType src_4 =
        svld1(pg, &src_rows.at(border_offsets.c4())[index]);
    vertical_vector_path(pg, src_0, src_1, src_2, src_3, src_4,
                         &dst_rows[index]);
  }

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void vertical_vector_path(svbool_t pg, svuint8_t src_0, svuint8_t src_1,
                            svuint8_t src_2, svuint8_t src_3, svuint8_t src_4,
                            BufferType *dst) const
      KLEIDICV_STREAMING_COMPATIBLE {
    svuint16_t acc_0_4_b = svaddlb_u16(src_0, src_4);
    svuint16_t acc_0_4_t = svaddlt_u16(src_0, src_4);
    svuint16_t acc_1_3_b = svaddlb_u16(src_1, src_3);
    svuint16_t acc_1_3_t = svaddlt_u16(src_1, src_3);

    svuint16_t acc_u16_b = svmlalb_n_u16(acc_0_4_b, src_2, 6);
    svuint16_t acc_u16_t = svmlalt_n_u16(acc_0_4_t, src_2, 6);
    acc_u16_b = svmla_n_u16_x(pg, acc_u16_b, acc_1_3_b, 4);
    acc_u16_t = svmla_n_u16_x(pg, acc_u16_t, acc_1_3_t, 4);

    svuint16x2_t interleaved = svcreate2(acc_u16_b, acc_u16_t);
    svst2(pg, &dst[0], interleaved);
  }

  void horizontal_vector_path_2x(
      svbool_t pg_src_0, svbool_t pg_src_1, Rows<const BufferType> src_rows,
      svbool_t pg_dst, Rows<DestinationType> dst_rows,
      BorderOffsets border_offsets,
      ptrdiff_t index) const KLEIDICV_STREAMING_COMPATIBLE {
    const auto *src_0 = &src_rows.at(0, border_offsets.c0())[index];
    const auto *src_1 = &src_rows.at(0, border_offsets.c1())[index];
    const auto *src_2 = &src_rows.at(0, border_offsets.c2())[index];
    const auto *src_3 = &src_rows.at(0, border_offsets.c3())[index];
    const auto *src_4 = &src_rows.at(0, border_offsets.c4())[index];

    BufferVectorType src_0_0 = svld1(pg_src_0, &src_0[0]);
    BufferVectorType src_1_0 = svld1_vnum(pg_src_1, &src_0[0], 1);
    BufferVectorType src_0_1 = svld1(pg_src_0, &src_1[0]);
    BufferVectorType src_1_1 = svld1_vnum(pg_src_1, &src_1[0], 1);
    BufferVectorType src_0_2 = svld1(pg_src_0, &src_2[0]);
    BufferVectorType src_1_2 = svld1_vnum(pg_src_1, &src_2[0], 1);
    BufferVectorType src_0_3 = svld1(pg_src_0, &src_3[0]);
    BufferVectorType src_1_3 = svld1_vnum(pg_src_1, &src_3[0], 1);
    BufferVectorType src_0_4 = svld1(pg_src_0, &src_4[0]);
    BufferVectorType src_1_4 = svld1_vnum(pg_src_1, &src_4[0], 1);

    svuint16_t res_0 = horizontal_vector_path(pg_src_0, src_0_0, src_0_1,
                                              src_0_2, src_0_3, src_0_4);
    svuint16_t res_1 = horizontal_vector_path(pg_src_1, src_1_0, src_1_1,
                                              src_1_2, src_1_3, src_1_4);

    svuint16_t res_even_only = svuzp1(res_0, res_1);
    svst1b(pg_dst, &dst_rows[index / 2], res_even_only);
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/256 * [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  svuint16_t horizontal_vector_path(
      svbool_t pg, svuint16_t src_0, svuint16_t src_1, svuint16_t src_2,
      svuint16_t src_3, svuint16_t src_4) const KLEIDICV_STREAMING_COMPATIBLE {
    svuint16_t acc_0_4 = svadd_x(pg, src_0, src_4);
    svuint16_t acc_1_3 = svadd_x(pg, src_1, src_3);
    svuint16_t acc = svmla_n_u16_x(pg, acc_0_4, src_2, 6);
    acc = svmla_n_u16_x(pg, acc, acc_1_3, 4);
    acc = svrshr_x(pg, acc, 8);
    return acc;
  }

  // Applies horizontal filtering for the borders using SIMD operations.
  //
  // DST = 1/256 * [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void horizontal_border_path(
      svbool_t pg, Rows<const BufferType> src_rows,
      Rows<DestinationType> dst_rows, BorderOffsets border_offsets,
      ptrdiff_t index) const KLEIDICV_STREAMING_COMPATIBLE {
    BufferVectorType src_0 =
        svld1(pg, &src_rows.at(0, border_offsets.c0())[index]);
    BufferVectorType src_1 =
        svld1(pg, &src_rows.at(0, border_offsets.c1())[index]);
    BufferVectorType src_2 =
        svld1(pg, &src_rows.at(0, border_offsets.c2())[index]);
    BufferVectorType src_3 =
        svld1(pg, &src_rows.at(0, border_offsets.c3())[index]);
    BufferVectorType src_4 =
        svld1(pg, &src_rows.at(0, border_offsets.c4())[index]);

    svuint16_t acc_0_4 = svadd_x(pg, src_0, src_4);
    svuint16_t acc_1_3 = svadd_x(pg, src_1, src_3);
    svuint16_t acc = svmla_n_u16_x(pg, acc_0_4, src_2, 6);
    acc = svmla_n_u16_x(pg, acc, acc_1_3, 4);
    acc = svrshr_x(pg, acc, 8);

    svst1b(pg, &dst_rows[index / 2], acc);
  }
};  // end of class BlurAndDownsample

// Does not include checks for whether the operation is implemented.
// This must be done earlier, by blur_and_downsample_is_implemented.
static kleidicv_error_t blur_and_downsample_checks(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t channels,
    BlurAndDownsampleFilterWorkspace *workspace) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTERS(workspace);
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, (src_height + 1) / 2);
  CHECK_IMAGE_SIZE(src_width, src_height);

  Rectangle rect{src_width, src_height};
  const Rectangle &context_rect = workspace->image_size();
  if (context_rect.width() < src_width || context_rect.height() < src_height) {
    return KLEIDICV_ERROR_CONTEXT_MISMATCH;
  }

  // Currently supports only one channel, so it cannot be tested.
  // GCOVR_EXCL_START
  if (workspace->channels() < channels) {
    return KLEIDICV_ERROR_CONTEXT_MISMATCH;
  }
  // GCOVR_EXCL_STOP

  return KLEIDICV_OK;
}

static kleidicv_error_t blur_and_downsample_stripe_u8_sc(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t y_begin, size_t y_end,
    size_t channels, FixedBorderType fixed_border_type,
    kleidicv_filter_context_t *context) KLEIDICV_STREAMING_COMPATIBLE {
  // Does not include checks for whether the operation is implemented.
  // This must be done earlier, by blur_and_downsample_is_implemented.
  auto *workspace =
      reinterpret_cast<BlurAndDownsampleFilterWorkspace *>(context);

  if (auto check_result =
          blur_and_downsample_checks(src, src_stride, src_width, src_height,
                                     dst, dst_stride, channels, workspace)) {
    return check_result;
  }

  Rectangle rect{src_width, src_height};

  Rows<const uint8_t> src_rows{src, src_stride, channels};
  Rows<uint8_t> dst_rows{dst, dst_stride, channels};
  workspace->process(rect, y_begin, y_end, src_rows, dst_rows, channels,
                     fixed_border_type, BlurAndDownsample{});

  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE
