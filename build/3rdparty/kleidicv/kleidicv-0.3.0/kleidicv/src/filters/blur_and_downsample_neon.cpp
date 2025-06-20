// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/ctypes.h"
#include "kleidicv/filters/blur_and_downsample.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/utils.h"
#include "kleidicv/workspace/blur_and_downsample_ws.h"
#include "kleidicv/workspace/border_5x5.h"

namespace kleidicv::neon {

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
  using SourceVecTraits = typename neon::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using BufferVecTraits = typename neon::VecTraits<BufferType>;
  using BufferVectorType = typename BufferVecTraits::VectorType;
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo5x5<SourceType>;
  using BorderType = FixedBorderType;
  using BorderOffsets = typename BorderInfoType::Offsets;

  BlurAndDownsample()
      : const_6_u8_half_{vdup_n_u8(6)},
        const_6_u16_{vdupq_n_u16(6)},
        const_4_u16_{vdupq_n_u16(4)} {}

  static constexpr size_t margin = 2UL;

  void process_vertical(size_t width, Rows<const SourceType> src_rows,
                        Rows<BufferType> dst_rows,
                        BorderOffsets border_offsets) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         SourceVecTraits::num_lanes()};

    loop.unroll_twice([&](ptrdiff_t index) {
      const auto *src_0 = &src_rows.at(border_offsets.c0())[index];
      const auto *src_1 = &src_rows.at(border_offsets.c1())[index];
      const auto *src_2 = &src_rows.at(border_offsets.c2())[index];
      const auto *src_3 = &src_rows.at(border_offsets.c3())[index];
      const auto *src_4 = &src_rows.at(border_offsets.c4())[index];

      SourceVectorType src_a[5], src_b[5];
      src_a[0] = vld1q(&src_0[0]);
      src_b[0] = vld1q(&src_0[SourceVecTraits::num_lanes()]);
      src_a[1] = vld1q(&src_1[0]);
      src_b[1] = vld1q(&src_1[SourceVecTraits::num_lanes()]);
      src_a[2] = vld1q(&src_2[0]);
      src_b[2] = vld1q(&src_2[SourceVecTraits::num_lanes()]);
      src_a[3] = vld1q(&src_3[0]);
      src_b[3] = vld1q(&src_3[SourceVecTraits::num_lanes()]);
      src_a[4] = vld1q(&src_4[0]);
      src_b[4] = vld1q(&src_4[SourceVecTraits::num_lanes()]);
      vertical_vector_path(src_a, &dst_rows[index]);
      vertical_vector_path(
          src_b, &dst_rows[index + static_cast<ptrdiff_t>(
                                       SourceVecTraits::num_lanes())]);
    });

    loop.unroll_once([&](ptrdiff_t index) {
      SourceVectorType src[5];
      src[0] = vld1q(&src_rows.at(border_offsets.c0())[index]);
      src[1] = vld1q(&src_rows.at(border_offsets.c1())[index]);
      src[2] = vld1q(&src_rows.at(border_offsets.c2())[index]);
      src[3] = vld1q(&src_rows.at(border_offsets.c3())[index]);
      src[4] = vld1q(&src_rows.at(border_offsets.c4())[index]);
      vertical_vector_path(src, &dst_rows[index]);
    });

    loop.tail([&](ptrdiff_t index) {
      SourceType src[5];
      src[0] = src_rows.at(border_offsets.c0())[index];
      src[1] = src_rows.at(border_offsets.c1())[index];
      src[2] = src_rows.at(border_offsets.c2())[index];
      src[3] = src_rows.at(border_offsets.c3())[index];
      src[4] = src_rows.at(border_offsets.c4())[index];
      vertical_scalar_path(src, &dst_rows[index]);
    });
  }

  void process_horizontal(size_t width, Rows<const BufferType> src_rows,
                          Rows<DestinationType> dst_rows,
                          BorderOffsets border_offsets) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         BufferVecTraits::num_lanes()};

    loop.unroll_twice([&](ptrdiff_t index) {
      const auto *src_0 = &src_rows.at(0, border_offsets.c0())[index];
      const auto *src_1 = &src_rows.at(0, border_offsets.c1())[index];
      const auto *src_2 = &src_rows.at(0, border_offsets.c2())[index];
      const auto *src_3 = &src_rows.at(0, border_offsets.c3())[index];
      const auto *src_4 = &src_rows.at(0, border_offsets.c4())[index];

      BufferVectorType src_a[5], src_b[5];
      src_a[0] = vld1q(&src_0[0]);
      src_b[0] = vld1q(&src_0[BufferVecTraits::num_lanes()]);
      src_a[1] = vld1q(&src_1[0]);
      src_b[1] = vld1q(&src_1[BufferVecTraits::num_lanes()]);
      src_a[2] = vld1q(&src_2[0]);
      src_b[2] = vld1q(&src_2[BufferVecTraits::num_lanes()]);
      src_a[3] = vld1q(&src_3[0]);
      src_b[3] = vld1q(&src_3[BufferVecTraits::num_lanes()]);
      src_a[4] = vld1q(&src_4[0]);
      src_b[4] = vld1q(&src_4[BufferVecTraits::num_lanes()]);

      uint8x8_t res_a = horizontal_vector_path(src_a);
      uint8x8_t res_b = horizontal_vector_path(src_b);

      // Only store even indices
      vst1(&dst_rows[index / 2], vuzp1_u8(res_a, res_b));
    });

    loop.remaining([&](ptrdiff_t index, size_t max_index) {
      index = align_up(index, 2);
      while (index < static_cast<ptrdiff_t>(max_index)) {
        process_horizontal_scalar(src_rows, dst_rows, border_offsets, index);
        index += 2;
      }
    });
  }

  void process_horizontal_borders(Rows<const BufferType> src_rows,
                                  Rows<DestinationType> dst_rows,
                                  BorderOffsets border_offsets) const {
    for (ptrdiff_t index = 0;
         index < static_cast<ptrdiff_t>(src_rows.channels()); ++index) {
      disable_loop_vectorization();
      process_horizontal_scalar(src_rows, dst_rows, border_offsets, index);
    }
  }

 private:
  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void vertical_vector_path(uint8x16_t src[5], BufferType *dst) const {
    uint16x8_t acc_0_4_l = vaddl_u8(vget_low_u8(src[0]), vget_low_u8(src[4]));
    uint16x8_t acc_0_4_h = vaddl_u8(vget_high_u8(src[0]), vget_high_u8(src[4]));
    uint16x8_t acc_1_3_l = vaddl_u8(vget_low_u8(src[1]), vget_low_u8(src[3]));
    uint16x8_t acc_1_3_h = vaddl_u8(vget_high_u8(src[1]), vget_high_u8(src[3]));
    uint16x8_t acc_l =
        vmlal_u8(acc_0_4_l, vget_low_u8(src[2]), const_6_u8_half_);
    uint16x8_t acc_h =
        vmlal_u8(acc_0_4_h, vget_high_u8(src[2]), const_6_u8_half_);
    acc_l = vmlaq_u16(acc_l, acc_1_3_l, const_4_u16_);
    acc_h = vmlaq_u16(acc_h, acc_1_3_h, const_4_u16_);
    vst1q(&dst[0], acc_l);
    vst1q(&dst[8], acc_h);
  }

  // Applies vertical filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void vertical_scalar_path(const SourceType src[5], BufferType *dst) const {
    dst[0] = src[0] + src[4] + 4 * (src[1] + src[3]) + 6 * src[2];
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/256 * [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  uint8x8_t horizontal_vector_path(uint16x8_t src[5]) const {
    uint16x8_t acc_0_4 = vaddq_u16(src[0], src[4]);
    uint16x8_t acc_1_3 = vaddq_u16(src[1], src[3]);
    uint16x8_t acc_u16 = vmlaq_u16(acc_0_4, src[2], const_6_u16_);
    acc_u16 = vmlaq_u16(acc_u16, acc_1_3, const_4_u16_);
    return vrshrn_n_u16(acc_u16, 8);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/256 * [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void process_horizontal_scalar(Rows<const BufferType> src_rows,
                                 Rows<DestinationType> dst_rows,
                                 BorderOffsets border_offsets,
                                 ptrdiff_t index) const {
    BufferType src[5];
    src[0] = src_rows.at(0, border_offsets.c0())[index];
    src[1] = src_rows.at(0, border_offsets.c1())[index];
    src[2] = src_rows.at(0, border_offsets.c2())[index];
    src[3] = src_rows.at(0, border_offsets.c3())[index];
    src[4] = src_rows.at(0, border_offsets.c4())[index];

    auto acc = src[0] + src[4] + 4 * (src[1] + src[3]) + 6 * src[2];
    dst_rows[index / 2] = rounding_shift_right(acc, 8);
  }

  uint8x8_t const_6_u8_half_;
  uint16x8_t const_6_u16_;
  uint16x8_t const_4_u16_;
};  // end of class BlurAndDownsample

// Does not include checks for whether the operation is implemented.
// This must be done earlier, by blur_and_downsample_is_implemented.
static kleidicv_error_t blur_and_downsample_checks(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t channels,
    BlurAndDownsampleFilterWorkspace *workspace) {
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

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t kleidicv_blur_and_downsample_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t y_begin, size_t y_end,
    size_t channels, FixedBorderType fixed_border_type,
    kleidicv_filter_context_t *context) {
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

}  // namespace kleidicv::neon
