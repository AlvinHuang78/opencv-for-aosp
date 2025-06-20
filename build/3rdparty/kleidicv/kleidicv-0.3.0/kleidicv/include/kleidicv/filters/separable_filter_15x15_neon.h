// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SEPARABLE_FILTER_15X15_NEON_H
#define KLEIDICV_SEPARABLE_FILTER_15X15_NEON_H

#include "kleidicv/neon.h"
#include "kleidicv/workspace/border_15x15.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Template for drivers of separable NxM filters.
template <typename FilterType, const size_t S>
class SeparableFilter;

// Driver for a separable 15x15 filter.
template <typename FilterType>
class SeparableFilter<FilterType, 15UL> {
 public:
  using SourceType = typename FilterType::SourceType;
  using BufferType = typename FilterType::BufferType;
  using DestinationType = typename FilterType::DestinationType;
  using SourceVecTraits = typename neon::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using BufferVecTraits = typename neon::VecTraits<BufferType>;
  using BufferVectorType = typename BufferVecTraits::VectorType;
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo15x15<SourceType>;
  using BorderType = FixedBorderType;
  using BorderOffsets = typename BorderInfoType::Offsets;

  explicit SeparableFilter(FilterType filter) : filter_{filter} {}

  static constexpr size_t margin = 7UL;

  void process_vertical(size_t width, Rows<const SourceType> src_rows,
                        Rows<BufferType> dst_rows,
                        BorderOffsets border_offsets) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         SourceVecTraits::num_lanes()};

    loop.unroll_once([&](size_t index) {
      SourceVectorType src[15];
      src[0] = vld1q(&src_rows.at(border_offsets.c0())[index]);
      src[1] = vld1q(&src_rows.at(border_offsets.c1())[index]);
      src[2] = vld1q(&src_rows.at(border_offsets.c2())[index]);
      src[3] = vld1q(&src_rows.at(border_offsets.c3())[index]);
      src[4] = vld1q(&src_rows.at(border_offsets.c4())[index]);
      src[5] = vld1q(&src_rows.at(border_offsets.c5())[index]);
      src[6] = vld1q(&src_rows.at(border_offsets.c6())[index]);
      src[7] = vld1q(&src_rows.at(border_offsets.c7())[index]);
      src[8] = vld1q(&src_rows.at(border_offsets.c8())[index]);
      src[9] = vld1q(&src_rows.at(border_offsets.c9())[index]);
      src[10] = vld1q(&src_rows.at(border_offsets.c10())[index]);
      src[11] = vld1q(&src_rows.at(border_offsets.c11())[index]);
      src[12] = vld1q(&src_rows.at(border_offsets.c12())[index]);
      src[13] = vld1q(&src_rows.at(border_offsets.c13())[index]);
      src[14] = vld1q(&src_rows.at(border_offsets.c14())[index]);
      filter_.vertical_vector_path(src, &dst_rows[index]);
    });

    loop.tail([&](size_t index) {
      SourceType src[15];
      src[0] = src_rows.at(border_offsets.c0())[index];
      src[1] = src_rows.at(border_offsets.c1())[index];
      src[2] = src_rows.at(border_offsets.c2())[index];
      src[3] = src_rows.at(border_offsets.c3())[index];
      src[4] = src_rows.at(border_offsets.c4())[index];
      src[5] = src_rows.at(border_offsets.c5())[index];
      src[6] = src_rows.at(border_offsets.c6())[index];
      src[7] = src_rows.at(border_offsets.c7())[index];
      src[8] = src_rows.at(border_offsets.c8())[index];
      src[9] = src_rows.at(border_offsets.c9())[index];
      src[10] = src_rows.at(border_offsets.c10())[index];
      src[11] = src_rows.at(border_offsets.c11())[index];
      src[12] = src_rows.at(border_offsets.c12())[index];
      src[13] = src_rows.at(border_offsets.c13())[index];
      src[14] = src_rows.at(border_offsets.c14())[index];
      filter_.vertical_scalar_path(src, &dst_rows[index]);
    });
  }

  void process_horizontal(size_t width, Rows<const BufferType> src_rows,
                          Rows<DestinationType> dst_rows,
                          BorderOffsets border_offsets) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         BufferVecTraits::num_lanes()};

    loop.unroll_twice([&](size_t index) {
      auto src_0 = &src_rows.at(0, border_offsets.c0())[index];
      auto src_1 = &src_rows.at(0, border_offsets.c1())[index];
      auto src_2 = &src_rows.at(0, border_offsets.c2())[index];
      auto src_3 = &src_rows.at(0, border_offsets.c3())[index];
      auto src_4 = &src_rows.at(0, border_offsets.c4())[index];
      auto src_5 = &src_rows.at(0, border_offsets.c5())[index];
      auto src_6 = &src_rows.at(0, border_offsets.c6())[index];
      auto src_7 = &src_rows.at(0, border_offsets.c7())[index];
      auto src_8 = &src_rows.at(0, border_offsets.c8())[index];
      auto src_9 = &src_rows.at(0, border_offsets.c9())[index];
      auto src_10 = &src_rows.at(0, border_offsets.c10())[index];
      auto src_11 = &src_rows.at(0, border_offsets.c11())[index];
      auto src_12 = &src_rows.at(0, border_offsets.c12())[index];
      auto src_13 = &src_rows.at(0, border_offsets.c13())[index];
      auto src_14 = &src_rows.at(0, border_offsets.c14())[index];

      BufferVectorType src_a[15], src_b[15];
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
      src_a[5] = vld1q(&src_5[0]);
      src_b[5] = vld1q(&src_5[BufferVecTraits::num_lanes()]);
      src_a[6] = vld1q(&src_6[0]);
      src_b[6] = vld1q(&src_6[BufferVecTraits::num_lanes()]);
      src_a[7] = vld1q(&src_7[0]);
      src_b[7] = vld1q(&src_7[BufferVecTraits::num_lanes()]);
      src_a[8] = vld1q(&src_8[0]);
      src_b[8] = vld1q(&src_8[BufferVecTraits::num_lanes()]);
      src_a[9] = vld1q(&src_9[0]);
      src_b[9] = vld1q(&src_9[BufferVecTraits::num_lanes()]);
      src_a[10] = vld1q(&src_10[0]);
      src_b[10] = vld1q(&src_10[BufferVecTraits::num_lanes()]);
      src_a[11] = vld1q(&src_11[0]);
      src_b[11] = vld1q(&src_11[BufferVecTraits::num_lanes()]);
      src_a[12] = vld1q(&src_12[0]);
      src_b[12] = vld1q(&src_12[BufferVecTraits::num_lanes()]);
      src_a[13] = vld1q(&src_13[0]);
      src_b[13] = vld1q(&src_13[BufferVecTraits::num_lanes()]);
      src_a[14] = vld1q(&src_14[0]);
      src_b[14] = vld1q(&src_14[BufferVecTraits::num_lanes()]);

      filter_.horizontal_vector_path(src_a, &dst_rows[index]);
      filter_.horizontal_vector_path(
          src_b, &dst_rows[index + BufferVecTraits::num_lanes()]);
    });

    loop.unroll_once([&](size_t index) {
      BufferVectorType src[15];
      src[0] = vld1q(&src_rows.at(0, border_offsets.c0())[index]);
      src[1] = vld1q(&src_rows.at(0, border_offsets.c1())[index]);
      src[2] = vld1q(&src_rows.at(0, border_offsets.c2())[index]);
      src[3] = vld1q(&src_rows.at(0, border_offsets.c3())[index]);
      src[4] = vld1q(&src_rows.at(0, border_offsets.c4())[index]);
      src[5] = vld1q(&src_rows.at(0, border_offsets.c5())[index]);
      src[6] = vld1q(&src_rows.at(0, border_offsets.c6())[index]);
      src[7] = vld1q(&src_rows.at(0, border_offsets.c7())[index]);
      src[8] = vld1q(&src_rows.at(0, border_offsets.c8())[index]);
      src[9] = vld1q(&src_rows.at(0, border_offsets.c9())[index]);
      src[10] = vld1q(&src_rows.at(0, border_offsets.c10())[index]);
      src[11] = vld1q(&src_rows.at(0, border_offsets.c11())[index]);
      src[12] = vld1q(&src_rows.at(0, border_offsets.c12())[index]);
      src[13] = vld1q(&src_rows.at(0, border_offsets.c13())[index]);
      src[14] = vld1q(&src_rows.at(0, border_offsets.c14())[index]);
      filter_.horizontal_vector_path(src, &dst_rows[index]);
    });

    loop.tail([&](size_t index) {
      process_horizontal_scalar(src_rows, dst_rows, border_offsets, index);
    });
  }

  void process_horizontal_borders(Rows<const BufferType> src_rows,
                                  Rows<DestinationType> dst_rows,
                                  BorderOffsets border_offsets) const {
    for (size_t index = 0; index < src_rows.channels(); ++index) {
      disable_loop_vectorization();
      process_horizontal_scalar(src_rows, dst_rows, border_offsets, index);
    }
  }

 private:
  void process_horizontal_scalar(Rows<const BufferType> src_rows,
                                 Rows<DestinationType> dst_rows,
                                 BorderOffsets border_offsets,
                                 size_t index) const {
    BufferType src[15];
    src[0] = src_rows.at(0, border_offsets.c0())[index];
    src[1] = src_rows.at(0, border_offsets.c1())[index];
    src[2] = src_rows.at(0, border_offsets.c2())[index];
    src[3] = src_rows.at(0, border_offsets.c3())[index];
    src[4] = src_rows.at(0, border_offsets.c4())[index];
    src[5] = src_rows.at(0, border_offsets.c5())[index];
    src[6] = src_rows.at(0, border_offsets.c6())[index];
    src[7] = src_rows.at(0, border_offsets.c7())[index];
    src[8] = src_rows.at(0, border_offsets.c8())[index];
    src[9] = src_rows.at(0, border_offsets.c9())[index];
    src[10] = src_rows.at(0, border_offsets.c10())[index];
    src[11] = src_rows.at(0, border_offsets.c11())[index];
    src[12] = src_rows.at(0, border_offsets.c12())[index];
    src[13] = src_rows.at(0, border_offsets.c13())[index];
    src[14] = src_rows.at(0, border_offsets.c14())[index];
    filter_.horizontal_scalar_path(src, &dst_rows[index]);
  }

  FilterType filter_;
};  // end of class SeparableFilter<FilterType, 15UL>

// Shorthand for 15x15 separable filters driver type.
template <class FilterType>
using SeparableFilter15x15 = SeparableFilter<FilterType, 15UL>;

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SEPARABLE_FILTER_15X15_NEON_H
