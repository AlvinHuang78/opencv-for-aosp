// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SEPARABLE_FILTER_3X3_NEON_H
#define KLEIDICV_SEPARABLE_FILTER_3X3_NEON_H

#include "kleidicv/neon.h"
#include "kleidicv/workspace/border_3x3.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Template for drivers of separable NxM filters.
template <typename FilterType, const size_t S>
class SeparableFilter;

// Driver for a separable 3x3 filter.
template <typename FilterType>
class SeparableFilter<FilterType, 3UL> {
 public:
  using SourceType = typename FilterType::SourceType;
  using BufferType = typename FilterType::BufferType;
  using DestinationType = typename FilterType::DestinationType;
  using SourceVecTraits = typename neon::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using BufferVecTraits = typename neon::VecTraits<BufferType>;
  using BufferVectorType = typename BufferVecTraits::VectorType;
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo3x3<SourceType>;
  using BorderType = FixedBorderType;
  using BorderOffsets = typename BorderInfoType::Offsets;

  explicit SeparableFilter(FilterType filter) : filter_{filter} {}

  static constexpr size_t margin = 1UL;

  void process_vertical(size_t width, Rows<const SourceType> src_rows,
                        Rows<BufferType> dst_rows,
                        BorderOffsets border_offsets) const {
    LoopUnroll2<TryToAvoidTailLoop> loop{width * src_rows.channels(),
                                         SourceVecTraits::num_lanes()};

    loop.unroll_twice([&](size_t index) {
      auto src_0 = &src_rows.at(border_offsets.c0())[index];
      auto src_1 = &src_rows.at(border_offsets.c1())[index];
      auto src_2 = &src_rows.at(border_offsets.c2())[index];

      auto src_0_x2 = vld1q_x2(&src_0[0]);
      auto src_1_x2 = vld1q_x2(&src_1[0]);
      auto src_2_x2 = vld1q_x2(&src_2[0]);

      SourceVectorType src_a[3], src_b[3];
      src_a[0] = src_0_x2.val[0];
      src_b[0] = src_0_x2.val[1];
      src_a[1] = src_1_x2.val[0];
      src_b[1] = src_1_x2.val[1];
      src_a[2] = src_2_x2.val[0];
      src_b[2] = src_2_x2.val[1];

      filter_.vertical_vector_path(src_a, &dst_rows[index]);
      filter_.vertical_vector_path(
          src_b, &dst_rows[index + SourceVecTraits::num_lanes()]);
    });

    loop.unroll_once([&](size_t index) {
      SourceVectorType src[3];
      src[0] = vld1q(&src_rows.at(border_offsets.c0())[index]);
      src[1] = vld1q(&src_rows.at(border_offsets.c1())[index]);
      src[2] = vld1q(&src_rows.at(border_offsets.c2())[index]);
      filter_.vertical_vector_path(src, &dst_rows[index]);
    });

    loop.tail([&](size_t index) {
      SourceType src[3];
      src[0] = src_rows.at(border_offsets.c0())[index];
      src[1] = src_rows.at(border_offsets.c1())[index];
      src[2] = src_rows.at(border_offsets.c2())[index];
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

      auto src_0_x2 = vld1q_x2(&src_0[0]);
      auto src_1_x2 = vld1q_x2(&src_1[0]);
      auto src_2_x2 = vld1q_x2(&src_2[0]);

      BufferVectorType src_a[3], src_b[3];
      src_a[0] = src_0_x2.val[0];
      src_b[0] = src_0_x2.val[1];
      src_a[1] = src_1_x2.val[0];
      src_b[1] = src_1_x2.val[1];
      src_a[2] = src_2_x2.val[0];
      src_b[2] = src_2_x2.val[1];

      filter_.horizontal_vector_path(src_a, &dst_rows[index]);
      filter_.horizontal_vector_path(
          src_b, &dst_rows[index + BufferVecTraits::num_lanes()]);
    });

    loop.unroll_once([&](size_t index) {
      BufferVectorType src[3];
      src[0] = vld1q(&src_rows.at(0, border_offsets.c0())[index]);
      src[1] = vld1q(&src_rows.at(0, border_offsets.c1())[index]);
      src[2] = vld1q(&src_rows.at(0, border_offsets.c2())[index]);
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
    BufferType src[3];
    src[0] = src_rows.at(0, border_offsets.c0())[index];
    src[1] = src_rows.at(0, border_offsets.c1())[index];
    src[2] = src_rows.at(0, border_offsets.c2())[index];
    filter_.horizontal_scalar_path(src, &dst_rows[index]);
  }

  FilterType filter_;
};  // end of class SeparableFilter<FilterType, 3UL>

// Shorthand for 3x3 separable filters driver type.
template <class FilterType>
using SeparableFilter3x3 = SeparableFilter<FilterType, 3UL>;

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SEPARABLE_FILTER_3X3_NEON_H
