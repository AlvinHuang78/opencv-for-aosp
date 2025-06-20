// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SEPARABLE_FILTER_5X5_SC_H
#define KLEIDICV_SEPARABLE_FILTER_5X5_SC_H

#include "kleidicv/sve2.h"
#include "kleidicv/workspace/border_5x5.h"

// It is used by SVE2 and SME2, the actual namespace will reflect it.
namespace KLEIDICV_TARGET_NAMESPACE {

// Template for drivers of separable NxM filters.
template <typename FilterType, const size_t S>
class SeparableFilter;

// Driver for a separable 5x5 filter.
template <typename FilterType>
class SeparableFilter<FilterType, 5UL> {
 public:
  using SourceType = typename FilterType::SourceType;
  using BufferType = typename FilterType::BufferType;
  using DestinationType = typename FilterType::DestinationType;
  using SourceVecTraits =
      typename ::KLEIDICV_TARGET_NAMESPACE::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using BufferVecTraits =
      typename ::KLEIDICV_TARGET_NAMESPACE::VecTraits<BufferType>;
  using BufferVectorType = typename BufferVecTraits::VectorType;
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo5x5<SourceType>;
  using BorderType = FixedBorderType;
  using BorderOffsets = typename BorderInfoType::Offsets;

  explicit SeparableFilter(FilterType filter) KLEIDICV_STREAMING_COMPATIBLE
      : filter_{filter} {}

  static constexpr size_t margin = 2UL;

  void process_vertical(
      size_t width, Rows<const SourceType> src_rows, Rows<BufferType> dst_rows,
      BorderOffsets border_offsets) const KLEIDICV_STREAMING_COMPATIBLE {
    LoopUnroll2 loop{width * src_rows.channels(), SourceVecTraits::num_lanes()};

    loop.unroll_once([&](size_t index) KLEIDICV_STREAMING_COMPATIBLE {
      svbool_t pg_all = SourceVecTraits::svptrue();
      vertical_vector_path(pg_all, src_rows, dst_rows, border_offsets, index);
    });

    loop.remaining(
        [&](size_t index, size_t length) KLEIDICV_STREAMING_COMPATIBLE {
          svbool_t pg = SourceVecTraits::svwhilelt(index, length);
          vertical_vector_path(pg, src_rows, dst_rows, border_offsets, index);
        });
  }

  void process_horizontal(size_t width, Rows<const BufferType> src_rows,
                          Rows<DestinationType> dst_rows,
                          BorderOffsets border_offsets) const
      KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t pg_all = BufferVecTraits::svptrue();
    LoopUnroll2 loop{width * src_rows.channels(), BufferVecTraits::num_lanes()};

    loop.unroll_twice([&](size_t index) KLEIDICV_STREAMING_COMPATIBLE {
      horizontal_vector_path_2x(pg_all, src_rows, dst_rows, border_offsets,
                                index);
    });

    loop.unroll_once([&](size_t index) KLEIDICV_STREAMING_COMPATIBLE {
      horizontal_vector_path(pg_all, src_rows, dst_rows, border_offsets, index);
    });

    loop.remaining(
        [&](size_t index, size_t length) KLEIDICV_STREAMING_COMPATIBLE {
          svbool_t pg = BufferVecTraits::svwhilelt(index, length);
          horizontal_vector_path(pg, src_rows, dst_rows, border_offsets, index);
        });
  }

  // Processing of horizontal borders is always scalar because border offsets
  // change for each and every element in the border.
  void process_horizontal_borders(
      Rows<const BufferType> src_rows, Rows<DestinationType> dst_rows,
      BorderOffsets border_offsets) const KLEIDICV_STREAMING_COMPATIBLE {
    for (size_t index = 0; index < src_rows.channels(); ++index) {
      disable_loop_vectorization();
      process_horizontal_border(src_rows, dst_rows, border_offsets, index);
    }
  }

 private:
  void vertical_vector_path(svbool_t pg, Rows<const SourceType> src_rows,
                            Rows<BufferType> dst_rows,
                            BorderOffsets border_offsets,
                            size_t index) const KLEIDICV_STREAMING_COMPATIBLE {
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
    filter_.vertical_vector_path(pg, src_0, src_1, src_2, src_3, src_4,
                                 &dst_rows[index]);
  }

  void horizontal_vector_path_2x(
      svbool_t pg, Rows<const BufferType> src_rows,
      Rows<DestinationType> dst_rows, BorderOffsets border_offsets,
      size_t index) const KLEIDICV_STREAMING_COMPATIBLE {
    auto src_0 = &src_rows.at(0, border_offsets.c0())[index];
    auto src_1 = &src_rows.at(0, border_offsets.c1())[index];
    auto src_2 = &src_rows.at(0, border_offsets.c2())[index];
    auto src_3 = &src_rows.at(0, border_offsets.c3())[index];
    auto src_4 = &src_rows.at(0, border_offsets.c4())[index];

    BufferVectorType src_0_0 = svld1(pg, &src_0[0]);
    BufferVectorType src_1_0 = svld1_vnum(pg, &src_0[0], 1);
    BufferVectorType src_0_1 = svld1(pg, &src_1[0]);
    BufferVectorType src_1_1 = svld1_vnum(pg, &src_1[0], 1);
    BufferVectorType src_0_2 = svld1(pg, &src_2[0]);
    BufferVectorType src_1_2 = svld1_vnum(pg, &src_2[0], 1);
    BufferVectorType src_0_3 = svld1(pg, &src_3[0]);
    BufferVectorType src_1_3 = svld1_vnum(pg, &src_3[0], 1);
    BufferVectorType src_0_4 = svld1(pg, &src_4[0]);
    BufferVectorType src_1_4 = svld1_vnum(pg, &src_4[0], 1);

    filter_.horizontal_vector_path(pg, src_0_0, src_0_1, src_0_2, src_0_3,
                                   src_0_4, &dst_rows[index]);
    filter_.horizontal_vector_path(
        pg, src_1_0, src_1_1, src_1_2, src_1_3, src_1_4,
        &dst_rows[index + BufferVecTraits::num_lanes()]);
  }

  void horizontal_vector_path(svbool_t pg, Rows<const BufferType> src_rows,
                              Rows<DestinationType> dst_rows,
                              BorderOffsets border_offsets, size_t index) const
      KLEIDICV_STREAMING_COMPATIBLE {
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
    filter_.horizontal_vector_path(pg, src_0, src_1, src_2, src_3, src_4,
                                   &dst_rows[index]);
  }

  void process_horizontal_border(
      Rows<const BufferType> src_rows, Rows<DestinationType> dst_rows,
      BorderOffsets border_offsets,
      size_t index) const KLEIDICV_STREAMING_COMPATIBLE {
    BufferType src[5];
    src[0] = src_rows.at(0, border_offsets.c0())[index];
    src[1] = src_rows.at(0, border_offsets.c1())[index];
    src[2] = src_rows.at(0, border_offsets.c2())[index];
    src[3] = src_rows.at(0, border_offsets.c3())[index];
    src[4] = src_rows.at(0, border_offsets.c4())[index];
    filter_.horizontal_scalar_path(src, &dst_rows[index]);
  }

  FilterType filter_;
};  // end of class SeparableFilter<FilterType, 5UL>

// Shorthand for 5x5 separable filters driver type.
template <class FilterType>
using SeparableFilter5x5 = SeparableFilter<FilterType, 5UL>;

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SEPARABLE_FILTER_5X5_SC_H
