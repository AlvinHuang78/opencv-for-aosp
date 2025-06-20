// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SEPARABLE_FILTER_15X15_SC_H
#define KLEIDICV_SEPARABLE_FILTER_15X15_SC_H

#include "kleidicv/sve2.h"
#include "kleidicv/workspace/border_15x15.h"

// It is used by SVE2 and SME2, the actual namespace will reflect it.
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
  using SourceVecTraits =
      typename ::KLEIDICV_TARGET_NAMESPACE::VecTraits<SourceType>;
  using SourceVectorType = typename SourceVecTraits::VectorType;
  using BufferVecTraits =
      typename ::KLEIDICV_TARGET_NAMESPACE::VecTraits<BufferType>;
  using BufferVectorType = typename BufferVecTraits::VectorType;
  using BorderInfoType =
      typename ::KLEIDICV_TARGET_NAMESPACE::FixedBorderInfo15x15<SourceType>;
  using BorderType = FixedBorderType;
  using BorderOffsets = typename BorderInfoType::Offsets;

  explicit SeparableFilter(FilterType filter) KLEIDICV_STREAMING_COMPATIBLE
      : filter_{filter} {}

  static constexpr size_t margin = 7UL;

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
    SourceVectorType src_5 =
        svld1(pg, &src_rows.at(border_offsets.c5())[index]);
    SourceVectorType src_6 =
        svld1(pg, &src_rows.at(border_offsets.c6())[index]);
    SourceVectorType src_7 =
        svld1(pg, &src_rows.at(border_offsets.c7())[index]);
    SourceVectorType src_8 =
        svld1(pg, &src_rows.at(border_offsets.c8())[index]);
    SourceVectorType src_9 =
        svld1(pg, &src_rows.at(border_offsets.c9())[index]);
    SourceVectorType src_10 =
        svld1(pg, &src_rows.at(border_offsets.c10())[index]);
    SourceVectorType src_11 =
        svld1(pg, &src_rows.at(border_offsets.c11())[index]);
    SourceVectorType src_12 =
        svld1(pg, &src_rows.at(border_offsets.c12())[index]);
    SourceVectorType src_13 =
        svld1(pg, &src_rows.at(border_offsets.c13())[index]);
    SourceVectorType src_14 =
        svld1(pg, &src_rows.at(border_offsets.c14())[index]);
    filter_.vertical_vector_path(pg, src_0, src_1, src_2, src_3, src_4, src_5,
                                 src_6, src_7, src_8, src_9, src_10, src_11,
                                 src_12, src_13, src_14, &dst_rows[index]);
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
    BufferVectorType src_0_5 = svld1(pg, &src_5[0]);
    BufferVectorType src_1_5 = svld1_vnum(pg, &src_5[0], 1);
    BufferVectorType src_0_6 = svld1(pg, &src_6[0]);
    BufferVectorType src_1_6 = svld1_vnum(pg, &src_6[0], 1);
    BufferVectorType src_0_7 = svld1(pg, &src_7[0]);
    BufferVectorType src_1_7 = svld1_vnum(pg, &src_7[0], 1);
    BufferVectorType src_0_8 = svld1(pg, &src_8[0]);
    BufferVectorType src_1_8 = svld1_vnum(pg, &src_8[0], 1);
    BufferVectorType src_0_9 = svld1(pg, &src_9[0]);
    BufferVectorType src_1_9 = svld1_vnum(pg, &src_9[0], 1);
    BufferVectorType src_0_10 = svld1(pg, &src_10[0]);
    BufferVectorType src_1_10 = svld1_vnum(pg, &src_10[0], 1);
    BufferVectorType src_0_11 = svld1(pg, &src_11[0]);
    BufferVectorType src_1_11 = svld1_vnum(pg, &src_11[0], 1);
    BufferVectorType src_0_12 = svld1(pg, &src_12[0]);
    BufferVectorType src_1_12 = svld1_vnum(pg, &src_12[0], 1);
    BufferVectorType src_0_13 = svld1(pg, &src_13[0]);
    BufferVectorType src_1_13 = svld1_vnum(pg, &src_13[0], 1);
    BufferVectorType src_0_14 = svld1(pg, &src_14[0]);
    BufferVectorType src_1_14 = svld1_vnum(pg, &src_14[0], 1);

    filter_.horizontal_vector_path(pg, src_0_0, src_0_1, src_0_2, src_0_3,
                                   src_0_4, src_0_5, src_0_6, src_0_7, src_0_8,
                                   src_0_9, src_0_10, src_0_11, src_0_12,
                                   src_0_13, src_0_14, &dst_rows[index]);
    filter_.horizontal_vector_path(
        pg, src_1_0, src_1_1, src_1_2, src_1_3, src_1_4, src_1_5, src_1_6,
        src_1_7, src_1_8, src_1_9, src_1_10, src_1_11, src_1_12, src_1_13,
        src_1_14, &dst_rows[index + BufferVecTraits::num_lanes()]);
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
    BufferVectorType src_5 =
        svld1(pg, &src_rows.at(0, border_offsets.c5())[index]);
    BufferVectorType src_6 =
        svld1(pg, &src_rows.at(0, border_offsets.c6())[index]);
    BufferVectorType src_7 =
        svld1(pg, &src_rows.at(0, border_offsets.c7())[index]);
    BufferVectorType src_8 =
        svld1(pg, &src_rows.at(0, border_offsets.c8())[index]);
    BufferVectorType src_9 =
        svld1(pg, &src_rows.at(0, border_offsets.c9())[index]);
    BufferVectorType src_10 =
        svld1(pg, &src_rows.at(0, border_offsets.c10())[index]);
    BufferVectorType src_11 =
        svld1(pg, &src_rows.at(0, border_offsets.c11())[index]);
    BufferVectorType src_12 =
        svld1(pg, &src_rows.at(0, border_offsets.c12())[index]);
    BufferVectorType src_13 =
        svld1(pg, &src_rows.at(0, border_offsets.c13())[index]);
    BufferVectorType src_14 =
        svld1(pg, &src_rows.at(0, border_offsets.c14())[index]);
    filter_.horizontal_vector_path(pg, src_0, src_1, src_2, src_3, src_4, src_5,
                                   src_6, src_7, src_8, src_9, src_10, src_11,
                                   src_12, src_13, src_14, &dst_rows[index]);
  }

  void process_horizontal_border(
      Rows<const BufferType> src_rows, Rows<DestinationType> dst_rows,
      BorderOffsets border_offsets,
      size_t index) const KLEIDICV_STREAMING_COMPATIBLE {
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

#endif  // KLEIDICV_SEPARABLE_FILTER_15X15_SC_H
