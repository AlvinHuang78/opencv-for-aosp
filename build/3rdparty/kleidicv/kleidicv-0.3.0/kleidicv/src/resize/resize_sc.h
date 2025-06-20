// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RESIZE_SC_H
#define KLEIDICV_RESIZE_SC_H

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

static inline svuint8_t resize_parallel_vectors(svbool_t pg, svuint8_t top_row,
                                                svuint8_t bottom_row)
    KLEIDICV_STREAMING_COMPATIBLE {
  svuint16_t result_before_averaging_b = svaddlb(top_row, bottom_row);
  svuint16_t result_before_averaging_t = svaddlt(top_row, bottom_row);
  svuint16_t result_before_averaging =
      svadd_x(pg, result_before_averaging_b, result_before_averaging_t);
  return svrshrnb(result_before_averaging, 2);
}

static inline void parallel_rows_vectors_path_2x(
    svbool_t pg, Rows<const uint8_t> src_rows,
    Rows<uint8_t> dst_rows) KLEIDICV_STREAMING_COMPATIBLE {
  svuint8_t top_row_0 = svld1(pg, &src_rows.at(0)[0]);
  svuint8_t bottom_row_0 = svld1(pg, &src_rows.at(1)[0]);
  svuint8_t top_row_1 = svld1_vnum(pg, &src_rows.at(0)[0], 1);
  svuint8_t bottom_row_1 = svld1_vnum(pg, &src_rows.at(1)[0], 1);
  svuint16_t sum0b = svaddlb(top_row_0, bottom_row_0);
  svuint16_t sum0t = svaddlt(top_row_0, bottom_row_0);
  svuint16_t sum1b = svaddlb(top_row_1, bottom_row_1);
  svuint16_t sum1t = svaddlt(top_row_1, bottom_row_1);
  svuint8_t res0 = svrshrnb(svadd_x(pg, sum0b, sum0t), 2);
  svuint8_t res1 = svrshrnb(svadd_x(pg, sum1b, sum1t), 2);
  svuint8_t result = svuzp1(res0, res1);
  svst1(pg, &dst_rows[0], result);
}

static inline void parallel_rows_vectors_path(
    svbool_t pg, Rows<const uint8_t> src_rows,
    Rows<uint8_t> dst_rows) KLEIDICV_STREAMING_COMPATIBLE {
  svuint8_t top_line = svld1(pg, &src_rows.at(0)[0]);
  svuint8_t bottom_line = svld1(pg, &src_rows.at(1)[0]);
  svuint8_t result = resize_parallel_vectors(pg, top_line, bottom_line);
  svst1b(pg, &dst_rows[0], svreinterpret_u16_u8(result));
}

template <typename ScalarType>
static inline void process_parallel_rows(
    Rows<const ScalarType> src_rows, size_t src_width,
    Rows<ScalarType> dst_rows, size_t dst_width) KLEIDICV_STREAMING_COMPATIBLE {
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  const size_t size_mask = ~static_cast<size_t>(1U);

  // Process rows up to the last even pixel index.
  LoopUnroll2{src_width & size_mask, VecTraits::num_lanes()}
      // Process double vector chunks.
      .unroll_twice([&](size_t index) KLEIDICV_STREAMING_COMPATIBLE {
        auto pg = VecTraits::svptrue();
        parallel_rows_vectors_path_2x(pg, src_rows.at(0, index),
                                      dst_rows.at(0, index / 2));
      })
      .unroll_once([&](size_t index) KLEIDICV_STREAMING_COMPATIBLE {
        auto pg = VecTraits::svptrue();
        parallel_rows_vectors_path(pg, src_rows.at(0, index),
                                   dst_rows.at(0, index / 2));
      })
      // Process the remaining chunk of the row.
      .remaining([&](size_t index, size_t length)
                     KLEIDICV_STREAMING_COMPATIBLE {
                       auto pg = VecTraits::svwhilelt(index, length);
                       parallel_rows_vectors_path(pg, src_rows.at(0, index),
                                                  dst_rows.at(0, index / 2));
                     });

  // Handle the last odd column, if any.
  if (dst_width > (src_width / 2)) {
    dst_rows[dst_width - 1] = rounding_shift_right<uint16_t>(
        static_cast<const uint16_t>(src_rows.at(0, src_width - 1)[0]) +
            src_rows.at(1, src_width - 1)[0],
        1);
  }
}

static inline svuint8_t resize_single_row(svbool_t pg, svuint8_t row)
    KLEIDICV_STREAMING_COMPATIBLE {
  return svrshrnb(svadalp_x(pg, svdup_u16(0), row), 1);
}

static inline void single_row_vector_path_2x(
    svbool_t pg, Rows<const uint8_t> src_rows,
    Rows<uint8_t> dst_rows) KLEIDICV_STREAMING_COMPATIBLE {
  svuint8_t line0 = svld1(pg, &src_rows[0]);
  svuint8_t line1 = svld1_vnum(pg, &src_rows[0], 1);
  svuint8_t result0 = svrshrnb(svadalp_x(pg, svdup_u16(0), line0), 1);
  svuint8_t result1 = svrshrnb(svadalp_x(pg, svdup_u16(0), line1), 1);
  svst1b(pg, &dst_rows[0], svreinterpret_u16_u8(result0));
  svst1b_vnum(pg, &dst_rows[0], 1, svreinterpret_u16_u8(result1));
}

static inline void single_row_vector_path(
    svbool_t pg, Rows<const uint8_t> src_rows,
    Rows<uint8_t> dst_rows) KLEIDICV_STREAMING_COMPATIBLE {
  svuint8_t line = svld1(pg, &src_rows.at(0)[0]);
  svuint8_t result = svrshrnb(svadalp_x(pg, svdup_u16(0), line), 1);
  svst1b(pg, &dst_rows[0], svreinterpret_u16_u8(result));
}

template <typename ScalarType>
static inline void process_single_row(
    Rows<const ScalarType> src_rows, size_t src_width,
    Rows<ScalarType> dst_rows, size_t dst_width) KLEIDICV_STREAMING_COMPATIBLE {
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  const size_t size_mask = ~static_cast<size_t>(1U);

  // Process rows up to the last even pixel index.
  LoopUnroll2{src_width & size_mask, VecTraits::num_lanes()}
      // Process full vector chunks.
      .unroll_twice([&](size_t index) KLEIDICV_STREAMING_COMPATIBLE {
        auto pg = VecTraits::svptrue();
        single_row_vector_path_2x(pg, src_rows.at(0, index),
                                  dst_rows.at(0, index / 2));
      })
      .unroll_once([&](size_t index) KLEIDICV_STREAMING_COMPATIBLE {
        auto pg = VecTraits::svptrue();
        single_row_vector_path(pg, src_rows.at(0, index),
                               dst_rows.at(0, index / 2));
      })
      // Process the remaining chunk of the row.
      .remaining([&](size_t index, size_t length)
                     KLEIDICV_STREAMING_COMPATIBLE {
                       auto pg = VecTraits::svwhilelt(index, length);
                       single_row_vector_path(pg, src_rows.at(0, index),
                                              dst_rows.at(0, index / 2));
                     });

  // Handle the last odd column, if any.
  if (dst_width > (src_width / 2)) {
    dst_rows[dst_width - 1] = src_rows[src_width - 1];
  }
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t check_dimensions(size_t src_dim, size_t dst_dim)
    KLEIDICV_STREAMING_COMPATIBLE {
  size_t half_src_dim = src_dim / 2;

  if ((src_dim % 2) == 0) {
    if (dst_dim == half_src_dim) {
      return KLEIDICV_OK;
    }
  } else {
    if (dst_dim == half_src_dim || dst_dim == (half_src_dim + 1)) {
      return KLEIDICV_OK;
    }
  }

  return KLEIDICV_ERROR_RANGE;
}

KLEIDICV_TARGET_FN_ATTRS static kleidicv_error_t resize_to_quarter_u8_sc(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width,
    size_t dst_height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_IMAGE_SIZE(src_width, src_height);

  if (kleidicv_error_t ret = check_dimensions(src_width, dst_width)) {
    return ret;
  }

  if (kleidicv_error_t ret = check_dimensions(src_height, dst_height)) {
    return ret;
  }

  Rows<const uint8_t> src_rows{src, src_stride, /* channels*/ 1};
  Rows<uint8_t> dst_rows{dst, dst_stride, /* channels*/ 1};
  LoopUnroll2 loop{src_height, /* Process two rows */ 2};

  // Process two rows at once.
  loop.unroll_once([&](size_t)  // NOLINT(readability/casting)
                   KLEIDICV_STREAMING_COMPATIBLE {
                     process_parallel_rows(src_rows, src_width, dst_rows,
                                           dst_width);
                     src_rows += 2;
                     ++dst_rows;
                   });

  // Handle an odd row, if any.
  if (dst_height > (src_height / 2)) {
    loop.remaining([&](size_t, size_t) KLEIDICV_STREAMING_COMPATIBLE {
      process_single_row(src_rows, src_width, dst_rows, dst_width);
    });
  }
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_RESIZE_SC_H
