// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_GRAY_TO_RGB_SC_H
#define KLEIDICV_GRAY_TO_RGB_SC_H

#include "kleidicv/conversions/gray_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType>
class GrayToRGB final :
#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    public UsesTailPath,
#endif
    public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  void vector_path(ContextType ctx, VectorType src_vect,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    svuint8x3_t dst_vect = svcreate3(src_vect, src_vect, src_vect);
    svst3(pg, dst, dst_vect);
  }
#else   // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  explicit GrayToRGB(svuint8x3_t &indices) KLEIDICV_STREAMING_COMPATIBLE
      : indices_{indices} {
    initialize_indices();
  }

  void vector_path(ContextType ctx, VectorType src_vect,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    // Call the common vector path.
    auto pg = ctx.predicate();
    common_vector_path(pg, pg, pg, src_vect, dst);
  }

  void tail_path(ContextType ctx, VectorType src_vect,
                 ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    // Predicates for consecutive stores.
    svbool_t pg_0, pg_1, pg_2;
    VecTraits::make_consecutive_predicates(pg, pg_0, pg_1, pg_2);
    // Call the common vector path.
    common_vector_path(pg_0, pg_1, pg_2, src_vect, dst);
  }

 private:
  void common_vector_path(svbool_t pg_0, svbool_t pg_1, svbool_t pg_2,
                          VectorType src_vect,
                          ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    // Convert from gray to RGB using table-lookups.
    VectorType dst_vec_0 = svtbl(src_vect, svget3(indices_, 0));
    VectorType dst_vec_1 = svtbl(src_vect, svget3(indices_, 1));
    VectorType dst_vec_2 = svtbl(src_vect, svget3(indices_, 2));

    svst1(pg_0, &dst[0], dst_vec_0);
    svst1_vnum(pg_1, &dst[0], 1, dst_vec_1);
    svst1_vnum(pg_2, &dst[0], 2, dst_vec_2);
  }

  void initialize_indices() KLEIDICV_STREAMING_COMPATIBLE {
    // All-true predicate to shorten code.
    svbool_t pg_all = VecTraits::svptrue();
    // Constant used for division by 3.
    VectorType const_171 = VecTraits::svdup(171);
    // Generated indices.
    VectorType indices_0, indices_1, indices_2;

    indices_0 = svindex_u8(0, 1);

    if (KLEIDICV_UNLIKELY(svcntb() == 256)) {
      indices_1 = svext(
          svdup_u8(0),
          svqadd(svindex_u8(svcntb() % 3, 1), static_cast<uint8_t>(2)), 254);
      indices_2 = svext(svdup_u8(0),
                        svqadd(svindex_u8(0, 1), static_cast<uint8_t>(3)), 255);
    } else {
      indices_1 = svindex_u8(svcntb() % 3, 1);
      indices_2 = svindex_u8((svcntb() * 2) % 3, 1);
    }

    indices_0 = svlsr_x(pg_all, svmulh_x(pg_all, indices_0, const_171), 1);
    indices_1 = svqadd_x(
        pg_all, svlsr_x(pg_all, svmulh_x(pg_all, indices_1, const_171), 1),
        static_cast<ScalarType>(svcntb() / 3));
    indices_2 = svqadd_x(
        pg_all, svlsr_x(pg_all, svmulh_x(pg_all, indices_2, const_171), 1),
        static_cast<ScalarType>((svcntb() * 2) / 3));

    indices_ = svcreate3(indices_0, indices_1, indices_2);
  }

  svuint8x3_t &indices_;
#endif  // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
};  // end of class GrayToRGB<ScalarType>

template <typename ScalarType>
class GrayToRGBAWithInterleaving final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  void vector_path(ContextType ctx, VectorType src_vect,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    svuint8_t alpha = svdup_u8(0xff);
    svuint8x4_t dst_vect = svcreate4(src_vect, src_vect, src_vect, alpha);

    svst4(pg, dst, dst_vect);
  }
};  // end of class GrayToRGBAWithInterleaving<ScalarType>

#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
template <typename ScalarType>
class GrayToRGBAWithLookUpTable final : public UnrollTwice,
                                        public UsesTailPath {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  explicit GrayToRGBAWithLookUpTable(svuint8x4_t &indices)
      KLEIDICV_STREAMING_COMPATIBLE : indices_{indices} {
    initialize_indices();
  }

  void vector_path(ContextType ctx, VectorType src_vect,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    // Call the common vector path.
    auto pg = ctx.predicate();
    common_vector_path(pg, pg, pg, pg, src_vect, dst);
  }

  void tail_path(ContextType ctx, VectorType src_vect,
                 ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    // Predicates for consecutive stores.
    svbool_t pg_0, pg_1, pg_2, pg_3;
    VecTraits::make_consecutive_predicates(pg, pg_0, pg_1, pg_2, pg_3);
    // Call the common vector path.
    common_vector_path(pg_0, pg_1, pg_2, pg_3, src_vect, dst);
  }

 private:
  void common_vector_path(svbool_t pg_0, svbool_t pg_1, svbool_t pg_2,
                          svbool_t pg_3, VectorType src_vect,
                          ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    svuint8x2_t src_and_alpha = svcreate2(src_vect, VecTraits::svdup(-1));

    // Convert from gray to RGBA using table-lookups.
    VectorType dst_vec_0 = svtbl2(src_and_alpha, svget4(indices_, 0));
    VectorType dst_vec_1 = svtbl2(src_and_alpha, svget4(indices_, 1));
    VectorType dst_vec_2 = svtbl2(src_and_alpha, svget4(indices_, 2));
    VectorType dst_vec_3 = svtbl2(src_and_alpha, svget4(indices_, 3));

    svst1(pg_0, &dst[0], dst_vec_0);
    svst1_vnum(pg_1, &dst[0], 1, dst_vec_1);
    svst1_vnum(pg_2, &dst[0], 2, dst_vec_2);
    svst1_vnum(pg_3, &dst[0], 3, dst_vec_3);
  }

  void initialize_indices() KLEIDICV_STREAMING_COMPATIBLE {
    // Number of four-tuple elements.
    uint64_t num_four_tuples = VecTraits::num_lanes() / 4;
    // Index of alpha.
    uint64_t idx_alpha = VecTraits::num_lanes();
    // Start index.
    uint64_t start_index = idx_alpha << 24;

    // Index generation is similar to that of GrayToRGB above.
    VectorType indices_0 =
        svreinterpret_u8_u32(svindex_u32(start_index, 0x10101));

    // Repeat for 'indices_1' but add number of 4-tuple elements.
    start_index += 0x10101 * num_four_tuples;
    VectorType indices_1 =
        svreinterpret_u8_u32(svindex_u32(start_index, 0x10101));

    // Similarly to 'indices_1', but add twice the number of 4-tuple elements.
    start_index += 0x10101 * num_four_tuples;
    VectorType indices_2 =
        svreinterpret_u8_u32(svindex_u32(start_index, 0x10101));

    // Similarly to 'indices_1', but add three times the number of 4-tuple
    // elements.
    start_index += 0x10101 * num_four_tuples;
    VectorType indices_3 =
        svreinterpret_u8_u32(svindex_u32(start_index, 0x10101));

    indices_ = svcreate4(indices_0, indices_1, indices_2, indices_3);
  }

  svuint8x4_t &indices_;
};  // end of class GrayToRGBAWithLookUpTable<ScalarType>
#endif  // !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE

KLEIDICV_TARGET_FN_ATTRS static kleidicv_error_t gray_to_rgb_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* RGB */};
#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  GrayToRGB<uint8_t> operation;
#else
  svuint8x3_t table_indices;
  GrayToRGB<uint8_t> operation{table_indices};
#endif
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS static kleidicv_error_t gray_to_rgba_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* RGBA */};

#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  GrayToRGBAWithInterleaving<int8_t> operation{};
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
#else
  if (svcntb() > 128) {
    GrayToRGBAWithInterleaving<uint8_t> operation{};
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  } else {
    svuint8x4_t table_indices;
    GrayToRGBAWithLookUpTable<uint8_t> operation{table_indices};
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  }
#endif
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_GRAY_TO_RGB_SC_H
