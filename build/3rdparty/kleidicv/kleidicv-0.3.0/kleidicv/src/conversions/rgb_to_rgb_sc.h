// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RGB_TO_RGB_SC_H
#define KLEIDICV_RGB_TO_RGB_SC_H

#include "kleidicv/conversions/rgb_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType>
class RGBToBGR final :
#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE && KLEIDICV_ASSUME_128BIT_SVE2
    public UsesTailPath,
#endif
    public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE || !KLEIDICV_ASSUME_128BIT_SVE2
  void vector_path(ContextType ctx, const ScalarType *src,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    svuint8x3_t src_vect = svld3(pg, src);
    svuint8x3_t dst_vect = svcreate3(svget3(src_vect, 2), svget3(src_vect, 1),
                                     svget3(src_vect, 0));

    svst3(pg, dst, dst_vect);
  }
#else   // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE ||
        // !KLEIDICV_ASSUME_128BIT_SVE2
  explicit RGBToBGR(svuint8x4_t &indices) KLEIDICV_STREAMING_COMPATIBLE
      : indices_{indices} {
    initialize_indices();
  }

  void vector_path(ContextType ctx, const ScalarType *src,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    // Call the common vector path.
    auto pg = ctx.predicate();
    common_vector_path(pg, pg, pg, src, dst);
  }

  void tail_path(ContextType ctx, const ScalarType *src,
                 ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    // Predicates for consecutive stores.
    svbool_t pg_0, pg_1, pg_2;
    VecTraits::make_consecutive_predicates(pg, pg_0, pg_1, pg_2);
    // Call the common vector path.
    common_vector_path(pg_0, pg_1, pg_2, src, dst);
  }

 private:
  void common_vector_path(svbool_t pg_0, svbool_t pg_1, svbool_t pg_2,
                          const ScalarType *src,
                          ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_0 = svld1(pg_0, &src[0]);
    VectorType src_1 = svld1_vnum(pg_1, &src[0], 1);
    VectorType src_2 = svld1_vnum(pg_2, &src[0], 2);

    svuint8x2_t src_vect_0_1 = svcreate2(src_0, src_1);
    svuint8x2_t src_vect_1_2 = svcreate2(src_1, src_2);

    svuint8_t dst_vec_0 = svtbl2(src_vect_0_1, svget4(indices_, 0));
    svuint8_t dst_vec_2 = svtbl2(src_vect_1_2, svget4(indices_, 3));
    svuint8_t dst_vec_1 = svtbl2(src_vect_0_1, svget4(indices_, 1));
    src_vect_1_2 = svcreate2(dst_vec_1, src_2);
    dst_vec_1 = svtbl2(src_vect_1_2, svget4(indices_, 2));

    svst1(pg_0, &dst[0], dst_vec_0);
    svst1_vnum(pg_1, &dst[0], 1, dst_vec_1);
    svst1_vnum(pg_2, &dst[0], 2, dst_vec_2);
  }

  void initialize_indices() KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t pg = VecTraits::svptrue();
    indices_ = svcreate4(svld1(pg, &kTableIndices[0]),
                         svld1_vnum(pg, &kTableIndices[0], 1),
                         svld1_vnum(pg, &kTableIndices[0], 2),
                         svld1_vnum(pg, &kTableIndices[0], 3));
  }

  static constexpr uint8_t kTableIndices[64] = {
      2,  1,  0,  5,  4,  3,  8,  7,  6,  11, 10, 9,  14, 13, 12, 17,
      16, 15, 20, 19, 18, 23, 22, 21, 26, 25, 24, 29, 28, 27, 32, 31,
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 16, 15,
      14, 19, 18, 17, 22, 21, 20, 25, 24, 23, 28, 27, 26, 31, 30, 29};

  // Hold a reference because a sizeless types cannot be members.
  svuint8x4_t &indices_;
#endif  // !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE ||
        // !KLEIDICV_ASSUME_128BIT_SVE2
};  // end of class RGBToBGR<ScalarType>

template <typename ScalarType>
class RGBAToBGRA final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;

  void vector_path(ContextType ctx, const ScalarType *src,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    svuint8x4_t src_vect = svld4(pg, src);
    svuint8x4_t dst_vect = svcreate4(svget4(src_vect, 2), svget4(src_vect, 1),
                                     svget4(src_vect, 0), svget4(src_vect, 3));

    svst4(pg, dst, dst_vect);
  }
};  // end of class RGBAToBGRA<ScalarType>

template <typename ScalarType>
class RGBToBGRA final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;

  void vector_path(ContextType ctx, const ScalarType *src,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    svuint8x3_t src_vect = svld3(pg, src);
    svuint8x4_t dst_vect = svcreate4(svget3(src_vect, 2), svget3(src_vect, 1),
                                     svget3(src_vect, 0), svdup_u8(0xff));

    svst4(pg, dst, dst_vect);
  }
};  // end of class RGBToBGRA<ScalarType>

template <typename ScalarType>
class RGBToRGBA final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;

  void vector_path(ContextType ctx, const ScalarType *src,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    svuint8x3_t src_vect = svld3(pg, src);
    svuint8x4_t dst_vect = svcreate4(svget3(src_vect, 0), svget3(src_vect, 1),
                                     svget3(src_vect, 2), svdup_u8(0xff));

    svst4(pg, dst, dst_vect);
  }
};  // end of class RGBToRGBA<ScalarType>

template <typename ScalarType>
class RGBAToBGR final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;

  void vector_path(ContextType ctx, const ScalarType *src,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    svuint8x4_t src_vect = svld4(pg, src);
    svuint8x3_t dst_vect = svcreate3(svget4(src_vect, 2), svget4(src_vect, 1),
                                     svget4(src_vect, 0));

    svst3(pg, dst, dst_vect);
  }
};  // end of class RGBAToBGR<ScalarType>

template <typename ScalarType>
class RGBAToRGB final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;

  void vector_path(ContextType ctx, const ScalarType *src,
                   ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    svuint8x4_t src_vect = svld4(pg, src);
    svuint8x3_t dst_vect = svcreate3(svget4(src_vect, 0), svget4(src_vect, 1),
                                     svget4(src_vect, 2));

    svst3(pg, dst, dst_vect);
  }
};  // end of class RGBAToRGB<ScalarType>

KLEIDICV_TARGET_FN_ATTRS static kleidicv_error_t rgb_to_bgr_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 3 /* RGB */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* BGR */};
#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE || !KLEIDICV_ASSUME_128BIT_SVE2
  RGBToBGR<uint8_t> operation;
#else
  svuint8x4_t table_indices;
  RGBToBGR<uint8_t> operation{table_indices};
#endif
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t rgba_to_bgra_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 4 /* RGBA */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* BGRA */};
  RGBAToBGRA<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t rgb_to_bgra_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 3 /* RGB */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* BGRA */};
  RGBToBGRA<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t rgb_to_rgba_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 3 /* RGB */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* RGBA */};
  RGBToRGBA<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t rgba_to_bgr_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 4 /* RGBA */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* BGR */};
  RGBAToBGR<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t rgba_to_rgb_u8_sc(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 4 /* RGBA */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* RGB */};
  RGBAToRGB<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_RGB_TO_RGB_SC_H
