// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/merge.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

// ----------------------------------------
// ------------ Two-way merge -------------
// ----------------------------------------

// Generic 2-way merge implementation.
//
// Algorithm description
//
//  Elements are identified by their intended final position in the output.
//  The description is for 32-bit elements, but it works just the same way
//  for different element sizes.
//
//    VECTOR / LANE:   0  1  2  3
//            src_a: [ 0, 2, 4, 6 ]
//            src_b: [ 1, 3, 5, 7 ]
//
//       zip1(a, b): [ 0, 1, 2, 3 ] -> d0
//       zip2(a, b): [ 4, 5, 6, 7 ] -> d1
//
//  Continuous store of { d0, d1 } gives the expected order.
template <typename ScalarType>
class Merge2 final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using Vector2Type = typename VecTraits::Vector2Type;

  void vector_path(VectorType src_a, VectorType src_b, ScalarType *dst) {
#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    Vector2Type dst_vect;
    dst_vect.val[0] = src_a;
    dst_vect.val[1] = src_b;
    vst2q(&dst[0], dst_vect);
#else   // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    Vector2Type dst_vect;
    dst_vect.val[0] = vzip1q(src_a, src_b);
    dst_vect.val[1] = vzip2q(src_a, src_b);
    vst1q_x2(&dst[0], dst_vect);
#endif  // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  }

  void scalar_path(const ScalarType *src_a, const ScalarType *src_b,
                   ScalarType *dst) {
    dst[0] = src_a[0];
    dst[1] = src_b[0];
  }
};  // end of class Merge2<ScalarType>

// ----------------------------------------
// ---------- Three-way merge -------------
// ----------------------------------------

template <typename ScalarType>
class Merge3 final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using Vector3Type = typename VecTraits::Vector3Type;

#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  Merge3() : table_indices_{vld1q_u8_x3(lookup_table(ScalarType()))} {}
#endif

  void vector_path(VectorType src_a, VectorType src_b, VectorType src_c,
                   ScalarType *dst) {
#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    Vector3Type dst_vect;
    dst_vect.val[0] = src_a;
    dst_vect.val[1] = src_b;
    dst_vect.val[2] = src_c;
    vst3q(&dst[0], dst_vect);
#else   // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    uint8x16x3_t src_vect, dst_vect;
    src_vect.val[0] = vreinterpretq_u8(src_a);
    src_vect.val[1] = vreinterpretq_u8(src_b);
    src_vect.val[2] = vreinterpretq_u8(src_c);
    dst_vect.val[0] = vqtbl3q_u8(src_vect, table_indices_.val[0]);
    dst_vect.val[1] = vqtbl3q_u8(src_vect, table_indices_.val[1]);
    dst_vect.val[2] = vqtbl3q_u8(src_vect, table_indices_.val[2]);
    vst1q_u8_x3(reinterpret_cast<uint8_t *>(&dst[0]), dst_vect);
#endif  // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  }

  void scalar_path(const ScalarType *src_a, const ScalarType *src_b,
                   const ScalarType *src_c, ScalarType *dst) {
    dst[0] = src_a[0];
    dst[1] = src_b[0];
    dst[2] = src_c[0];
  }

 private:
#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  static const uint8_t *lookup_table(uint8_t) {
    // clang-format off
    static constexpr uint8_t kIndices[48] = {
       0, 16, 32,  1, 17, 33,  2, 18, 34,  3, 19, 35,  4, 20, 36,  5,
      21, 37,  6, 22, 38,  7, 23, 39,  8, 24, 40,  9, 25, 41, 10, 26,
      42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47,
    };
    // clang-format on
    return &kIndices[0];
  }

  // Lookup table for 16-bit inputs.
  static const uint8_t *lookup_table(uint16_t) {
    // clang-format off
    static constexpr uint8_t kIndices[48] = {
       0,  1, 16, 17, 32, 33,  2,  3, 18, 19, 34, 35,  4,  5, 20, 21,
      36, 37,  6,  7, 22, 23, 38, 39,  8,  9, 24, 25, 40, 41, 10, 11,
      26, 27, 42, 43, 12, 13, 28, 29, 44, 45, 14, 15, 30, 31, 46, 47,
    };
    // clang-format on
    return &kIndices[0];
  }

  uint8x16x3_t table_indices_;
#endif
};  // end of class Merge3<ScalarType>

#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE

// Specialized 3-way merge implementation for 32-bit elements.
//
// Algorithm description
//
//  Elements are identified by their intended final position in the output.
//
//             VECTOR / LANE:    0   1   2   3
//                     src_a: [  0,  3,  6,  9 ]
//                     src_b: [  1,  4,  7, 10 ]
//                     src_c: [  2,  5,  8, 11 ]
//
//                trn2(a, b): [  3,  4,  9, 10 ] -> w
//                trn1(c, w): [  2,  3,  8,  9 ] -> x
//                trn2(w, c): [  4,  5, 10, 11 ] -> y
//                trn1(a, b): [  0,  1,  6,  7 ] -> z
//
//            zip1_u64(z, x): [  0,  1,  2,  3 ] -> d0
//    [ y_u64[0], z_u64[1] ]: [  4,  5,  6,  7 ] -> d1
//            zip2_u64(x, y): [  8,  9, 10, 11 ] -> d2
//
//  Continuous store of { d0, d1, d2 } gives the expected order.
template <>
class Merge3<uint32_t> final : public UnrollTwice {
 public:
  using ScalarType = uint32_t;
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using Vector3Type = typename VecTraits::Vector3Type;

  void vector_path(VectorType src_a, VectorType src_b, VectorType src_c,
                   ScalarType *dst) {
    uint32x4_t w = vtrn2q_u32(src_a, src_b);
    uint32x4_t x = vtrn1q_u32(src_c, w);
    uint32x4_t y = vtrn2q_u32(w, src_c);
    uint32x4_t z = vtrn1q_u32(src_a, src_b);

    uint32x4_t dst_vect_0 = vzip1q_u64(z, x);
    uint64x2_t dst_vect_1 = y;
    dst_vect_1[1] = vreinterpretq_u64_u32(z)[1];
    uint32x4_t dst_vect_2 = vzip2q_u64(x, y);

    // Not using vst1q_u32_x3, becuse the requirement on continuous vector
    // register allocation may result in longer code.
    vst1q_u32(&dst[0 * VecTraits::num_lanes()], dst_vect_0);
    vst1q_u32(&dst[1 * VecTraits::num_lanes()], dst_vect_1);
    vst1q_u32(&dst[2 * VecTraits::num_lanes()], dst_vect_2);
  }

  void scalar_path(const ScalarType *src_a, const ScalarType *src_b,
                   const ScalarType *src_c, ScalarType *dst) {
    dst[0] = src_a[0];
    dst[1] = src_b[0];
    dst[2] = src_c[0];
  }
};  // end of class Merge3<uint32_t>

// Specialized 3-way merge implementation for 64-bit elements.
//
// Algorithm description
//
//  Elements are identified by their intended final position in the output.
//
//             VECTOR / LANE:   0  1
//                     src_a: [ 0, 3 ]
//                     src_b: [ 1, 4 ]
//                     src_c: [ 2, 5 ]
//
//                zip1(a, b): [ 0, 1 ] -> d0
//    [ src_c[0], src_a[1] ]: [ 2, 3 ] -> d1
//                zip2(b, c): [ 4, 5 ] -> d2
//
//  Continuous store of { d0, d1, d2 } gives the expected order.
template <>
class Merge3<uint64_t> final : public UnrollTwice {
 public:
  using ScalarType = uint64_t;
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  void vector_path(VectorType src_a, VectorType src_b, VectorType src_c,
                   ScalarType *dst) {
    uint64x2x3_t dst_vect;
    dst_vect.val[0] = vzip1q_u64(src_a, src_b);
    dst_vect.val[1] = src_c;
    dst_vect.val[1][1] = src_a[1];
    dst_vect.val[2] = vzip2q_u64(src_b, src_c);
    vst1q_u64_x3(&dst[0], dst_vect);
  }

  void scalar_path(const ScalarType *src_a, const ScalarType *src_b,
                   const ScalarType *src_c, ScalarType *dst) {
    dst[0] = src_a[0];
    dst[1] = src_b[0];
    dst[2] = src_c[0];
  }
};  // end of class Merge3<uint64_t>

#endif  // !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE

// ----------------------------------------
// ----------- Four-way merge -------------
// ----------------------------------------

// Generic 4-way merge implementation.
//
// Algorithm description
//
//  Elements are identified by their intended final position in the output.
//  The description is for 32-bit elements, but it works just the same way
//  for smaller element sizes.
//
//     VECTOR / LANE:    0   1   2   3
//             src_a: [  0,  4,  8, 12 ]
//             src_b: [  1,  5,  9, 13 ]
//             src_c: [  2,  6, 10, 14 ]
//             src_d: [  3,  7, 11, 15 ]
//
//    zip1_u32(a, b): [  0,  1,  4,  5 ] -> w
//    zip1_u32(c, d): [  2,  3,  6,  7 ] -> x
//    zip2_u32(a, b): [  8,  9, 12, 13 ] -> y
//    zip2_u32(c, d): [ 10, 11, 14, 15 ] -> z
//
//    zip1_u64(w, x): [  0,  1,  2,  3 ] -> d0
//    zip2_u64(w, x): [  4,  5,  6,  7 ] -> d1
//    zip1_u64(y, z): [  8,  9, 10, 11 ] -> d2
//    zip2_u64(y, z): [ 12, 13, 14, 15 ] -> d3
//
//  Continuous store of { d0, d1, d2, d3 } gives the expected order.
template <typename ScalarType>
class Merge4 final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using Vector4Type = typename VecTraits::Vector4Type;

  void vector_path(VectorType src_a, VectorType src_b, VectorType src_c,
                   VectorType src_d, ScalarType *dst) {
#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    Vector4Type dst_vect;
    dst_vect.val[0] = src_a;
    dst_vect.val[1] = src_b;
    dst_vect.val[2] = src_c;
    dst_vect.val[3] = src_d;
    vst4q(&dst[0], dst_vect);
#else   // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    auto zip1_a_b = double_width(vzip1q(src_a, src_b));
    auto zip1_c_d = double_width(vzip1q(src_c, src_d));
    auto zip2_a_b = double_width(vzip2q(src_a, src_b));
    auto zip2_c_d = double_width(vzip2q(src_c, src_d));

    // Compilers tend to replace zip instructions with mov, resulting in
    // longer generated code. Omitting a bitcast appears to help.
    using DoubleScalarType = double_element_width_t<ScalarType>;
    typename neon::VecTraits<DoubleScalarType>::Vector4Type dst_vect;
    dst_vect.val[0] = vzip1q(zip1_a_b, zip1_c_d);
    dst_vect.val[1] = vzip2q(zip1_a_b, zip1_c_d);
    dst_vect.val[2] = vzip1q(zip2_a_b, zip2_c_d);
    dst_vect.val[3] = vzip2q(zip2_a_b, zip2_c_d);
    vst1q_x4(reinterpret_cast<DoubleScalarType *>(&dst[0]), dst_vect);
#endif  // KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  }

  void scalar_path(const ScalarType *src_a, const ScalarType *src_b,
                   const ScalarType *src_c, const ScalarType *src_d,
                   ScalarType *dst) {
    dst[0] = src_a[0];
    dst[1] = src_b[0];
    dst[2] = src_c[0];
    dst[3] = src_d[0];
  }

 private:
#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  // Polymorphic retinterpret_cast<>() between vector types where the element
  // size is doubled. For example, if 'VectorType' is 'uint8x16_t', this
  // method returns 'reinterpret_cast<uint16x8_t>(vector)'.
  static double_element_width_t<VectorType> double_width(VectorType vector) {
    return reinterpret_cast<double_element_width_t<VectorType>>(vector);
  }
#endif
};  // end of class Merge4<ScalarType>

#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE

// Specialized 4-way merge implementation for 64-bit elements.
//
// Algorithm description
//
//  Elements are identified by their intended final position in the output.
//
//    VECTOR / LANE:   0  1
//            src_a: [ 0, 4 ]
//            src_b: [ 1, 5 ]
//            src_c: [ 2, 6 ]
//            src_d: [ 3, 7 ]
//
//       zip1(a, b): [ 0, 1 ] -> d0
//       zip1(c, d): [ 2, 3 ] -> d1
//       zip2(a, b): [ 4, 5 ] -> d2
//       zip2(c, d): [ 6, 7 ] -> d3
//
//  Continuous store of { d0, d1, d2, d3 } gives the expected order.
template <>
class Merge4<uint64_t> final : public UnrollTwice {
 public:
  using ScalarType = uint64_t;
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using Vector4Type = typename VecTraits::Vector4Type;

  void vector_path(VectorType src_a, VectorType src_b, VectorType src_c,
                   VectorType src_d, ScalarType *dst) {
    Vector4Type dst_vect;
    dst_vect.val[0] = vzip1q(src_a, src_b);
    dst_vect.val[1] = vzip1q(src_c, src_d);
    dst_vect.val[2] = vzip2q(src_a, src_b);
    dst_vect.val[3] = vzip2q(src_c, src_d);
    vst1q_x4(&dst[0], dst_vect);
  }

  void scalar_path(const ScalarType *src_a, const ScalarType *src_b,
                   const ScalarType *src_c, const ScalarType *src_d,
                   ScalarType *dst) {
    dst[0] = src_a[0];
    dst[1] = src_b[0];
    dst[2] = src_c[0];
    dst[3] = src_d[0];
  }
};  // end of class Merge4<uint64_t>

#endif  // !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE

// Most of the complexity comes from parameter checking.
// NOLINTBEGIN(readability-function-cognitive-complexity)
template <typename ScalarType>
kleidicv_error_t merge(const void **srcs, const size_t *src_strides,
                       void *dst_void, size_t dst_stride, size_t width,
                       size_t height, size_t channels) {
  if (channels < 2) {
    return KLEIDICV_ERROR_RANGE;
  }
  CHECK_POINTERS(srcs, src_strides);
  MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src0, srcs[0]);
  MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src1, srcs[1]);
  MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst, dst_void);
  CHECK_POINTER_AND_STRIDE(src0, src_strides[0], height);
  CHECK_POINTER_AND_STRIDE(src1, src_strides[1], height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const ScalarType> src_a_rows{src0, src_strides[0]};
  Rows<const ScalarType> src_b_rows{src1, src_strides[1]};
  Rows<ScalarType> dst_rows{dst, dst_stride, channels};

  switch (channels) {
    case 2: {
      Merge2<ScalarType> operation;
      apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows,
                              dst_rows);
    } break;

    case 3: {
      MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src2, srcs[2]);
      CHECK_POINTER_AND_STRIDE(src2, src_strides[2], height);
      Merge3<ScalarType> operation;
      Rows<const ScalarType> src_c_rows{src2, src_strides[2]};
      apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows,
                              src_c_rows, dst_rows);
    } break;

    case 4: {
      MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src2, srcs[2]);
      MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src3, srcs[3]);
      CHECK_POINTER_AND_STRIDE(src2, src_strides[2], height);
      CHECK_POINTER_AND_STRIDE(src3, src_strides[3], height);
      Merge4<ScalarType> operation;
      Rows<const ScalarType> src_c_rows{src2, src_strides[2]};
      Rows<const ScalarType> src_d_rows{src3, src_strides[3]};
      apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows,
                              src_c_rows, src_d_rows, dst_rows);
    } break;

    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  return KLEIDICV_OK;
}
// NOLINTEND(readability-function-cognitive-complexity)

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t merge(const void **srcs, const size_t *src_strides, void *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t element_size) {
  switch (element_size) {
    case sizeof(uint8_t):
      return merge<uint8_t>(srcs, src_strides, dst, dst_stride, width, height,
                            channels);

    case sizeof(uint16_t):
      return merge<uint16_t>(srcs, src_strides, dst, dst_stride, width, height,
                             channels);

    case sizeof(uint32_t):
      return merge<uint32_t>(srcs, src_strides, dst, dst_stride, width, height,
                             channels);

    case sizeof(uint64_t):
      return merge<uint64_t>(srcs, src_strides, dst, dst_stride, width, height,
                             channels);

    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace kleidicv::neon
