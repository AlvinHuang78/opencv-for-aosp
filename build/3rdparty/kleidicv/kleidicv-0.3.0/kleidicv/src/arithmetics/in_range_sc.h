// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_IN_RANGE_SC_H
#define KLEIDICV_IN_RANGE_SC_H

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType>
class InRange;

template <>
class InRange<uint8_t> : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<uint8_t>;
  using VectorType = typename VecTraits::VectorType;
  using SignedScalarType = typename std::make_signed<uint8_t>::type;
  using SignedVecTraits =
      KLEIDICV_TARGET_NAMESPACE::VecTraits<SignedScalarType>;
  using SignedVectorType = typename SignedVecTraits::VectorType;

  InRange(VectorType &vec_lower_bound,
          VectorType &vec_upper_bound) KLEIDICV_STREAMING_COMPATIBLE
      : vec_lower_bound_(vec_lower_bound),
        vec_upper_bound_(vec_upper_bound) {}

  // NOLINTBEGIN(readability-make-member-function-const)
  VectorType vector_path(ContextType ctx,
                         VectorType src) KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t pg = ctx.predicate();

    VectorType diff_low = svsub_x(pg, src, vec_lower_bound_);
    // Shift subtraction result 7 bits to the right, i.e. divide by 2^7 to keep
    // sign bit only.
    VectorType result_within_low =
        VecTraits::svreinterpret(SignedVecTraits::svasr_n(
            pg, SignedVecTraits::svreinterpret(diff_low), 7));

    VectorType diff_up = svsub_x(pg, vec_upper_bound_, src);
    VectorType result_within_up =
        VecTraits::svreinterpret(SignedVecTraits::svasr_n(
            pg, SignedVecTraits::svreinterpret(diff_up), 7));

    // src[i] < lower_bound OR src[i] > upper_bound
    VectorType out_of_range = svorr_x(pg, result_within_low, result_within_up);
    // NOT(out_of_range) to set within elements to 1 and the rest to 0.
    VectorType within_range = svcnot_x(pg, out_of_range);
    // Negate to set elements within to 0xFF (all 1s).
    return VecTraits::svreinterpret(
        svqneg_x(pg, SignedVecTraits::svreinterpret(within_range)));
  }
  // NOLINTEND(readability-make-member-function-const)

 private:
  VectorType &vec_lower_bound_;
  VectorType &vec_upper_bound_;
};  // end of class InRange<uint8_t>

template <>
class InRange<float> {
 public:
  using SrcVecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<float>;
  using SrcVectorType = typename SrcVecTraits::VectorType;
  using DstVecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<uint8_t>;
  using DstVectorType = typename DstVecTraits::VectorType;

  InRange(float lower_bound, float upper_bound) KLEIDICV_STREAMING_COMPATIBLE
      : lower_bound_(lower_bound),
        upper_bound_(upper_bound) {}

  void process_row(size_t width, Columns<const float> src,
                   Columns<uint8_t> dst) KLEIDICV_STREAMING_COMPATIBLE {
    LoopUnroll{width, SrcVecTraits::num_lanes()}
        .unroll_n_times<4>([&](size_t step) KLEIDICV_STREAMING_COMPATIBLE {
          svbool_t pg_src = SrcVecTraits::svptrue();
          SrcVectorType src_v0 = svld1(pg_src, &src[0]);
          SrcVectorType src_v1 = svld1_vnum(pg_src, &src[0], 1);
          SrcVectorType src_v2 = svld1_vnum(pg_src, &src[0], 2);
          SrcVectorType src_v3 = svld1_vnum(pg_src, &src[0], 3);
          DstVectorType res0 =
              vector_path(pg_src, src_v0, src_v1, src_v2, src_v3);
          svbool_t pg_dst = DstVecTraits::svptrue();
          svst1(pg_dst, &dst[0], res0);
          src += ptrdiff_t(step);
          dst += ptrdiff_t(step);
        })
        .remaining([&](size_t length, size_t) KLEIDICV_STREAMING_COMPATIBLE {
          size_t index = 0;
          svbool_t pg = SrcVecTraits::svwhilelt(index, length);
          while (svptest_first(SrcVecTraits::svptrue(), pg)) {
            SrcVectorType src_vector = svld1(pg, &src[ptrdiff_t(index)]);
            DstVectorType result_vector = remaining_path(pg, src_vector);
            svst1b(pg, &dst[ptrdiff_t(index)],
                   svreinterpret_u32(result_vector));
            // Update loop counter and calculate the next governing predicate.
            index += SrcVecTraits::num_lanes();
            pg = SrcVecTraits::svwhilelt(index, length);
          }
        });
  }

 private:
  // NOLINTBEGIN(readability-make-member-function-const)
  DstVectorType vector_path(svbool_t full_pg, SrcVectorType fsrc0,
                            SrcVectorType fsrc1, SrcVectorType fsrc2,
                            SrcVectorType fsrc3) KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t pred0 = svand_z(full_pg, svcmpge(full_pg, fsrc0, lower_bound_),
                             svcmple(full_pg, fsrc0, upper_bound_));
    auto res00 = svsel(pred0, svdup_u32(0xFF), svdup_u32(0));

    svbool_t pred1 = svand_z(full_pg, svcmpge(full_pg, fsrc1, lower_bound_),
                             svcmple(full_pg, fsrc1, upper_bound_));
    auto res01 = svsel(pred1, svdup_u32(0xFF), svdup_u32(0));

    svbool_t pred2 = svand_z(full_pg, svcmpge(full_pg, fsrc2, lower_bound_),
                             svcmple(full_pg, fsrc2, upper_bound_));
    auto res10 = svsel(pred2, svdup_u32(0xFF), svdup_u32(0));

    svbool_t pred3 = svand_z(full_pg, svcmpge(full_pg, fsrc3, lower_bound_),
                             svcmple(full_pg, fsrc3, upper_bound_));
    auto res11 = svsel(pred3, svdup_u32(0xFF), svdup_u32(0));

    auto res0 =
        svuzp1(svreinterpret_u16_u32(res00), svreinterpret_u16_u32(res01));
    auto res1 =
        svuzp1(svreinterpret_u16_u32(res10), svreinterpret_u16_u32(res11));
    return svuzp1(svreinterpret_u8_u16(res0), svreinterpret_u8_u16(res1));
  }
  // NOLINTEND(readability-make-member-function-const)

  // NOLINTBEGIN(readability-make-member-function-const)
  DstVectorType remaining_path(svbool_t &pg, SrcVectorType src)
      KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t predicate = svand_z(pg, svcmpge(pg, src, lower_bound_),
                                 svcmple(pg, src, upper_bound_));
    return svsel(predicate, DstVecTraits::svdup(0xFF), DstVecTraits::svdup(0));
  }
  // NOLINTEND(readability-make-member-function-const)

  float lower_bound_;
  float upper_bound_;
};  // end of class InRange<float>

template <typename T>
kleidicv_error_t in_range_sc(const T *src, size_t src_stride, uint8_t *dst,
                             size_t dst_stride, size_t width, size_t height,
                             T lower_bound,
                             T upper_bound) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const T> src_rows{src, src_stride};
  Rows<uint8_t> dst_rows{dst, dst_stride};

  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<T>;
  using VectorType = typename VecTraits::VectorType;

  if constexpr (std::is_same_v<T, uint8_t>) {
    VectorType vec_lower_bound = VecTraits::svdup(lower_bound);
    VectorType vec_upper_bound = VecTraits::svdup(upper_bound);
    InRange<T> operation{vec_lower_bound, vec_upper_bound};
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  } else {
    InRange<T> operation{lower_bound, upper_bound};
    zip_rows(operation, rect, src_rows, dst_rows);
  }

  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_IN_RANGE_SC_H
