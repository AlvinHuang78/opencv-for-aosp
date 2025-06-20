// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_EXP_SC_H
#define KLEIDICV_EXP_SC_H

#include "kleidicv/arithmetics/exp_constants.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {
template <typename ScalarType, bool TryShortPath>
class Exp;

template <bool TryShortPath>
class Exp<float, TryShortPath> final : public UnrollOnce {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<float>;
  using VectorType = typename VecTraits::VectorType;

  VectorType vector_path(ContextType ctx,
                         VectorType src) KLEIDICV_STREAMING_COMPATIBLE {
    svfloat32_t n, r, poly, z;
    svuint32_t e;

    /* exp(x) = 2^n * poly(r), with poly(r) in [1/sqrt(2),sqrt(2)]
      x = ln2*n + r, with r in [-ln2/2, ln2/2].  */
    z = svmla_x(ctx.predicate(), svdup_f32(exp_f32::kShift), src,
                exp_f32::kInvLn2);
    n = svsub_x(ctx.predicate(), z, exp_f32::kShift);
    r = svmla_x(ctx.predicate(), src, n, -exp_f32::kLn2Hi);
    r = svmla_x(ctx.predicate(), r, n, -exp_f32::kLn2Lo);
    e = svlsl_x(ctx.predicate(), svreinterpret_u32(z), 23);
    poly = svmla_x(ctx.predicate(), svdup_f32(exp_f32::kPoly[1]),
                   svdup_f32(exp_f32::kPoly[0]), r);
    poly = svmla_x(ctx.predicate(), svdup_f32(exp_f32::kPoly[2]), poly, r);
    poly = svmla_x(ctx.predicate(), svdup_f32(exp_f32::kPoly[3]), poly, r);
    poly = svmla_x(ctx.predicate(), svdup_f32(exp_f32::kPoly[4]), poly, r);
    poly = svmla_x(ctx.predicate(), svdup_f32(1.0F), poly, r);
    poly = svmla_x(ctx.predicate(), svdup_f32(1.0F), poly, r);

    if constexpr (TryShortPath) {
      svbool_t cmp = svacgt(ctx.predicate(), n, 126.0F);
      if (KLEIDICV_UNLIKELY(svptest_any(ctx.predicate(), cmp))) {
        return specialcase(ctx.predicate(), poly, n, e);
      }
      svfloat32_t scale =
          svreinterpret_f32(svadd_x(ctx.predicate(), e, 0x3f800000U));
      return svmul_x(ctx.predicate(), scale, poly);
    }

    return specialcase(ctx.predicate(), poly, n, e);
  }

 private:
  static svfloat32_t specialcase(svbool_t pg, svfloat32_t poly, svfloat32_t n,
                                 svuint32_t e) KLEIDICV_STREAMING_COMPATIBLE {
    /* 2^n may overflow, break it up into s1*s2.  */
    svuint32_t b = svsel(svcmple(pg, n, svdup_f32(0.0F)),
                         svdup_u32(0x83000000U), svdup_u32(0.0F));
    svfloat32_t s1 = svreinterpret_f32(svadd_x(pg, b, 0x7f000000U));
    svfloat32_t s2 = svreinterpret_f32(svsub_x(pg, e, b));
    svbool_t cmp = svacgt(pg, n, 192.0F);
    svfloat32_t r1 = svmul_x(pg, s1, s1);
    svfloat32_t r0 = svmul_x(pg, s2, svmul_x(pg, poly, s1));

    return svsel(cmp, r1, r0);
  }
};  // end of class Exp<float>

template <typename T>
using ExpNoShortPath = Exp<T, false>;

template <typename T>
using ExpTryShortPath = Exp<T, true>;

template <typename T, typename Operation>
static kleidicv_error_t exp_sc(const T* src, size_t src_stride, T* dst,
                               size_t dst_stride, size_t width,
                               size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Operation operation;
  Rectangle rect{width, height};
  Rows<const T> src_rows{src, src_stride};
  Rows<T> dst_rows{dst, dst_stride};
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_EXP_SC_H
