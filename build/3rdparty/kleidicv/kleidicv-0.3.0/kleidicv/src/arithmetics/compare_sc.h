// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_COMPARE_SC_H
#define KLEIDICV_COMPARE_SC_H

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType>
class ComparatorEqual : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using SignedScalarType = typename std::make_signed<ScalarType>::type;
  using SignedVecTraits =
      KLEIDICV_TARGET_NAMESPACE::VecTraits<SignedScalarType>;
  using SignedVectorType = typename SignedVecTraits::VectorType;

  // NOLINTBEGIN(readability-make-member-function-const)
  VectorType vector_path(ContextType ctx, VectorType src_a,
                         VectorType src_b) KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t pg = ctx.predicate();
    svbool_t predicate = svcmpeq(pg, src_a, src_b);
    return svsel(predicate, VecTraits::svdup(255), VecTraits::svdup(0));
  }
  // NOLINTEND(readability-make-member-function-const)
};  // end of class ComparatorEqual

template <typename ScalarType>
class ComparatorGreater : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using SignedScalarType = typename std::make_signed<ScalarType>::type;
  using SignedVecTraits =
      KLEIDICV_TARGET_NAMESPACE::VecTraits<SignedScalarType>;
  using SignedVectorType = typename SignedVecTraits::VectorType;

  // NOLINTBEGIN(readability-make-member-function-const)
  VectorType vector_path(ContextType ctx, VectorType src_a,
                         VectorType src_b) KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t pg = ctx.predicate();
    svbool_t predicate = svcmpgt(pg, src_a, src_b);
    return svsel(predicate, VecTraits::svdup(255), VecTraits::svdup(0));
  }
  // NOLINTEND(readability-make-member-function-const)
};  // end of class ComparatorGreater

template <typename Comparator, typename ScalarType>
kleidicv_error_t compare_sc(const ScalarType *src_a, size_t src_a_stride,
                            const ScalarType *src_b, size_t src_b_stride,
                            ScalarType *dst, size_t dst_stride, size_t width,
                            size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Comparator operation{};
  Rectangle rect{width, height};
  Rows<const ScalarType> src_a_rows{src_a, src_a_stride};
  Rows<const ScalarType> src_b_rows{src_b, src_b_stride};
  Rows<ScalarType> dst_rows{dst, dst_stride};

  apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows, dst_rows);

  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_COMPARE_SC_H
