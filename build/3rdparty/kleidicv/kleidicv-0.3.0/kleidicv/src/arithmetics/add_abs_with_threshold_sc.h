// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SATURATING_ADD_ABS_WITH_THRESHOLD_SC_H
#define KLEIDICV_SATURATING_ADD_ABS_WITH_THRESHOLD_SC_H

#include <limits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType>
class SaturatingAddAbsWithThreshold final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  explicit SaturatingAddAbsWithThreshold(ScalarType threshold)
      KLEIDICV_STREAMING_COMPATIBLE : threshold_(threshold) {}

  VectorType vector_path(ContextType ctx, VectorType src_a,
                         VectorType src_b) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    VectorType add_abs = svqadd_x(pg, svqabs_x(pg, src_a), svqabs_x(pg, src_b));
    svbool_t predicate = svcmpgt(pg, add_abs, threshold_);
    return svsel(predicate, add_abs, VecTraits::svdup(0));
  }

 private:
  ScalarType threshold_;
};  // end of class SaturatingAddAbsWithThreshold<ScalarType>

template <typename T>
kleidicv_error_t saturating_add_abs_with_threshold_sc(
    const T *src_a, size_t src_a_stride, const T *src_b, size_t src_b_stride,
    T *dst, size_t dst_stride, size_t width, size_t height,
    T threshold) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  SaturatingAddAbsWithThreshold<T> operation{threshold};
  Rectangle rect{width, height};
  Rows<const T> src_a_rows{src_a, src_a_stride};
  Rows<const T> src_b_rows{src_b, src_b_stride};
  Rows<T> dst_rows{dst, dst_stride};
  apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows, dst_rows);
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SATURATING_ADD_ABS_WITH_THRESHOLD_SC_H
