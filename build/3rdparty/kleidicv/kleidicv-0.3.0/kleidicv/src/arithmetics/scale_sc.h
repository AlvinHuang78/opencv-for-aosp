// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SCALE_SC_H
#define KLEIDICV_SCALE_SC_H

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

class AddFloat final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<float>;
  using VectorType = typename VecTraits::VectorType;

  explicit AddFloat(const svfloat32_t &svshift) KLEIDICV_STREAMING_COMPATIBLE
      : svshift_{svshift} {}

  // NOLINTBEGIN(readability-make-member-function-const)
  VectorType vector_path(ContextType ctx,
                         VectorType src) KLEIDICV_STREAMING_COMPATIBLE {
    return svadd_x(ctx.predicate(), src, svshift_);
  }
  // NOLINTEND(readability-make-member-function-const)

 private:
  const svfloat32_t &svshift_;
};  // end of class AddFloat

class ScaleFloat final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<float>;
  using VectorType = typename VecTraits::VectorType;

  ScaleFloat(const svfloat32_t &svscale,
             const svfloat32_t &svshift) KLEIDICV_STREAMING_COMPATIBLE
      : svscale_{svscale},
        svshift_{svshift} {}

  // NOLINTBEGIN(readability-make-member-function-const)
  VectorType vector_path(ContextType ctx,
                         VectorType src) KLEIDICV_STREAMING_COMPATIBLE {
    return svmla_x(ctx.predicate(), svshift_, src, svscale_);
  }
  // NOLINTEND(readability-make-member-function-const)

 private:
  const svfloat32_t &svscale_, &svshift_;
};  // end of class ScaleFloat

template <typename T>
kleidicv_error_t scale_sc(const T *src, size_t src_stride, T *dst,
                          size_t dst_stride, size_t width, size_t height,
                          float scale,
                          float shift) KLEIDICV_STREAMING_COMPATIBLE;

// Specialization for float
template <>
kleidicv_error_t scale_sc(const float *src, size_t src_stride, float *dst,
                          size_t dst_stride, size_t width, size_t height,
                          float scale,
                          float shift) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const float> src_rows{src, src_stride};
  Rows<float> dst_rows{dst, dst_stride};
  svfloat32_t svscale = svdup_f32(scale);
  svfloat32_t svshift = svdup_f32(shift);
  if (scale == 1.0) {
    AddFloat operation(svshift);
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  } else {
    ScaleFloat operation(svscale, svshift);
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  }
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SCALE_SC_H
