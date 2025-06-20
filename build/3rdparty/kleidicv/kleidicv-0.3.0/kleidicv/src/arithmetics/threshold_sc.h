// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_THRESHOLD_SC_H
#define KLEIDICV_THRESHOLD_SC_H

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType>
class BinaryThreshold final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  BinaryThreshold(ScalarType threshold,
                  ScalarType value) KLEIDICV_STREAMING_COMPATIBLE
      : threshold_(threshold),
        value_(value) {}

  VectorType vector_path(ContextType ctx,
                         VectorType src) KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t predicate = svcmpgt(ctx.predicate(), src, threshold_);
    return svsel_u8(predicate, svdup_u8(value_), svdup_u8(0));
  }

 private:
  ScalarType threshold_;
  ScalarType value_;
};  // end of class BinaryThreshold<ScalarType>

template <typename T>
kleidicv_error_t threshold_binary_sc(const T *src, size_t src_stride, T *dst,
                                     size_t dst_stride, size_t width,
                                     size_t height, T threshold,
                                     T value) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const T> src_rows{src, src_stride};
  Rows<T> dst_rows{dst, dst_stride};
  BinaryThreshold<T> operation{threshold, value};
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_THRESHOLD_SC_H
