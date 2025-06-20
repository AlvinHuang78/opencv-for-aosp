// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace kleidicv::sme2 {

template <typename ScalarType>
class SaturatingAdd final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  VectorType vector_path(ContextType ctx, VectorType src_a,
                         VectorType src_b) KLEIDICV_STREAMING_COMPATIBLE {
    return svqadd_m(ctx.predicate(), src_a, src_b);
  }
};  // end of class SaturatingAdd<ScalarType>

template <typename T>
KLEIDICV_LOCALLY_STREAMING kleidicv_error_t saturating_add(
    const T *src_a, size_t src_a_stride, const T *src_b, size_t src_b_stride,
    T *dst, size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  SaturatingAdd<T> operation;
  Rectangle rect{width, height};
  Rows<const T> src_a_rows{src_a, src_a_stride};
  Rows<const T> src_b_rows{src_b, src_b_stride};
  Rows<T> dst_rows{dst, dst_stride};
  apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows, dst_rows);
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t saturating_add<type>( \
      const type *src_a, size_t src_a_stride, const type *src_b,           \
      size_t src_b_stride, type *dst, size_t dst_stride, size_t width,     \
      size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int64_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint64_t);

}  // namespace kleidicv::sme2
