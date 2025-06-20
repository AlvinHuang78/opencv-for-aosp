// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <typename ScalarType>
class BitwiseAnd final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  VectorType vector_path(VectorType src_a, VectorType src_b) {
    return vandq(src_a, src_b);
  }

  ScalarType scalar_path(ScalarType src_a, ScalarType src_b) {
    return src_a & src_b;
  }
};  // end of class BitwiseAnd<ScalarType>

template <typename T>
kleidicv_error_t bitwise_and(const T *src_a, size_t src_a_stride,
                             const T *src_b, size_t src_b_stride, T *dst,
                             size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  BitwiseAnd<T> operation;
  Rectangle rect{width, height};
  Rows<const T> src_a_rows{src_a, src_a_stride};
  Rows<const T> src_b_rows{src_b, src_b_stride};
  Rows<T> dst_rows{dst, dst_stride};
  neon::apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows,
                                dst_rows);
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                             \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t bitwise_and<type>( \
      const type *src_a, size_t src_a_stride, const type *src_b,        \
      size_t src_b_stride, type *dst, size_t dst_stride, size_t width,  \
      size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);

}  // namespace kleidicv::neon
