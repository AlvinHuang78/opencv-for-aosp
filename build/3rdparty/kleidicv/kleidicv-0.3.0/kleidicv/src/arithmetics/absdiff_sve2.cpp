// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace kleidicv::sve2 {

template <typename ScalarType, typename VectorType,
          std::enable_if_t<std::is_signed<ScalarType>::value, bool> = true>
VectorType vector_path_impl(svbool_t pg, VectorType src_a, VectorType src_b) {
  // Results of SABD may be outside the signed range so use two
  // saturating instructions instead.
  return svqabs_x(pg, svqsub_m(pg, src_a, src_b));
}

template <typename ScalarType, typename VectorType,
          std::enable_if_t<std::is_unsigned<ScalarType>::value, bool> = true>
VectorType vector_path_impl(svbool_t pg, VectorType src_a, VectorType src_b) {
  return svabd_m(pg, src_a, src_b);
}

template <typename ScalarType>
class SaturatingAbsDiff final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  VectorType vector_path(ContextType ctx, VectorType src_a, VectorType src_b) {
    return vector_path_impl<ScalarType>(ctx.predicate(), src_a, src_b);
  }
};  // end of class SaturatingAbsDiff<ScalarType>

template <typename T>
kleidicv_error_t saturating_absdiff(const T *src_a, size_t src_a_stride,
                                    const T *src_b, size_t src_b_stride, T *dst,
                                    size_t dst_stride, size_t width,
                                    size_t height) {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  SaturatingAbsDiff<T> operation;
  Rectangle rect{width, height};
  Rows<const T> src_a_rows{src_a, src_a_stride};
  Rows<const T> src_b_rows{src_b, src_b_stride};
  Rows<T> dst_rows{dst, dst_stride};
  apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows, dst_rows);
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                    \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t saturating_absdiff<type>( \
      const type *src_a, size_t src_a_stride, const type *src_b,               \
      size_t src_b_stride, type *dst, size_t dst_stride, size_t width,         \
      size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);

}  // namespace kleidicv::sve2
