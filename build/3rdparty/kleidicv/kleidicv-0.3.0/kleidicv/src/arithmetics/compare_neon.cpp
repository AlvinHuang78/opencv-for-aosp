// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <typename ScalarType>
class ComparatorEqual : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  VectorType vector_path(VectorType src_a, VectorType src_b) {
    return vceqq_u8(src_a, src_b);
  }

  // NOLINTBEGIN(readability-make-member-function-const)
  ScalarType scalar_path(ScalarType src_a, ScalarType src_b) {
    return src_a == src_b ? 255 : 0;
  }
  // NOLINTEND(readability-make-member-function-const)
};  // end of class ComparatorEqual

template <typename ScalarType>
class ComparatorGreater : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  VectorType vector_path(VectorType src_a, VectorType src_b) {
    return vcgtq_u8(src_a, src_b);
  }

  // NOLINTBEGIN(readability-make-member-function-const)
  ScalarType scalar_path(ScalarType src_a, ScalarType src_b) {
    return src_a > src_b ? 255 : 0;
  }
  // NOLINTEND(readability-make-member-function-const)
};  // end of class ComparatorGreater

template <typename Comparator, typename ScalarType>
static kleidicv_error_t compare(const ScalarType *src_a, size_t src_a_stride,
                                const ScalarType *src_b, size_t src_b_stride,
                                ScalarType *dst, size_t dst_stride,
                                size_t width, size_t height) {
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

template <typename ScalarType>
kleidicv_error_t compare_equal(const ScalarType *src_a, size_t src_a_stride,
                               const ScalarType *src_b, size_t src_b_stride,
                               ScalarType *dst, size_t dst_stride, size_t width,
                               size_t height) {
  return compare<ComparatorEqual<ScalarType>>(
      src_a, src_a_stride, src_b, src_b_stride, dst, dst_stride, width, height);
}

template <typename ScalarType>
kleidicv_error_t compare_greater(const ScalarType *src_a, size_t src_a_stride,
                                 const ScalarType *src_b, size_t src_b_stride,
                                 ScalarType *dst, size_t dst_stride,
                                 size_t width, size_t height) {
  return compare<ComparatorGreater<ScalarType>>(
      src_a, src_a_stride, src_b, src_b_stride, dst, dst_stride, width, height);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(name, stype)                      \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t name<stype>(       \
      const stype *src_a, size_t src_a_stride, const stype *src_b,      \
      size_t src_b_stride, stype *dst, size_t dst_stride, size_t width, \
      size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(compare_equal, uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(compare_greater, uint8_t);

}  // namespace kleidicv::neon
