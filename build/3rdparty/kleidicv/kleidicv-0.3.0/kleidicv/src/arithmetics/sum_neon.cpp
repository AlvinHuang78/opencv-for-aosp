// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/utils.h"

namespace kleidicv::neon {

template <typename ScalarType>
class Sum final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  VectorType vector_sum;
  ScalarType scalar_sum;

  Sum() : vector_sum(VectorType{0}), scalar_sum(0) {}

  void vector_path(VectorType src) { vector_sum = vaddq(src, vector_sum); }

  void scalar_path(ScalarType src) { scalar_sum += src; }

  ScalarType get_sum() { return vaddvq(vector_sum) + scalar_sum; }
};

template <typename ScalarType>
kleidicv_error_t sum(const ScalarType *src, size_t src_stride, size_t width,
                     size_t height, ScalarType *sum) {
  CHECK_POINTERS(sum);
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const ScalarType> src_rows{src, src_stride};
  Sum<ScalarType> operation;
  apply_operation_by_rows(operation, rect, src_rows);

  *sum = operation.get_sum();

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t sum<type>(        \
      const type *src, size_t src_stride, size_t width, size_t height, \
      type *sum)

KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::neon
