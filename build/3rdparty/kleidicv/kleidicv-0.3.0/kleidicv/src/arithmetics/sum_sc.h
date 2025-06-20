// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SUM_SC_H
#define KLEIDICV_SUM_SC_H

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"
#include "kleidicv/utils.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType>
class Sum final : public UnrollTwice {
 public:
  using ContextType = Context;
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  explicit Sum(VectorType &accumulator) KLEIDICV_STREAMING_COMPATIBLE
      : accumulator_{accumulator} {
    accumulator_ = VecTraits::svdup(0);
  }

  void vector_path(ContextType ctx,
                   VectorType src) KLEIDICV_STREAMING_COMPATIBLE {
    accumulator_ = svadd_m(ctx.predicate(), accumulator_, src);
  }

  ScalarType get_sum() KLEIDICV_STREAMING_COMPATIBLE {
    ScalarType accumulator_final[VecTraits::max_num_lanes()] = {0};
    svst1(VecTraits::svptrue(), accumulator_final, accumulator_);

    ScalarType sum = 0;
    for (size_t i = 0; i != VecTraits::num_lanes(); ++i) {
      sum += accumulator_final[i];
    }
    return sum;
  }

 private:
  VectorType &accumulator_;
};

template <typename ScalarType>
kleidicv_error_t sum_sc(const ScalarType *src, size_t src_stride, size_t width,
                        size_t height,
                        ScalarType *sum) KLEIDICV_STREAMING_COMPATIBLE {
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  CHECK_POINTERS(sum);
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const ScalarType> src_rows{src, src_stride};

  VectorType accumulator;
  Sum<ScalarType> operation{accumulator};

  apply_operation_by_rows(operation, rect, src_rows);

  *sum = operation.get_sum();

  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SUM_SC_H
