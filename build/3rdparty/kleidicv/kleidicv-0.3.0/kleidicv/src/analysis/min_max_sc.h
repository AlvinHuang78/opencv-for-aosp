// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_MIN_MAX_SC_H
#define KLEIDICV_MIN_MAX_SC_H

#include <limits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename ScalarType>
class MinMax final : public UnrollTwice {
 public:
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using ContextType = Context;

  MinMax(VectorType &vmin, VectorType &vmax) KLEIDICV_STREAMING_COMPATIBLE
      : vmin_{vmin},
        vmax_{vmax} {}

  void vector_path(ContextType ctx,
                   VectorType src) KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = ctx.predicate();
    vmin_ = svmin_m(pg, vmin_, src);
    vmax_ = svmax_m(pg, vmax_, src);
  }

  ScalarType get_min() const KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = VecTraits::svptrue();
    return svminv(pg, vmin_);
  }

  ScalarType get_max() const KLEIDICV_STREAMING_COMPATIBLE {
    auto pg = VecTraits::svptrue();
    return svmaxv(pg, vmax_);
  }

 private:
  VectorType &vmin_, &vmax_;
};  // end of class MinMax<T>

template <typename ScalarType>
kleidicv_error_t min_max_sc(const ScalarType *src, size_t src_stride,
                            size_t width, size_t height, ScalarType *min_value,
                            ScalarType *max_value)
    KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  if (KLEIDICV_UNLIKELY(width == 0 || height == 0)) {
    return KLEIDICV_ERROR_RANGE;
  }

  Rectangle rect{width, height};
  Rows<const ScalarType> src_rows{src, src_stride};

  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  VectorType vmin = VecTraits::svdup(std::numeric_limits<ScalarType>::max());
  VectorType vmax = VecTraits::svdup(std::numeric_limits<ScalarType>::lowest());

  MinMax<ScalarType> operation{vmin, vmax};

  apply_operation_by_rows(operation, rect, src_rows);
  if (min_value) {
    *min_value = operation.get_min();
  }
  if (max_value) {
    *max_value = operation.get_max();
  }
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_MIN_MAX_SC_H
