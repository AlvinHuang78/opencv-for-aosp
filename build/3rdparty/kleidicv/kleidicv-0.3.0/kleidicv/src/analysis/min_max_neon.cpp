// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <typename ScalarType>
class MinMax final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  MinMax()
      : vmin_(vdupq_n(std::numeric_limits<ScalarType>::max())),
        vmax_(vdupq_n(std::numeric_limits<ScalarType>::lowest())),
        min_(std::numeric_limits<ScalarType>::max()),
        max_(std::numeric_limits<ScalarType>::lowest()) {}

  void vector_path(VectorType src) {
    vmin_ = vminq(vmin_, src);
    vmax_ = vmaxq(vmax_, src);
  }

  void scalar_path(ScalarType src) {
    min_ = std::min(min_, src);
    max_ = std::max(max_, src);
  }

  ScalarType get_min() const {
    return std::min<ScalarType>(min_, vminvq(vmin_));
  }

  ScalarType get_max() const {
    return std::max<ScalarType>(max_, vmaxvq(vmax_));
  }

 private:
  VectorType vmin_, vmax_;
  ScalarType min_, max_;
};  // end of class MinMax<T>

template <typename ScalarType>
kleidicv_error_t min_max(const ScalarType *src, size_t src_stride, size_t width,
                         size_t height, ScalarType *min_value,
                         ScalarType *max_value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  if (KLEIDICV_UNLIKELY(width == 0 || height == 0)) {
    return KLEIDICV_ERROR_RANGE;
  }

  Rectangle rect{width, height};
  Rows<const ScalarType> src_rows{src, src_stride};
  MinMax<ScalarType> operation;
  apply_operation_by_rows(operation, rect, src_rows);
  if (min_value) {
    *min_value = operation.get_min();
  }
  if (max_value) {
    *max_value = operation.get_max();
  }
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t min_max<type>(    \
      const type *src, size_t src_stride, size_t width, size_t height, \
      type *min_value, type *max_value)

KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  //  namespace kleidicv::neon
