// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <typename ScalarType>
class SaturatingAddAbsWithThreshold final : public UnrollOnce,
                                            public UnrollTwice,
                                            public TryToAvoidTailLoop {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  explicit SaturatingAddAbsWithThreshold(ScalarType threshold)
      : threshold_{threshold}, threshold_vec_{vdupq_n_s16(threshold)} {}

  VectorType vector_path(VectorType src_a, VectorType src_b) {
    VectorType add_abs = vqaddq_s16(vqabsq_s16(src_a), vqabsq_s16(src_b));
    return vandq_s16(add_abs, vcgtq_s16(add_abs, threshold_vec_));
  }

  ScalarType scalar_path(ScalarType src_a, ScalarType src_b) {
    ScalarType add_abs = 0;

    if (__builtin_add_overflow(saturate_abs(src_a), saturate_abs(src_b),
                               &add_abs)) {
      add_abs = std::numeric_limits<ScalarType>::max();
    }
    return add_abs > threshold_ ? add_abs : 0;
  }

 private:
  ScalarType saturate_abs(ScalarType input) {
    if (std::numeric_limits<ScalarType>::is_signed &&
        input == std::numeric_limits<ScalarType>::lowest()) {
      return std::numeric_limits<ScalarType>::max();
    }
    return std::abs(input);
  }

  ScalarType threshold_;
  VectorType threshold_vec_;
};  // end of class SaturatingAddAbsWithThreshold<ScalarType>

template <typename T>
kleidicv_error_t saturating_add_abs_with_threshold(
    const T *src_a, size_t src_a_stride, const T *src_b, size_t src_b_stride,
    T *dst, size_t dst_stride, size_t width, size_t height, T threshold) {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  SaturatingAddAbsWithThreshold<T> operation{threshold};
  Rectangle rect{width, height};
  Rows<const T> src_a_rows{src_a, src_a_stride};
  Rows<const T> src_b_rows{src_b, src_b_stride};
  Rows<T> dst_rows{dst, dst_stride};
  apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows, dst_rows);
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                   \
  saturating_add_abs_with_threshold<type>(                             \
      const type *src_a, size_t src_a_stride, const type *src_b,       \
      size_t src_b_stride, type *dst, size_t dst_stride, size_t width, \
      size_t height, type threshold)

KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);

}  // namespace kleidicv::neon
