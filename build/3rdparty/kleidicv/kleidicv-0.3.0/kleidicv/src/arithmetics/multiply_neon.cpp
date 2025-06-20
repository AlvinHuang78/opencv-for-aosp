// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/types.h"

namespace kleidicv::neon {

template <typename ScalarType>
class SaturatingMultiply final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  explicit SaturatingMultiply(double scale = 1.0) : scale_{scale} {};

  VectorType vector_path(VectorType src_a, VectorType src_b) {
    VectorType result;

    // multiply-widen high part of the vectors.
    // results in e.g. int16x8 -> int32x4
    auto high_part = vmull_high(src_a, src_b);

    // get low part of the vectors and multiply-widen these.
    auto low_part = vmull(vget_low(src_a), vget_low(src_b));

    // narrow-saturate low_part back to int16x4
    // narrow-saturate high_part back to int16x4
    // and stitch them back together into int16x8
    result = vqmovn_high(vqmovn(low_part), high_part);

    /* TODO: figure out the way to multiply by double or some
      fixed supported scale. Note that vmulq_n does not support 8x8 vectors.
      */
    // result = vmulq_n(result, this->scale_);

    return result;
  }

  ScalarType scalar_path(ScalarType src_a, ScalarType src_b) {
    ScalarType result;
    if (std::numeric_limits<ScalarType>::is_signed) {
      if (__builtin_mul_overflow(src_a, src_b, &result)) {
        return (src_a < 0 && src_b > 0) || (src_a > 0 && src_b < 0)
                   ? std::numeric_limits<ScalarType>::lowest()
                   : std::numeric_limits<ScalarType>::max();
      }
      return result;
    }

    if (__builtin_mul_overflow(src_a, src_b, &result)) {
      return std::numeric_limits<ScalarType>::max();
    }
    return result;
  }

 private:
  double scale_;
};

template <typename T>
kleidicv_error_t saturating_multiply(const T *src_a, size_t src_a_stride,
                                     const T *src_b, size_t src_b_stride,
                                     T *dst, size_t dst_stride, size_t width,
                                     size_t height, double scale) {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  (void)scale;  // TODO: figure out the way to process the scale.
  SaturatingMultiply<T> operation;
  Rectangle rect{width, height};
  Rows<const T> src_a_rows{src_a, src_a_stride};
  Rows<const T> src_b_rows{src_b, src_b_stride};
  Rows<T> dst_rows{dst, dst_stride};
  neon::apply_operation_by_rows(operation, rect, src_a_rows, src_b_rows,
                                dst_rows);
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                    \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                           \
  saturating_multiply<type>(const type *src_a, size_t src_a_stride,            \
                            const type *src_b, size_t src_b_stride, type *dst, \
                            size_t dst_stride, size_t width, size_t height,    \
                            double scale)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);

}  // namespace kleidicv::neon
