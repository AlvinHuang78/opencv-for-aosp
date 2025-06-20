// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <typename ScalarType>
class BinaryThreshold final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  BinaryThreshold(ScalarType threshold, ScalarType value)
      : threshold_vect_{vdupq_n_u8(threshold)},
        value_vect_{vdupq_n_u8(value)},
        zero_vect_{vdupq_n_u8(0)},
        threshold_{threshold},
        value_{value} {}

  VectorType vector_path(VectorType src) {
    VectorType predicate = vcgtq_u8(src, threshold_vect_);
    return vbslq_u8(predicate, value_vect_, zero_vect_);
  }

  ScalarType scalar_path(ScalarType src) {
    return src > threshold_ ? value_ : 0;
  }

 private:
  VectorType threshold_vect_;
  VectorType value_vect_;
  VectorType zero_vect_;
  ScalarType threshold_;
  ScalarType value_;
};  // end of class BinaryThreshold<ScalarType>

template <typename T>
kleidicv_error_t threshold_binary(const T *src, size_t src_stride, T *dst,
                                  size_t dst_stride, size_t width,
                                  size_t height, T threshold, T value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const T> src_rows{src, src_stride};
  Rows<T> dst_rows{dst, dst_stride};
  BinaryThreshold<T> operation{threshold, value};
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                  \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t threshold_binary<type>( \
      const type *src, size_t src_stride, type *dst, size_t dst_stride,      \
      size_t width, size_t height, type threshold, type value)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);

}  // namespace kleidicv::neon
