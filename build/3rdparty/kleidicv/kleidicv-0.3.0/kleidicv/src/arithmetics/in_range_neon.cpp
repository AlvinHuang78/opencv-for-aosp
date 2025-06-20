// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <typename ScalarType>
class InRange;

template <>
class InRange<uint8_t> : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<uint8_t>;
  using VectorType = typename VecTraits::VectorType;

  InRange(uint8_t lower_bound, uint8_t upper_bound)
      : lower_bound_vect_{vdupq_n(lower_bound)},
        upper_bound_vect_{vdupq_n(upper_bound)},
        lower_bound_{lower_bound},
        upper_bound_{upper_bound} {}

  VectorType vector_path(VectorType src) {
    return vandq(vcgeq(src, lower_bound_vect_), vcleq(src, upper_bound_vect_));
  }

  // NOLINTBEGIN(readability-make-member-function-const)
  uint8_t scalar_path(uint8_t src) {
    return (src >= lower_bound_ && src <= upper_bound_) ? 0xFF : 0;
  }
  // NOLINTEND(readability-make-member-function-const)

 private:
  VectorType lower_bound_vect_;
  VectorType upper_bound_vect_;
  uint8_t lower_bound_;
  uint8_t upper_bound_;
};  // end of class InRange<uint8_t>

template <>
class InRange<float> {
 public:
  using SrcVecTraits = neon::VecTraits<float>;
  using SrcVectorType = typename SrcVecTraits::VectorType;
  using SrcVector4Type = typename SrcVecTraits::Vector4Type;
  using DstVecTraits = neon::VecTraits<uint8_t>;
  using DstVectorType = typename DstVecTraits::VectorType;

  InRange(float lower_bound, float upper_bound)
      : lower_bound_vect_{vdupq_n(lower_bound)},
        upper_bound_vect_{vdupq_n(upper_bound)},
        lower_bound_{lower_bound},
        upper_bound_{upper_bound} {}

  void process_row(size_t width, Columns<const float> src,
                   Columns<uint8_t> dst) {
    LoopUnroll{width, SrcVecTraits::num_lanes()}
        .unroll_n_times<4>([&](size_t step) {
          SrcVector4Type src_vector = vld1q_f32_x4(&src[0]);
          DstVectorType result_vector = vector_path(src_vector);
          vst1q(&dst[0], result_vector);
          src += ptrdiff_t(step);
          dst += ptrdiff_t(step);
        })
        .remaining([&](size_t length, size_t) {
          for (size_t index = 0; index < length; ++index) {
            disable_loop_vectorization();
            float f = src[ptrdiff_t(index)];
            dst[ptrdiff_t(index)] =
                (f >= lower_bound_ && f <= upper_bound_) ? 0xFF : 0;
          }
        });
  }

 private:
  DstVectorType vector_path(SrcVector4Type src) {
    SrcVectorType src0 = src.val[0];
    SrcVectorType src1 = src.val[1];
    SrcVectorType src2 = src.val[2];
    SrcVectorType src3 = src.val[3];
    uint32x4_t res00 =
        vandq(vcgeq(src0, lower_bound_vect_), vcleq(src0, upper_bound_vect_));
    uint32x4_t res01 =
        vandq(vcgeq(src1, lower_bound_vect_), vcleq(src1, upper_bound_vect_));
    uint32x4_t res11 =
        vandq(vcgeq(src2, lower_bound_vect_), vcleq(src2, upper_bound_vect_));
    uint32x4_t res12 =
        vandq(vcgeq(src3, lower_bound_vect_), vcleq(src3, upper_bound_vect_));
    // AND-ing the results of the compare ops sets all 32 bits to all 0's or all
    // 1's. Unzipping them twice chooses 8 bits from those 32.
    uint16x8_t res0 =
        vuzp1q_u16(vreinterpretq_u16_u32(res00), vreinterpretq_u16_u32(res01));
    uint16x8_t res1 =
        vuzp1q_u16(vreinterpretq_u16_u32(res11), vreinterpretq_u16_u32(res12));
    return vuzp1q_u8(vreinterpretq_u8_u16(res0), vreinterpretq_u8_u16(res1));
  }

  SrcVectorType lower_bound_vect_;
  SrcVectorType upper_bound_vect_;
  float lower_bound_;
  float upper_bound_;
};  // end of class InRange<float>

template <typename T>
kleidicv_error_t in_range(const T *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          T lower_bound, T upper_bound) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  InRange<T> operation{lower_bound, upper_bound};
  Rectangle rect{width, height};
  Rows<const T> src_rows{src, src_stride};
  Rows<uint8_t> dst_rows{dst, dst_stride};

  if constexpr (std::is_same_v<T, uint8_t>) {
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  } else {
    zip_rows(operation, rect, src_rows, dst_rows);
  }

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t in_range<type>(       \
      const type *src, size_t src_stride, uint8_t *dst, size_t dst_stride, \
      size_t width, size_t height, type lower_bound, type upper_bound)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::neon
