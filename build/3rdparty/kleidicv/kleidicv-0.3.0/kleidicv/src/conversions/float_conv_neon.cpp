// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "kleidicv/conversions/float_conversion.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <typename InputType, typename OutputType>
class float_conversion_operation;

template <typename OutputType>
class float_conversion_operation<float, OutputType> {
 public:
  using SrcVecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<float>;
  using SrcVectorType = typename SrcVecTraits::VectorType;
  using IntermediateVecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<
      std::conditional_t<std::is_signed_v<OutputType>, int32_t, uint32_t>>;
  using IntermediateVectorType = typename IntermediateVecTraits::VectorType;

  void process_row(size_t width, Columns<const float> src,
                   Columns<OutputType> dst) {
    LoopUnroll{width, SrcVecTraits::num_lanes()}
        .unroll_twice([&](size_t step) {
          SrcVectorType src_vector1 = vld1q_f32(&src[0]);
          SrcVectorType src_vector2 =
              vld1q_f32(&src[SrcVecTraits::num_lanes()]);
          IntermediateVectorType result_vector1 =
              vector_path<OutputType>(src_vector1);
          IntermediateVectorType result_vector2 =
              vector_path<OutputType>(src_vector2);
          vst1(&dst[0], vqmovn(vcombine(vqmovn(result_vector1),
                                        vqmovn(result_vector2))));
          src += ptrdiff_t(step);
          dst += ptrdiff_t(step);
        })
        .remaining([&](size_t length, size_t) {
          for (size_t index = 0; index < length; ++index) {
            disable_loop_vectorization();
            float f = std::nearbyint(src[ptrdiff_t(index)]);
            if (f > std::numeric_limits<OutputType>::max()) {
              f = std::numeric_limits<OutputType>::max();
            } else if (f < std::numeric_limits<OutputType>::lowest()) {
              f = std::numeric_limits<OutputType>::lowest();
            }
            dst[index] = static_cast<OutputType>(f);
          }
        });
  }

 private:
  template <
      typename O,
      std::enable_if_t<std::is_integral_v<O> && std::is_signed_v<O>, int> = 0>
  IntermediateVectorType vector_path(SrcVectorType src) {
    IntermediateVectorType result = vcvtnq_s32_f32(src);
    return result;
  }

  template <
      typename O,
      std::enable_if_t<std::is_integral_v<O> && !std::is_signed_v<O>, int> = 0>
  IntermediateVectorType vector_path(SrcVectorType src) {
    IntermediateVectorType result = vcvtnq_u32_f32(src);
    return result;
  }
};  // end of class float_conversion_operation<float, OutputType>

template <typename InputType, typename OutputType>
kleidicv_error_t float_conversion(const InputType* src, size_t src_stride,
                                  OutputType* dst, size_t dst_stride,
                                  size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  float_conversion_operation<InputType, OutputType> operation;
  Rectangle rect{width, height};
  Rows<const InputType> src_rows{src, src_stride};
  Rows<OutputType> dst_rows{dst, dst_stride};
  zip_rows(operation, rect, src_rows, dst_rows);

  return KLEIDICV_OK;
}

kleidicv_error_t f32_to_s8(const float* src, size_t src_stride, int8_t* dst,
                           size_t dst_stride, size_t width, size_t height) {
  return float_conversion(src, src_stride, dst, dst_stride, width, height);
}

kleidicv_error_t f32_to_u8(const float* src, size_t src_stride, uint8_t* dst,
                           size_t dst_stride, size_t width, size_t height) {
  return float_conversion(src, src_stride, dst, dst_stride, width, height);
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
s8_to_f32(const int8_t* src, size_t src_stride, float* dst, size_t dst_stride,
          size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  // Indices used with the TBL instruction to widen from 8-bit to 32-bit in a
  // single instruction.
  const uint8x16_t index0 = vcombine_u8(vcreate_u8(0x01ffffff00ffffffULL),
                                        vcreate_u8(0x03ffffff02ffffffULL));
  const uint8x16_t index1 = vcombine_u8(vcreate_u8(0x05ffffff04ffffffULL),
                                        vcreate_u8(0x07ffffff06ffffffULL));
  const uint8x16_t index2 = vcombine_u8(vcreate_u8(0x09ffffff08ffffffULL),
                                        vcreate_u8(0x0bffffff0affffffULL));
  const uint8x16_t index3 = vcombine_u8(vcreate_u8(0x0dffffff0cffffffULL),
                                        vcreate_u8(0x0fffffff0effffffULL));
  for (size_t y = 0; y != height; ++y) {
    size_t x = 0;
    for (; x + 16 <= width; x += 16) {
      int8x16_t input = vld1q(src + x);
      // Widen from 8-bit to 32-bit and shift right 24 bits instead of
      // sign-extending.
      int32x4_t a = vreinterpretq_s32_s8(vqtbl1q_s8(input, index0));
      int32x4_t b = vreinterpretq_s32_s8(vqtbl1q_s8(input, index1));
      int32x4_t c = vreinterpretq_s32_s8(vqtbl1q_s8(input, index2));
      int32x4_t d = vreinterpretq_s32_s8(vqtbl1q_s8(input, index3));
      // Convert to float and divide by 2^24.
      float32x4x4_t output = {
          vcvtq_n_f32_s32(a, 24),
          vcvtq_n_f32_s32(b, 24),
          vcvtq_n_f32_s32(c, 24),
          vcvtq_n_f32_s32(d, 24),
      };
      vst1q_f32_x4(dst + x, output);
    }
    for (; x != width; ++x) {
      disable_loop_vectorization();
      dst[x] = src[x];
    }

    src += src_stride / sizeof(*src);
    dst += dst_stride / sizeof(*dst);
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
u8_to_f32(const uint8_t* src, size_t src_stride, float* dst, size_t dst_stride,
          size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  // Indices used with the TBL instruction to widen from 8-bit to 32-bit in a
  // single instruction.
  const uint8x16_t index0 = vcombine_u8(vcreate_u8(0xffffff01ffffff00ULL),
                                        vcreate_u8(0xffffff03ffffff02ULL));
  const uint8x16_t index1 = vcombine_u8(vcreate_u8(0xffffff05ffffff04ULL),
                                        vcreate_u8(0xffffff07ffffff06ULL));
  const uint8x16_t index2 = vcombine_u8(vcreate_u8(0xffffff09ffffff08ULL),
                                        vcreate_u8(0xffffff0bffffff0aULL));
  const uint8x16_t index3 = vcombine_u8(vcreate_u8(0xffffff0dffffff0cULL),
                                        vcreate_u8(0xffffff0fffffff0eULL));
  for (size_t y = 0; y != height; ++y) {
    size_t x = 0;
    for (; x + 16 <= width; x += 16) {
      uint8x16_t input = vld1q(src + x);
      // Widen from 8-bit to 32-bit
      uint32x4_t a = vreinterpretq_u32_u8(vqtbl1q_u8(input, index0));
      uint32x4_t b = vreinterpretq_u32_u8(vqtbl1q_u8(input, index1));
      uint32x4_t c = vreinterpretq_u32_u8(vqtbl1q_u8(input, index2));
      uint32x4_t d = vreinterpretq_u32_u8(vqtbl1q_u8(input, index3));
      float32x4x4_t output = {
          vcvtq_f32_u32(a),
          vcvtq_f32_u32(b),
          vcvtq_f32_u32(c),
          vcvtq_f32_u32(d),
      };
      vst1q_f32_x4(dst + x, output);
    }
    for (; x != width; ++x) {
      disable_loop_vectorization();
      dst[x] = src[x];
    }

    src += src_stride / sizeof(*src);
    dst += dst_stride / sizeof(*dst);
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
