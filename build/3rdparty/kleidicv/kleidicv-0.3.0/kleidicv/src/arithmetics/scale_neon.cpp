// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <climits>
#include <cmath>
#include <limits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/traits.h"

namespace kleidicv::neon {

// Scale algorithm: for each value in the source,
//   dst[i] = src[i] * scale + shift   (floating point operation)
//
// Unsigned 8-bit implementation
//
// Since converting from uint8 to float32 and back takes more steps,
// 'ScaleTbx' saves time by pre-calculating all 256 values and uses TBLs
// and TBXs to map the values directly from uint8 to uint8:
//   i: 0 to 255:    tbl[i] = i * scale + shift
//
// Since a single TBL intruction can map only 16 values, more TBX instructions
// needed for the remaining 240 values. After the first TBL (that replaces
// 0-15 values with indexed values from the table) 16 is subtracted from all
// lanes in the source vector before the next TBX is done, so when indexing 0
// to 15, actually 16 to 31 values are replaced from the original source vector.
//
// Example:
//   scale = 1
//   shift = 100
// Initialization: (it also takes time, so for short inputs it's not used)
//   tbl = [ 100, 101, 102, ..., 255, <100 times 255, it's saturated>]
// Copy table to vector registers:
//   t0  = [ 100, ..., 115 ]
//   t1  = [ 116, ..., 131 ]
//   t2  = [ 132, ..., 147 ]
//   ...
//   t15 = [ 255, ..., 255 ]
//
//   input:    v = [  21,   3,  39,   6 ]
//   TBL(t0):  d = [   0, 103,   0, 106 ] // index > 16 result in 0
//   SUB:      v = [   5, 243,  23, 246 ] // subtracted 16 --> next table
//   TBX(t1):  d = [ 121, 103,   0, 106 ] // index > 16 are ignored
//   SUB:      v = [ 245, 227,   7, 230 ] // subtracted 16 --> next table
//   TBX(t2):  d = [ 121, 103, 107, 106 ] // index > 16 are ignored
//   ... etc.
//
// Bigger index tables (32, 48 or 64 values) can be used by TBX2 - TBX3 - TBX4.
// In this case, instead of 16, 2/3/4 * 16 have to be subtracted from source.
// The below solution (combining TBX2-TBX3) gives a good compromise between code
// size and speed.

template <typename ScalarType>
class ScaleIntBase : public UnrollTwice {
 public:
  ScaleIntBase(float scale, float shift) : scale_{scale}, shift_{shift} {}

 protected:
  static constexpr ScalarType ScalarMax =
      std::numeric_limits<ScalarType>::max();
  inline ScalarType scale_value(ScalarType value) {
    int64_t v = lrintf(value * scale_ + shift_);
    if (static_cast<uint64_t>(v) <= ScalarMax) {
      return static_cast<ScalarType>(v);
    }
    return static_cast<ScalarType>(v > 0 ? ScalarMax : 0);
  }

 private:
  float scale_, shift_;
};

class ScaleUint8Tbx final : public ScaleIntBase<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using Vector2Type = typename VecTraits::Vector2Type;
  using Vector3Type = typename VecTraits::Vector3Type;

  ScaleUint8Tbx(float scale, float shift)
      : ScaleIntBase<uint8_t>(scale, shift) {
    constexpr size_t TableLength = 1 << (CHAR_BIT * sizeof(ScalarType));
    ScalarType values[TableLength];
    for (size_t i = 0; i < TableLength; ++i) {
      values[i] = this->scale_value(i);
    }
    t0_3_ = vld1q_u8_x3(values);
    t1_3_ = vld1q_u8_x3(values + 3 * VecTraits::num_lanes());
    t2_2_ = vld1q_u8_x2(values + (3 + 3) * VecTraits::num_lanes());
    t3_3_ = vld1q_u8_x3(values + (3 + 3 + 2) * VecTraits::num_lanes());
    t4_2_ = vld1q_u8_x2(values + (3 + 3 + 2 + 3) * VecTraits::num_lanes());
    t5_3_ = vld1q_u8_x3(values + (3 + 3 + 2 + 3 + 2) * VecTraits::num_lanes());
    v_step3_ = vdupq_n_u8(3 * VecTraits::num_lanes());
    v_step2_ = vdupq_n_u8(2 * VecTraits::num_lanes());
  }

  VectorType vector_path(VectorType src) {
    VectorType dst = vqtbl3q_u8(t0_3_, src);
    src = vsubq_u8(src, v_step3_);
    dst = vqtbx3q_u8(dst, t1_3_, src);
    src = vsubq_u8(src, v_step3_);
    dst = vqtbx2q_u8(dst, t2_2_, src);
    src = vsubq_u8(src, v_step2_);
    dst = vqtbx3q_u8(dst, t3_3_, src);
    src = vsubq_u8(src, v_step3_);
    dst = vqtbx2q_u8(dst, t4_2_, src);
    src = vsubq_u8(src, v_step2_);
    dst = vqtbx3q_u8(dst, t5_3_, src);
    return dst;
  }

  ScalarType scalar_path(ScalarType src) { return this->scale_value(src); }

 private:
  Vector3Type t0_3_{}, t1_3_{}, t3_3_{}, t5_3_{};
  Vector2Type t2_2_{}, t4_2_{};
  VectorType v_step3_, v_step2_;
};  // end of class ScaleUint8Tbx<T>

// Opposite to ScaleUint8Tbx, ScaleUint8Calc is the direct approach:
// - calculate dst[i] = src[i] * scale + shift  using vector instructions
class ScaleUint8Calc final : public ScaleIntBase<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  ScaleUint8Calc(float scale, float shift)
      : ScaleIntBase<ScalarType>(scale, shift),
        vscale_{vdupq_n_f32(scale)},
        vshift_{vdupq_n_f32(shift)} {}

  VectorType vector_path(VectorType src) {
    // For scaling, uint8 values have to be converted to uint32
    // i.e. create four vectors from one
    uint32x4_t res11 = scale_shift(vqtbl1q_u8(src, w0));
    uint32x4_t res12 = scale_shift(vqtbl1q_u8(src, w1));
    uint32x4_t res21 = scale_shift(vqtbl1q_u8(src, w2));
    uint32x4_t res22 = scale_shift(vqtbl1q_u8(src, w3));
    // Convert back from 32-bit: top two bytes are 0 for sure, unzip them
    uint16x8_t res1 =
        vuzp1q_u16(vreinterpretq_u16_u32(res11), vreinterpretq_u16_u32(res12));
    uint16x8_t res2 =
        vuzp1q_u16(vreinterpretq_u16_u32(res21), vreinterpretq_u16_u32(res22));

    // Saturating narrowing from 16 to 8 bits
    return vqmovn_high_u16(vqmovn_u16(res1), res2);
  }

  ScalarType scalar_path(ScalarType src) { return this->scale_value(src); }

 private:
  static constexpr ScalarType FF = std::numeric_limits<uint8_t>::max();
  // clang-format off
  static constexpr uint8x16_t w0 = { 0, FF, FF, FF,  1, FF, FF, FF,  2, FF, FF, FF,  3, FF, FF, FF};
  static constexpr uint8x16_t w1 = { 4, FF, FF, FF,  5, FF, FF, FF,  6, FF, FF, FF,  7, FF, FF, FF};
  static constexpr uint8x16_t w2 = { 8, FF, FF, FF,  9, FF, FF, FF, 10, FF, FF, FF, 11, FF, FF, FF};
  static constexpr uint8x16_t w3 = {12, FF, FF, FF, 13, FF, FF, FF, 14, FF, FF, FF, 15, FF, FF, FF};
  // clang-format on

  // Convert from uint32 to float32, scale and convert back with rounding
  inline uint32x4_t scale_shift(VectorType src) {
    float32x4_t fx = vcvtq_f32_u32(vreinterpretq_u32_u8(src));
    // scale + shift is done by MLA
    return vcvtnq_u32_f32(vmlaq_f32(vshift_, fx, vscale_));
  }

  float32x4_t vscale_, vshift_;
};  // end of class ScaleUint8Calc<T>

// -----------------------------------------------------------------------
// Float implementation
// -----------------------------------------------------------------------

class AddFloat final : public UnrollTwice,
                       public UnrollOnce,
                       public TryToAvoidTailLoop {
 public:
  using ScalarType = float;
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  explicit AddFloat(float shift) : shift_{shift}, vshift_{vdupq_n_f32(shift)} {}

  VectorType vector_path(VectorType src) { return vaddq_f32(vshift_, src); }

  // NOLINTBEGIN(readability-make-member-function-const)
  ScalarType scalar_path(ScalarType src) { return src + shift_; }
  // NOLINTEND(readability-make-member-function-const)

 private:
  float shift_;
  float32x4_t vshift_;
};  // end of class AddFloat

class ScaleFloat final : public UnrollTwice,
                         public UnrollOnce,
                         public TryToAvoidTailLoop {
 public:
  using ScalarType = float;
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  ScaleFloat(float scale, float shift)
      : scale_{scale},
        shift_{shift},
        vscale_{vdupq_n_f32(scale)},
        vshift_{vdupq_n_f32(shift)} {}

  VectorType vector_path(VectorType src) {
    return vmlaq_f32(vshift_, src, vscale_);
  }

  // NOLINTBEGIN(readability-make-member-function-const)
  ScalarType scalar_path(ScalarType src) { return src * scale_ + shift_; }
  // NOLINTEND(readability-make-member-function-const)

 private:
  float scale_, shift_;
  float32x4_t vscale_, vshift_;
};  // end of class ScaleFloat

template <typename T>
kleidicv_error_t scale(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       float scale, float shift);

// Specialization for uint8_t
template <>
kleidicv_error_t scale(const uint8_t *src, size_t src_stride, uint8_t *dst,
                       size_t dst_stride, size_t width, size_t height,
                       float scale, float shift) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride};
  Rows<uint8_t> dst_rows{dst, dst_stride};
  // For smaller inputs, the full calculation is the faster
  if (width * height < 2500) {  // empirical value
    ScaleUint8Calc operation(scale, shift);
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  } else {
    // For bigger inputs, it's faster to pre-calculate the table
    // and map those values during the run
    ScaleUint8Tbx operation(scale, shift);
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  }
  return KLEIDICV_OK;
}

// Specialization for float
template <>
kleidicv_error_t scale(const float *src, size_t src_stride, float *dst,
                       size_t dst_stride, size_t width, size_t height,
                       float scale, float shift) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const float> src_rows{src, src_stride};
  Rows<float> dst_rows{dst, dst_stride};
  if (scale == 1.0) {
    AddFloat operation(shift);
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  } else {
    ScaleFloat operation(scale, shift);
    apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  }
  return KLEIDICV_OK;
}

}  //  namespace kleidicv::neon
