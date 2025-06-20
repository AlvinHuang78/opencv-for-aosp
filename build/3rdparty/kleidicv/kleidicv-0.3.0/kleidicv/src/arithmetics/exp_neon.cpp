// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "kleidicv/arithmetics/exp_constants.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <typename ScalarType>
class Exp;

template <>
class Exp<float> final : public UnrollOnce {
 public:
  using VecTraits = neon::VecTraits<float>;
  using VectorType = typename VecTraits::VectorType;

  VectorType vector_path(VectorType src) {
    float32x4_t n, r, scale, poly, z;
    uint32x4_t cmp, e;

    /* exp(x) = 2^n * poly(r), with poly(r) in [1/sqrt(2),sqrt(2)]
      x = ln2*n + r, with r in [-ln2/2, ln2/2].  */
    z = vfmaq_f32(vdupq_n(exp_f32::kShift), src, vdupq_n(exp_f32::kInvLn2));
    n = z - vdupq_n(exp_f32::kShift);
    r = vfmaq_f32(src, n, vdupq_n(-exp_f32::kLn2Hi));
    r = vfmaq_f32(r, n, vdupq_n(-exp_f32::kLn2Lo));
    e = vreinterpretq_u32_f32(z) << 23;
    scale = vreinterpretq_f32_u32(e + vdupq_n(0x3f800000));
    cmp = vcagtq_f32(n, vdupq_n(126.0F));
    poly = vfmaq_f32(vdupq_n(exp_f32::kPoly[1]), vdupq_n(exp_f32::kPoly[0]), r);
    poly = vfmaq_f32(vdupq_n(exp_f32::kPoly[2]), poly, r);
    poly = vfmaq_f32(vdupq_n(exp_f32::kPoly[3]), poly, r);
    poly = vfmaq_f32(vdupq_n(exp_f32::kPoly[4]), poly, r);
    poly = vfmaq_f32(vdupq_n(1.0F), poly, r);
    poly = vfmaq_f32(vdupq_n(1.0F), poly, r);
    if (KLEIDICV_UNLIKELY(v_any_u32(cmp))) {
      return specialcase(poly, n, e);
    }
    return scale * poly;
  }

  float scalar_path(float src) { return expf(src); }

 private:
  static int v_any_u32(uint32x4_t x) {
    /* assume elements in x are either 0 or -1u.  */
    return vpaddd_u64(vreinterpretq_u64_u32(x)) != 0;
  }

  static float32x4_t specialcase(float32x4_t poly, float32x4_t n,
                                 uint32x4_t e) {
    /* 2^n may overflow, break it up into s1*s2.  */
    uint32x4_t b = (n <= vdupq_n(0.0F)) & vdupq_n(0x83000000);
    float32x4_t s1 = vreinterpretq_f32_u32(vdupq_n(0x7f000000) + b);
    float32x4_t s2 = vreinterpretq_f32_u32(e - b);
    uint32x4_t cmp = vcagtq_f32(n, vdupq_n(192.0F));
    float32x4_t r1 = s1 * s1;
    float32x4_t r0 = (poly * s1) * s2;
    return vreinterpretq_f32_u32((cmp & vreinterpretq_u32_f32(r1)) |
                                 (~cmp & vreinterpretq_u32_f32(r0)));
  }
};  // end of class Exp<float>

template <typename T>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t exp(const T* src, size_t src_stride,
                                              T* dst, size_t dst_stride,
                                              size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Exp<T> operation;
  Rectangle rect{width, height};
  Rows<const T> src_rows{src, src_stride};
  Rows<T> dst_rows{dst, dst_stride};
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                             \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t exp<type>(         \
      const type* src, size_t src_stride, type* dst, size_t dst_stride, \
      size_t width, size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::neon
