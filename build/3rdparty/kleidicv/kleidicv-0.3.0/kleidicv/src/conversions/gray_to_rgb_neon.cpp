// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/gray_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <typename ScalarType>
class GrayToRGB final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  GrayToRGB() : indices_{vld1q_u8_x3(kGrayToRGBTableIndices)} {}
#else
  GrayToRGB() = default;
#endif

  void vector_path(VectorType src_vect, ScalarType *dst) {
    uint8x16x3_t dst_vect;
#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    dst_vect.val[0] = src_vect;
    dst_vect.val[1] = src_vect;
    dst_vect.val[2] = src_vect;
    vst3q_u8(dst, dst_vect);
#else
    dst_vect.val[0] = vqtbl1q_u8(src_vect, indices_.val[0]);
    dst_vect.val[1] = vqtbl1q_u8(src_vect, indices_.val[1]);
    dst_vect.val[2] = vqtbl1q_u8(src_vect, indices_.val[2]);
    vst1q_u8_x3(dst, dst_vect);
#endif
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    dst[0] = dst[1] = dst[2] = src[0];
  }

 private:
#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  static constexpr uint8_t kGrayToRGBTableIndices[48] = {
      0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,  5,
      5,  5,  6,  6,  6,  7,  7,  7,  8,  8,  8,  9,  9,  9,  10, 10,
      10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15};

  uint8x16x3_t indices_;
#endif
};  // end of class GrayToRGB<ScalarType>

template <typename ScalarType>
class GrayToRGBA final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  GrayToRGBA() : alpha_{vdupq_n_u8(0xff)} {}
#else
  // NOLINTBEGIN(hicpp-member-init)
  GrayToRGBA() : indices_{vld1q_u8_x4(kGrayToRGBATableIndices)} {
    src_and_alpha_.val[1] = vdupq_n_u8(0xff);
  }
  // NOLINTEND(hicpp-member-init)
#endif

  void vector_path(VectorType src_vect, ScalarType *dst) {
    uint8x16x4_t dst_vect;
#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    dst_vect.val[0] = src_vect;
    dst_vect.val[1] = src_vect;
    dst_vect.val[2] = src_vect;
    dst_vect.val[3] = alpha_;
    vst4q_u8(dst, dst_vect);
#else
    src_and_alpha_.val[0] = src_vect;
    dst_vect.val[0] = vqtbl2q_u8(src_and_alpha_, indices_.val[0]);
    dst_vect.val[1] = vqtbl2q_u8(src_and_alpha_, indices_.val[1]);
    dst_vect.val[2] = vqtbl2q_u8(src_and_alpha_, indices_.val[2]);
    dst_vect.val[3] = vqtbl2q_u8(src_and_alpha_, indices_.val[3]);
    vst1q_u8_x4(dst, dst_vect);
#endif
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    dst[0] = dst[1] = dst[2] = src[0];
    dst[3] = 0xff;
  }

 private:
#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  uint8x16_t alpha_;
#else
  uint8x16x4_t indices_;
  uint8x16x2_t src_and_alpha_;

  static constexpr uint8_t kGrayToRGBATableIndices[64] = {
      0,  0,  0,  16, 1,  1,  1,  16, 2,  2,  2,  16, 3,  3,  3,  16,
      4,  4,  4,  16, 5,  5,  5,  16, 6,  6,  6,  16, 7,  7,  7,  16,
      8,  8,  8,  16, 9,  9,  9,  16, 10, 10, 10, 16, 11, 11, 11, 16,
      12, 12, 12, 16, 13, 13, 13, 16, 14, 14, 14, 16, 15, 15, 15, 16};
#endif
};  // end of class GrayToRGBA<ScalarType>

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t gray_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* RGB */};
  GrayToRGB<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t gray_to_rgba_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* RGBA */};
  GrayToRGBA<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
