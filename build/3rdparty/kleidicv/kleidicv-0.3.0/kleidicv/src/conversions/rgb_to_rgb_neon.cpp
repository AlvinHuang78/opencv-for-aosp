// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/rgb_to_rgb.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <typename ScalarType>
class RGBToBGR final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  RGBToBGR() : indices_{vld1q_u8_x3(kRGBToBGRTableIndices)} {}
#else
  RGBToBGR() = default;
#endif

  void vector_path(const ScalarType *src, ScalarType *dst) {
#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
    uint8x16x3_t src_vect = vld3q_u8(src);
    uint8x16x3_t dst_vect;

    dst_vect.val[0] = src_vect.val[2];
    dst_vect.val[1] = src_vect.val[1];
    dst_vect.val[2] = src_vect.val[0];

    vst3q_u8(dst, dst_vect);
#else
    uint8x16x3_t src_vect = vld1q_u8_x3(src);
    uint8x16x3_t dst_vect;

    uint8x16x2_t src_vect_0_1;
    src_vect_0_1.val[0] = src_vect.val[0];
    src_vect_0_1.val[1] = src_vect.val[1];

    uint8x16x2_t src_vect_1_2;
    src_vect_1_2.val[0] = src_vect.val[1];
    src_vect_1_2.val[1] = src_vect.val[2];

    dst_vect.val[0] = vqtbl2q_u8(src_vect_0_1, indices_.val[0]);
    dst_vect.val[1] = vqtbl3q_u8(src_vect, indices_.val[1]);
    dst_vect.val[2] = vqtbl2q_u8(src_vect_1_2, indices_.val[2]);

    vst1q_u8_x3(dst, dst_vect);
#endif
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    auto tmp = src[0];
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = tmp;
  }

 private:
#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  static constexpr uint8_t kRGBToBGRTableIndices[48] = {
      2,  1,  0,  5,  4,  3,  8,  7,  6,  11, 10, 9,  14, 13, 12, 17,
      16, 15, 20, 19, 18, 23, 22, 21, 26, 25, 24, 29, 28, 27, 32, 31,
      14, 19, 18, 17, 22, 21, 20, 25, 24, 23, 28, 27, 26, 31, 30, 29};

  uint8x16x3_t indices_;
#endif
};  // end of class RGBToBGR<ScalarType>

template <typename ScalarType>
class RGBAToBGRA final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  void vector_path(const ScalarType *src, ScalarType *dst) {
    uint8x16x4_t src_vect = vld4q_u8(src);
    uint8x16x4_t dst_vect;

    dst_vect.val[0] = src_vect.val[2];
    dst_vect.val[1] = src_vect.val[1];
    dst_vect.val[2] = src_vect.val[0];
    dst_vect.val[3] = src_vect.val[3];

    vst4q_u8(dst, dst_vect);
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    auto tmp = src[0];
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = tmp;
    dst[3] = src[3];
  }
};  // end of class RGBAToBGRA<ScalarType>

template <typename ScalarType>
class RGBToBGRA final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  void vector_path(const ScalarType *src, ScalarType *dst) {
    uint8x16x3_t src_vect = vld3q_u8(src);
    uint8x16x4_t dst_vect;

    dst_vect.val[0] = src_vect.val[2];
    dst_vect.val[1] = src_vect.val[1];
    dst_vect.val[2] = src_vect.val[0];
    dst_vect.val[3] = vdupq_n_u8(0xff);

    vst4q_u8(dst, dst_vect);
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    auto tmp = src[0];
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = tmp;
    dst[3] = 0xff;
  }
};  // end of class RGBToBGRA<ScalarType>

template <typename ScalarType>
class RGBToRGBA final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  void vector_path(const ScalarType *src, ScalarType *dst) {
    uint8x16x3_t src_vect = vld3q_u8(src);
    uint8x16x4_t dst_vect;

    dst_vect.val[0] = src_vect.val[0];
    dst_vect.val[1] = src_vect.val[1];
    dst_vect.val[2] = src_vect.val[2];
    dst_vect.val[3] = vdupq_n_u8(0xff);

    vst4q_u8(dst, dst_vect);
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    memcpy(static_cast<void *>(dst), static_cast<const void *>(src), 3);
    dst[3] = 0xff;
  }
};  // end of class RGBToRGBA<ScalarType>

template <typename ScalarType>
class RGBAToBGR final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  void vector_path(const ScalarType *src, ScalarType *dst) {
    uint8x16x4_t src_vect = vld4q_u8(src);
    uint8x16x3_t dst_vect;

    dst_vect.val[0] = src_vect.val[2];
    dst_vect.val[1] = src_vect.val[1];
    dst_vect.val[2] = src_vect.val[0];

    vst3q_u8(dst, dst_vect);
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    auto tmp = src[0];
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = tmp;
  }
};  // end of class RGBAToBGR<ScalarType>

template <typename ScalarType>
class RGBAToRGB final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  void vector_path(const ScalarType *src, ScalarType *dst) {
    uint8x16x4_t src_vect = vld4q_u8(src);
    uint8x16x3_t dst_vect;

    dst_vect.val[0] = src_vect.val[0];
    dst_vect.val[1] = src_vect.val[1];
    dst_vect.val[2] = src_vect.val[2];

    vst3q_u8(dst, dst_vect);
  }

  void scalar_path(const ScalarType *src, ScalarType *dst) {
    memcpy(static_cast<void *>(dst), static_cast<const void *>(src), 3);
  }
};  // end of class RGBAToRGB<ScalarType>

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgb_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 3 /* RGB */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* BGR */};
  RGBToBGR<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 4 /* RGBA */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* BGRA */};
  RGBAToBGRA<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
rgb_to_bgra_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
               size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 3 /* RGB */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* BGRA */};
  RGBToBGRA<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
rgb_to_rgba_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
               size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 3 /* RGB */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* RGBA */};
  RGBToRGBA<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_bgr_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 4 /* RGBA */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* BGR */};
  RGBAToBGR<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 4 /* RGBA */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* RGB */};
  RGBAToRGB<uint8_t> operation;
  apply_operation_by_rows(operation, rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
