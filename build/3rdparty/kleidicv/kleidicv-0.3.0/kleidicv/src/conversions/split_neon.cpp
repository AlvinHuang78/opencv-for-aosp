// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/split.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

// Generic 2-channel split
//
// Split the source into two channel:
// vsrc=[0,1,2,3,4,5,6,7]
// -> vdst0=[0,2,4,6]
// -> vdst1=[1,3,5,7]
template <typename ScalarType>
class Split2 final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using Vector2Type = typename VecTraits::Vector2Type;

#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  void vector_path(const ScalarType *src, ScalarType *dst0, ScalarType *dst1) {
    Vector2Type vsrc;
    vsrc = vld2q(src);
    vst1q(dst0, vsrc.val[0]);
    vst1q(dst1, vsrc.val[1]);
  }
#else
  void vector_path(const Vector2Type vsrc, VectorType &vdst0,
                   VectorType &vdst1) {
    vdst0 = vuzp1q(vsrc.val[0], vsrc.val[1]);
    vdst1 = vuzp2q(vsrc.val[0], vsrc.val[1]);
  }
#endif

  void scalar_path(const ScalarType *src, ScalarType *dst0, ScalarType *dst1) {
    dst0[0] = src[0];
    dst1[0] = src[1];
  }
};

// Generic 3-channel split
//
// Split the loaded triple vector size source into 3 channels
// vsrc=[0,1,2,3,4,5,6,7,8,9]
// -> vdst0=[0,3,6,9]
// -> vdst1=[1,4,7,10]
// -> vdst2=[2,5,8,11]
template <typename ScalarType>
class Split3 final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using Vector3Type = typename VecTraits::Vector3Type;
  using VectorType = typename VecTraits::VectorType;

#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  // NOLINTBEGIN(hicpp-member-init)
  Split3() { Split3Init(ScalarType()); }
// NOLINTEND(hicpp-member-init)
#endif

#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  void vector_path(const ScalarType *src, ScalarType *dst0, ScalarType *dst1,
                   ScalarType *dst2) {
    Vector3Type vsrc;
    vsrc = vld3q(src);
    vst1q(dst0, vsrc.val[0]);
    vst1q(dst1, vsrc.val[1]);
    vst1q(dst2, vsrc.val[2]);
  }
#else
  void vector_path(Vector3Type vsrc, VectorType &vdst0, VectorType &vdst1,
                   VectorType &vdst2) {
    uint8x16x3_t tmp;
    tmp.val[0] = reinterpret_cast<uint8x16_t>(vsrc.val[0]);
    tmp.val[1] = reinterpret_cast<uint8x16_t>(vsrc.val[1]);
    tmp.val[2] = reinterpret_cast<uint8x16_t>(vsrc.val[2]);
    vdst0 = vqtbl3q_u8(tmp, index1_);
    vdst1 = vqtbl3q_u8(tmp, index2_);
    vdst2 = vqtbl3q_u8(tmp, index3_);
  }
#endif

  void scalar_path(const ScalarType *src, ScalarType *dst0, ScalarType *dst1,
                   ScalarType *dst2) {
    dst0[0] = src[0];
    dst1[0] = src[1];
    dst2[0] = src[2];
  }

 private:
#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  uint8x16_t index1_, index2_, index3_;

  void Split3Init(uint8_t) {
    // clang-format off
    const uint8_t kIndices[3][16] = {
    {0, 3, 6, 9,  12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45},
    {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46},
    {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47}};
    // clang-format on

    index1_ = vld1q_u8(kIndices[0]);
    index2_ = vld1q_u8(kIndices[1]);
    index3_ = vld1q_u8(kIndices[2]);
  }

  void Split3Init(uint16_t) {
    // clang-format off
    const uint8_t kIndices[3][16] = {
    {0, 1, 6,  7,  12, 13, 18, 19, 24, 25, 30, 31, 36, 37, 42, 43},
    {2, 3, 8,  9,  14, 15, 20, 21, 26, 27, 32, 33, 38, 39, 44, 45},
    {4, 5, 10, 11, 16, 17, 22, 23, 28, 29, 34, 35, 40, 41, 46, 47}};
    // clang-format on

    index1_ = vld1q_u8(kIndices[0]);
    index2_ = vld1q_u8(kIndices[1]);
    index3_ = vld1q_u8(kIndices[2]);
  }

  void Split3Init(uint32_t) {
    // clang-format off
    const uint8_t kIndices[3][16] = {
    {0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39},
    {4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43},
    {8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47}};
    // clang-format on

    index1_ = vld1q_u8(kIndices[0]);
    index2_ = vld1q_u8(kIndices[1]);
    index3_ = vld1q_u8(kIndices[2]);
  }

  void Split3Init(uint64_t) {
    // clang-format off
    const uint8_t kIndices[3][16] = {
    {0,  1,  2,  3,  4,  5,  6,  7,  24, 25, 26, 27, 28, 29, 30, 31},
    {8,  9,  10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39},
    {16, 17, 18, 19, 20, 21, 22, 23, 40, 41, 42, 43, 44, 45, 46, 47}};
    // clang-format on

    index1_ = vld1q_u8(kIndices[0]);
    index2_ = vld1q_u8(kIndices[1]);
    index3_ = vld1q_u8(kIndices[2]);
  }
#endif
};

// Generic 4-channel split
//
// Split the source first into two part, using double size cast:
// vsrc=[0,1,2,3,4,5,6,7]
// -> halfway_unzipped_1=[0,1,4,5]
// -> halfway_unzipped_2=[2,3,6,7]
// Then split these parts into final channels:
// halfway_unzipped_1=[0,1,4,5]
// -> vdst0=[0,4]
// -> vdst1=[1,5]
// halfway_unzipped_2=[2,3,6,7]
// -> vdst0=[2,6]
// -> vdst1=[3,7]
template <typename ScalarType>
class Split4 final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using Vector2Type = typename VecTraits::Vector2Type;
  using Vector4Type = typename VecTraits::Vector4Type;

#if KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
  void vector_path(const ScalarType *src, ScalarType *dst0, ScalarType *dst1,
                   ScalarType *dst2, ScalarType *dst3) {
    Vector4Type vsrc;
    vsrc = vld4q(src);
    vst1q(dst0, vsrc.val[0]);
    vst1q(dst1, vsrc.val[1]);
    vst1q(dst2, vsrc.val[2]);
    vst1q(dst3, vsrc.val[3]);
  }
#else
  void vector_path(const Vector4Type vsrc, VectorType &dst0, VectorType &dst1,
                   VectorType &dst2, VectorType &dst3) {
    VectorType halfway_unzipped_1, halfway_unzipped_2;

    halfway_unzipped_1 = vuzp1q(
        reinterpret_cast<double_element_width_t<VectorType> >(vsrc.val[0]),
        reinterpret_cast<double_element_width_t<VectorType> >(vsrc.val[1]));
    halfway_unzipped_2 = vuzp1q(
        reinterpret_cast<double_element_width_t<VectorType> >(vsrc.val[2]),
        reinterpret_cast<double_element_width_t<VectorType> >(vsrc.val[3]));

    dst0 = vuzp1q(halfway_unzipped_1, halfway_unzipped_2);
    dst1 = vuzp2q(halfway_unzipped_1, halfway_unzipped_2);

    halfway_unzipped_1 = vuzp2q(
        reinterpret_cast<double_element_width_t<VectorType> >(vsrc.val[0]),
        reinterpret_cast<double_element_width_t<VectorType> >(vsrc.val[1]));
    halfway_unzipped_2 = vuzp2q(
        reinterpret_cast<double_element_width_t<VectorType> >(vsrc.val[2]),
        reinterpret_cast<double_element_width_t<VectorType> >(vsrc.val[3]));

    dst2 = vuzp1q(halfway_unzipped_1, halfway_unzipped_2);
    dst3 = vuzp2q(halfway_unzipped_1, halfway_unzipped_2);
  }
#endif

  void scalar_path(const ScalarType *src, ScalarType *dst0, ScalarType *dst1,
                   ScalarType *dst2, ScalarType *dst3) {
    dst0[0] = src[0];
    dst1[0] = src[1];
    dst2[0] = src[2];
    dst3[0] = src[3];
  }
};

// Specialized split implementation for 4 channels with 64 bits data
// As in case of 64 bits we have 2 values in one q-sized vector,
// the implementation is simpler, no need for cast to double size
#if !KLEIDICV_PREFER_INTERLEAVING_LOAD_STORE
template <>
class Split4<uint64_t> final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<uint64_t>;
  using VectorType = typename VecTraits::VectorType;
  using Vector2Type = typename VecTraits::Vector2Type;
  using Vector4Type = typename VecTraits::Vector4Type;

  void vector_path(const Vector4Type vsrc, VectorType &dst0, VectorType &dst1,
                   VectorType &dst2, VectorType &dst3) {
    dst0 = vuzp1q(vsrc.val[0], vsrc.val[2]);
    dst1 = vuzp2q(vsrc.val[0], vsrc.val[2]);
    dst2 = vuzp1q(vsrc.val[1], vsrc.val[3]);
    dst3 = vuzp2q(vsrc.val[1], vsrc.val[3]);
  }

  void scalar_path(const uint64_t *src, uint64_t *dst0, uint64_t *dst1,
                   uint64_t *dst2, uint64_t *dst3) {
    dst0[0] = src[0];
    dst1[0] = src[1];
    dst2[0] = src[2];
    dst3[0] = src[3];
  }
};
#endif

// Most of the complexity comes from parameter checking.
// NOLINTBEGIN(readability-function-cognitive-complexity)
template <typename ScalarType>
kleidicv_error_t split(const void *src_void, const size_t src_stride,
                       void **dst_data, const size_t *dst_strides, size_t width,
                       size_t height, size_t channels) {
  if (channels < 2) {
    return KLEIDICV_ERROR_RANGE;
  }

  CHECK_POINTERS(dst_data, dst_strides);
  MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src_data, src_void);
  MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst0, dst_data[0]);
  MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst1, dst_data[1]);
  CHECK_POINTER_AND_STRIDE(src_data, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst0, dst_strides[0], height);
  CHECK_POINTER_AND_STRIDE(dst1, dst_strides[1], height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<ScalarType> src_rows{const_cast<ScalarType *>(src_data), src_stride,
                            channels};
  Rows<ScalarType> dst_rows0{dst0, dst_strides[0]};
  Rows<ScalarType> dst_rows1{dst1, dst_strides[1]};
  switch (channels) {
    case 2: {
      Split2<ScalarType> operation;
      apply_operation_by_rows(operation, rect, src_rows, dst_rows0, dst_rows1);
    } break;
    case 3: {
      MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst2, dst_data[2]);
      CHECK_POINTER_AND_STRIDE(dst2, dst_strides[2], height);
      Rows<ScalarType> dst_rows2{dst2, dst_strides[2]};
      Split3<ScalarType> operation;
      apply_operation_by_rows(operation, rect, src_rows, dst_rows0, dst_rows1,
                              dst_rows2);
    } break;
    case 4: {
      MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst2, dst_data[2]);
      MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst3, dst_data[3]);
      CHECK_POINTER_AND_STRIDE(dst2, dst_strides[2], height);
      CHECK_POINTER_AND_STRIDE(dst3, dst_strides[3], height);
      Rows<ScalarType> dst_rows2{dst2, dst_strides[2]};
      Rows<ScalarType> dst_rows3{dst3, dst_strides[3]};
      Split4<ScalarType> operation;
      apply_operation_by_rows(operation, rect, src_rows, dst_rows0, dst_rows1,
                              dst_rows2, dst_rows3);
    } break;
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  return KLEIDICV_OK;
}
// NOLINTEND(readability-function-cognitive-complexity)

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t split(const void *src_data, size_t src_stride, void **dst_data,
                       const size_t *dst_strides, size_t width, size_t height,
                       size_t channels, size_t element_size) {
  switch (element_size) {
    case sizeof(uint8_t):
      return split<uint8_t>(src_data, src_stride, dst_data, dst_strides, width,
                            height, channels);

    case sizeof(uint16_t):
      return split<uint16_t>(src_data, src_stride, dst_data, dst_strides, width,
                             height, channels);

    case sizeof(uint32_t):
      return split<uint32_t>(src_data, src_stride, dst_data, dst_strides, width,
                             height, channels);

    case sizeof(uint64_t):
      return split<uint64_t>(src_data, src_stride, dst_data, dst_strides, width,
                             height, channels);

    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}
}  // namespace kleidicv::neon
