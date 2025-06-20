// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cassert>

#include "kleidicv/ctypes.h"
#include "kleidicv/filters/gaussian_blur.h"
#include "kleidicv/filters/separable_filter_15x15_neon.h"
#include "kleidicv/filters/separable_filter_3x3_neon.h"
#include "kleidicv/filters/separable_filter_5x5_neon.h"
#include "kleidicv/filters/separable_filter_7x7_neon.h"
#include "kleidicv/filters/sigma.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/workspace/separable.h"

namespace kleidicv::neon {

// Primary template for Gaussian Blur filters.
template <typename ScalarType, size_t KernelSize, bool IsBinomial>
class GaussianBlur;

// Template for 3x3 Gaussian Blur binomial filters.
//
//             [ 1, 2, 1 ]          [ 1 ]
//  F = 1/16 * [ 2, 4, 2 ] = 1/16 * [ 2 ] * [ 1, 2, 1 ]
//             [ 1, 2, 1 ]          [ 1 ]
template <>
class GaussianBlur<uint8_t, 3, true> {
 public:
  using ScalarType = uint8_t;
  using SourceType = ScalarType;
  using SourceVectorType = typename VecTraits<SourceType>::VectorType;
  using BufferType = double_element_width_t<ScalarType>;
  using BufferVectorType = typename VecTraits<BufferType>::VectorType;
  using DestinationType = ScalarType;

  explicit GaussianBlur([[maybe_unused]] float sigma) {}

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void vertical_vector_path(SourceVectorType src[3], BufferType *dst) const {
    // acc_0_2 = src[0] + src[2]
    BufferVectorType acc_0_2_l = vaddl(vget_low(src[0]), vget_low(src[2]));
    BufferVectorType acc_0_2_h = vaddl(vget_high(src[0]), vget_high(src[2]));
    // acc_1 = src[1] + src[1]
    BufferVectorType acc_1_l = vshll_n<1>(vget_low(src[1]));
    BufferVectorType acc_1_h = vshll_n<1>(vget_high(src[1]));
    // acc = acc_0_2 + acc_1
    BufferVectorType acc_l = vaddq(acc_0_2_l, acc_1_l);
    BufferVectorType acc_h = vaddq(acc_0_2_h, acc_1_h);

    VecTraits<BufferType>::store_consecutive(acc_l, acc_h, &dst[0]);
  }

  // Applies vertical filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void vertical_scalar_path(const SourceType src[3], BufferType *dst) const {
    dst[0] = src[0] + 2 * src[1] + src[2];
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/16 * [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void horizontal_vector_path(BufferVectorType src[3],
                              DestinationType *dst) const {
    BufferVectorType acc_wide = vaddq(src[0], src[2]);
    acc_wide = vaddq(acc_wide, vshlq_n<1>(src[1]));
    auto acc_narrow = vrshrn_n<4>(acc_wide);
    vst1(&dst[0], acc_narrow);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/16 * [ SRC0, SRC1, SRC2 ] * [ 1, 2, 1 ]T
  void horizontal_scalar_path(const BufferType src[3],
                              DestinationType *dst) const {
    auto acc = src[0] + 2 * src[1] + src[2];
    dst[0] = rounding_shift_right(acc, 4);
  }
};  // end of class GaussianBlur<uint8_t, 3, true>

// Template for 5x5 Gaussian Blur binomial filters.
//
//              [ 1,  4,  6,  4, 1 ]           [ 1 ]
//              [ 4, 16, 24, 16, 4 ]           [ 4 ]
//  F = 1/256 * [ 6, 24, 36, 24, 6 ] = 1/256 * [ 6 ] * [ 1,  4,  6,  4, 1 ]
//              [ 4, 16, 24, 16, 4 ]           [ 4 ]
//              [ 1,  4,  6,  4, 1 ]           [ 1 ]
template <>
class GaussianBlur<uint8_t, 5, true> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint16_t;
  using DestinationType = uint8_t;

  explicit GaussianBlur([[maybe_unused]] float sigma)
      : const_6_u8_half_{vdup_n_u8(6)},
        const_6_u16_{vdupq_n_u16(6)},
        const_4_u16_{vdupq_n_u16(4)} {}

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void vertical_vector_path(uint8x16_t src[5], BufferType *dst) const {
    uint16x8_t acc_0_4_l = vaddl_u8(vget_low_u8(src[0]), vget_low_u8(src[4]));
    uint16x8_t acc_0_4_h = vaddl_u8(vget_high_u8(src[0]), vget_high_u8(src[4]));
    uint16x8_t acc_1_3_l = vaddl_u8(vget_low_u8(src[1]), vget_low_u8(src[3]));
    uint16x8_t acc_1_3_h = vaddl_u8(vget_high_u8(src[1]), vget_high_u8(src[3]));
    uint16x8_t acc_l =
        vmlal_u8(acc_0_4_l, vget_low_u8(src[2]), const_6_u8_half_);
    uint16x8_t acc_h =
        vmlal_u8(acc_0_4_h, vget_high_u8(src[2]), const_6_u8_half_);
    acc_l = vmlaq_u16(acc_l, acc_1_3_l, const_4_u16_);
    acc_h = vmlaq_u16(acc_h, acc_1_3_h, const_4_u16_);
    vst1q(&dst[0], acc_l);
    vst1q(&dst[8], acc_h);
  }

  // Applies vertical filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void vertical_scalar_path(const SourceType src[5], BufferType *dst) const {
    dst[0] = src[0] + src[4] + 4 * (src[1] + src[3]) + 6 * src[2];
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/256 * [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void horizontal_vector_path(uint16x8_t src[5], DestinationType *dst) const {
    uint16x8_t acc_0_4 = vaddq_u16(src[0], src[4]);
    uint16x8_t acc_1_3 = vaddq_u16(src[1], src[3]);
    uint16x8_t acc_u16 = vmlaq_u16(acc_0_4, src[2], const_6_u16_);
    acc_u16 = vmlaq_u16(acc_u16, acc_1_3, const_4_u16_);
    uint8x8_t acc_u8 = vrshrn_n_u16(acc_u16, 8);
    vst1(&dst[0], acc_u8);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/256 * [ SRC0, SRC1, SRC2, SRC3, SRC4 ] * [ 1, 4, 6, 4, 1 ]T
  void horizontal_scalar_path(const BufferType src[5],
                              DestinationType *dst) const {
    auto acc = src[0] + src[4] + 4 * (src[1] + src[3]) + 6 * src[2];
    dst[0] = rounding_shift_right(acc, 8);
  }

 private:
  uint8x8_t const_6_u8_half_;
  uint16x8_t const_6_u16_;
  uint16x8_t const_4_u16_;
};  // end of class GaussianBlur<uint8_t, 5, true>

// Template for 7x7 Gaussian Blur binomial filters.
//
//               [  4,  14,  28,  36,  28,  14,  4 ]
//               [ 14,  49,  98, 126,  98,  49, 14 ]
//               [ 28,  98, 196, 252, 196,  98, 28 ]
//  F = 1/4096 * [ 36, 126, 252, 324, 252, 126, 36 ] =
//               [ 28,  98, 196, 252, 196,  98, 28 ]
//               [ 14,  49,  98, 126,  98,  49, 14 ]
//               [  4,  14,  28,  36,  28,  14,  4 ]
//
//               [  2 ]
//               [  7 ]
//               [ 14 ]
//  = 1/4096  *  [ 18 ] * [ 2, 7, 14, 18, 14, 7, 2 ]
//               [ 14 ]
//               [  7 ]
//               [  2 ]
template <>
class GaussianBlur<uint8_t, 7, true> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint16_t;
  using DestinationType = uint8_t;

  explicit GaussianBlur([[maybe_unused]] float sigma)
      : const_7_u16_{vdupq_n_u16(7)},
        const_7_u32_{vdupq_n_u32(7)},
        const_9_u16_{vdupq_n_u16(9)} {}

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6 ] *
  //     * [ 2, 7, 14, 18, 14, 7, 2 ]T
  void vertical_vector_path(uint8x16_t src[7], BufferType *dst) const {
    uint16x8_t acc_0_6_l = vaddl_u8(vget_low_u8(src[0]), vget_low_u8(src[6]));
    uint16x8_t acc_0_6_h = vaddl_u8(vget_high_u8(src[0]), vget_high_u8(src[6]));

    uint16x8_t acc_1_5_l = vaddl_u8(vget_low_u8(src[1]), vget_low_u8(src[5]));
    uint16x8_t acc_1_5_h = vaddl_u8(vget_high_u8(src[1]), vget_high_u8(src[5]));

    uint16x8_t acc_2_4_l = vaddl_u8(vget_low_u8(src[2]), vget_low_u8(src[4]));
    uint16x8_t acc_2_4_h = vaddl_u8(vget_high_u8(src[2]), vget_high_u8(src[4]));

    uint16x8_t acc_3_l = vmovl_u8(vget_low_u8(src[3]));
    uint16x8_t acc_3_h = vmovl_u8(vget_high_u8(src[3]));

    uint16x8_t acc_0_2_4_6_l = vmlaq_u16(acc_0_6_l, acc_2_4_l, const_7_u16_);
    uint16x8_t acc_0_2_4_6_h = vmlaq_u16(acc_0_6_h, acc_2_4_h, const_7_u16_);

    uint16x8_t acc_0_2_3_4_6_l =
        vmlaq_u16(acc_0_2_4_6_l, acc_3_l, const_9_u16_);
    uint16x8_t acc_0_2_3_4_6_h =
        vmlaq_u16(acc_0_2_4_6_h, acc_3_h, const_9_u16_);

    acc_0_2_3_4_6_l = vshlq_n_u16(acc_0_2_3_4_6_l, 1);
    acc_0_2_3_4_6_h = vshlq_n_u16(acc_0_2_3_4_6_h, 1);

    uint16x8_t acc_0_1_2_3_4_5_6_l =
        vmlaq_u16(acc_0_2_3_4_6_l, acc_1_5_l, const_7_u16_);
    uint16x8_t acc_0_1_2_3_4_5_6_h =
        vmlaq_u16(acc_0_2_3_4_6_h, acc_1_5_h, const_7_u16_);

    vst1q(&dst[0], acc_0_1_2_3_4_5_6_l);
    vst1q(&dst[8], acc_0_1_2_3_4_5_6_h);
  }

  // Applies vertical filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6 ] *
  //     * [ 2, 7, 14, 18, 14, 7, 2 ]T
  void vertical_scalar_path(const SourceType src[7], BufferType *dst) const {
    uint16_t acc = src[0] * 2 + src[1] * 7 + src[2] * 14 + src[3] * 18 +
                   src[4] * 14 + src[5] * 7 + src[6] * 2;
    dst[0] = acc;
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/4096 * [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6 ] *
  //              * [ 2, 7, 14, 18, 14, 7, 2 ]T
  void horizontal_vector_path(uint16x8_t src[7], DestinationType *dst) const {
    uint32x4_t acc_0_6_l =
        vaddl_u16(vget_low_u16(src[0]), vget_low_u16(src[6]));
    uint32x4_t acc_0_6_h =
        vaddl_u16(vget_high_u16(src[0]), vget_high_u16(src[6]));

    uint32x4_t acc_1_5_l =
        vaddl_u16(vget_low_u16(src[1]), vget_low_u16(src[5]));
    uint32x4_t acc_1_5_h =
        vaddl_u16(vget_high_u16(src[1]), vget_high_u16(src[5]));

    uint16x8_t acc_2_4 = vaddq_u16(src[2], src[4]);

    uint32x4_t acc_0_2_4_6_l =
        vmlal_u16(acc_0_6_l, vget_low_u16(acc_2_4), vget_low_u16(const_7_u16_));
    uint32x4_t acc_0_2_4_6_h = vmlal_u16(acc_0_6_h, vget_high_u16(acc_2_4),
                                         vget_high_u16(const_7_u16_));

    uint32x4_t acc_0_2_3_4_6_l = vmlal_u16(acc_0_2_4_6_l, vget_low_u16(src[3]),
                                           vget_low_u16(const_9_u16_));
    uint32x4_t acc_0_2_3_4_6_h = vmlal_u16(acc_0_2_4_6_h, vget_high_u16(src[3]),
                                           vget_high_u16(const_9_u16_));

    acc_0_2_3_4_6_l = vshlq_n_u32(acc_0_2_3_4_6_l, 1);
    acc_0_2_3_4_6_h = vshlq_n_u32(acc_0_2_3_4_6_h, 1);

    uint32x4_t acc_0_1_2_3_4_5_6_l =
        vmlaq_u32(acc_0_2_3_4_6_l, acc_1_5_l, const_7_u32_);
    uint32x4_t acc_0_1_2_3_4_5_6_h =
        vmlaq_u32(acc_0_2_3_4_6_h, acc_1_5_h, const_7_u32_);

    uint16x4_t acc_0_1_2_3_4_5_6_u16_l = vrshrn_n_u32(acc_0_1_2_3_4_5_6_l, 12);
    uint16x4_t acc_0_1_2_3_4_5_6_u16_h = vrshrn_n_u32(acc_0_1_2_3_4_5_6_h, 12);

    uint16x8_t acc_0_1_2_3_4_5_6_u16 =
        vcombine_u16(acc_0_1_2_3_4_5_6_u16_l, acc_0_1_2_3_4_5_6_u16_h);
    uint8x8_t acc_0_1_2_3_4_5_6_u8 = vmovn_u16(acc_0_1_2_3_4_5_6_u16);

    vst1(&dst[0], acc_0_1_2_3_4_5_6_u8);
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/4096 * [ SRC0, SRC1, SRC2, SRC3, SRC4, SRC5, SRC6 ] *
  //              * [ 2, 7, 14, 18, 14, 7, 2 ]T
  void horizontal_scalar_path(const BufferType src[7],
                              DestinationType *dst) const {
    uint32_t acc = src[0] * 2 + src[1] * 7 + src[2] * 14 + src[3] * 18 +
                   src[4] * 14 + src[5] * 7 + src[6] * 2;
    dst[0] = static_cast<DestinationType>(rounding_shift_right(acc, 12));
  }

 private:
  uint16x8_t const_7_u16_;
  uint32x4_t const_7_u32_;
  uint16x8_t const_9_u16_;
};  // end of class GaussianBlur<uint8_t, 7, true>

// Template for 15x15 Gaussian Blur binomial filters.
//
//                  [  16,   44,  100,  192 ...  192,  100,   44,  16 ]
//                  [  44,  121,  275,  528 ...  528,  275,  121,  44 ]
//                  [ 100,  275,  625, 1200 ... 1200,  625,  275, 100 ]
//                  [ 192,  528, 1200, 2304 ... 2304, 1200,  528, 192 ]
//  F = 1/1048576 * [  |     |     |     |  ...   |     |     |    |  ] =
//                  [ 192,  528, 1200, 2304 ... 2304, 1200,  528, 192 ]
//                  [ 100,  275,  625, 1200 ... 1200,  625,  275, 100 ]
//                  [  44,  121,  275,  528 ...  528,  275,  121,  44 ]
//                  [  16,   44,  100,  192 ...  192,  100,   44,  16 ]
//
//                  [   4 ]
//                  [  11 ]
//                  [  25 ]
//                  [  48 ]
//                  [  81 ]
//                  [ 118 ]
//                  [ 146 ]
//  = 1/1048576  *  [ 158 ] * [4,11,25,48,81,118,146,158,146,118,81,48,25,11,4]
//                  [ 146 ]
//                  [ 118 ]
//                  [  81 ]
//                  [  48 ]
//                  [  25 ]
//                  [  11 ]
//                  [   4 ]
template <>
class GaussianBlur<uint8_t, 15, true> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint32_t;
  using DestinationType = uint8_t;

  explicit GaussianBlur([[maybe_unused]] float sigma)
      : const_11_u16_{vdupq_n_u16(11)},
        const_11_u32_{vdupq_n_u32(11)},
        const_25_u16_{vdupq_n_u16(25)},
        const_25_u32_{vdupq_n_u32(25)},
        const_81_u16_{vdupq_n_u16(81)},
        const_81_u32_{vdupq_n_u32(81)},
        const_118_u16_half_{vdup_n_u16(118)},
        const_118_u32_{vdupq_n_u32(118)},
        const_146_u16_half_{vdup_n_u16(146)},
        const_146_u32_{vdupq_n_u32(146)},
        const_158_u16_half_{vdup_n_u16(158)},
        const_158_u32_{vdupq_n_u32(158)} {}

  // Applies vertical filtering vector using SIMD operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3...SRC11, SRC12, SRC13, SRC14 ] *
  //     * [ 4, 11, 25, 48 ... 48, 25, 11, 4 ]T
  void vertical_vector_path(uint8x16_t src[15], BufferType *dst) const {
    uint16x8_t acc_7_l = vmovl_u8(vget_low_u8(src[7]));
    uint16x8_t acc_7_h = vmovl_u8(vget_high_u8(src[7]));

    uint16x8_t acc_1_13_l = vaddl_u8(vget_low_u8(src[1]), vget_low_u8(src[13]));
    uint16x8_t acc_1_13_h =
        vaddl_u8(vget_high_u8(src[1]), vget_high_u8(src[13]));

    uint16x8_t acc_2_12_l = vaddl_u8(vget_low_u8(src[2]), vget_low_u8(src[12]));
    uint16x8_t acc_2_12_h =
        vaddl_u8(vget_high_u8(src[2]), vget_high_u8(src[12]));

    uint16x8_t acc_6_8_l = vaddl_u8(vget_low_u8(src[6]), vget_low_u8(src[8]));
    uint16x8_t acc_6_8_h = vaddl_u8(vget_high_u8(src[6]), vget_high_u8(src[8]));

    uint16x8_t acc_5_9_l = vaddl_u8(vget_low_u8(src[5]), vget_low_u8(src[9]));
    uint16x8_t acc_5_9_h = vaddl_u8(vget_high_u8(src[5]), vget_high_u8(src[9]));

    uint16x8_t acc_0_14_l = vaddl_u8(vget_low_u8(src[0]), vget_low_u8(src[14]));
    uint16x8_t acc_0_14_h =
        vaddl_u8(vget_high_u8(src[0]), vget_high_u8(src[14]));

    uint16x8_t acc_3_11_l = vaddl_u8(vget_low_u8(src[3]), vget_low_u8(src[11]));
    uint16x8_t acc_3_11_h =
        vaddl_u8(vget_high_u8(src[3]), vget_high_u8(src[11]));

    uint16x8_t acc_4_10_l = vaddl_u8(vget_low_u8(src[4]), vget_low_u8(src[10]));
    uint16x8_t acc_4_10_h =
        vaddl_u8(vget_high_u8(src[4]), vget_high_u8(src[10]));

    acc_0_14_l = vshlq_n_u16(acc_0_14_l, 2);
    acc_0_14_h = vshlq_n_u16(acc_0_14_h, 2);

    acc_3_11_l = vshlq_n_u16(acc_3_11_l, 2);
    acc_3_11_h = vshlq_n_u16(acc_3_11_h, 2);

    acc_4_10_l = vmulq_u16(acc_4_10_l, const_81_u16_);
    acc_4_10_h = vmulq_u16(acc_4_10_h, const_81_u16_);

    uint16x8_t acc_1_3_11_13_l = vaddq_u16(acc_3_11_l, acc_1_13_l);
    uint16x8_t acc_1_3_11_13_h = vaddq_u16(acc_3_11_h, acc_1_13_h);
    acc_1_3_11_13_l = vmlaq_u16(acc_3_11_l, acc_1_3_11_13_l, const_11_u16_);
    acc_1_3_11_13_h = vmlaq_u16(acc_3_11_h, acc_1_3_11_13_h, const_11_u16_);

    uint16x8_t acc_0_1_3_11_13_14_l = vaddq_u16(acc_1_3_11_13_l, acc_0_14_l);
    uint16x8_t acc_0_1_3_11_13_14_h = vaddq_u16(acc_1_3_11_13_h, acc_0_14_h);

    uint16x8_t acc_2_4_10_12_l =
        vmlaq_u16(acc_4_10_l, acc_2_12_l, const_25_u16_);
    uint16x8_t acc_2_4_10_12_h =
        vmlaq_u16(acc_4_10_h, acc_2_12_h, const_25_u16_);

    uint32x4x4_t acc = {{
        vaddl_u16(vget_low_u16(acc_2_4_10_12_l),
                  vget_low_u16(acc_0_1_3_11_13_14_l)),
        vaddl_u16(vget_high_u16(acc_2_4_10_12_l),
                  vget_high_u16(acc_0_1_3_11_13_14_l)),
        vaddl_u16(vget_low_u16(acc_2_4_10_12_h),
                  vget_low_u16(acc_0_1_3_11_13_14_h)),
        vaddl_u16(vget_high_u16(acc_2_4_10_12_h),
                  vget_high_u16(acc_0_1_3_11_13_14_h)),
    }};

    acc.val[0] =
        vmlal_u16(acc.val[0], vget_low_u16(acc_6_8_l), const_146_u16_half_);
    acc.val[1] =
        vmlal_u16(acc.val[1], vget_high_u16(acc_6_8_l), const_146_u16_half_);
    acc.val[2] =
        vmlal_u16(acc.val[2], vget_low_u16(acc_6_8_h), const_146_u16_half_);
    acc.val[3] =
        vmlal_u16(acc.val[3], vget_high_u16(acc_6_8_h), const_146_u16_half_);

    acc.val[0] =
        vmlal_u16(acc.val[0], vget_low_u16(acc_5_9_l), const_118_u16_half_);
    acc.val[1] =
        vmlal_u16(acc.val[1], vget_high_u16(acc_5_9_l), const_118_u16_half_);
    acc.val[2] =
        vmlal_u16(acc.val[2], vget_low_u16(acc_5_9_h), const_118_u16_half_);
    acc.val[3] =
        vmlal_u16(acc.val[3], vget_high_u16(acc_5_9_h), const_118_u16_half_);

    acc.val[0] =
        vmlal_u16(acc.val[0], vget_low_u16(acc_7_l), const_158_u16_half_);
    acc.val[1] =
        vmlal_u16(acc.val[1], vget_high_u16(acc_7_l), const_158_u16_half_);
    acc.val[2] =
        vmlal_u16(acc.val[2], vget_low_u16(acc_7_h), const_158_u16_half_);
    acc.val[3] =
        vmlal_u16(acc.val[3], vget_high_u16(acc_7_h), const_158_u16_half_);

    vst1q_u32_x4(&dst[0], acc);
  }

  // Applies vertical filtering vector using scalar operations.
  //
  // DST = [ SRC0, SRC1, SRC2, SRC3...SRC11, SRC12, SRC13, SRC14 ] *
  //     * [ 4, 11, 25, 48 ... 48, 25, 11, 4 ]T
  void vertical_scalar_path(const SourceType src[15], BufferType *dst) const {
    uint32_t acc = (static_cast<uint32_t>(src[3]) + src[11]) * 4;
    acc += (acc + src[1] + src[13]) * 11;
    acc += (src[0] + src[14]) * 4 + (src[2] + src[12]) * 25 +
           (src[4] + src[10]) * 81;
    acc += (src[5] + src[9]) * 118 + (src[6] + src[8]) * 146 + src[7] * 158;
    dst[0] = acc;
  }

  // Applies horizontal filtering vector using SIMD operations.
  //
  // DST = 1/1048576 * [ SRC0, SRC1, SRC2, SRC3...SRC11, SRC12, SRC13, SRC14 ] *
  //                 * [ 4, 11, 25, 48 ... 48, 25, 11, 4 ]T
  void horizontal_vector_path(uint32x4_t src[15], DestinationType *dst) const {
    uint32x4_t acc_1_13 = vaddq_u32(src[1], src[13]);
    uint32x4_t acc_2_12 = vaddq_u32(src[2], src[12]);
    uint32x4_t acc_6_8 = vaddq_u32(src[6], src[8]);
    uint32x4_t acc_5_9 = vaddq_u32(src[5], src[9]);
    uint32x4_t acc_0_14 = vaddq_u32(src[0], src[14]);
    uint32x4_t acc_3_11 = vaddq_u32(src[3], src[11]);
    uint32x4_t acc_4_10 = vaddq_u32(src[4], src[10]);

    acc_0_14 = vshlq_n_u32(acc_0_14, 2);
    acc_3_11 = vshlq_n_u32(acc_3_11, 2);
    acc_4_10 = vmulq_u32(acc_4_10, const_81_u32_);

    uint32x4_t acc_1_3_11_13 = vaddq_u32(acc_3_11, acc_1_13);
    acc_1_3_11_13 = vmlaq_u32(acc_3_11, acc_1_3_11_13, const_11_u32_);
    uint32x4_t acc_0_1_3_11_13_14 = vaddq_u32(acc_1_3_11_13, acc_0_14);
    uint32x4_t acc_2_4_10_12 = vmlaq_u32(acc_4_10, acc_2_12, const_25_u32_);

    uint32x4_t acc = vaddq_u32(acc_2_4_10_12, acc_0_1_3_11_13_14);
    acc = vmlaq_u32(acc, acc_6_8, const_146_u32_);
    acc = vmlaq_u32(acc, acc_5_9, const_118_u32_);
    acc = vmlaq_u32(acc, src[7], const_158_u32_);
    acc = vrshrq_n_u32(acc, 20);

    uint16x4_t narrowed = vmovn_u32(acc);
    uint8x8_t interleaved =
        vuzp1_u8(vreinterpret_u8_u16(narrowed), vreinterpret_u8_u16(narrowed));
    uint32_t result = vget_lane_u32(vreinterpret_u32_u8(interleaved), 0);
    memcpy(&dst[0], &result, sizeof(result));
  }

  // Applies horizontal filtering vector using scalar operations.
  //
  // DST = 1/1048576 * [ SRC0, SRC1, SRC2, SRC3...SRC11, SRC12, SRC13, SRC14 ] *
  //                 * [ 4, 11, 25, 48 ... 48, 25, 11, 4 ]T
  void horizontal_scalar_path(const BufferType src[15],
                              DestinationType *dst) const {
    uint32_t acc = (static_cast<uint32_t>(src[3]) + src[11]) * 4;
    acc += (acc + src[1] + src[13]) * 11;
    acc += (src[0] + src[14]) * 4 + (src[2] + src[12]) * 25 +
           (src[4] + src[10]) * 81;
    acc += (src[5] + src[9]) * 118 + (src[6] + src[8]) * 146 + src[7] * 158;
    dst[0] = static_cast<DestinationType>(rounding_shift_right(acc, 20));
  }

 private:
  uint16x8_t const_11_u16_;
  uint32x4_t const_11_u32_;
  uint16x8_t const_25_u16_;
  uint32x4_t const_25_u32_;
  uint16x8_t const_81_u16_;
  uint32x4_t const_81_u32_;
  uint16x4_t const_118_u16_half_;
  uint32x4_t const_118_u32_;
  uint16x4_t const_146_u16_half_;
  uint32x4_t const_146_u32_;
  uint16x4_t const_158_u16_half_;
  uint32x4_t const_158_u32_;
};  // end of class GaussianBlur<uint8_t, 15, true>

template <size_t KernelSize>
class GaussianBlur<uint8_t, KernelSize, false> {
 public:
  using SourceType = uint8_t;
  using BufferType = uint32_t;
  using DestinationType = uint8_t;

  static constexpr size_t kHalfKernelSize = get_half_kernel_size(KernelSize);

  // Ignored because vectors are initialized in the constructor body.
  // NOLINTNEXTLINE - hicpp-member-init
  explicit GaussianBlur(float sigma)
      : half_kernel_(generate_gaussian_half_kernel<kHalfKernelSize>(sigma)) {
    for (size_t i = 0; i < kHalfKernelSize; i++) {
      half_kernel_u16_[i] = vdupq_n_u16(half_kernel_[i]);
      half_kernel_u32_[i] = vdupq_n_u32(half_kernel_[i]);
    }
  }

  void vertical_vector_path(uint8x16_t src[KernelSize], BufferType *dst) const {
    uint16x8_t initial_l = vmovl_u8(vget_low_u8(src[KernelSize >> 1]));
    uint16x8_t initial_h = vmovl_high_u8(src[KernelSize >> 1]);

    uint32x4_t acc_l_l =
        vmull_u16(vget_low_u16(initial_l),
                  vget_low_u16(half_kernel_u16_[KernelSize >> 1]));
    uint32x4_t acc_l_h =
        vmull_high_u16(initial_l, half_kernel_u16_[KernelSize >> 1]);
    uint32x4_t acc_h_l =
        vmull_u16(vget_low_u16(initial_h),
                  vget_low_u16(half_kernel_u16_[KernelSize >> 1]));
    uint32x4_t acc_h_h =
        vmull_high_u16(initial_h, half_kernel_u16_[KernelSize >> 1]);

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 0; i < (KernelSize >> 1); i++) {
      const size_t j = KernelSize - i - 1;
      uint16x8_t vec_l = vaddl_u8(vget_low_u8(src[i]), vget_low_u8(src[j]));
      uint16x8_t vec_h = vaddl_high_u8(src[i], src[j]);

      acc_l_l = vmlal_u16(acc_l_l, vget_low_u16(vec_l),
                          vget_low_u16(half_kernel_u16_[i]));
      acc_l_h = vmlal_high_u16(acc_l_h, vec_l, half_kernel_u16_[i]);
      acc_h_l = vmlal_u16(acc_h_l, vget_low_u16(vec_h),
                          vget_low_u16(half_kernel_u16_[i]));
      acc_h_h = vmlal_high_u16(acc_h_h, vec_h, half_kernel_u16_[i]);
    }

    uint32x4x4_t result = {acc_l_l, acc_l_h, acc_h_l, acc_h_h};

    vst1q_u32_x4(&dst[0], result);
  }

  void vertical_scalar_path(const SourceType src[KernelSize],
                            BufferType *dst) const {
    BufferType acc = static_cast<BufferType>(src[0]) * half_kernel_[0];

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 1; i <= (KernelSize >> 1); i++) {
      acc += static_cast<BufferType>(src[i]) * half_kernel_[i];
    }

    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = (KernelSize >> 1) + 1; i < KernelSize; i++) {
      size_t j = KernelSize - i - 1;
      acc += static_cast<BufferType>(src[i]) * half_kernel_[j];
    }

    dst[0] = acc;
  }

  void horizontal_vector_path(uint32x4_t src[KernelSize],
                              DestinationType *dst) const {
    uint32x4_t acc =
        vmulq_u32(src[KernelSize >> 1], half_kernel_u32_[KernelSize >> 1]);

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 0; i < (KernelSize >> 1); i++) {
      const size_t j = KernelSize - i - 1;
      uint32x4_t vec_inner = vaddq_u32(src[i], src[j]);
      acc = vmlaq_u32(acc, vec_inner, half_kernel_u32_[i]);
    }

    uint32x4_t acc_u32 = vrshrq_n_u32(acc, 16);
    uint16x4_t narrowed = vmovn_u32(acc_u32);
    uint8x8_t interleaved =
        vuzp1_u8(vreinterpret_u8_u16(narrowed), vreinterpret_u8_u16(narrowed));
    uint32_t result = vget_lane_u32(vreinterpret_u32_u8(interleaved), 0);
    memcpy(&dst[0], &result, sizeof(result));
  }

  void horizontal_scalar_path(const BufferType src[KernelSize],
                              DestinationType *dst) const {
    BufferType acc = src[0] * half_kernel_[0];

    // Optimization to avoid unnecessary branching in vector code.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = 1; i <= (KernelSize >> 1); i++) {
      acc += src[i] * half_kernel_[i];
    }

    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t i = (KernelSize >> 1) + 1; i < KernelSize; i++) {
      size_t j = KernelSize - i - 1;
      acc += src[i] * half_kernel_[j];
    }

    dst[0] = static_cast<DestinationType>(rounding_shift_right(acc, 16));
  }

 private:
  const std::array<uint16_t, kHalfKernelSize> half_kernel_;
  uint16x8_t half_kernel_u16_[kHalfKernelSize];
  uint32x4_t half_kernel_u32_[kHalfKernelSize];
};  // end of class GaussianBlur<uint8_t, KernelSize, false>

template <size_t KernelSize, bool IsBinomial, typename ScalarType>
static kleidicv_error_t gaussian_blur_fixed_kernel_size(
    const ScalarType *src, size_t src_stride, ScalarType *dst,
    size_t dst_stride, Rectangle &rect, size_t y_begin, size_t y_end,
    size_t channels, float sigma, FixedBorderType border_type,
    SeparableFilterWorkspace *workspace) {
  using GaussianBlurFilter = GaussianBlur<ScalarType, KernelSize, IsBinomial>;

  GaussianBlurFilter blur{sigma};
  SeparableFilter<GaussianBlurFilter, KernelSize> filter{blur};

  Rows<const ScalarType> src_rows{src, src_stride, channels};
  Rows<ScalarType> dst_rows{dst, dst_stride, channels};
  workspace->process(rect, y_begin, y_end, src_rows, dst_rows, channels,
                     border_type, filter);

  return KLEIDICV_OK;
}

template <bool IsBinomial, typename ScalarType>
static kleidicv_error_t gaussian_blur(size_t kernel_size, const ScalarType *src,
                                      size_t src_stride, ScalarType *dst,
                                      size_t dst_stride, Rectangle &rect,
                                      size_t y_begin, size_t y_end,
                                      size_t channels, float sigma,
                                      FixedBorderType border_type,
                                      SeparableFilterWorkspace *workspace) {
  switch (kernel_size) {
    case 3:
      return gaussian_blur_fixed_kernel_size<3, IsBinomial>(
          src, src_stride, dst, dst_stride, rect, y_begin, y_end, channels,
          sigma, border_type, workspace);
    case 5:
      return gaussian_blur_fixed_kernel_size<5, IsBinomial>(
          src, src_stride, dst, dst_stride, rect, y_begin, y_end, channels,
          sigma, border_type, workspace);
    case 7:
      return gaussian_blur_fixed_kernel_size<7, IsBinomial>(
          src, src_stride, dst, dst_stride, rect, y_begin, y_end, channels,
          sigma, border_type, workspace);
    case 15:
      return gaussian_blur_fixed_kernel_size<15, IsBinomial>(
          src, src_stride, dst, dst_stride, rect, y_begin, y_end, channels,
          sigma, border_type, workspace);
    // gaussian_blur_is_implemented checked the kernel size already.
    // GCOVR_EXCL_START
    default:
      assert(!"kernel size not implemented");
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
      // GCOVR_EXCL_STOP
  }
}

// Does not include checks for whether the operation is implemented.
// This must be done earlier, by gaussian_blur_is_implemented.
template <typename T>
static kleidicv_error_t gaussian_blur_checks(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t channels, SeparableFilterWorkspace *workspace) {
  CHECK_POINTERS(workspace);

  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return KLEIDICV_ERROR_RANGE;
  }

  if (workspace->channels() < channels) {
    return KLEIDICV_ERROR_CONTEXT_MISMATCH;
  }

  Rectangle rect{width, height};
  const Rectangle &context_rect = workspace->image_size();
  if (context_rect.width() < width || context_rect.height() < height) {
    return KLEIDICV_ERROR_CONTEXT_MISMATCH;
  }

  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t gaussian_blur_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t /*kernel_height*/, float sigma_x,
    float /*sigma_y*/, FixedBorderType fixed_border_type,
    kleidicv_filter_context_t *context) {
  auto *workspace = reinterpret_cast<SeparableFilterWorkspace *>(context);
  kleidicv_error_t checks_result = gaussian_blur_checks(
      src, src_stride, dst, dst_stride, width, height, channels, workspace);

  if (checks_result != KLEIDICV_OK) {
    return checks_result;
  }

  Rectangle rect{width, height};

  if (sigma_x == 0.0) {
    return gaussian_blur<true>(kernel_width, src, src_stride, dst, dst_stride,
                               rect, y_begin, y_end, channels, sigma_x,
                               fixed_border_type, workspace);
  }

  return gaussian_blur<false>(kernel_width, src, src_stride, dst, dst_stride,
                              rect, y_begin, y_end, channels, sigma_x,
                              fixed_border_type, workspace);
}

}  // namespace kleidicv::neon
