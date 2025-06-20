// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/resize/resize.h"

namespace kleidicv::neon {

KLEIDICV_TARGET_FN_ATTRS
static kleidicv_error_t check_dimensions(size_t src_dim, size_t dst_dim) {
  size_t half_src_dim = src_dim / 2;

  if ((src_dim % 2) == 0) {
    if (dst_dim == half_src_dim) {
      return KLEIDICV_OK;
    }
  } else {
    if (dst_dim == half_src_dim || dst_dim == (half_src_dim + 1)) {
      return KLEIDICV_OK;
    }
  }

  return KLEIDICV_ERROR_RANGE;
}

// Disable the warning, as the complexity is just above the threshold, it's
// better to leave it in one piece.
// NOLINTBEGIN(readability-function-cognitive-complexity)
KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t resize_to_quarter_u8(const uint8_t *src, size_t src_stride,
                                      size_t src_width, size_t src_height,
                                      uint8_t *dst, size_t dst_stride,
                                      size_t dst_width, size_t dst_height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_IMAGE_SIZE(src_width, src_height);

  if (kleidicv_error_t ret = check_dimensions(src_width, dst_width)) {
    return ret;
  }

  if (kleidicv_error_t ret = check_dimensions(src_height, dst_height)) {
    return ret;
  }

  for (; src_height >= 2; src_height -= 2, src += (src_stride * 2),
                          --dst_height, dst += dst_stride) {
    const uint8_t *src_l = src;
    uint8_t *dst_l = dst;
    size_t src_width_l = src_width;
    size_t dst_width_l = dst_width;

    for (; src_width_l >= 32;
         src_width_l -= 32, dst_width_l -= 16, dst_l += 16, src_l += 32) {
      uint8x16_t top_line_0 = vld1q_u8(src_l);
      uint8x16_t top_line_1 = vld1q_u8(&src_l[16]);
      uint8x16_t bottom_line_0 = vld1q_u8(&src_l[src_stride]);
      uint8x16_t bottom_line_1 = vld1q_u8(&src_l[src_stride + 16]);

      uint16x8_t top_line_pairs_summed_0 = vpaddlq_u8(top_line_0);
      uint16x8_t top_line_pairs_summed_1 = vpaddlq_u8(top_line_1);
      uint16x8_t bottom_line_pairs_summed_0 = vpaddlq_u8(bottom_line_0);
      uint16x8_t bottom_line_pairs_summed_1 = vpaddlq_u8(bottom_line_1);

      uint16x8_t result_before_averaging_0 =
          vaddq_u16(top_line_pairs_summed_0, bottom_line_pairs_summed_0);
      uint16x8_t result_before_averaging_1 =
          vaddq_u16(top_line_pairs_summed_1, bottom_line_pairs_summed_1);

      uint8x8_t result_0 = vrshrn_n_u16(result_before_averaging_0, 2);
      uint8x8_t result_1 = vrshrn_n_u16(result_before_averaging_1, 2);

      vst1_u8(&dst_l[0], result_0);
      vst1_u8(&dst_l[8], result_1);
    }

    for (; src_width_l > 1;
         src_width_l -= 2, src_l += 2, --dst_width_l, ++dst_l) {
      disable_loop_vectorization();
      *dst_l = rounding_shift_right<uint16_t>(
          static_cast<uint16_t>(*src_l) + *(src_l + 1) + *(src_l + src_stride) +
              *(src_l + src_stride + 1),
          2);
    }

    if (dst_width_l) {
      *dst_l = rounding_shift_right<uint16_t>(
          static_cast<uint16_t>(*src_l) + *(src_l + src_stride), 1);
    }
  }

  if (dst_height) {
    for (; src_width >= 32;
         src_width -= 32, dst_width -= 16, dst += 16, src += 32) {
      uint8x16_t vsrc_0 = vld1q_u8(&src[0]);
      uint8x16_t vsrc_1 = vld1q_u8(&src[16]);

      uint16x8_t vsrc_line_pairs_summed_0 = vpaddlq_u8(vsrc_0);
      uint16x8_t vsrc_line_pairs_summed_1 = vpaddlq_u8(vsrc_1);

      uint8x8_t result_0 = vrshrn_n_u16(vsrc_line_pairs_summed_0, 1);
      uint8x8_t result_1 = vrshrn_n_u16(vsrc_line_pairs_summed_1, 1);

      vst1_u8(&dst[0], result_0);
      vst1_u8(&dst[8], result_1);
    }

    for (; src_width > 1; src_width -= 2, src += 2, --dst_width, ++dst) {
      disable_loop_vectorization();
      *dst = rounding_shift_right<uint16_t>(
          static_cast<uint16_t>(*src) + *(src + 1), 1);
    }

    if (dst_width) {
      *dst = *src;
    }
  }
  return KLEIDICV_OK;
}
// NOLINTEND(readability-function-cognitive-complexity)

}  // namespace kleidicv::neon
