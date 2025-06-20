// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RESIZE_RESIZE_LINEAR_H
#define KLEIDICV_RESIZE_RESIZE_LINEAR_H

#include <algorithm>
#include <array>

#include "kleidicv/kleidicv.h"

namespace kleidicv {

inline bool resize_linear_u8_is_implemented(size_t src_width, size_t src_height,
                                            size_t dst_width,
                                            size_t dst_height) {
  if (src_width == 0 || src_height == 0) {
    return true;
  }
  const std::array<size_t, 2> implemented_ratios = {2, 4};
  return std::any_of(implemented_ratios.begin(), implemented_ratios.end(),
                     [&](size_t ratio) {
                       return src_width * ratio == dst_width &&
                              src_height * ratio == dst_height;
                     });
}

inline bool resize_linear_f32_is_implemented(size_t src_width,
                                             size_t src_height,
                                             size_t dst_width,
                                             size_t dst_height) {
  if (src_width == 0 || src_height == 0) {
    return true;
  }
  const std::array<size_t, 3> implemented_ratios = {2, 4, 8};
  return std::any_of(implemented_ratios.begin(), implemented_ratios.end(),
                     [&](size_t ratio) {
                       return src_width * ratio == dst_width &&
                              src_height * ratio == dst_height;
                     });
}

namespace neon {
kleidicv_error_t resize_linear_stripe_u8(const uint8_t *src, size_t src_stride,
                                         size_t src_width, size_t src_height,
                                         size_t y_begin, size_t y_end,
                                         uint8_t *dst, size_t dst_stride,
                                         size_t dst_width, size_t dst_height);
kleidicv_error_t resize_linear_stripe_f32(const float *src, size_t src_stride,
                                          size_t src_width, size_t src_height,
                                          size_t y_begin, size_t y_end,
                                          float *dst, size_t dst_stride,
                                          size_t dst_width, size_t dst_height);
}  // namespace neon

namespace sve2 {
kleidicv_error_t resize_linear_stripe_u8(const uint8_t *src, size_t src_stride,
                                         size_t src_width, size_t src_height,
                                         size_t y_begin, size_t y_end,
                                         uint8_t *dst, size_t dst_stride,
                                         size_t dst_width, size_t dst_height);
kleidicv_error_t resize_linear_stripe_f32(const float *src, size_t src_stride,
                                          size_t src_width, size_t src_height,
                                          size_t y_begin, size_t y_end,
                                          float *dst, size_t dst_stride,
                                          size_t dst_width, size_t dst_height);
}  // namespace sve2

namespace sme2 {
kleidicv_error_t resize_linear_stripe_u8(const uint8_t *src, size_t src_stride,
                                         size_t src_width, size_t src_height,
                                         size_t y_begin, size_t y_end,
                                         uint8_t *dst, size_t dst_stride,
                                         size_t dst_width, size_t dst_height);
kleidicv_error_t resize_linear_stripe_f32(const float *src, size_t src_stride,
                                          size_t src_width, size_t src_height,
                                          size_t y_begin, size_t y_end,
                                          float *dst, size_t dst_stride,
                                          size_t dst_width, size_t dst_height);
}  // namespace sme2

}  // namespace kleidicv

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
/// Internal - not part of the public API and its direct use is not supported.
/// It is used by the multithreaded function.
extern kleidicv_error_t (*kleidicv_resize_linear_stripe_u8)(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height);

/// Internal - not part of the public API and its direct use is not supported.
/// It is used by the multithreaded function.
extern kleidicv_error_t (*kleidicv_resize_linear_stripe_f32)(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, float *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // KLEIDICV_RESIZE_RESIZE_H
