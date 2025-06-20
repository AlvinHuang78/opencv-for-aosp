// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_CONVERSIONS_RGB_TO_RGB_H
#define KLEIDICV_CONVERSIONS_RGB_TO_RGB_H

#include "kleidicv/kleidicv.h"

namespace kleidicv {

namespace neon {

kleidicv_error_t rgb_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);
kleidicv_error_t rgba_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height);
kleidicv_error_t rgb_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);
kleidicv_error_t rgb_to_rgba_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);
kleidicv_error_t rgba_to_bgr_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);
kleidicv_error_t rgba_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);

}  // namespace neon

namespace sve2 {

kleidicv_error_t rgb_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);
kleidicv_error_t rgba_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height);
kleidicv_error_t rgb_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);
kleidicv_error_t rgb_to_rgba_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);
kleidicv_error_t rgba_to_bgr_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);
kleidicv_error_t rgba_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);

}  // namespace sve2

namespace sme2 {

kleidicv_error_t rgb_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height);

kleidicv_error_t rgba_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height);

kleidicv_error_t rgb_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);

kleidicv_error_t rgb_to_rgba_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);

kleidicv_error_t rgba_to_bgr_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);

kleidicv_error_t rgba_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height);

}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_CONVERSIONS_RGB_TO_RGB_H
