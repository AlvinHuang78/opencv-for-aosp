// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/rgb_to_rgb.h"
#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"

#define KLEIDICV_DEFINE_C_API(name, partialname)           \
  KLEIDICV_MULTIVERSION_C_API(                             \
      name, &kleidicv::neon::partialname,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::partialname), \
      &kleidicv::sme2::partialname)

KLEIDICV_DEFINE_C_API(kleidicv_rgb_to_bgr_u8, rgb_to_bgr_u8);
KLEIDICV_DEFINE_C_API(kleidicv_rgba_to_bgra_u8, rgba_to_bgra_u8);
KLEIDICV_DEFINE_C_API(kleidicv_rgb_to_bgra_u8, rgb_to_bgra_u8);
KLEIDICV_DEFINE_C_API(kleidicv_rgb_to_rgba_u8, rgb_to_rgba_u8);
KLEIDICV_DEFINE_C_API(kleidicv_rgba_to_bgr_u8, rgba_to_bgr_u8);
KLEIDICV_DEFINE_C_API(kleidicv_rgba_to_rgb_u8, rgba_to_rgb_u8);

extern "C" {

using KLEIDICV_TARGET_NAMESPACE::CopyRows;
using KLEIDICV_TARGET_NAMESPACE::Rectangle;
using KLEIDICV_TARGET_NAMESPACE::Rows;

static kleidicv_error_t kleidicv_rgb_to_rgb_u8_impl(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 3 /* RGB */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 3 /* BGR */};
  CopyRows<uint8_t>::copy_rows(rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

decltype(kleidicv_rgb_to_rgb_u8_impl) *kleidicv_rgb_to_rgb_u8 =
    kleidicv_rgb_to_rgb_u8_impl;

static kleidicv_error_t kleidicv_rgba_to_rgba_u8_impl(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle rect{width, height};
  Rows<const uint8_t> src_rows{src, src_stride, 4 /* RGBA */};
  Rows<uint8_t> dst_rows{dst, dst_stride, 4 /* RGBA */};
  CopyRows<uint8_t>::copy_rows(rect, src_rows, dst_rows);
  return KLEIDICV_OK;
}

decltype(kleidicv_rgba_to_rgba_u8_impl) *kleidicv_rgba_to_rgba_u8 =
    kleidicv_rgba_to_rgba_u8_impl;

}  // extern "C"
