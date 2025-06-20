// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/filters/separable_filter_2d.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/workspace/separable.h"

namespace kleidicv {

namespace neon {

template <typename T>
kleidicv_error_t separable_filter_2d_stripe(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    const T *kernel_x, size_t kernel_width, const T *kernel_y,
    size_t kernel_height, FixedBorderType border_type,
    kleidicv_filter_context_t *context);

}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t separable_filter_2d_stripe(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    const T *kernel_x, size_t kernel_width, const T *kernel_y,
    size_t kernel_height, FixedBorderType border_type,
    kleidicv_filter_context_t *context);

}  // namespace sve2

namespace sme2 {

template <typename T>
kleidicv_error_t separable_filter_2d_stripe(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    const T *kernel_x, size_t kernel_width, const T *kernel_y,
    size_t kernel_height, FixedBorderType border_type,
    kleidicv_filter_context_t *context);

}  // namespace sme2

}  // namespace kleidicv

#define KLEIDICV_DEFINE_C_API(name, type)                                      \
  KLEIDICV_MULTIVERSION_C_API(                                                 \
      name, &kleidicv::neon::separable_filter_2d_stripe<type>,                 \
      KLEIDICV_SVE2_IMPL_IF(kleidicv::sve2::separable_filter_2d_stripe<type>), \
      &kleidicv::sme2::separable_filter_2d_stripe<type>)

KLEIDICV_DEFINE_C_API(kleidicv_separable_filter_2d_stripe_u8, uint8_t);
KLEIDICV_DEFINE_C_API(kleidicv_separable_filter_2d_stripe_u16, uint16_t);
KLEIDICV_DEFINE_C_API(kleidicv_separable_filter_2d_stripe_s16, int16_t);

extern "C" {

using KLEIDICV_TARGET_NAMESPACE::Rectangle;
using KLEIDICV_TARGET_NAMESPACE::SeparableFilterWorkspace;

kleidicv_error_t kleidicv_filter_context_create(
    kleidicv_filter_context_t **context, size_t max_channels,
    size_t max_kernel_width, size_t max_kernel_height, size_t max_image_width,
    size_t max_image_height) {
  CHECK_POINTERS(context);

  if (max_kernel_width != max_kernel_height) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  if (max_channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return KLEIDICV_ERROR_RANGE;
  }

  CHECK_IMAGE_SIZE(max_image_width, max_image_height);

  // As we cannot predict the intermediate size based on the parameters given,
  // just use the largest possible size out of all available operations.
  constexpr size_t intermediate_size = sizeof(uint32_t);
  auto workspace = SeparableFilterWorkspace::create(
      Rectangle{max_image_width, max_image_height}, max_channels,
      intermediate_size);
  if (!workspace) {
    *context = nullptr;
    return KLEIDICV_ERROR_ALLOCATION;
  }

  *context = reinterpret_cast<kleidicv_filter_context_t *>(workspace.release());
  return KLEIDICV_OK;
}

kleidicv_error_t kleidicv_filter_context_release(
    kleidicv_filter_context_t *context) {
  CHECK_POINTERS(context);

  // Deliberately create and immediately destroy a unique_ptr to delete the
  // workspace.
  // NOLINTBEGIN(bugprone-unused-raii)
  SeparableFilterWorkspace::Pointer{
      reinterpret_cast<SeparableFilterWorkspace *>(context)};
  // NOLINTEND(bugprone-unused-raii)
  return KLEIDICV_OK;
}

kleidicv_error_t kleidicv_separable_filter_2d_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint8_t *kernel_x,
    size_t kernel_width, const uint8_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context) {
  if (!kleidicv::separable_filter_2d_is_implemented(width, height, kernel_width,
                                                    kernel_height)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  auto fixed_border_type = kleidicv::get_fixed_border_type(border_type);
  if (!fixed_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  return kleidicv_separable_filter_2d_stripe_u8(
      src, src_stride, dst, dst_stride, width, height, 0, height, channels,
      kernel_x, kernel_width, kernel_y, kernel_height, *fixed_border_type,
      context);
}

kleidicv_error_t kleidicv_separable_filter_2d_u16(
    const uint16_t *src, size_t src_stride, uint16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint16_t *kernel_x,
    size_t kernel_width, const uint16_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context) {
  if (!kleidicv::separable_filter_2d_is_implemented(width, height, kernel_width,
                                                    kernel_height)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  auto fixed_border_type = kleidicv::get_fixed_border_type(border_type);
  if (!fixed_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  return kleidicv_separable_filter_2d_stripe_u16(
      src, src_stride, dst, dst_stride, width, height, 0, height, channels,
      kernel_x, kernel_width, kernel_y, kernel_height, *fixed_border_type,
      context);
}

kleidicv_error_t kleidicv_separable_filter_2d_s16(
    const int16_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const int16_t *kernel_x,
    size_t kernel_width, const int16_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context) {
  if (!kleidicv::separable_filter_2d_is_implemented(width, height, kernel_width,
                                                    kernel_height)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  auto fixed_border_type = kleidicv::get_fixed_border_type(border_type);
  if (!fixed_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  return kleidicv_separable_filter_2d_stripe_s16(
      src, src_stride, dst, dst_stride, width, height, 0, height, channels,
      kernel_x, kernel_width, kernel_y, kernel_height, *fixed_border_type,
      context);
}

}  // extern "C"
