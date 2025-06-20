// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/morphology/workspace.h"

namespace kleidicv {

namespace neon {

template <typename T>
kleidicv_error_t dilate(const T *src, size_t src_stride, T *dst,
                        size_t dst_stride, size_t width, size_t height,
                        kleidicv_morphology_context_t *context);

template <typename T>
kleidicv_error_t erode(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       kleidicv_morphology_context_t *context);

}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t dilate(const T *src, size_t src_stride, T *dst,
                        size_t dst_stride, size_t width, size_t height,
                        kleidicv_morphology_context_t *context);

template <typename T>
kleidicv_error_t erode(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       kleidicv_morphology_context_t *context);

}  // namespace sve2

namespace sme2 {

template <typename T>
kleidicv_error_t dilate(const T *src, size_t src_stride, T *dst,
                        size_t dst_stride, size_t width, size_t height,
                        kleidicv_morphology_context_t *context);

template <typename T>
kleidicv_error_t erode(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       kleidicv_morphology_context_t *context);

}  // namespace sme2

}  // namespace kleidicv

extern "C" {

using KLEIDICV_TARGET_NAMESPACE::MorphologyWorkspace;

kleidicv_error_t kleidicv_morphology_create(
    kleidicv_morphology_context_t **context, kleidicv_rectangle_t kernel,
    kleidicv_point_t anchor, kleidicv_border_type_t border_type,
    kleidicv_border_values_t border_values, size_t channels, size_t iterations,
    size_t type_size, kleidicv_rectangle_t image) {
  CHECK_POINTERS(context);
  *context = nullptr;
  CHECK_RECTANGLE_SIZE(kernel);
  CHECK_RECTANGLE_SIZE(image);

  if (type_size > KLEIDICV_MAXIMUM_TYPE_SIZE) {
    return KLEIDICV_ERROR_RANGE;
  }

  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return KLEIDICV_ERROR_RANGE;
  }

  auto morphology_border_type =
      MorphologyWorkspace::get_border_type(border_type);

  if (!morphology_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  MorphologyWorkspace::Pointer workspace;
  if (kleidicv_error_t error = MorphologyWorkspace::create(
          workspace, kernel, anchor, *morphology_border_type, border_values,
          channels, iterations, type_size, image)) {
    return error;
  }

  *context =
      reinterpret_cast<kleidicv_morphology_context_t *>(workspace.release());
  return KLEIDICV_OK;
}

kleidicv_error_t kleidicv_morphology_release(
    kleidicv_morphology_context_t *context) {
  CHECK_POINTERS(context);

  // Deliberately create and immediately destroy a unique_ptr to delete the
  // workspace.
  // NOLINTBEGIN(bugprone-unused-raii)
  MorphologyWorkspace::Pointer{
      reinterpret_cast<MorphologyWorkspace *>(context)};
  // NOLINTEND(bugprone-unused-raii)
  return KLEIDICV_OK;
}

}  // extern "C"

#define KLEIDICV_DEFINE_C_API(name, tname, type)           \
  KLEIDICV_MULTIVERSION_C_API(                             \
      name, &kleidicv::neon::tname<type>,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::tname<type>), \
      &kleidicv::sme2::tname<type>)

KLEIDICV_DEFINE_C_API(kleidicv_dilate_u8, dilate, uint8_t);
KLEIDICV_DEFINE_C_API(kleidicv_erode_u8, erode, uint8_t);
