// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "morphology_sc.h"

namespace kleidicv::sve2 {

template <typename T>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
dilate(const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
       size_t height, kleidicv_morphology_context_t *context) {
  return dilate_sc<T, MorphologyWorkspace::CopyDataMemcpy<T> >(
      src, src_stride, dst, dst_stride, width, height, context);
}

template <typename T>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
erode(const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
      size_t height, kleidicv_morphology_context_t *context) {
  return erode_sc<T, MorphologyWorkspace::CopyDataMemcpy<T> >(
      src, src_stride, dst, dst_stride, width, height, context);
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(name, type)                       \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t name<type>(        \
      const type *src, size_t src_stride, type *dst, size_t dst_stride, \
      size_t width, size_t height, kleidicv_morphology_context_t *context)

KLEIDICV_INSTANTIATE_TEMPLATE(dilate, uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(erode, uint8_t);

}  // namespace kleidicv::sve2
