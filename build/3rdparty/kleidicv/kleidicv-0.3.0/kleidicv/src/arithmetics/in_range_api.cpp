// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv {

namespace neon {
template <typename T>
kleidicv_error_t in_range(const T *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          T lower_bound, T upper_bound);
}  // namespace neon

namespace sve2 {
template <typename T>
kleidicv_error_t in_range(const T *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          T lower_bound, T upper_bound);
}  // namespace sve2

namespace sme2 {
template <typename T>
kleidicv_error_t in_range(const T *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          T lower_bound, T upper_bound);
}  // namespace sme2

}  // namespace kleidicv

#define KLEIDICV_DEFINE_C_API(name, type)                     \
  KLEIDICV_MULTIVERSION_C_API(                                \
      name, &kleidicv::neon::in_range<type>,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::in_range<type>), \
      KLEIDICV_SME2_IMPL_IF(&kleidicv::sme2::in_range<type>))

KLEIDICV_DEFINE_C_API(kleidicv_in_range_u8, uint8_t);
KLEIDICV_DEFINE_C_API(kleidicv_in_range_f32, float);
