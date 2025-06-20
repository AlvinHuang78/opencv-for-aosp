// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"

namespace kleidicv {

namespace neon {

template <typename T>
kleidicv_error_t bitwise_and(const T *src_a, size_t src_a_stride,
                             const T *src_b, size_t src_b_stride, T *dst,
                             size_t dst_stride, size_t width, size_t height);

}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t bitwise_and(const T *src_a, size_t src_a_stride,
                             const T *src_b, size_t src_b_stride, T *dst,
                             size_t dst_stride, size_t width, size_t height);

}  // namespace sve2

namespace sme2 {
template <typename T>
kleidicv_error_t bitwise_and(const T *src_a, size_t src_a_stride,
                             const T *src_b, size_t src_b_stride, T *dst,
                             size_t dst_stride, size_t width, size_t height);

}  // namespace sme2

}  // namespace kleidicv

#define KLEIDICV_DEFINE_C_API(name, type)                        \
  KLEIDICV_MULTIVERSION_C_API(                                   \
      name, &kleidicv::neon::bitwise_and<type>,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::bitwise_and<type>), \
      KLEIDICV_SME2_IMPL_IF(&kleidicv::sme2::bitwise_and<type>))

KLEIDICV_DEFINE_C_API(kleidicv_bitwise_and, uint8_t);
