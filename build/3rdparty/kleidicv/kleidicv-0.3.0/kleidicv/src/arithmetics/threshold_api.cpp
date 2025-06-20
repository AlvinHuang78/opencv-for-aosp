// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv {

namespace neon {
template <typename T>
kleidicv_error_t threshold_binary(const T *src, size_t src_stride, T *dst,
                                  size_t dst_stride, size_t width,
                                  size_t height, T threshold, T value);
}  // namespace neon

namespace sve2 {
template <typename T>
kleidicv_error_t threshold_binary(const T *src, size_t src_stride, T *dst,
                                  size_t dst_stride, size_t width,
                                  size_t height, T threshold, T value);
}  // namespace sve2

namespace sme2 {
template <typename T>
kleidicv_error_t threshold_binary(const T *src, size_t src_stride, T *dst,
                                  size_t dst_stride, size_t width,
                                  size_t height, T threshold, T value);
}  // namespace sme2

}  // namespace kleidicv

#define KLEIDICV_DEFINE_C_API(name, type)                             \
  KLEIDICV_MULTIVERSION_C_API(                                        \
      name, &kleidicv::neon::threshold_binary<type>,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::threshold_binary<type>), \
      &kleidicv::sme2::threshold_binary<type>)

KLEIDICV_DEFINE_C_API(kleidicv_threshold_binary_u8, uint8_t);
