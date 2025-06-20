// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv {

namespace neon {
template <typename ScalarType>
kleidicv_error_t compare_equal(const ScalarType *src_a, size_t src_a_stride,
                               const ScalarType *src_b, size_t src_b_stride,
                               ScalarType *dst, size_t dst_stride, size_t width,
                               size_t height);

template <typename ScalarType>
kleidicv_error_t compare_greater(const ScalarType *src_a, size_t src_a_stride,
                                 const ScalarType *src_b, size_t src_b_stride,
                                 ScalarType *dst, size_t dst_stride,
                                 size_t width, size_t height);

}  // namespace neon

namespace sve2 {
template <typename ScalarType>
kleidicv_error_t compare_equal(const ScalarType *src_a, size_t src_a_stride,
                               const ScalarType *src_b, size_t src_b_stride,
                               ScalarType *dst, size_t dst_stride, size_t width,
                               size_t height);

template <typename ScalarType>
kleidicv_error_t compare_greater(const ScalarType *src_a, size_t src_a_stride,
                                 const ScalarType *src_b, size_t src_b_stride,
                                 ScalarType *dst, size_t dst_stride,
                                 size_t width, size_t height);
}  // namespace sve2

namespace sme2 {
template <typename ScalarType>
kleidicv_error_t compare_equal(const ScalarType *src_a, size_t src_a_stride,
                               const ScalarType *src_b, size_t src_b_stride,
                               ScalarType *dst, size_t dst_stride, size_t width,
                               size_t height);

template <typename ScalarType>
kleidicv_error_t compare_greater(const ScalarType *src_a, size_t src_a_stride,
                                 const ScalarType *src_b, size_t src_b_stride,
                                 ScalarType *dst, size_t dst_stride,
                                 size_t width, size_t height);
}  // namespace sme2

}  // namespace kleidicv

#define KLEIDICV_DEFINE_CMP_EQ_API(name, type)                     \
  KLEIDICV_MULTIVERSION_C_API(                                     \
      name, &kleidicv::neon::compare_equal<type>,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::compare_equal<type>), \
      KLEIDICV_SME2_IMPL_IF(&kleidicv::sme2::compare_equal<type>))

#define KLEIDICV_DEFINE_CMP_GT_API(name, type)                       \
  KLEIDICV_MULTIVERSION_C_API(                                       \
      name, &kleidicv::neon::compare_greater<type>,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::compare_greater<type>), \
      KLEIDICV_SME2_IMPL_IF(&kleidicv::sme2::compare_greater<type>))

KLEIDICV_DEFINE_CMP_EQ_API(kleidicv_compare_equal_u8, uint8_t);
KLEIDICV_DEFINE_CMP_GT_API(kleidicv_compare_greater_u8, uint8_t);
