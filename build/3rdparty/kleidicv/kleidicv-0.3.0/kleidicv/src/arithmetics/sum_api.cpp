// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv {

namespace neon {

template <typename T>
kleidicv_error_t sum(const T *src, size_t src_stride, size_t width,
                     size_t height, T *sum);

}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t sum(const T *src, size_t src_stride, size_t width,
                     size_t height, T *sum);

}  // namespace sve2

namespace sme2 {

template <typename T>
kleidicv_error_t sum(const T *src, size_t src_stride, size_t width,
                     size_t height, T *sum);

}  // namespace sme2

}  // namespace kleidicv

KLEIDICV_MULTIVERSION_C_API(kleidicv_sum_f32, &kleidicv::neon::sum<float>,
                            KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::sum<float>),
                            KLEIDICV_SME2_IMPL_IF(&kleidicv::sme2::sum<float>));
