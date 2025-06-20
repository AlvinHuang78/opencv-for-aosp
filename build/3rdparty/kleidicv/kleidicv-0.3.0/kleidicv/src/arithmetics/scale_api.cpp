// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"

namespace kleidicv {

namespace neon {

template <typename T>
kleidicv_error_t scale(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       float scale, float shift);
}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t scale(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       float scale, float shift);

}  // namespace sve2

namespace sme2 {
template <typename T>
kleidicv_error_t scale(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       float scale, float shift);

}  // namespace sme2

}  // namespace kleidicv

KLEIDICV_MULTIVERSION_C_API(kleidicv_scale_u8, &kleidicv::neon::scale<uint8_t>,
                            nullptr, nullptr);
KLEIDICV_MULTIVERSION_C_API(
    kleidicv_scale_f32, &kleidicv::neon::scale<float>,
    KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::scale<float>),
    KLEIDICV_SME2_IMPL_IF(&kleidicv::sme2::scale<float>));
