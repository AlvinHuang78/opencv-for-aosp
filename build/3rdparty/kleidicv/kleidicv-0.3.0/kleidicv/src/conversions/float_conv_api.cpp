// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/float_conversion.h"
#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"

namespace kleidicv {

namespace sve2 {

template <typename InputType, typename OutputType>
kleidicv_error_t float_conversion(const InputType* src, size_t src_stride,
                                  OutputType* dst, size_t dst_stride,
                                  size_t width, size_t height);

}  // namespace sve2

namespace sme2 {

template <typename InputType, typename OutputType>
kleidicv_error_t float_conversion(const InputType* src, size_t src_stride,
                                  OutputType* dst, size_t dst_stride,
                                  size_t width, size_t height);

}  // namespace sme2

}  // namespace kleidicv

KLEIDICV_MULTIVERSION_C_API(
    kleidicv_f32_to_s8, &kleidicv::neon::f32_to_s8,
    KLEIDICV_SVE2_IMPL_IF((&kleidicv::sve2::float_conversion<float, int8_t>)),
    (&kleidicv::sme2::float_conversion<float, int8_t>));

KLEIDICV_MULTIVERSION_C_API(
    kleidicv_f32_to_u8, &kleidicv::neon::f32_to_u8,
    KLEIDICV_SVE2_IMPL_IF((&kleidicv::sve2::float_conversion<float, uint8_t>)),
    (&kleidicv::sme2::float_conversion<float, uint8_t>));

KLEIDICV_MULTIVERSION_C_API(kleidicv_s8_to_f32, &kleidicv::neon::s8_to_f32,
                            (&kleidicv::sve2::float_conversion<int8_t, float>),
                            (&kleidicv::sme2::float_conversion<int8_t, float>));

KLEIDICV_MULTIVERSION_C_API(
    kleidicv_u8_to_f32, &kleidicv::neon::u8_to_f32,
    (&kleidicv::sve2::float_conversion<uint8_t, float>),
    (&kleidicv::sme2::float_conversion<uint8_t, float>));
