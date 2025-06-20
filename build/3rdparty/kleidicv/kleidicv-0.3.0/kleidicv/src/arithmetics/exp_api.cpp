// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"

namespace kleidicv {

namespace neon {

template <typename T>
kleidicv_error_t exp(const T* src, size_t src_stride, T* dst, size_t dst_stride,
                     size_t width, size_t height);

}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t exp(const T* src, size_t src_stride, T* dst, size_t dst_stride,
                     size_t width, size_t height);

}  // namespace sve2

namespace sme2 {

template <typename T>
kleidicv_error_t exp(const T* src, size_t src_stride, T* dst, size_t dst_stride,
                     size_t width, size_t height);

}  // namespace sme2

}  // namespace kleidicv

#define KLEIDICV_DEFINE_C_API(name, type)                       \
  KLEIDICV_MULTIVERSION_C_API(name, &kleidicv::neon::exp<type>, \
                              &kleidicv::sve2::exp<type>,       \
                              &kleidicv::sme2::exp<type>)

KLEIDICV_DEFINE_C_API(kleidicv_exp_f32, float);
