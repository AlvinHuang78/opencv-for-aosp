// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_ARITHMETICS_TRANSPOSE_H
#define KLEIDICV_ARITHMETICS_TRANSPOSE_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/ctypes.h"

namespace kleidicv {

namespace neon {
kleidicv_error_t transpose(const void *src, size_t src_stride, void *dst,
                           size_t dst_stride, size_t width, size_t height,
                           size_t element_size);
}  // namespace neon

namespace sve2 {}  // namespace sve2

namespace sme2 {}  // namespace sme2

}  // namespace kleidicv

#endif  // KLEIDICV_ARITHMETICS_TRANSPOSE_H
