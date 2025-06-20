// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_CONVERSIONS_MERGE_H
#define KLEIDICV_CONVERSIONS_MERGE_H

#include "kleidicv/kleidicv.h"

namespace kleidicv {

namespace neon {

kleidicv_error_t merge(const void **srcs, const size_t *src_strides, void *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t element_size);

}  // namespace neon

}  // namespace kleidicv

#endif  // KLEIDICV_CONVERSIONS_MERGE_H
