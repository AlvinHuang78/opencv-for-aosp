// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"

namespace kleidicv {

namespace neon {

template <typename T>
kleidicv_error_t min_max(const T *src, size_t src_stride, size_t width,
                         size_t height, T *min_value, T *max_value);

template <typename T>
kleidicv_error_t min_max_loc(const T *src, size_t src_stride, size_t width,
                             size_t height, size_t *min_offset,
                             size_t *max_offset);

}  // namespace neon

namespace sve2 {

template <typename T>
kleidicv_error_t min_max(const T *src, size_t src_stride, size_t width,
                         size_t height, T *min_value, T *max_value);
}  // namespace sve2

namespace sme2 {

template <typename T>
kleidicv_error_t min_max(const T *src, size_t src_stride, size_t width,
                         size_t height, T *min_value, T *max_value);
}  // namespace sme2

}  // namespace kleidicv

#define KLEIDICV_DEFINE_MINMAX_API(name, type)               \
  KLEIDICV_MULTIVERSION_C_API(                               \
      name, &kleidicv::neon::min_max<type>,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::min_max<type>), \
      &kleidicv::sme2::min_max<type>)

KLEIDICV_DEFINE_MINMAX_API(kleidicv_min_max_u8, uint8_t);
KLEIDICV_DEFINE_MINMAX_API(kleidicv_min_max_s8, int8_t);
KLEIDICV_DEFINE_MINMAX_API(kleidicv_min_max_u16, uint16_t);
KLEIDICV_DEFINE_MINMAX_API(kleidicv_min_max_s16, int16_t);
KLEIDICV_DEFINE_MINMAX_API(kleidicv_min_max_s32, int32_t);
KLEIDICV_DEFINE_MINMAX_API(kleidicv_min_max_f32, float);

#define KLEIDICV_DEFINE_MINMAXLOC_API(name, type)                       \
  KLEIDICV_MULTIVERSION_C_API(name, &kleidicv::neon::min_max_loc<type>, \
                              nullptr, nullptr)

KLEIDICV_DEFINE_MINMAXLOC_API(kleidicv_min_max_loc_u8, uint8_t);
