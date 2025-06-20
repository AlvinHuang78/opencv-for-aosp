// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/rgb_to_yuv.h"
#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"

#define KLEIDICV_DEFINE_C_API(name, partialname)           \
  KLEIDICV_MULTIVERSION_C_API(                             \
      name, &kleidicv::neon::partialname,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::partialname), \
      &kleidicv::sme2::partialname)

KLEIDICV_DEFINE_C_API(kleidicv_rgb_to_yuv_u8, rgb_to_yuv_u8);
KLEIDICV_DEFINE_C_API(kleidicv_bgr_to_yuv_u8, bgr_to_yuv_u8);
KLEIDICV_DEFINE_C_API(kleidicv_rgba_to_yuv_u8, rgba_to_yuv_u8);
KLEIDICV_DEFINE_C_API(kleidicv_bgra_to_yuv_u8, bgra_to_yuv_u8);
