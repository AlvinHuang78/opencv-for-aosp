// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/gray_to_rgb.h"
#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"

#define KLEIDICV_DEFINE_C_API(name, partialname)           \
  KLEIDICV_MULTIVERSION_C_API(                             \
      name, &kleidicv::neon::partialname,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::partialname), \
      &kleidicv::sme2::partialname)

KLEIDICV_DEFINE_C_API(kleidicv_gray_to_rgb_u8, gray_to_rgb_u8);
KLEIDICV_DEFINE_C_API(kleidicv_gray_to_rgba_u8, gray_to_rgba_u8);
