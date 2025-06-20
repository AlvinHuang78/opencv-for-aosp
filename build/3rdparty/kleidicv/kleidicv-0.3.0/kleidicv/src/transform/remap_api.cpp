// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/transform/remap.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
KLEIDICV_MULTIVERSION_C_API(kleidicv_remap_s16_u8,
                            &kleidicv::neon::remap_s16<uint8_t>,
                            &kleidicv::sve2::remap_s16<uint8_t>, nullptr);

KLEIDICV_MULTIVERSION_C_API(
    kleidicv_remap_s16point5_u8, &kleidicv::neon::remap_s16point5<uint8_t>,
    &kleidicv::sve2::remap_s16point5<uint8_t>,
    KLEIDICV_SME2_IMPL_IF(&kleidicv::sme2::remap_s16point5<uint8_t>));
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
