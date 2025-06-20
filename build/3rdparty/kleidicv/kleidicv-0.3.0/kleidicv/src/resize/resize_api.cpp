// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/resize/resize.h"

KLEIDICV_MULTIVERSION_C_API(
    kleidicv_resize_to_quarter_u8, &kleidicv::neon::resize_to_quarter_u8,
    KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::resize_to_quarter_u8),
    &kleidicv::sme2::resize_to_quarter_u8);
