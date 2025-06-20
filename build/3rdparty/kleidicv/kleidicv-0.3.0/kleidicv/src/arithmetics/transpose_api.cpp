// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/arithmetics/transpose.h"
#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"

KLEIDICV_MULTIVERSION_C_API(kleidicv_transpose, &kleidicv::neon::transpose,
                            nullptr, nullptr);
