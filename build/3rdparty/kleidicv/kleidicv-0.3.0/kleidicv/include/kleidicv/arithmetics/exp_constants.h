// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_EXP_CONSTANTS_H
#define KLEIDICV_EXP_CONSTANTS_H

namespace kleidicv::exp_f32 {

constexpr float kShift = 0x1.8p23F;
constexpr float kInvLn2 = 0x1.715476p+0F;
constexpr float kLn2Hi = 0x1.62e4p-1F;
constexpr float kLn2Lo = 0x1.7f7d1cp-20F;
constexpr float kPoly[] = {
    /*  maxerr: 0.36565 +0.5 ulp.  */
    0x1.6a6000p-10F, 0x1.12718ep-7F, 0x1.555af0p-5F,
    0x1.555430p-3F,  0x1.fffff4p-2F,
};

}  // namespace kleidicv::exp_f32

#endif  // KLEIDICV_EXP_CONSTANTS_H
