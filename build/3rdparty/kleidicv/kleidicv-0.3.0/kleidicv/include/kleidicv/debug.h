// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_DEBUG_H
#define KLEIDICV_DEBUG_H

#include "kleidicv/config.h"
#if KLEIDICV_TARGET_NEON
#include "kleidicv/neon.h"
#elif KLEIDICV_TARGET_SVE2
#include "kleidicv/sve2.h"
#endif
#include <iostream>

namespace kleidicv {

#if KLEIDICV_TARGET_NEON

namespace neon {

template <typename ScalarType>
[[maybe_unused]] static void debug_print(
    const char *msg, typename neon::VecTraits<ScalarType>::VectorType vector) {
  using VecTraits = neon::VecTraits<ScalarType>;

  static ScalarType array[VecTraits::num_lanes()];
  vst1q(&array[0], vector);

  printf("%s\n", msg);
  for (size_t index = 0; index < VecTraits::num_lanes(); ++index) {
    printf("0x%0*x ", static_cast<int>(2 * sizeof(ScalarType)),
           static_cast<uint32_t>(array[index]) &
               static_cast<uint32_t>(((1UL << (8 * sizeof(ScalarType))) - 1)));
  }

  printf("\n");
}

}  // namespace neon

#endif  // KLEIDICV_TARGET_NEON

#if KLEIDICV_TARGET_SVE2

namespace sve2 {

template <typename ScalarType>
[[maybe_unused]] static void debug_print(
    const char *msg, typename sve2::VecTraits<ScalarType>::VectorType vector) {
  using VecTraits = sve2::VecTraits<ScalarType>;

  static ScalarType array[512 * sizeof(ScalarType)];
  svst1(VecTraits::svptrue(), &array[0], vector);

  printf("%s\n", msg);
  for (size_t index = 0; index < VecTraits::num_lanes(); ++index) {
    printf("0x%0*x ", static_cast<int>(2 * sizeof(ScalarType)),
           static_cast<uint32_t>(array[index]) &
               static_cast<uint32_t>(((1UL << (8 * sizeof(ScalarType))) - 1)));
  }

  printf("\n");
}

}  // namespace sve2

#endif  // KLEIDICV_TARGET_SVE2

}  // namespace kleidicv

#endif  // KLEIDICV_DEBUG_H
