// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_DISPATCH_H
#define KLEIDICV_DISPATCH_H

#include "kleidicv/config.h"

#if defined(KLEIDICV_HAVE_SVE2) || defined(KLEIDICV_HAVE_SME2)
#include <sys/auxv.h>

#include <cstdint>
#include <type_traits>

namespace KLEIDICV_TARGET_NAMESPACE {

using HwCapTy = uint64_t;

struct HwCaps final {
  HwCapTy hwcap1;
  HwCapTy hwcap2;
};

static inline HwCaps get_hwcaps() {
  return HwCaps{getauxval(AT_HWCAP), getauxval(AT_HWCAP2)};
}

#ifdef KLEIDICV_HAVE_SVE2
static inline bool hwcaps_has_sve2(HwCaps hwcaps) {
  return hwcaps.hwcap2 & (1 << 1);
}

#define KLEIDICV_SVE2_RESOLVE(sve2_impl)                    \
  if (!std::is_null_pointer_v<decltype(sve2_impl)> &&       \
      KLEIDICV_TARGET_NAMESPACE::hwcaps_has_sve2(hwcaps)) { \
    return sve2_impl;                                       \
  }
#else
#define KLEIDICV_SVE2_RESOLVE(x)
#endif  // KLEIDICV_HAVE_SVE2

#ifdef KLEIDICV_HAVE_SME2
static inline bool hwcaps_has_sme2(HwCaps hwcaps) {
  // Actually checks for SME, not SME2, but this will be changed to check for
  // SME2 in future.
  const int kSMEBit = 23;
  return hwcaps.hwcap2 & (1UL << kSMEBit);
}

#define KLEIDICV_SME2_RESOLVE(sme2_impl)                    \
  if (!std::is_null_pointer_v<decltype(sme2_impl)> &&       \
      KLEIDICV_TARGET_NAMESPACE::hwcaps_has_sme2(hwcaps)) { \
    return sme2_impl;                                       \
  }
#else
#define KLEIDICV_SME2_RESOLVE(x)
#endif  // KLEIDICV_HAVE_SME2

}  // namespace KLEIDICV_TARGET_NAMESPACE

#define KLEIDICV_MULTIVERSION_C_API(api_name, neon_impl, sve2_impl, sme2_impl) \
  static decltype(neon_impl) api_name##_resolver() {                           \
    [[maybe_unused]] KLEIDICV_TARGET_NAMESPACE::HwCaps hwcaps =                \
        KLEIDICV_TARGET_NAMESPACE::get_hwcaps();                               \
    KLEIDICV_SME2_RESOLVE(sme2_impl);                                          \
    KLEIDICV_SVE2_RESOLVE(sve2_impl);                                          \
    return neon_impl;                                                          \
  }                                                                            \
  extern "C" {                                                                 \
  decltype(neon_impl) api_name = api_name##_resolver();                        \
  }

#else  // KLEIDICV_HAVE_SVE2 || KLEIDICV_HAVE_SME2

#define KLEIDICV_MULTIVERSION_C_API(api_name, neon_impl, sve2_impl, sme2_impl) \
                                                                               \
  extern "C" {                                                                 \
  decltype(neon_impl) api_name = neon_impl;                                    \
  }

#endif  // KLEIDICV_HAVE_SVE2 || KLEIDICV_HAVE_SME2

#if KLEIDICV_ALWAYS_ENABLE_SME2
#define KLEIDICV_SME2_IMPL_IF(func) func
#else
#define KLEIDICV_SME2_IMPL_IF(func) nullptr
#endif  // KLEIDICV_ALWAYS_ENABLE_SME2

#if KLEIDICV_ALWAYS_ENABLE_SVE2
#define KLEIDICV_SVE2_IMPL_IF(func) func
#else
#define KLEIDICV_SVE2_IMPL_IF(func) nullptr
#endif  // KLEIDICV_ALWAYS_ENABLE_SVE2

#endif  // KLEIDICV_DISPATCH_H
