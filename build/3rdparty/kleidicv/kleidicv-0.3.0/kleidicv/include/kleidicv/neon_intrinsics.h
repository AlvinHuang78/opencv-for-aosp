// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_NEON_INTRINSICS_H
#define KLEIDICV_NEON_INTRINSICS_H

#ifndef KLEIDICV_NEON_H
#error "Please include neon.h instead."
#endif

#include <arm_neon.h>

#include <cinttypes>

namespace kleidicv::neon {

// -----------------------------------------------------------------------------
// NEON binary operations
// -----------------------------------------------------------------------------

#define NEON_BINARY_OP_Q_B8_B16_B32_B64(name)                     \
  static inline int8x16_t name(int8x16_t lhs, int8x16_t rhs) {    \
    return name##_s8(lhs, rhs);                                   \
  }                                                               \
                                                                  \
  static inline uint8x16_t name(uint8x16_t lhs, uint8x16_t rhs) { \
    return name##_u8(lhs, rhs);                                   \
  }                                                               \
                                                                  \
  static inline int16x8_t name(int16x8_t lhs, int16x8_t rhs) {    \
    return name##_s16(lhs, rhs);                                  \
  }                                                               \
                                                                  \
  static inline uint16x8_t name(uint16x8_t lhs, uint16x8_t rhs) { \
    return name##_u16(lhs, rhs);                                  \
  }                                                               \
                                                                  \
  static inline int32x4_t name(int32x4_t lhs, int32x4_t rhs) {    \
    return name##_s32(lhs, rhs);                                  \
  }                                                               \
                                                                  \
  static inline uint32x4_t name(uint32x4_t lhs, uint32x4_t rhs) { \
    return name##_u32(lhs, rhs);                                  \
  }                                                               \
                                                                  \
  static inline int64x2_t name(int64x2_t lhs, int64x2_t rhs) {    \
    return name##_s64(lhs, rhs);                                  \
  }                                                               \
                                                                  \
  static inline uint64x2_t name(uint64x2_t lhs, uint64x2_t rhs) { \
    return name##_u64(lhs, rhs);                                  \
  }

// Alphabetical order
NEON_BINARY_OP_Q_B8_B16_B32_B64(vaddq);
NEON_BINARY_OP_Q_B8_B16_B32_B64(vcleq);
NEON_BINARY_OP_Q_B8_B16_B32_B64(vcgeq);
NEON_BINARY_OP_Q_B8_B16_B32_B64(vqaddq);
NEON_BINARY_OP_Q_B8_B16_B32_B64(vqsubq);
NEON_BINARY_OP_Q_B8_B16_B32_B64(vtrn1q);
NEON_BINARY_OP_Q_B8_B16_B32_B64(vtrn2q);
NEON_BINARY_OP_Q_B8_B16_B32_B64(vuzp1q);
NEON_BINARY_OP_Q_B8_B16_B32_B64(vuzp2q);
NEON_BINARY_OP_Q_B8_B16_B32_B64(vzip1q);
NEON_BINARY_OP_Q_B8_B16_B32_B64(vzip2q);

#undef NEON_BINARY_OP_Q_B8_B16_B32_B64

#define NEON_BINARY_OP_Q_F32_F64(name)                               \
  static inline float32x4_t name(float32x4_t lhs, float32x4_t rhs) { \
    return name##_f32(lhs, rhs);                                     \
  }                                                                  \
                                                                     \
  static inline float64x2_t name(float64x2_t lhs, float64x2_t rhs) { \
    return name##_f64(lhs, rhs);                                     \
  }

NEON_BINARY_OP_Q_F32_F64(vaddq);

#undef NEON_BINARY_OP_Q_F32_F64

// clang-format off

// -----------------------------------------------------------------------------
// vaddv*
// -----------------------------------------------------------------------------

static inline int8_t vaddvq(int8x16_t vec) { return vaddvq_s8(vec); }
static inline uint8_t vaddvq(uint8x16_t vec) { return vaddvq_u8(vec); }
static inline int16_t vaddvq(int16x8_t vec) { return vaddvq_s16(vec); }
static inline uint16_t vaddvq(uint16x8_t vec) { return vaddvq_u16(vec); }
static inline int32_t vaddvq(int32x4_t vec) { return vaddvq_s32(vec); }
static inline uint32_t vaddvq(uint32x4_t vec) { return vaddvq_u32(vec); }
static inline int64_t vaddvq(int64x2_t vec) { return vaddvq_s64(vec); }
static inline uint64_t vaddvq(uint64x2_t vec) { return vaddvq_u64(vec); }
static inline float32_t vaddvq(float32x4_t vec) { return vaddvq_f32(vec); }
static inline float64_t vaddvq(float64x2_t vec) { return vaddvq_f64(vec); }

// -----------------------------------------------------------------------------
// vabd*
// -----------------------------------------------------------------------------

static inline int8x16_t  vabdq(int8x16_t  lhs, int8x16_t  rhs) { return  vabdq_s8(lhs, rhs); }
static inline uint8x16_t vabdq(uint8x16_t lhs, uint8x16_t rhs) { return  vabdq_u8(lhs, rhs); }
static inline int16x8_t  vabdq(int16x8_t  lhs, int16x8_t  rhs) { return vabdq_s16(lhs, rhs); }
static inline uint16x8_t vabdq(uint16x8_t lhs, uint16x8_t rhs) { return vabdq_u16(lhs, rhs); }
static inline int32x4_t  vabdq(int32x4_t  lhs, int32x4_t  rhs) { return vabdq_s32(lhs, rhs); }
static inline uint32x4_t vabdq(uint32x4_t lhs, uint32x4_t rhs) { return vabdq_u32(lhs, rhs); }

// -----------------------------------------------------------------------------
// vand*
// -----------------------------------------------------------------------------

static inline uint8x16_t vandq(uint8x16_t lhs, uint8x16_t rhs) { return vandq_u8(lhs, rhs); }
static inline uint16x8_t vandq(uint16x8_t lhs, uint16x8_t rhs) { return vandq_u16(lhs, rhs); }
static inline uint32x4_t vandq(uint32x4_t lhs, uint32x4_t rhs) { return vandq_u32(lhs, rhs); }

// -----------------------------------------------------------------------------
// vqabs*
// -----------------------------------------------------------------------------

static inline int8x16_t vqabsq(int8x16_t vec) { return  vqabsq_s8(vec); }
static inline int16x8_t vqabsq(int16x8_t vec) { return vqabsq_s16(vec); }
static inline int32x4_t vqabsq(int32x4_t vec) { return vqabsq_s32(vec); }
static inline int64x2_t vqabsq(int64x2_t vec) { return vqabsq_s64(vec); }

// -----------------------------------------------------------------------------
// vaddl*
// -----------------------------------------------------------------------------

static inline int16x8_t  vaddl(int8x8_t   lhs, int8x8_t   rhs) { return  vaddl_s8(lhs, rhs); }
static inline uint16x8_t vaddl(uint8x8_t  lhs, uint8x8_t  rhs) { return  vaddl_u8(lhs, rhs); }
static inline int32x4_t  vaddl(int16x4_t  lhs, int16x4_t  rhs) { return vaddl_s16(lhs, rhs); }
static inline uint32x4_t vaddl(uint16x4_t lhs, uint16x4_t rhs) { return vaddl_u16(lhs, rhs); }
static inline int64x2_t  vaddl(int32x2_t  lhs, int32x2_t  rhs) { return vaddl_s32(lhs, rhs); }
static inline uint64x2_t vaddl(uint32x2_t lhs, uint32x2_t rhs) { return vaddl_u32(lhs, rhs); }

// -----------------------------------------------------------------------------
// vbslq*
// -----------------------------------------------------------------------------

static inline int8x16_t vbslq(int8x16_t a, int8x16_t b, int8x16_t c) { return vbslq_s8(a, b, c); }
static inline uint8x16_t vbslq(uint8x16_t a, uint8x16_t b, uint8x16_t c) { return vbslq_u8(a, b, c); }
static inline int16x8_t vbslq(int16x8_t a, int16x8_t b, int16x8_t c) { return vbslq_s16(a, b, c); }
static inline uint16x8_t vbslq(uint16x8_t a, uint16x8_t b, uint16x8_t c) { return vbslq_u16(a, b, c); }
static inline int32x4_t vbslq(int32x4_t a, int32x4_t b, int32x4_t c) { return vbslq_s32(a, b, c); }
static inline uint32x4_t vbslq(uint32x4_t a, uint32x4_t b, uint32x4_t c) { return vbslq_u32(a, b, c); }
static inline float32x4_t vbslq(uint32x4_t a, float32x4_t b, float32x4_t c) { return vbslq_f32(a, b, c); }

// -----------------------------------------------------------------------------
// vget_high*
// -----------------------------------------------------------------------------

static inline int8x8_t   vget_high(int8x16_t  vec) { return  vget_high_s8(vec); }
static inline uint8x8_t  vget_high(uint8x16_t vec) { return  vget_high_u8(vec); }
static inline int16x4_t  vget_high(int16x8_t  vec) { return vget_high_s16(vec); }
static inline uint16x4_t vget_high(uint16x8_t vec) { return vget_high_u16(vec); }
static inline int32x2_t  vget_high(int32x4_t  vec) { return vget_high_s32(vec); }
static inline uint32x2_t vget_high(uint32x4_t vec) { return vget_high_u32(vec); }
static inline int64x1_t  vget_high(int64x2_t  vec) { return vget_high_s64(vec); }
static inline uint64x1_t vget_high(uint64x2_t vec) { return vget_high_u64(vec); }

// -----------------------------------------------------------------------------
// vcgeq*
// -----------------------------------------------------------------------------

static inline uint32x4_t vcgeq(float32x4_t lhs, float32x4_t rhs) { return vcgeq_f32(lhs, rhs); }

// -----------------------------------------------------------------------------
// vget_low*
// -----------------------------------------------------------------------------

static inline int8x8_t   vget_low(int8x16_t  vec) { return  vget_low_s8(vec); }
static inline uint8x8_t  vget_low(uint8x16_t vec) { return  vget_low_u8(vec); }
static inline int16x4_t  vget_low(int16x8_t  vec) { return vget_low_s16(vec); }
static inline uint16x4_t vget_low(uint16x8_t vec) { return vget_low_u16(vec); }
static inline int32x2_t  vget_low(int32x4_t  vec) { return vget_low_s32(vec); }
static inline uint32x2_t vget_low(uint32x4_t vec) { return vget_low_u32(vec); }
static inline int64x1_t  vget_low(int64x2_t  vec) { return vget_low_s64(vec); }
static inline uint64x1_t vget_low(uint64x2_t vec) { return vget_low_u64(vec); }

// -----------------------------------------------------------------------------
// vminq*
// -----------------------------------------------------------------------------

static inline int8x16_t   vminq(int8x16_t   lhs, int8x16_t   rhs) { return  vminq_s8(lhs, rhs); }
static inline uint8x16_t  vminq(uint8x16_t  lhs, uint8x16_t  rhs) { return  vminq_u8(lhs, rhs); }
static inline int16x8_t   vminq(int16x8_t   lhs, int16x8_t   rhs) { return vminq_s16(lhs, rhs); }
static inline uint16x8_t  vminq(uint16x8_t  lhs, uint16x8_t  rhs) { return vminq_u16(lhs, rhs); }
static inline int32x4_t   vminq(int32x4_t   lhs, int32x4_t   rhs) { return vminq_s32(lhs, rhs); }
static inline uint32x4_t  vminq(uint32x4_t  lhs, uint32x4_t  rhs) { return vminq_u32(lhs, rhs); }
static inline float32x4_t vminq(float32x4_t lhs, float32x4_t rhs) { return vminq_f32(lhs, rhs); }

// -----------------------------------------------------------------------------
// vmaxq*
// -----------------------------------------------------------------------------

static inline int8x16_t   vmaxq(int8x16_t   lhs, int8x16_t   rhs) { return  vmaxq_s8(lhs, rhs); }
static inline uint8x16_t  vmaxq(uint8x16_t  lhs, uint8x16_t  rhs) { return  vmaxq_u8(lhs, rhs); }
static inline int16x8_t   vmaxq(int16x8_t   lhs, int16x8_t   rhs) { return vmaxq_s16(lhs, rhs); }
static inline uint16x8_t  vmaxq(uint16x8_t  lhs, uint16x8_t  rhs) { return vmaxq_u16(lhs, rhs); }
static inline int32x4_t   vmaxq(int32x4_t   lhs, int32x4_t   rhs) { return vmaxq_s32(lhs, rhs); }
static inline uint32x4_t  vmaxq(uint32x4_t  lhs, uint32x4_t  rhs) { return vmaxq_u32(lhs, rhs); }
static inline float32x4_t vmaxq(float32x4_t lhs, float32x4_t rhs) { return vmaxq_f32(lhs, rhs); }

// -----------------------------------------------------------------------------
// vminvq*
// -----------------------------------------------------------------------------

static inline int8_t    vminvq(int8x16_t   src) { return  vminvq_s8(src); }
static inline uint8_t   vminvq(uint8x16_t  src) { return  vminvq_u8(src); }
static inline int16_t   vminvq(int16x8_t   src) { return vminvq_s16(src); }
static inline uint16_t  vminvq(uint16x8_t  src) { return vminvq_u16(src); }
static inline int32_t   vminvq(int32x4_t   src) { return vminvq_s32(src); }
static inline uint32_t  vminvq(uint32x4_t  src) { return vminvq_u32(src); }
static inline float32_t vminvq(float32x4_t src) { return vminvq_f32(src); }

// -----------------------------------------------------------------------------
// vmaxvq*
// -----------------------------------------------------------------------------

static inline int8_t    vmaxvq(int8x16_t   src) { return  vmaxvq_s8(src); }
static inline uint8_t   vmaxvq(uint8x16_t  src) { return  vmaxvq_u8(src); }
static inline int16_t   vmaxvq(int16x8_t   src) { return vmaxvq_s16(src); }
static inline uint16_t  vmaxvq(uint16x8_t  src) { return vmaxvq_u16(src); }
static inline int32_t   vmaxvq(int32x4_t   src) { return vmaxvq_s32(src); }
static inline uint32_t  vmaxvq(uint32x4_t  src) { return vmaxvq_u32(src); }
static inline float32_t vmaxvq(float32x4_t src) { return vmaxvq_f32(src); }

// -----------------------------------------------------------------------------
// vcleq*
// -----------------------------------------------------------------------------

static inline uint32x4_t vcleq(float32x4_t lhs, float32x4_t rhs) { return vcleq_f32(lhs, rhs); }

// -----------------------------------------------------------------------------
// vrshrn_n*
// -----------------------------------------------------------------------------

template <int n> static inline int8x8_t   vrshrn_n(int16x8_t  vec) { return vrshrn_n_s16(vec, n); }
template <int n> static inline uint8x8_t  vrshrn_n(uint16x8_t vec) { return vrshrn_n_u16(vec, n); }
template <int n> static inline int16x4_t  vrshrn_n(int32x4_t  vec) { return vrshrn_n_s32(vec, n); }
template <int n> static inline uint16x4_t vrshrn_n(uint32x4_t vec) { return vrshrn_n_u32(vec, n); }
template <int n> static inline int32x2_t  vrshrn_n(int64x2_t  vec) { return vrshrn_n_s64(vec, n); }
template <int n> static inline uint32x2_t vrshrn_n(uint64x2_t vec) { return vrshrn_n_u64(vec, n); }

// -----------------------------------------------------------------------------
// vshll_n*
// -----------------------------------------------------------------------------

template <int n> static inline int16x8_t  vshll_n(int8x8_t   vec) {  return vshll_n_s8(vec, n); }
template <int n> static inline uint16x8_t vshll_n(uint8x8_t  vec) {  return vshll_n_u8(vec, n); }
template <int n> static inline int32x4_t  vshll_n(int16x4_t  vec) { return vshll_n_s16(vec, n); }
template <int n> static inline uint32x4_t vshll_n(uint16x4_t vec) { return vshll_n_u16(vec, n); }
template <int n> static inline int64x2_t  vshll_n(int32x2_t  vec) { return vshll_n_s32(vec, n); }
template <int n> static inline uint64x2_t vshll_n(uint32x2_t vec) { return vshll_n_u32(vec, n); }

// -----------------------------------------------------------------------------
// vshlq_n*
// -----------------------------------------------------------------------------

template <int n> static inline int8x16_t  vshlq_n(int8x16_t  vec) { return  vshlq_n_s8(vec, n); }
template <int n> static inline uint8x16_t vshlq_n(uint8x16_t vec) { return  vshlq_n_u8(vec, n); }
template <int n> static inline int16x8_t  vshlq_n(int16x8_t  vec) { return vshlq_n_s16(vec, n); }
template <int n> static inline uint16x8_t vshlq_n(uint16x8_t vec) { return vshlq_n_u16(vec, n); }
template <int n> static inline int32x4_t  vshlq_n(int32x4_t  vec) { return vshlq_n_s32(vec, n); }
template <int n> static inline uint32x4_t vshlq_n(uint32x4_t vec) { return vshlq_n_u32(vec, n); }
template <int n> static inline int64x2_t  vshlq_n(int64x2_t  vec) { return vshlq_n_s64(vec, n); }
template <int n> static inline uint64x2_t vshlq_n(uint64x2_t vec) { return vshlq_n_u64(vec, n); }

// -----------------------------------------------------------------------------
// vdupq*
// -----------------------------------------------------------------------------

static inline int8x16_t   vdupq_n(int8_t    src) { return  vdupq_n_s8(src); }
static inline uint8x16_t  vdupq_n(uint8_t   src) { return  vdupq_n_u8(src); }
static inline int16x8_t   vdupq_n(int16_t   src) { return vdupq_n_s16(src); }
static inline uint16x8_t  vdupq_n(uint16_t  src) { return vdupq_n_u16(src); }
static inline int32x4_t   vdupq_n(int32_t   src) { return vdupq_n_s32(src); }
static inline uint32x4_t  vdupq_n(uint32_t  src) { return vdupq_n_u32(src); }
static inline int64x2_t   vdupq_n(int64_t   src) { return vdupq_n_s64(src); }
static inline uint64x2_t  vdupq_n(uint64_t  src) { return vdupq_n_u64(src); }
static inline float32x4_t vdupq_n(float32_t src) { return vdupq_n_f32(src); }

// -----------------------------------------------------------------------------
// vmull*
// -----------------------------------------------------------------------------

static inline int16x8_t  vmull(int8x8_t   lhs, int8x8_t   rhs) { return  vmull_s8(lhs, rhs); }
static inline uint16x8_t vmull(uint8x8_t  lhs, uint8x8_t  rhs) { return  vmull_u8(lhs, rhs); }
static inline int32x4_t  vmull(int16x4_t  lhs, int16x4_t  rhs) { return vmull_s16(lhs, rhs); }
static inline uint32x4_t vmull(uint16x4_t lhs, uint16x4_t rhs) { return vmull_u16(lhs, rhs); }
static inline int64x2_t  vmull(int32x2_t  lhs, int32x2_t  rhs) { return vmull_s32(lhs, rhs); }

// -----------------------------------------------------------------------------
// vmull_high*
// -----------------------------------------------------------------------------

static inline int16x8_t  vmull_high(int8x16_t  lhs, int8x16_t  rhs) { return  vmull_high_s8(lhs, rhs); }
static inline uint16x8_t vmull_high(uint8x16_t lhs, uint8x16_t rhs) { return  vmull_high_u8(lhs, rhs); }
static inline int32x4_t  vmull_high(int16x8_t  lhs, int16x8_t  rhs) { return vmull_high_s16(lhs, rhs); }
static inline uint32x4_t vmull_high(uint16x8_t lhs, uint16x8_t rhs) { return vmull_high_u16(lhs, rhs); }
static inline int64x2_t  vmull_high(int32x4_t  lhs, int32x4_t  rhs) { return vmull_high_s32(lhs, rhs); }

// -----------------------------------------------------------------------------
// vqmovn*
// -----------------------------------------------------------------------------

static inline int8x8_t   vqmovn(int16x8_t  src) { return vqmovn_s16(src); }
static inline uint8x8_t  vqmovn(uint16x8_t src) { return vqmovn_u16(src); }
static inline int16x4_t  vqmovn(int32x4_t  src) { return vqmovn_s32(src); }
static inline uint16x4_t vqmovn(uint32x4_t src) { return vqmovn_u32(src); }
static inline int32x2_t  vqmovn(int64x2_t  src) { return vqmovn_s64(src); }

// -----------------------------------------------------------------------------
// vqmovn_high*
// -----------------------------------------------------------------------------

static inline int8x16_t  vqmovn_high(int8x8_t   low, int16x8_t  src) { return vqmovn_high_s16(low, src); }
static inline uint8x16_t vqmovn_high(uint8x8_t  low, uint16x8_t src) { return vqmovn_high_u16(low, src); }
static inline int16x8_t  vqmovn_high(int16x4_t  low, int32x4_t  src) { return vqmovn_high_s32(low, src); }
static inline uint16x8_t vqmovn_high(uint16x4_t low, uint32x4_t src) { return vqmovn_high_u32(low, src); }
static inline int32x4_t  vqmovn_high(int32x2_t  low, int64x2_t  src) { return vqmovn_high_s64(low, src); }

// -----------------------------------------------------------------------------
// NEON load operations
// -----------------------------------------------------------------------------

static inline int8x16_t   vld1q(const int8_t    *src) { return  vld1q_s8(src); }
static inline uint8x16_t  vld1q(const uint8_t   *src) { return  vld1q_u8(src); }
static inline int16x8_t   vld1q(const int16_t   *src) { return vld1q_s16(src); }
static inline uint16x8_t  vld1q(const uint16_t  *src) { return vld1q_u16(src); }
static inline int32x4_t   vld1q(const int32_t   *src) { return vld1q_s32(src); }
static inline uint32x4_t  vld1q(const uint32_t  *src) { return vld1q_u32(src); }
static inline int64x2_t   vld1q(const int64_t   *src) { return vld1q_s64(src); }
static inline uint64x2_t  vld1q(const uint64_t  *src) { return vld1q_u64(src); }
static inline float32x4_t vld1q(const float32_t *src) { return vld1q_f32(src); }

static inline int8x16x2_t   vld2q(const int8_t    *src) { return  vld2q_s8(src); }
static inline uint8x16x2_t  vld2q(const uint8_t   *src) { return  vld2q_u8(src); }
static inline int16x8x2_t   vld2q(const int16_t   *src) { return vld2q_s16(src); }
static inline uint16x8x2_t  vld2q(const uint16_t  *src) { return vld2q_u16(src); }
static inline int32x4x2_t   vld2q(const int32_t   *src) { return vld2q_s32(src); }
static inline uint32x4x2_t  vld2q(const uint32_t  *src) { return vld2q_u32(src); }
static inline int64x2x2_t   vld2q(const int64_t   *src) { return vld2q_s64(src); }
static inline uint64x2x2_t  vld2q(const uint64_t  *src) { return vld2q_u64(src); }
static inline float32x4x2_t vld2q(const float32_t *src) { return vld2q_f32(src); }

static inline int8x16x3_t   vld3q(const int8_t    *src) { return  vld3q_s8(src); }
static inline uint8x16x3_t  vld3q(const uint8_t   *src) { return  vld3q_u8(src); }
static inline int16x8x3_t   vld3q(const int16_t   *src) { return vld3q_s16(src); }
static inline uint16x8x3_t  vld3q(const uint16_t  *src) { return vld3q_u16(src); }
static inline int32x4x3_t   vld3q(const int32_t   *src) { return vld3q_s32(src); }
static inline uint32x4x3_t  vld3q(const uint32_t  *src) { return vld3q_u32(src); }
static inline int64x2x3_t   vld3q(const int64_t   *src) { return vld3q_s64(src); }
static inline uint64x2x3_t  vld3q(const uint64_t  *src) { return vld3q_u64(src); }
static inline float32x4x3_t vld3q(const float32_t *src) { return vld3q_f32(src); }

static inline int8x16x4_t   vld4q(const int8_t    *src) { return  vld4q_s8(src); }
static inline uint8x16x4_t  vld4q(const uint8_t   *src) { return  vld4q_u8(src); }
static inline int16x8x4_t   vld4q(const int16_t   *src) { return vld4q_s16(src); }
static inline uint16x8x4_t  vld4q(const uint16_t  *src) { return vld4q_u16(src); }
static inline int32x4x4_t   vld4q(const int32_t   *src) { return vld4q_s32(src); }
static inline uint32x4x4_t  vld4q(const uint32_t  *src) { return vld4q_u32(src); }
static inline int64x2x4_t   vld4q(const int64_t   *src) { return vld4q_s64(src); }
static inline uint64x2x4_t  vld4q(const uint64_t  *src) { return vld4q_u64(src); }
static inline float32x4x4_t vld4q(const float32_t *src) { return vld4q_f32(src); }

static inline int8x16x2_t   vld1q_x2(const int8_t    *src) { return  vld1q_s8_x2(src); }
static inline uint8x16x2_t  vld1q_x2(const uint8_t   *src) { return  vld1q_u8_x2(src); }
static inline int16x8x2_t   vld1q_x2(const int16_t   *src) { return vld1q_s16_x2(src); }
static inline uint16x8x2_t  vld1q_x2(const uint16_t  *src) { return vld1q_u16_x2(src); }
static inline int32x4x2_t   vld1q_x2(const int32_t   *src) { return vld1q_s32_x2(src); }
static inline uint32x4x2_t  vld1q_x2(const uint32_t  *src) { return vld1q_u32_x2(src); }
static inline int64x2x2_t   vld1q_x2(const int64_t   *src) { return vld1q_s64_x2(src); }
static inline uint64x2x2_t  vld1q_x2(const uint64_t  *src) { return vld1q_u64_x2(src); }
static inline float32x4x2_t vld1q_x2(const float32_t *src) { return vld1q_f32_x2(src); }

static inline int8x16x3_t   vld1q_x3(const int8_t    *src) { return  vld1q_s8_x3(src); }
static inline uint8x16x3_t  vld1q_x3(const uint8_t   *src) { return  vld1q_u8_x3(src); }
static inline int16x8x3_t   vld1q_x3(const int16_t   *src) { return vld1q_s16_x3(src); }
static inline uint16x8x3_t  vld1q_x3(const uint16_t  *src) { return vld1q_u16_x3(src); }
static inline int32x4x3_t   vld1q_x3(const int32_t   *src) { return vld1q_s32_x3(src); }
static inline uint32x4x3_t  vld1q_x3(const uint32_t  *src) { return vld1q_u32_x3(src); }
static inline int64x2x3_t   vld1q_x3(const int64_t   *src) { return vld1q_s64_x3(src); }
static inline uint64x2x3_t  vld1q_x3(const uint64_t  *src) { return vld1q_u64_x3(src); }
static inline float32x4x3_t vld1q_x3(const float32_t *src) { return vld1q_f32_x3(src); }

static inline int8x16x4_t   vld1q_x4(const int8_t    *src) { return  vld1q_s8_x4(src); }
static inline uint8x16x4_t  vld1q_x4(const uint8_t   *src) { return  vld1q_u8_x4(src); }
static inline int16x8x4_t   vld1q_x4(const int16_t   *src) { return vld1q_s16_x4(src); }
static inline uint16x8x4_t  vld1q_x4(const uint16_t  *src) { return vld1q_u16_x4(src); }
static inline int32x4x4_t   vld1q_x4(const int32_t   *src) { return vld1q_s32_x4(src); }
static inline uint32x4x4_t  vld1q_x4(const uint32_t  *src) { return vld1q_u32_x4(src); }
static inline int64x2x4_t   vld1q_x4(const int64_t   *src) { return vld1q_s64_x4(src); }
static inline uint64x2x4_t  vld1q_x4(const uint64_t  *src) { return vld1q_u64_x4(src); }
static inline float32x4x4_t vld1q_x4(const float32_t *src) { return vld1q_f32_x4(src); }

// -----------------------------------------------------------------------------
// NEON store operations
// -----------------------------------------------------------------------------

static inline void vst1(int8_t   *dst, int8x8_t   vec) {  vst1_s8(dst, vec); }
static inline void vst1(uint8_t  *dst, uint8x8_t  vec) {  vst1_u8(dst, vec); }
static inline void vst1(int16_t  *dst, int16x4_t  vec) { vst1_s16(dst, vec); }
static inline void vst1(uint16_t *dst, uint16x4_t vec) { vst1_u16(dst, vec); }
static inline void vst1(int32_t  *dst, int32x2_t  vec) { vst1_s32(dst, vec); }
static inline void vst1(uint32_t *dst, uint32x2_t vec) { vst1_u32(dst, vec); }
static inline void vst1(int64_t  *dst, int64x1_t  vec) { vst1_s64(dst, vec); }
static inline void vst1(uint64_t *dst, uint64x1_t vec) { vst1_u64(dst, vec); }

static inline void vst1q(int8_t    *dst, int8x16_t   vec) { vst1q_s8(dst, vec); }
static inline void vst1q(uint8_t   *dst, uint8x16_t  vec) { vst1q_u8(dst, vec); }
static inline void vst1q(int16_t   *dst, int16x8_t   vec) { vst1q_s16(dst, vec); }
static inline void vst1q(uint16_t  *dst, uint16x8_t  vec) { vst1q_u16(dst, vec); }
static inline void vst1q(int32_t   *dst, int32x4_t   vec) { vst1q_s32(dst, vec); }
static inline void vst1q(uint32_t  *dst, uint32x4_t  vec) { vst1q_u32(dst, vec); }
static inline void vst1q(int64_t   *dst, int64x2_t   vec) { vst1q_s64(dst, vec); }
static inline void vst1q(uint64_t  *dst, uint64x2_t  vec) { vst1q_u64(dst, vec); }
static inline void vst1q(float32_t *dst, float32x4_t vec) { vst1q_f32(dst, vec); }

static inline void vst2q(int8_t    *dst, int8x16x2_t   vec) { vst2q_s8(dst, vec); }
static inline void vst2q(uint8_t   *dst, uint8x16x2_t  vec) { vst2q_u8(dst, vec); }
static inline void vst2q(int16_t   *dst, int16x8x2_t   vec) { vst2q_s16(dst, vec); }
static inline void vst2q(uint16_t  *dst, uint16x8x2_t  vec) { vst2q_u16(dst, vec); }
static inline void vst2q(int32_t   *dst, int32x4x2_t   vec) { vst2q_s32(dst, vec); }
static inline void vst2q(uint32_t  *dst, uint32x4x2_t  vec) { vst2q_u32(dst, vec); }
static inline void vst2q(int64_t   *dst, int64x2x2_t   vec) { vst2q_s64(dst, vec); }
static inline void vst2q(uint64_t  *dst, uint64x2x2_t  vec) { vst2q_u64(dst, vec); }
static inline void vst2q(float32_t *dst, float32x4x2_t vec) { vst2q_f32(dst, vec); }

static inline void vst3q(int8_t    *dst, int8x16x3_t   vec) { vst3q_s8(dst, vec); }
static inline void vst3q(uint8_t   *dst, uint8x16x3_t  vec) { vst3q_u8(dst, vec); }
static inline void vst3q(int16_t   *dst, int16x8x3_t   vec) { vst3q_s16(dst, vec); }
static inline void vst3q(uint16_t  *dst, uint16x8x3_t  vec) { vst3q_u16(dst, vec); }
static inline void vst3q(int32_t   *dst, int32x4x3_t   vec) { vst3q_s32(dst, vec); }
static inline void vst3q(uint32_t  *dst, uint32x4x3_t  vec) { vst3q_u32(dst, vec); }
static inline void vst3q(int64_t   *dst, int64x2x3_t   vec) { vst3q_s64(dst, vec); }
static inline void vst3q(uint64_t  *dst, uint64x2x3_t  vec) { vst3q_u64(dst, vec); }
static inline void vst3q(float32_t *dst, float32x4x3_t vec) { vst3q_f32(dst, vec); }

static inline void vst4q(int8_t    *dst, int8x16x4_t   vec) { vst4q_s8(dst, vec); }
static inline void vst4q(uint8_t   *dst, uint8x16x4_t  vec) { vst4q_u8(dst, vec); }
static inline void vst4q(int16_t   *dst, int16x8x4_t   vec) { vst4q_s16(dst, vec); }
static inline void vst4q(uint16_t  *dst, uint16x8x4_t  vec) { vst4q_u16(dst, vec); }
static inline void vst4q(int32_t   *dst, int32x4x4_t   vec) { vst4q_s32(dst, vec); }
static inline void vst4q(uint32_t  *dst, uint32x4x4_t  vec) { vst4q_u32(dst, vec); }
static inline void vst4q(int64_t   *dst, int64x2x4_t   vec) { vst4q_s64(dst, vec); }
static inline void vst4q(uint64_t  *dst, uint64x2x4_t  vec) { vst4q_u64(dst, vec); }
static inline void vst4q(float32_t *dst, float32x4x4_t vec) { vst4q_f32(dst, vec); }

static inline void vst1q_x2(int8_t    *dst, int8x16x2_t   vec) { vst1q_s8_x2(dst, vec); }
static inline void vst1q_x2(uint8_t   *dst, uint8x16x2_t  vec) { vst1q_u8_x2(dst, vec); }
static inline void vst1q_x2(int16_t   *dst, int16x8x2_t   vec) { vst1q_s16_x2(dst, vec); }
static inline void vst1q_x2(uint16_t  *dst, uint16x8x2_t  vec) { vst1q_u16_x2(dst, vec); }
static inline void vst1q_x2(int32_t   *dst, int32x4x2_t   vec) { vst1q_s32_x2(dst, vec); }
static inline void vst1q_x2(uint32_t  *dst, uint32x4x2_t  vec) { vst1q_u32_x2(dst, vec); }
static inline void vst1q_x2(int64_t   *dst, int64x2x2_t   vec) { vst1q_s64_x2(dst, vec); }
static inline void vst1q_x2(uint64_t  *dst, uint64x2x2_t  vec) { vst1q_u64_x2(dst, vec); }
static inline void vst1q_x2(float32_t *dst, float32x4x2_t vec) { vst1q_f32_x2(dst, vec); }

static inline void vst1q_x4(int8_t    *dst, int8x16x4_t   vec) { vst1q_s8_x4(dst, vec); }
static inline void vst1q_x4(uint8_t   *dst, uint8x16x4_t  vec) { vst1q_u8_x4(dst, vec); }
static inline void vst1q_x4(int16_t   *dst, int16x8x4_t   vec) { vst1q_s16_x4(dst, vec); }
static inline void vst1q_x4(uint16_t  *dst, uint16x8x4_t  vec) { vst1q_u16_x4(dst, vec); }
static inline void vst1q_x4(int32_t   *dst, int32x4x4_t   vec) { vst1q_s32_x4(dst, vec); }
static inline void vst1q_x4(uint32_t  *dst, uint32x4x4_t  vec) { vst1q_u32_x4(dst, vec); }
static inline void vst1q_x4(int64_t   *dst, int64x2x4_t   vec) { vst1q_s64_x4(dst, vec); }
static inline void vst1q_x4(uint64_t  *dst, uint64x2x4_t  vec) { vst1q_u64_x4(dst, vec); }
static inline void vst1q_x4(float32_t *dst, float32x4x4_t vec) { vst1q_f32_x4(dst, vec); }

// -----------------------------------------------------------------------------
// vreinterpret*
// -----------------------------------------------------------------------------

static inline uint8x16_t vreinterpretq_u8(int8x16_t  vec) { return vreinterpretq_u8_s8(vec); }
static inline uint8x16_t vreinterpretq_u8(uint8x16_t vec) { return vec; }
static inline uint8x16_t vreinterpretq_u8(int16x8_t  vec) { return vreinterpretq_u8_s16(vec); }
static inline uint8x16_t vreinterpretq_u8(uint16x8_t vec) { return vreinterpretq_u8_u16(vec); }
static inline uint8x16_t vreinterpretq_u8(int32x4_t  vec) { return vreinterpretq_u8_s32(vec); }
static inline uint8x16_t vreinterpretq_u8(uint32x4_t vec) { return vreinterpretq_u8_u32(vec); }
static inline uint8x16_t vreinterpretq_u8(int64x2_t  vec) { return vreinterpretq_u8_s64(vec); }
static inline uint8x16_t vreinterpretq_u8(uint64x2_t vec) { return vreinterpretq_u8_u64(vec); }

static inline uint64x2_t vreinterpretq_u64(int8x16_t  vec) { return vreinterpretq_u64_s8(vec); }
static inline uint64x2_t vreinterpretq_u64(uint8x16_t vec) { return vreinterpretq_u64_u8(vec); }
static inline uint64x2_t vreinterpretq_u64(int16x8_t  vec) { return vreinterpretq_u64_s16(vec); }
static inline uint64x2_t vreinterpretq_u64(uint16x8_t vec) { return vreinterpretq_u64_u16(vec); }
static inline uint64x2_t vreinterpretq_u64(int32x4_t  vec) { return vreinterpretq_u64_s32(vec); }
static inline uint64x2_t vreinterpretq_u64(uint32x4_t vec) { return vreinterpretq_u64_u32(vec); }
static inline uint64x2_t vreinterpretq_u64(int64x2_t  vec) { return vreinterpretq_u64_s64(vec); }
static inline uint64x2_t vreinterpretq_u64(uint64x2_t vec) { return vec; }

// -----------------------------------------------------------------------------
// vcombine*
// -----------------------------------------------------------------------------

static inline int8x16_t  vcombine(int8x8_t   lhs, int8x8_t   rhs) { return vcombine_s8(lhs, rhs); }
static inline uint8x16_t vcombine(uint8x8_t  lhs, uint8x8_t  rhs) { return vcombine_u8(lhs, rhs); }
static inline int16x8_t  vcombine(int16x4_t  lhs, int16x4_t  rhs) { return vcombine_s16(lhs, rhs); }
static inline uint16x8_t vcombine(uint16x4_t lhs, uint16x4_t rhs) { return vcombine_u16(lhs, rhs); }
static inline int32x4_t  vcombine(int32x2_t  lhs, int32x2_t  rhs) { return vcombine_s32(lhs, rhs); }
static inline uint32x4_t vcombine(uint32x2_t lhs, uint32x2_t rhs) { return vcombine_u32(lhs, rhs); }
static inline int64x2_t  vcombine(int64x1_t  lhs, int64x1_t  rhs) { return vcombine_s64(lhs, rhs); }
static inline uint64x2_t vcombine(uint64x1_t lhs, uint64x1_t rhs) { return vcombine_u64(lhs, rhs); }

// clang-format on

}  // namespace kleidicv::neon

#endif  // KLEIDICV_NEON_INTRINSICS_H
