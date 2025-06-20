// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/containers/stack.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_CANNY

namespace kleidicv::neon {

// Container to hold strong edges.
using StrongEdgeStack = Stack<uint8_t *>;

// A single-channel buffer which width is extended to be the multiple of the
// Neon vector length.
template <typename ScalarType>
class SobelBuffer final : public RowsOverUniquePtr<ScalarType> {
 public:
  explicit SobelBuffer(Rectangle rect)
      : RowsOverUniquePtr<ScalarType>(rect, get_margin(rect)) {}

 private:
  // Align the width up to an integral number of lanes in a vector.
  static Margin get_margin(Rectangle rect) {
    size_t width = align_up(rect.width(), VecTraits<ScalarType>::num_lanes());
    return Margin{0, 0, width - rect.width(), 0};
  }
};  // end of class SobelBuffer<ScalarType>

// A buffer for a single-channel magnitude data, which is padded with Neon
// vector length at left and right, and with one extra line at the top and the
// bottom.
template <typename ScalarType>
class MagnitudeBuffer final : public RowsOverUniquePtr<ScalarType> {
 public:
  explicit MagnitudeBuffer(Rectangle rect)
      : RowsOverUniquePtr<ScalarType>(rect, get_margin()) {}

 private:
  // Pad with Neon vector length at left and right, and with one extra row at
  // the top and the bottom.
  static Margin get_margin() {
    return Margin{VecTraits<ScalarType>::num_lanes(), 1,
                  VecTraits<ScalarType>::num_lanes(), 1};
  }
};  // end of class MagnitudeBuffer<ScalarType>

// A single-channel buffer for running hysteresis data, which is padded with one
// byte at left and right, and with one extra line at the top and the bottom.
template <typename T>
class HysteresisBuffer final : public RowsOverUniquePtr<T> {
 public:
  explicit HysteresisBuffer(Rectangle rect)
      : RowsOverUniquePtr<T>(rect, Margin{1}) {}
};  // end of class HysteresisBuffer<T>

// Classification of atan(Gy/Gx) into 6 non-overlapping categories.
//
//  0: [    0,  PI/8) and [    PI,  9PI/8)
//  2: [ PI/8, 3PI/8) and [ 9PI/8, 11PI/8)
//  4: [3PI/8, 4PI/8) and [11PI/8, 12PI/8)
//  5: [4PI/8, 5PI/8) and [12PI/8, 13PI/8)
//  3: [5PI/8, 7PI/8) and [13PI/8, 15PI/8)
//  1: [7PI/8, 8PI/8) and [15PI/8, 2PI)
//
// Details and key values
// ----------------------
//
// This algorithm compares Gy and Gx to approximate Gy/Gx value. The exact
// value of Gy/Gx is insignificant.
//
// 1) Scale is 15 bits:
//    kScale = 32.768
//
// 2) Given at PI/8 (22.5 degrees)
//      tan(PI/8) = 0.41421
//    therefore
//      0.41421 * 32,768 = 13,572.95001
//    yielding
//      k22p5 = 13,573
//
// 3) Given at 3PI/8 (67.5 degrees)
//      tan(3PI/8) = 2.41421
//    and
//      2.41421 = 0.41421 + 2.0
//    alternatively
//      2.41421 = tan(PI/8) + 2.0
//    so that
//      k67p5 = 2.0 * 32,768 = 65,536
//    which is equivalent to a simple bitwise shift left by 'kScale + 1'.
//
// Limitations
// -----------
//
// 1) Input values must be in the range [-27,145, 27,145], because calculations
//    are performed with 32-bit signed integer representation, and
//      27,145 * (k22p5 + 2^(15+1)) = 27,145 * 79,109 = 2,147,413,805
//    which is
//      0x7FFEEF2D
//    and
//      27,146 * 79,109 = 2,147,492,914
//    which is
//      0x80002432
//    and that would mean an overflow.
//
// 2) Gx = 0, Gy = 0 is classified as category 2 (similar to f(x) = y).
//
// 3) Gx = 0, Gy != 0 is classified as category 4.
static int16x8_t atan_classification(const int16_t *gx_ptr,
                                     const int16_t *gy_ptr) {
  static constexpr int16_t kScale = 15;
  static constexpr int16_t k22p5 = 13573;

  int16x8_t gx = vld1q_s16(gx_ptr);
  int16x8_t gy = vld1q_s16(gy_ptr);
  // Calculate |Gx| and |Gy|.
  int16x8_t abs_gx = vabsq_s16(gx);
  int16x8_t abs_gy = vabsq_s16(gy);
  // Grab low and high halves of |Gx|.
  int16x4_t abs_gx_l = vget_low_s16(abs_gx);
  int16x4_t abs_gx_h = vget_high_s16(abs_gx);
  // Calculate the compare value for |Gx| at PI/8.
  int32x4_t gx_22p5_l = vmull_n_s16(abs_gx_l, k22p5);
  int32x4_t gx_22p5_h = vmull_n_s16(abs_gx_h, k22p5);
  // Scale up |Gy| with the scaling factor.
  int32x4_t gy_scaled_l = vshll_n_s16(vget_low_s16(abs_gy), kScale);
  int32x4_t gy_scaled_h = vshll_n_s16(vget_high_s16(abs_gy), kScale);
  // [1] Assess whether atan(Gy/Gx) is within categories 0 or 1.
  int32x4_t cmp_22p5_l = vcgeq_s32(gy_scaled_l, gx_22p5_l);
  int32x4_t cmp_22p5_h = vcgeq_s32(gy_scaled_h, gx_22p5_h);
  // Calculate the compare value for |Gx| at 3PI/8.
  int32x4_t gx_67p5_l = vaddq_s32(gx_22p5_l, vshll_n_s16(abs_gx_l, kScale + 1));
  int32x4_t gx_67p5_h = vaddq_s32(gx_22p5_h, vshll_n_s16(abs_gx_h, kScale + 1));
  // [2] Assess whether atan(Gy/Gx) is within categories 4 or 5.
  int32x4_t cmp_67p5_l = vcgtq_s32(gy_scaled_l, gx_67p5_l);
  int32x4_t cmp_67p5_h = vcgtq_s32(gy_scaled_h, gx_67p5_h);
  // [3] Check the direction, which depends on the sign bits.
  int16x8_t xor_gx_gy = veorq_s16(gx, gy);
  int16x8_t cmp_sign = vcltzq_s16(xor_gx_gy);
  // Compose partial results from [1] and [2], and scale up by a factor of 2.
  int32x4_t res_l = vaddq_s32(cmp_22p5_l, cmp_67p5_l);
  int32x4_t res_h = vaddq_s32(cmp_22p5_h, cmp_67p5_h);
  res_l = vshlq_n_s32(res_l, 1);
  res_h = vshlq_n_s32(res_h, 1);
  // Compose final results accounting in for the sign bits [3], which creates
  // the odd-numbered categories.
  int16x8_t res = vmovn_high_s32(vmovn_s32(res_l), res_h);
  res = vaddq_s16(res, cmp_sign);
  return vabsq_s16(res);
}

namespace workaround {
// Workaround: force LDP
template <typename T>
T *remove_constant_pool_usage(T *ptr) {
  asm("" : "+r"(ptr));
  return ptr;
}
}  // namespace workaround

// Creates two vectors out of nine governed by given directions of the center
// elements.
//
// Layout of input rows
// --------------------
//
//  prev[0] prev[1] prev[2]
//  curr[0] curr[1] curr[2]
//  next[0] next[1] next[2]
//
// Domain of directions
// --------------------
//
// The directions supplied in 'directions' are associated with elements in
// curr[1]. The values in 'directions' must be:
//  - 0, 1 for horizontal directions across {curr[0], curr[1], curr[2]}
//  - 2 for diagonal directions across {next[0], curr[1], prev[2]}
//  - 3 for diagonal directions across {prev[0], curr[1], next[2]}
//  - 4, 5 for vertical directions across {prev[0], curr[1], next[2]}
//
// Algorithm description
// ---------------------
//
// The vector 'lane_offsets' contains the number of the respective lane plus
// 'number of lanes'. For example, lane[0] will hold the value 0 + 16 = 16, and
// lane[15] will hold the value 15 + 16 = 31.
//
// 1.1 Perform a table lookup with 'directions' creating 'tmp_indices_0' using a
//     table with the following mapping rules:
//      - 2 becomes 2,
//      - 3 becomes -2,
//      - 4 and 5 are both replaced with 0 and
//      - other indexing value yields '2 * number of lanes'.
//
// 1.2 Add 'lane_offsets' to 'tmp_indices_0' forming 'tmp_indices_1'.
//
// 1.3 Perform a three-vector table lookup with 'tmp_indices_1' on vectors
//     {prev[0], prev[1], prev[2]}, yielding 'prev_row_by_directions'.
//
// 2.1 Perform a table lookup with 'directions' creating 'tmp_indices_0' using a
//     table with the following mapping rules:
//      - 2 becomes -2,
//      - 3 becomes 2,
//      - 4 and 5 are both replaced with 0 and
//      - other indexing value yields '2 * number of lanes'.
//
// 2.2 Add 'lane_offsets' to 'tmp_indices_0' forming 'tmp_indices_1'.
//
// 2.3 Perform a three-vector table lookup with 'tmp_indices_1' on vectors
//     {next[0], next[1], next[2]}, yielding 'next_row_by_directions'.
//
// 3.1 Perform a table lookup with 'directions' creating 'tmp_indices_0' using a
//     table with the following mapping rules:
//      - 0 and 1 become -2 and
//      - any other indexing value yields '2 * number of lanes'.
//
// 3.2 Add 'lane_offsets' to 'tmp_indices_0' forming 'tmp_indices_1'.
//
// 3.3 Perform a three-vector table lookup extension with 'tmp_indices_1' on
//     vectors {curr[0], curr[1], curr[2]} operating on
//     'prev_row_by_directions'.
//
// 4.1 Because of symmetry, calculate the absolute value of 'tmp_indices_0'
//     created in step 3.1.
//
// 4.2 Add 'lane_offsets' to 'tmp_indices_0' forming 'tmp_indices_1'.
//
// 4.3 Perform a three-vector table lookup extension with 'tmp_indices_1' on
//     vectors {curr[0], curr[1], curr[2]} operating on
//     'next_row_by_directions'.
//
// The outputs of the algorithm are 'prev_row_by_directions' and
// 'next_row_by_directions'. These vectors contain the proper values in the
// associated lanes: {top, bottom}, {top-left, bottom-left}, {top-right,
// bottom-left} and {left, right} neighbouring values as governed by the
// associated directions.
//
// If KLEIDICV_CANNY_ALGORITHM_CONFORM_OPENCV is set to 1:
//  - diagonal directions are swapped
//  - diagonal non-maxima-suppressions are calculated as:
//      curr > prev && curr > next
//    This is different from other directions where it is calculated as:
//      curr > prev && curr >= next
//    (To achieve this lanes in 'next_row_by_directions', where the direction is
//    diagonal, are incremented by 1 saturating.)
static void directional_masking(const int16_t *prev_rows,
                                const int16_t *curr_rows,
                                const int16_t *next_rows, int16x8_t directions,
                                int16x8_t &prev_row_by_directions,
                                int16x8_t &curr_row_by_directions,
                                int16x8_t &next_row_by_directions) {
  static constexpr int16_t kNumLanesS8 = 16;
  static constexpr int16_t kNumLanes = 8;
  // clang-format off
  // Tables containing the described mapping rules.
  static constexpr int8_t kIndices[4 * kNumLanesS8] = {
      /* Lane offsets holding 'lane number + VL' */
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
#if KLEIDICV_CANNY_ALGORITHM_CONFORM_OPENCV
      /* Table lookup indices for previous row */
      32, 32, -2, 2, 0, 0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      /* Table lookup indices for next row */
      32, 32, 2, -2, 0, 0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
#else
      /* Table lookup indices for previous row */
      32, 32, 2, -2, 0, 0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
      /* Table lookup indices for next row */
      32, 32, -2, 2, 0, 0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
#endif
      /* Table lookup indices for current row */
      -2, -2, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
  };
  // clang-format on

  // Compiler workaround to produce better code.
  const int8_t *indices = workaround::remove_constant_pool_usage(&kIndices[0]);

  // Directions
  const int8x16_t dir = vtrn1q_s8(vreinterpretq_s8_s16(directions),
                                  vreinterpretq_s8_s16(directions));

  // Load indices which are used to create prev and next rows.
  const int8x16_t lane_offsets = vld1q_s8(&indices[0UL * kNumLanesS8]);
  const int8x16_t prev_row_table = vld1q_s8(&indices[1UL * kNumLanesS8]);
  const int8x16_t next_row_table = vld1q_s8(&indices[2UL * kNumLanesS8]);
  const int8x16_t current_row_table = vld1q_s8(&indices[3UL * kNumLanesS8]);

  // Temporary indexing and other vectors.
  int8x16_t tmp_indices_0, tmp_indices_1;
  int8x16_t prev_row_by_dir, next_row_by_dir;

  // Load the previous row.
  int8x16x3_t prev_row;
  prev_row.val[0] =
      vreinterpretq_s8_s16(vld1q_s16(&prev_rows[0UL * kNumLanes]));
  prev_row.val[1] =
      vreinterpretq_s8_s16(vld1q_s16(&prev_rows[1UL * kNumLanes]));
  prev_row.val[2] =
      vreinterpretq_s8_s16(vld1q_s16(&prev_rows[2UL * kNumLanes]));

  // 1.1
  tmp_indices_0 = vqtbl1q_s8(prev_row_table, dir);
  // 1.2
  tmp_indices_1 = vaddq_s8(tmp_indices_0, lane_offsets);
  // 1.3
  prev_row_by_dir = vqtbl3q_s8(prev_row, tmp_indices_1);

  // Load the next row.
  int8x16x3_t next_row;
  next_row.val[0] =
      vreinterpretq_s8_s16(vld1q_s16(&next_rows[0UL * kNumLanes]));
  next_row.val[1] =
      vreinterpretq_s8_s16(vld1q_s16(&next_rows[1UL * kNumLanes]));
  next_row.val[2] =
      vreinterpretq_s8_s16(vld1q_s16(&next_rows[2UL * kNumLanes]));

  // 2.1
  tmp_indices_0 = vqtbl1q_s8(next_row_table, dir);
#if KLEIDICV_CANNY_ALGORITHM_CONFORM_OPENCV
  int8x16_t opencv_tmp_indices_0 = tmp_indices_0;
#endif
  // 2.2
  tmp_indices_1 = vaddq_s8(tmp_indices_0, lane_offsets);
  // 2.3
  next_row_by_dir = vqtbl3q_s8(next_row, tmp_indices_1);

  // Load the current row.
  int8x16x3_t curr_row;
  curr_row.val[0] =
      vreinterpretq_s8_s16(vld1q_s16(&curr_rows[0UL * kNumLanes]));
  curr_row.val[1] =
      vreinterpretq_s8_s16(vld1q_s16(&curr_rows[1UL * kNumLanes]));
  curr_row.val[2] =
      vreinterpretq_s8_s16(vld1q_s16(&curr_rows[2UL * kNumLanes]));

  // 3.1
  tmp_indices_0 = vqtbl1q_s8(current_row_table, dir);
  // 3.2
  tmp_indices_1 = vaddq_s8(tmp_indices_0, lane_offsets);
  // 3.3
  prev_row_by_dir = vqtbx3q_s8(prev_row_by_dir, curr_row, tmp_indices_1);

  // 4.1
  tmp_indices_0 = vabsq_s8(tmp_indices_0);
  // 4.2
  tmp_indices_1 = vaddq_s8(tmp_indices_0, lane_offsets);
  // 4.3
  next_row_by_dir = vqtbx3q_s8(next_row_by_dir, curr_row, tmp_indices_1);

  // Set output vectors.
  prev_row_by_directions = vreinterpretq_s16_s8(prev_row_by_dir);
  curr_row_by_directions = vreinterpretq_s16_s8(curr_row.val[1]);
  next_row_by_directions = vreinterpretq_s16_s8(next_row_by_dir);

#if KLEIDICV_CANNY_ALGORITHM_CONFORM_OPENCV
  // Reuse temporary indexing values from step 2.1 to saturating add one to
  // diagonal values in the next row. This works only because of the domain of
  // input values is restricted.
  uint16x8_t opencv_tmp_indices_1 =
      vshrq_n_u16(vreinterpretq_u16_s8(opencv_tmp_indices_0), 9);
  opencv_tmp_indices_1 = vandq_u16(opencv_tmp_indices_1, vdupq_n_u16(0xFF01));
  next_row_by_directions = vqaddq_s16(
      next_row_by_directions, vreinterpretq_s16_u16(opencv_tmp_indices_1));
#endif
}

static bool is_vect_len_memory_null(const int16_t *data) {
  int64_t data64[2];
  memcpy(data64, data, sizeof(data64));
  return data64[0] == 0 && data64[1] == 0;
}

static void non_maxima_suppression_and_high_thresholding(
    Rectangle size, Rows<const int16_t> horizontal_gradient_rows,
    Rows<const int16_t> vertical_gradient_rows,
    Rows<const int16_t> magnitude_rows, Rows<uint8_t> dst_rows,
    double high_threshold, StrongEdgeStack &strong_edge_pixels) {
  static constexpr ptrdiff_t kNumLanes = VecTraits<int16_t>::num_lanes();
  static const int16_t kPositionBits[8] = {
      0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
  };

  const int16x8_t strong_edge_positions = vld1q_s16(&kPositionBits[0]);
  const int16x8_t high_threshold_vec =
      vdupq_n_s16(static_cast<int16_t>(high_threshold));

  auto handle_row = [&](ptrdiff_t index) {
    if (is_vect_len_memory_null(&magnitude_rows.at(1)[index + kNumLanes])) {
      memset(&dst_rows[index], 0, kNumLanes);
      return;
    }

    int16x8_t directions = atan_classification(&horizontal_gradient_rows[index],
                                               &vertical_gradient_rows[index]);

    int16x8_t prev_row_by_directions;
    int16x8_t curr_row_by_directions;
    int16x8_t next_row_by_directions;

    directional_masking(
        &magnitude_rows.at(0)[index], &magnitude_rows.at(1)[index],
        &magnitude_rows.at(2)[index], directions, prev_row_by_directions,
        curr_row_by_directions, next_row_by_directions);

    int16x8_t is_greater_than_prev =
        vcgtq_s16(curr_row_by_directions, prev_row_by_directions);
    int16x8_t is_greaterequal_than_next =
        vcgeq_s16(curr_row_by_directions, next_row_by_directions);

    int16x8_t non_max_result =
        vandq_s16(curr_row_by_directions,
                  vandq_s16(is_greater_than_prev, is_greaterequal_than_next));

    uint16x8_t is_strong_edge =
        vreinterpretq_u16_s16(vcgtq_s16(non_max_result, high_threshold_vec));
    uint16x8_t is_weak_edge =
        vshrq_n_u16(vreinterpretq_u16_s16(vcgtzq_s16(non_max_result)), 15);

    uint8x8_t result = vmovn_u16(vorrq_u16(is_strong_edge, is_weak_edge));

    vst1_u8(&dst_rows[index], result);
    int16_t strong_edges = vaddvq_s16(vreinterpretq_s16_u16(
        vandq_s16(is_strong_edge, strong_edge_positions)));
    Columns<uint8_t> dst_columns = dst_rows.at(0, index).as_columns();
    while (strong_edges) {
      if (strong_edges & 1) {
        strong_edge_pixels.push_back(&dst_columns[0]);
      }

      strong_edges >>= 1;
      ++dst_columns;
    }
  };

  for (size_t index = 0; index < size.height(); ++index) {
    LoopUnroll2{size.width(), kNumLanes}.unroll_once(handle_row);
    ++vertical_gradient_rows;
    ++horizontal_gradient_rows;
    ++magnitude_rows;
    ++dst_rows;
  }
}

static void handle_potentially_strong_pixel(StrongEdgeStack &strong_edge_pixels,
                                            uint8_t *pixel) {
  if (*pixel == 1) {
    *pixel = 0xFF;
    strong_edge_pixels.push_back(pixel);
  }
}

static void perform_hysteresis(StrongEdgeStack &strong_edge_pixels,
                               size_t stride) {
  while (!strong_edge_pixels.empty()) {
    auto *strong_pixel = strong_edge_pixels.back();
    strong_edge_pixels.pop_back();

    handle_potentially_strong_pixel(strong_edge_pixels,
                                    strong_pixel - stride - 1);
    handle_potentially_strong_pixel(strong_edge_pixels, strong_pixel - stride);
    handle_potentially_strong_pixel(strong_edge_pixels,
                                    strong_pixel - stride + 1);
    handle_potentially_strong_pixel(strong_edge_pixels, strong_pixel - 1);
    handle_potentially_strong_pixel(strong_edge_pixels, strong_pixel + 1);
    handle_potentially_strong_pixel(strong_edge_pixels,
                                    strong_pixel + stride - 1);
    handle_potentially_strong_pixel(strong_edge_pixels, strong_pixel + stride);
    handle_potentially_strong_pixel(strong_edge_pixels,
                                    strong_pixel + stride + 1);
  }
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t canny_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, double low_threshold, double high_threshold) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  Rectangle dst_rect{width, height};

  // Allocate all temporary buffers in advance.
  SobelBuffer<int16_t> horizontal_gradient{dst_rect};
  SobelBuffer<int16_t> vertical_gradient{dst_rect};
  if (!horizontal_gradient.data() || !vertical_gradient.data()) {
    return KLEIDICV_ERROR_ALLOCATION;
  }

  MagnitudeBuffer<int16_t> magnitudes{horizontal_gradient.rect()};
  if (!magnitudes.data()) {
    return KLEIDICV_ERROR_ALLOCATION;
  }

  HysteresisBuffer<uint8_t> hysteresis{horizontal_gradient.rect()};
  if (!hysteresis.data()) {
    return KLEIDICV_ERROR_ALLOCATION;
  }

  // Calculate horizontal dervatives using 3x3 Sobel operator.
  if (auto err = kleidicv_sobel_3x3_horizontal_s16_u8(
          src, src_stride, horizontal_gradient.data(),
          horizontal_gradient.rows().stride(), width, height, 1)) {
    return err;
  }

  // Calculate vertical dervatives using 3x3 Sobel operator.
  if (auto err = kleidicv_sobel_3x3_vertical_s16_u8(
          src, src_stride, vertical_gradient.data(),
          vertical_gradient.rows().stride(), width, height, 1)) {
    return err;
  }

  // Calculate magnitude from the horizontal and vertical derivatives, and apply
  // lower threshold.
  if (auto err = kleidicv_saturating_add_abs_with_threshold_s16(
          &horizontal_gradient.rows()[0], horizontal_gradient.rows().stride(),
          &vertical_gradient.rows()[0], vertical_gradient.rows().stride(),
          &magnitudes.rows_without_margin()[0],
          magnitudes.rows_without_margin().stride(),
          horizontal_gradient.rect().width(),
          horizontal_gradient.rect().height(),
          static_cast<int16_t>(low_threshold))) {
    return err;
  }

  // Perform non-maxima supression and high thresholding.
  StrongEdgeStack strong_edge_pixels;
  non_maxima_suppression_and_high_thresholding(
      horizontal_gradient.rect(), horizontal_gradient.rows(),
      vertical_gradient.rows(), magnitudes.rows(),
      hysteresis.rows_without_margin(), high_threshold, strong_edge_pixels);

  // Perform hysteresis: promote weak edges, which are connected to strong
  // edges, to strong edges.
  perform_hysteresis(strong_edge_pixels,
                     hysteresis.rows_without_margin().stride());

  // Finalize results by supressing weak edges.
  return kleidicv_threshold_binary_u8(&hysteresis.rows_without_margin()[0],
                                      hysteresis.rows_without_margin().stride(),
                                      dst, dst_stride, width, height, 0x80,
                                      0xFF);
}

}  // namespace kleidicv::neon

extern "C" {

decltype(kleidicv::neon::canny_u8) *kleidicv_canny_u8 =
    kleidicv::neon::canny_u8;

}  // extern "C"

#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_CANNY
