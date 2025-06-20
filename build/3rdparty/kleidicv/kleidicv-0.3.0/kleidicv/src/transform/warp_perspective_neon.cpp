// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/traits.h"
#include "kleidicv/transform/warp_perspective.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE

namespace kleidicv::neon {

// Template for WarpPerspective transformation.
// Destination pixels are filled from the source, by taking pixels using the
// transformed coordinates that are calculated as follows:
//
//                    [ T0, T1, T2 ]   [ x ]
//      (x',y',w') =  [ T3, T4, T5 ] * [ y ]
//                    [ T6, T7, T8 ]   [ 1 ]
//  then
//
//      xt = x' / w'
//      yt = y' / w'
//
//  or putting it together:
//
//      xt = (T0*x + T1*y + T2) / (T6*x + T7*y + T8)
//      yt = (T3*x + T4*y + T5) / (T6*x + T7*y + T8)
//
template <typename ScalarType, bool IsLarge>
class WarpPerspective;

template <bool IsLarge>
class WarpPerspective<uint8_t, IsLarge> {
 public:
  using ScalarType = uint8_t;
  using CoordVecTraits = VecTraits<float>;
  using CoordVector = CoordVecTraits::VectorType;

  WarpPerspective(Rows<const ScalarType> src_rows, size_t src_width,
                  size_t src_height)
      : src_rows_{src_rows},
        src_height_{src_height},
        v_src_stride_{vdup_n_u32(static_cast<uint32_t>(src_rows_.stride()))},
        vq_src_stride_{vdupq_n_u32(static_cast<uint32_t>(src_rows_.stride()))},
        x0123_{vld1q(first_few_x)},
        v_xmax_{vdupq_n_f32(static_cast<float>(src_width - 1))},
        v_ymax_{vdupq_n_f32(static_cast<float>(src_height - 1))} {}

  void process_row(size_t y, size_t width, const float transform[9],
                   Columns<ScalarType> dst) {
    float dy = static_cast<float>(y);
    // Calculate half-transformed values at the first pixel (nominators)
    // tw =  T6*x + T7*y + T8
    // tx = (T0*x + T1*y + T2) / tw
    // ty = (T3*x + T4*y + T5) / tw
    float x0 = transform[1] * dy + transform[2];
    float y0 = transform[4] * dy + transform[5];
    float w0 = transform[7] * dy + transform[8];
    // The next few values can be calculated by adding the corresponding Tn*x
    CoordVector tx0 =
        vmlaq_f32(vdupq_n_f32(x0), x0123_, vdupq_n_f32(transform[0]));
    CoordVector ty0 =
        vmlaq_f32(vdupq_n_f32(y0), x0123_, vdupq_n_f32(transform[3]));
    CoordVector tw0 =
        vmlaq_f32(vdupq_n_f32(w0), x0123_, vdupq_n_f32(transform[6]));

    // (Nearest) integer part of transformed coordinates
    uint32x4_t xi, yi;

    auto calculate_coordinates = [&](size_t x) {
      float fx = static_cast<float>(x);
      // Calculate half-transformed values from the first few pixel values, plus
      // Tn*x, similarly to the one above
      CoordVector tx = vaddq_f32(tx0, vdupq_n_f32(transform[0] * fx));
      CoordVector ty = vaddq_f32(ty0, vdupq_n_f32(transform[3] * fx));
      CoordVector tw = vaddq_f32(tw0, vdupq_n_f32(transform[6] * fx));

      // Calculate inverse weight because division is expensive
      CoordVector iw = vdivq_f32(vdupq_n_f32(1.F), tw);
      // Calc and clamp coordinates to within the dimensions of the source image
      CoordVector xf =
          vmaxq_f32(vdupq_n_f32(0.F), vminq_f32(vmulq_f32(tx, iw), v_xmax_));
      CoordVector yf =
          vmaxq_f32(vdupq_n_f32(0.F), vminq_f32(vmulq_f32(ty, iw), v_ymax_));
      // Rounding convert to int
      xi = vcvtaq_u32_f32(xf);
      yi = vcvtaq_u32_f32(yf);
    };

    auto large_vector_path = [&](size_t x) {
      calculate_coordinates(x);
      // Calculate offsets from coordinates (y * stride + x)
      // To avoid losing precision, the final indices should be in 64 bits
      uint64x2_t indices_low = vmlal_u32(vmovl_u32(vget_low_u32(xi)),
                                         vget_low_u32(yi), v_src_stride_);
      uint64x2_t indices_high =
          vmlal_u32(vmovl_high_u32(xi), vget_high_u32(yi), v_src_stride_);
      // Copy pixels from source
      ptrdiff_t ix = static_cast<ptrdiff_t>(x);
      dst[ix] = src_rows_[vgetq_lane_u64(indices_low, 0)];
      dst[ix + 1] = src_rows_[vgetq_lane_u64(indices_low, 1)];
      dst[ix + 2] = src_rows_[vgetq_lane_u64(indices_high, 0)];
      dst[ix + 3] = src_rows_[vgetq_lane_u64(indices_high, 1)];
    };

    auto small_vector_path = [&](size_t x) {
      calculate_coordinates(x);
      // Calculate offsets from coordinates (y * stride + x)
      // Use this path only when the final indices fit into 32 bits
      uint32x4_t indices = vmlaq_u32(xi, yi, vq_src_stride_);
      // Copy pixels from source
      ptrdiff_t ix = static_cast<ptrdiff_t>(x);
      dst[ix] = src_rows_[vgetq_lane_u32(indices, 0)];
      dst[ix + 1] = src_rows_[vgetq_lane_u32(indices, 1)];
      dst[ix + 2] = src_rows_[vgetq_lane_u32(indices, 2)];
      dst[ix + 3] = src_rows_[vgetq_lane_u32(indices, 3)];
    };

    LoopUnroll2<TryToAvoidTailLoop> loop{width, CoordVecTraits::num_lanes()};
    if constexpr (IsLarge) {
      loop.unroll_once(large_vector_path);
    } else {
      loop.unroll_once(small_vector_path);
    }
  }

 private:
  static constexpr float first_few_x[] = {0.F, 1.F, 2.F, 3.F};
  Rows<const ScalarType> src_rows_;
  size_t src_height_;
  uint32x2_t v_src_stride_;
  uint32x4_t vq_src_stride_;
  CoordVector x0123_, v_xmax_, v_ymax_;
};  // end of class WarpPerspective<uint8_t>

// Most of the complexity comes from parameter checking.
// NOLINTBEGIN(readability-function-cognitive-complexity)
template <typename T>
kleidicv_error_t warp_perspective_stripe(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t y_begin, size_t y_end, const float transformation[9],
    size_t channels, kleidicv_interpolation_type_t, kleidicv_border_type_t,
    kleidicv_border_values_t) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_POINTERS(transformation);
  CHECK_IMAGE_SIZE(src_width, src_height);
  CHECK_IMAGE_SIZE(dst_width, dst_height);

  // Calculating in float32_t will only be precise until 24 bits, and
  // multiplication can only be done with 32x32 bits
  if (src_stride >= (1ULL << 32) || src_width >= (1ULL << 24) ||
      src_height >= (1ULL << 24)) {
    return KLEIDICV_ERROR_RANGE;
  }

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<T> dst_rows{dst, dst_stride, channels};
  Rectangle rect{dst_width, dst_height};
  dst_rows += y_begin;

  if (KLEIDICV_LIKELY(src_stride * src_height < (1ULL << 32))) {
    WarpPerspective<T, false> operation{src_rows, src_width, src_height};
    for (size_t y = y_begin; y < y_end; ++y) {
      operation.process_row(y, rect.width(), transformation,
                            dst_rows.as_columns());
      ++dst_rows;
    }
  } else {
    WarpPerspective<T, true> operation{src_rows, src_width, src_height};
    for (size_t y = y_begin; y < y_end; ++y) {
      operation.process_row(y, rect.width(), transformation,
                            dst_rows.as_columns());
      ++dst_rows;
    }
  }

  return KLEIDICV_OK;
}
// NOLINTEND(readability-function-cognitive-complexity)

#define KLEIDICV_INSTANTIATE_WARP_PERSPECTIVE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                           \
  warp_perspective_stripe<type>(                                               \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t y_begin, size_t y_end, const float transformation[9],             \
      size_t channels, kleidicv_interpolation_type_t interpolation,            \
      kleidicv_border_type_t border_type,                                      \
      kleidicv_border_values_t border_values)

KLEIDICV_INSTANTIATE_WARP_PERSPECTIVE(uint8_t);

}  // namespace kleidicv::neon
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
