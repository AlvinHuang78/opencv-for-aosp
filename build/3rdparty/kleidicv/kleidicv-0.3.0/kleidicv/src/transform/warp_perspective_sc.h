// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <arm_sve.h>

#include <cmath>

#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"
#include "kleidicv/traits.h"
#include "kleidicv/transform/warp_perspective.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
namespace KLEIDICV_TARGET_NAMESPACE {

// Gather load is not available in streaming mode, and in general random access
// is not recommended for SME.
#if !KLEIDICV_TARGET_SME2

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
template <typename ScalarType>
class WarpPerspective;

template <>
class WarpPerspective<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using DstVecTraits = VecTraits<ScalarType>;
  using CoordVecTraits = VecTraits<float>;
  using CoordVector = CoordVecTraits::VectorType;

  WarpPerspective(Rows<const ScalarType> src_rows, size_t src_width,
                  size_t src_height,
                  const float tr_[9]) KLEIDICV_STREAMING_COMPATIBLE
      : src_rows_{src_rows},
        src_width_{src_width},
        src_height_{src_height},
        tr_{tr_} {}

  void process_row(size_t y, size_t width, Columns<ScalarType> dst) {
    svfloat32_t svzero = svdup_n_f32(0.F);
    svfloat32_t sv_xmax = svdup_n_f32(static_cast<float>(src_width_ - 1));
    svfloat32_t sv_ymax = svdup_n_f32(static_cast<float>(src_height_ - 1));
    svuint32_t sv_src_stride = svdup_n_u32(src_rows_.stride());
    svbool_t pg_all64 = svptrue_b64();
    svbool_t pg_all32 = svptrue_b32();
    svbool_t pg_all8 = svptrue_b8();
    float fy = static_cast<float>(y);
    svfloat32_t vx0 = svcvt_f32_u32_z(pg_all32, svindex_u32(0, 1));
    // (Nearest) integer part of transformed coordinates
    svuint32_t xi, yi;
    // Calculate half-transformed base values (nominators) at the first vector
    // (x=0)
    //     tw =  T6*x + T7*y + T8
    //     tx = (T0*x + T1*y + T2) / tw
    //     ty = (T3*x + T4*y + T5) / tw
    svfloat32_t tw0 = svdup_n_f32(tr_[7] * fy + tr_[8]);
    svfloat32_t tx0 = svdup_n_f32(tr_[1] * fy + tr_[2]);
    svfloat32_t ty0 = svdup_n_f32(tr_[4] * fy + tr_[5]);
    auto calculate_coordinates = [&](size_t x) {
      svfloat32_t vx = svadd_n_f32_x(pg_all32, vx0, static_cast<float>(x));
      svfloat32_t tw = svmla_x(pg_all32, tw0, vx, tr_[6]);
      svfloat32_t tx = svmla_x(pg_all32, tx0, vx, tr_[0]);
      svfloat32_t ty = svmla_x(pg_all32, ty0, vx, tr_[3]);

      // Calculate inverse weight because division is expensive
      svfloat32_t iw = svdiv_f32_x(pg_all32, svdup_n_f32(1.F), tw);
      // Calc and clamp coordinates to within the dimensions of the source image
      svfloat32_t dx =
          svmax_x(pg_all32, svzero,
                  svmin_x(pg_all32, svmad_x(pg_all32, tx, iw, 0.5F), sv_xmax));
      svfloat32_t dy =
          svmax_x(pg_all32, svzero,
                  svmin_x(pg_all32, svmad_x(pg_all32, ty, iw, 0.5F), sv_ymax));

      xi = svcvt_u32_f32_x(pg_all32, dx);
      yi = svcvt_u32_f32_x(pg_all32, dy);
    };

    auto calculate_and_load_64bits = [&](svbool_t pg_b, svbool_t pg_t,
                                         size_t x) {
      calculate_coordinates(x);
      svuint64_t indices_b = svmlalb(svmovlb(xi), yi, sv_src_stride);
      svuint64_t indices_t = svmlalt(svmovlt(xi), yi, sv_src_stride);
      // Copy pixels from source
      svuint64_t result_b =
          svld1ub_gather_offset_u64(pg_b, &src_rows_[0], indices_b);
      svuint64_t result_t =
          svld1ub_gather_offset_u64(pg_t, &src_rows_[0], indices_t);
      return svtrn1_u32(svreinterpret_u32_u64(result_b),
                        svreinterpret_u32_u64(result_t));
    };

    auto calculate_and_load_32bits = [&](svbool_t pg, size_t x) {
      calculate_coordinates(x);
      svuint32_t indices = svmla_x(pg, xi, yi, sv_src_stride);
      return svld1ub_gather_offset_u32(pg, &src_rows_[0], indices);
    };

    auto vector_path_generic = [&](size_t x, size_t x_max) {
      size_t length = x_max - x;
      svbool_t pg_b = svwhilelt_b64(0ULL, (length + 1) / 2);
      svbool_t pg_t = svwhilelt_b64(0ULL, length / 2);
      svbool_t pg = svwhilelt_b32(0ULL, length);
      // To avoid losing precision, the final indices use 64 bits
      svuint32_t result = calculate_and_load_64bits(pg_b, pg_t, x);
      svst1b_u32(pg, &dst[0], result);
      dst += static_cast<ptrdiff_t>(length);
    };

    auto vector_path_4x_large = [&](size_t x) {
      // Copy pixels from source
      // To avoid losing precision, the final indices use 64 bits
      svuint32_t res32_0 = calculate_and_load_64bits(pg_all64, pg_all64, x);
      x += CoordVecTraits::num_lanes();
      svuint32_t res32_1 = calculate_and_load_64bits(pg_all64, pg_all64, x);
      svuint16_t result0 = svuzp1_u16(svreinterpret_u16_u32(res32_0),
                                      svreinterpret_u16_u32(res32_1));
      x += CoordVecTraits::num_lanes();
      res32_0 = calculate_and_load_64bits(pg_all64, pg_all64, x);
      x += CoordVecTraits::num_lanes();
      res32_1 = calculate_and_load_64bits(pg_all64, pg_all64, x);
      svuint16_t result1 = svuzp1_u16(svreinterpret_u16_u32(res32_0),
                                      svreinterpret_u16_u32(res32_1));
      svuint8_t result = svuzp1_u8(svreinterpret_u8_u16(result0),
                                   svreinterpret_u8_u16(result1));
      svst1(pg_all8, &dst[0], result);
      dst += static_cast<ptrdiff_t>(4 * CoordVecTraits::num_lanes());
    };

    auto vector_path_4x_small = [&](size_t x) {
      // Copy pixels from source
      // Use this path only when the final indices fit into 32 bits
      svuint32_t res32_0 = calculate_and_load_32bits(pg_all32, x);
      x += CoordVecTraits::num_lanes();
      svuint32_t res32_1 = calculate_and_load_32bits(pg_all32, x);
      svuint16_t result0 = svuzp1_u16(svreinterpret_u16_u32(res32_0),
                                      svreinterpret_u16_u32(res32_1));
      x += CoordVecTraits::num_lanes();
      res32_0 = calculate_and_load_32bits(pg_all32, x);
      x += CoordVecTraits::num_lanes();
      res32_1 = calculate_and_load_32bits(pg_all32, x);
      svuint16_t result1 = svuzp1_u16(svreinterpret_u16_u32(res32_0),
                                      svreinterpret_u16_u32(res32_1));
      svuint8_t result = svuzp1_u8(svreinterpret_u8_u16(result0),
                                   svreinterpret_u8_u16(result1));
      svst1(pg_all8, &dst[0], result);
      dst += static_cast<ptrdiff_t>(4 * CoordVecTraits::num_lanes());
    };

    LoopUnroll2 loop{width, CoordVecTraits::num_lanes()};
    if (KLEIDICV_LIKELY(src_rows_.stride() * src_height_ < (1ULL << 32))) {
      loop.unroll_four_times(vector_path_4x_small);
    } else {
      loop.unroll_four_times(vector_path_4x_large);
    }
    loop.unroll_once([&](size_t x) {
      vector_path_generic(x, x + CoordVecTraits::num_lanes());
    });
    loop.remaining(vector_path_generic);
  }

 private:
  Rows<const ScalarType> src_rows_;
  size_t src_width_, src_height_;
  const float *tr_;
};  // end of class WarpPerspective<uint8_t>

template <typename T>
KLEIDICV_LOCALLY_STREAMING kleidicv_error_t warp_perspective_stripe_sc(
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
  WarpPerspective<T> operation{src_rows, src_width, src_height, transformation};
  Rectangle rect{dst_width, dst_height};

  dst_rows += y_begin;
  for (size_t y = y_begin; y < y_end; ++y) {
    operation.process_row(y, rect.width(), dst_rows.as_columns());
    ++dst_rows;
  }

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_WARP_PERSPECTIVE_SC(type)                         \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                           \
  warp_perspective_stripe_sc<type>(                                            \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t y_begin, size_t y_end, const float transformation[9],             \
      size_t channels, kleidicv_interpolation_type_t interpolation,            \
      kleidicv_border_type_t border_type,                                      \
      kleidicv_border_values_t border_values)

KLEIDICV_INSTANTIATE_WARP_PERSPECTIVE_SC(uint8_t);

#endif

}  // namespace KLEIDICV_TARGET_NAMESPACE
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
