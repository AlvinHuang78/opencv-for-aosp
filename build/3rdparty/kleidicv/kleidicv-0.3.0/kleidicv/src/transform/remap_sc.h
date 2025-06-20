// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_REMAP_SC_H
#define KLEIDICV_REMAP_SC_H

#include <arm_sve.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>

#include "kleidicv/sve2.h"
#include "kleidicv/transform/remap.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
namespace KLEIDICV_TARGET_NAMESPACE {

#if !KLEIDICV_TARGET_SME2

template <typename ScalarType>
class RemapS16;

template <>
class RemapS16<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using MapVecTraits = VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;

  RemapS16(Rows<const ScalarType> src_rows, size_t src_width, size_t src_height,
           svuint16_t& v_src_stride, MapVectorType& v_x_max,
           MapVectorType& v_y_max)
      : src_rows_{src_rows},
        v_src_stride_{v_src_stride},
        v_xmax_{v_x_max},
        v_ymax_{v_y_max} {
    v_src_stride_ = svdup_u16(src_rows.stride());
    v_xmax_ = svdup_s16(static_cast<int16_t>(
        std::min<size_t>(std::numeric_limits<int16_t>::max(), src_width - 1)));
    v_ymax_ = svdup_s16(static_cast<int16_t>(
        std::min<size_t>(std::numeric_limits<int16_t>::max(), src_height - 1)));
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<ScalarType> dst) {
    svuint32_t offsets_b, offsets_t;
    svint16_t svzero = svdup_n_s16(0);
    auto load_offsets = [&](svbool_t pg) {
      MapVector2Type xy = svld2_s16(pg, &mapxy[0]);
      // Clamp coordinates to within the dimensions of the source image
      svuint16_t x = svreinterpret_u16_s16(
          svmax_x(pg, svzero, svmin_x(pg, svget2(xy, 0), v_xmax_)));
      svuint16_t y = svreinterpret_u16_s16(
          svmax_x(pg, svzero, svmin_x(pg, svget2(xy, 1), v_ymax_)));
      // Calculate offsets from coordinates (y * stride + x)
      offsets_b = svmlalb_u32(svmovlb_u32(x), y, v_src_stride_);
      offsets_t = svmlalt_u32(svmovlt_u32(x), y, v_src_stride_);
    };

    svbool_t pg_all16 = MapVecTraits::svptrue();
    svbool_t pg_all32 = svptrue_b32();

    auto gather_load_generic_vector_path = [&](svbool_t pg, ptrdiff_t step) {
      load_offsets(pg);
      svbool_t pg_b = svwhilelt_b32(int64_t{0}, (step + 1) / 2);
      svbool_t pg_t = svwhilelt_b32(int64_t{0}, step / 2);
      // Copy pixels from source
      svuint32_t result_b =
          svldnt1ub_gather_u32offset_u32(pg_b, &src_rows_[0], offsets_b);
      svuint32_t result_t =
          svldnt1ub_gather_u32offset_u32(pg_t, &src_rows_[0], offsets_t);
      svuint16_t result = svtrn1_u16(svreinterpret_u16_u32(result_b),
                                     svreinterpret_u16_u32(result_t));
      svst1b_u16(pg, &dst[0], result);
      mapxy += step;
      dst += step;
    };

    // NOTE: gather load is not available in streaming mode
    auto gather_load_full_vector_path = [&](ptrdiff_t step) {
      load_offsets(pg_all16);
      // Copy pixels from source
      svuint32_t result_b =
          svldnt1ub_gather_u32offset_u32(pg_all32, &src_rows_[0], offsets_b);
      svuint32_t result_t =
          svldnt1ub_gather_u32offset_u32(pg_all32, &src_rows_[0], offsets_t);
      svuint16_t result = svtrn1_u16(svreinterpret_u16_u32(result_b),
                                     svreinterpret_u16_u32(result_t));
      svst1b_u16(pg_all16, &dst[0], result);
      mapxy += step;
      dst += step;
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once([&](size_t step) {
      gather_load_full_vector_path(static_cast<ptrdiff_t>(step));
    });
    loop.remaining([&](size_t length, size_t step) {
      svbool_t pg = MapVecTraits::svwhilelt(step - length, step);
      gather_load_generic_vector_path(pg, static_cast<ptrdiff_t>(length));
    });
  }

 private:
  Rows<const ScalarType> src_rows_;
  svuint16_t& v_src_stride_;
  MapVectorType& v_xmax_;
  MapVectorType& v_ymax_;
};  // end of class RemapS16<uint8_t>

template <typename T>
kleidicv_error_t remap_s16_sc(const T* src, size_t src_stride, size_t src_width,
                              size_t src_height, T* dst, size_t dst_stride,
                              size_t dst_width, size_t dst_height,
                              size_t channels, const int16_t* mapxy,
                              size_t mapxy_stride,
                              kleidicv_border_type_t border_type,
                              kleidicv_border_values_t) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapxy, mapxy_stride, dst_height);
  CHECK_IMAGE_SIZE(src_width, src_height);
  CHECK_IMAGE_SIZE(dst_width, dst_height);

  if (!remap_s16_is_implemented<T>(dst_width, border_type, channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<const int16_t> mapxy_rows{mapxy, mapxy_stride, 2};
  Rows<T> dst_rows{dst, dst_stride, channels};
  svuint16_t sv_src_stride;
  svint16_t sv_xmax, sv_ymax;
  RemapS16<T> operation{src_rows,      src_width, src_height,
                        sv_src_stride, sv_xmax,   sv_ymax};
  Rectangle rect{dst_width, dst_height};
  zip_rows(operation, rect, mapxy_rows, dst_rows);
  return KLEIDICV_OK;
}

#endif  // KLEIDICV_TARGET_SME2

template <typename ScalarType>
class RemapS16Point5SVE2;

template <>
class RemapS16Point5SVE2<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using MapVecTraits = VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;
  using FracVecTraits = VecTraits<uint16_t>;
  using FracVectorType = typename FracVecTraits::VectorType;

  RemapS16Point5SVE2(Rows<const ScalarType> src_rows, size_t src_width,
                     size_t src_height, svuint16_t& v_src_stride,
                     MapVectorType& v_x_max, MapVectorType& v_y_max)
      : src_rows_{src_rows},
        v_src_stride_{v_src_stride},
        v_xmax_{v_x_max},
        v_ymax_{v_y_max} {
    v_src_stride_ = svdup_u16(src_rows.stride());
    v_xmax_ = svdup_s16(static_cast<int16_t>(
        std::min<size_t>(std::numeric_limits<int16_t>::max(), src_width - 1)));
    v_ymax_ = svdup_s16(static_cast<int16_t>(
        std::min<size_t>(std::numeric_limits<int16_t>::max(), src_height - 1)));
  }

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    svuint16_t src_a, src_b, src_c, src_d;

    svuint32_t bias = svdup_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2);
    auto vector_path = [&](svbool_t pg, ptrdiff_t step) {
      load_source(pg, step, mapxy, src_a, src_b, src_c, src_d);
      interpolate_and_store(pg, step, mapfrac, dst, src_a, src_b, src_c, src_d,
                            bias);
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once([&](size_t step) {
      svbool_t pg = MapVecTraits::svptrue();
      vector_path(pg, static_cast<ptrdiff_t>(step));
    });
    loop.remaining([&](size_t length, size_t step) {
      svbool_t pg = MapVecTraits::svwhilelt(step - length, step);
      vector_path(pg, static_cast<ptrdiff_t>(length));
    });
  }

 protected:
  svuint16_t gather_load_src(svbool_t pg_b, svuint32_t offsets_b, svbool_t pg_t,
                             svuint32_t offsets_t) {
    svuint32_t src_b =
        svldnt1ub_gather_u32offset_u32(pg_b, &src_rows_[0], offsets_b);
    svuint32_t src_t =
        svldnt1ub_gather_u32offset_u32(pg_t, &src_rows_[0], offsets_t);
    return svtrn1_u16(svreinterpret_u16_u32(src_b),
                      svreinterpret_u16_u32(src_t));
  }

  void load_source(svbool_t pg, ptrdiff_t step, Columns<const int16_t>& mapxy,
                   svuint16_t& src_a, svuint16_t& src_b, svuint16_t& src_c,
                   svuint16_t& src_d) {
    MapVector2Type xy = svld2_s16(pg, &mapxy[0]);

    // Clamp coordinates to within the dimensions of the source image
    svuint16_t x0 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0), svmin_x(pg, svget2(xy, 0), v_xmax_)));
    svuint16_t y0 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0), svmin_x(pg, svget2(xy, 1), v_ymax_)));

    // x1 = x0 + 1, and clamp it too
    svuint16_t x1 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0),
                svmin_x(pg, svqadd_n_s16_x(pg, svget2(xy, 0), 1), v_xmax_)));

    svuint16_t y1 = svreinterpret_u16_s16(
        svmax_x(pg, svdup_n_s16(0),
                svmin_x(pg, svqadd_n_s16_x(pg, svget2(xy, 1), 1), v_ymax_)));
    svbool_t pg_b = svwhilelt_b32(int64_t{0}, (step + 1) / 2);
    svbool_t pg_t = svwhilelt_b32(int64_t{0}, step / 2);

    // Calculate offsets from coordinates (y * stride + x)
    svuint32_t offsets_a_b = svmlalb_u32(svmovlb_u32(x0), y0, v_src_stride_);
    svuint32_t offsets_a_t = svmlalt_u32(svmovlt_u32(x0), y0, v_src_stride_);
    svuint32_t offsets_b_b = svmlalb_u32(svmovlb_u32(x1), y0, v_src_stride_);
    svuint32_t offsets_b_t = svmlalt_u32(svmovlt_u32(x1), y0, v_src_stride_);
    svuint32_t offsets_c_b = svmlalb_u32(svmovlb_u32(x0), y1, v_src_stride_);
    svuint32_t offsets_c_t = svmlalt_u32(svmovlt_u32(x0), y1, v_src_stride_);
    svuint32_t offsets_d_b = svmlalb_u32(svmovlb_u32(x1), y1, v_src_stride_);
    svuint32_t offsets_d_t = svmlalt_u32(svmovlt_u32(x1), y1, v_src_stride_);

    // Load pixels from source
    src_a = gather_load_src(pg_b, offsets_a_b, pg_t, offsets_a_t);
    src_b = gather_load_src(pg_b, offsets_b_b, pg_t, offsets_b_t);
    src_c = gather_load_src(pg_b, offsets_c_b, pg_t, offsets_c_t);
    src_d = gather_load_src(pg_b, offsets_d_b, pg_t, offsets_d_t);
    mapxy += step;
  }

  void interpolate_and_store(svbool_t pg, ptrdiff_t step,
                             Columns<const uint16_t>& mapfrac,
                             Columns<ScalarType>& dst, svuint16_t src_a,
                             svuint16_t src_b, svuint16_t src_c,
                             svuint16_t src_d,
                             svuint32_t bias) KLEIDICV_STREAMING_COMPATIBLE {
    FracVectorType frac = svld1_u16(pg, &mapfrac[0]);
    svuint16_t xfrac =
        svand_x(pg, frac, svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
    svuint16_t yfrac =
        svand_x(pg, svlsr_n_u16_x(pg, frac, REMAP16POINT5_FRAC_BITS),
                svdup_n_u16(REMAP16POINT5_FRAC_MAX - 1));
    svuint16_t nxfrac =
        svsub_u16_x(pg, svdup_n_u16(REMAP16POINT5_FRAC_MAX), xfrac);
    svuint16_t nyfrac =
        svsub_u16_x(pg, svdup_n_u16(REMAP16POINT5_FRAC_MAX), yfrac);
    svuint16_t line0 = svmla_x(pg, svmul_x(pg, xfrac, src_b), nxfrac, src_a);
    svuint16_t line1 = svmla_x(pg, svmul_x(pg, xfrac, src_d), nxfrac, src_c);

    svuint32_t acc_b = svmlalb_u32(bias, line0, nyfrac);
    svuint32_t acc_t = svmlalt_u32(bias, line0, nyfrac);
    acc_b = svmlalb_u32(acc_b, line1, yfrac);
    acc_t = svmlalt_u32(acc_t, line1, yfrac);

    svuint16_t result = svshrnt(svshrnb(acc_b, 2ULL * REMAP16POINT5_FRAC_BITS),
                                acc_t, 2ULL * REMAP16POINT5_FRAC_BITS);
    svst1b_u16(pg, &dst[0], result);
    mapfrac += step;
    dst += step;
  }

  Rows<const ScalarType> src_rows_;

 private:
  svuint16_t& v_src_stride_;
  MapVectorType& v_xmax_;
  MapVectorType& v_ymax_;
};  // end of class RemapS16Point5SVE2<uint8_t>

#if KLEIDICV_TARGET_SME2

template <typename ScalarType>
class RemapS16Point5SME2;

template <>
class RemapS16Point5SME2<uint8_t> : public RemapS16Point5SVE2<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using MapVecTraits = VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;
  using FracVecTraits = VecTraits<uint16_t>;
  using FracVectorType = typename FracVecTraits::VectorType;

  RemapS16Point5SME2(Rows<const ScalarType> src_rows, size_t src_width,
                     size_t src_height, svuint16_t& v_src_stride,
                     MapVectorType& v_x_max, MapVectorType& v_y_max)
      : RemapS16Point5SVE2<uint8_t>(src_rows, src_width, src_height,
                                    v_src_stride, v_x_max, v_y_max),
        rowbuffer_width_{0} {}

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    if (KLEIDICV_UNLIKELY(width != rowbuffer_width_)) {
      if (width > rowbuffer_width_) {
        rowbuffer_.reset(new ScalarType[4 * (width + 3)]);
        rowbuffer_width_ = width;
      }
    }

    Columns<ScalarType> demapped_src{rowbuffer_.get(),
                                     4 * src_rows_.channels()};

    svuint16_t src_a, src_b, src_c, src_d;

    auto vector_path = [&](svbool_t pg, ptrdiff_t step) {
      load_source(pg, step, mapxy, src_a, src_b, src_c, src_d);

      // Keep only the 8-bit data
      svuint8_t src_ab =
          svtrn1_u8(svreinterpret_u8_u16(src_a), svreinterpret_u8_u16(src_b));
      svuint8_t src_cd =
          svtrn1_u8(svreinterpret_u8_u16(src_c), svreinterpret_u8_u16(src_d));

      // Store them to rowbuffer
      // Interleaved store makes it abcdabcd
      svst2_u16(pg, reinterpret_cast<uint16_t*>(&demapped_src[0]),
                svcreate2(svreinterpret_u16_u8(src_ab),
                          svreinterpret_u16_u8(src_cd)));
      demapped_src += step;
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once([&](size_t step) {
      svbool_t pg = MapVecTraits::svptrue();
      vector_path(pg, static_cast<ptrdiff_t>(step));
    });
    loop.remaining([&](size_t length, size_t step) {
      svbool_t pg = MapVecTraits::svwhilelt(step - length, step);
      vector_path(pg, static_cast<ptrdiff_t>(length));
    });

    process_demapped_row(mapfrac, dst);
  }

 private:
  KLEIDICV_LOCALLY_STREAMING void process_demapped_row(
      Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    Columns<ScalarType> demapped_src{rowbuffer_.get(),
                                     4 * src_rows_.channels()};

    svuint32_t bias = svdup_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2);

    auto vector_path = [&](svbool_t pg16, svbool_t pg8,
                           ptrdiff_t step) KLEIDICV_STREAMING_COMPATIBLE {
      // Deinterleave abcd into two vectors, ac and bd
      svuint8x2_t src = svld2_u8(pg8, &demapped_src[0]);

      svuint16_t src_a = svmovlb_u16(svget2(src, 0));
      svuint16_t src_b = svmovlb_u16(svget2(src, 1));
      svuint16_t src_c = svmovlt_u16(svget2(src, 0));
      svuint16_t src_d = svmovlt_u16(svget2(src, 1));

      interpolate_and_store(pg16, step, mapfrac, dst, src_a, src_b, src_c,
                            src_d, bias);
      demapped_src += step;
    };

    svbool_t ptrue_16 = FracVecTraits::svptrue();
    svbool_t ptrue_8 = svptrue_b8();
    LoopUnroll loop{rowbuffer_width_, FracVecTraits::num_lanes()};
    loop.unroll_once([&](size_t step) KLEIDICV_STREAMING_COMPATIBLE {
      vector_path(ptrue_16, ptrue_8, static_cast<ptrdiff_t>(step));
    });
    loop.remaining([&](size_t length, size_t) KLEIDICV_STREAMING_COMPATIBLE {
      svbool_t pg16 = FracVecTraits::svwhilelt(size_t{0}, length);
      svbool_t pg8 = svwhilelt_b8(size_t{0}, 2 * length);
      vector_path(pg16, pg8, static_cast<ptrdiff_t>(length));
    });
  }

  size_t rowbuffer_width_;
  std::unique_ptr<ScalarType> rowbuffer_;
};  // end of class RemapS16Point5SME2<uint8_t>
#endif  // KLEIDICV_TARGET_SME2

template <typename T>
kleidicv_error_t remap_s16point5_sc(
    const T* src, size_t src_stride, size_t src_width, size_t src_height,
    T* dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels, const int16_t* mapxy, size_t mapxy_stride,
    const uint16_t* mapfrac, size_t mapfrac_stride,
    kleidicv_border_type_t border_type, kleidicv_border_values_t) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapxy, mapxy_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapfrac, mapfrac_stride, dst_height);
  CHECK_IMAGE_SIZE(src_width, src_height);
  CHECK_IMAGE_SIZE(dst_width, dst_height);

  if (!remap_s16point5_is_implemented<T>(dst_width, border_type, channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<const int16_t> mapxy_rows{mapxy, mapxy_stride, 2};
  Rows<const uint16_t> mapfrac_rows{mapfrac, mapfrac_stride, 1};
  Rows<T> dst_rows{dst, dst_stride, channels};
  svuint16_t sv_src_stride;
  svint16_t sv_xmax, sv_ymax;

#if KLEIDICV_TARGET_SME2
  RemapS16Point5SME2<T> operation{src_rows,      src_width, src_height,
                                  sv_src_stride, sv_xmax,   sv_ymax};
#else
  RemapS16Point5SVE2<T> operation{src_rows,      src_width, src_height,
                                  sv_src_stride, sv_xmax,   sv_ymax};
#endif  // KLEIDICV_TARGET_SME2
  Rectangle rect{dst_width, dst_height};
  zip_rows(operation, rect, mapxy_rows, mapfrac_rows, dst_rows);
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_REMAP

#endif  // KLEIDICV_REMAP_SC_H
