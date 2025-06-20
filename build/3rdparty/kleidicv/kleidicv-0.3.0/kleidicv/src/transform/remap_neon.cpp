// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"
#include "kleidicv/transform/remap.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
namespace kleidicv::neon {

template <typename ScalarType>
class RemapS16;

template <>
class RemapS16<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using MapVecTraits = neon::VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;

  RemapS16(Rows<const ScalarType> src_rows, size_t src_width, size_t src_height)
      : src_rows_{src_rows},
        v_src_stride_{vdupq_n_s16(static_cast<int16_t>(src_rows_.stride()))},
        v_xmax_{vdupq_n_s16(static_cast<int16_t>(src_width - 1))},
        v_ymax_{vdupq_n_s16(static_cast<int16_t>(src_height - 1))} {}

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      MapVector2Type xy = vld2q_s16(&mapxy[0]);
      // Clamp coordinates to within the dimensions of the source image
      uint16x8_t x = vreinterpretq_u16_s16(
          vmaxq_s16(vdupq_n_s16(0), vminq_s16(xy.val[0], v_xmax_)));
      uint16x8_t y = vreinterpretq_u16_s16(
          vmaxq_s16(vdupq_n_s16(0), vminq_s16(xy.val[1], v_ymax_)));
      // Calculate offsets from coordinates (y * stride + x)
      uint32x4_t indices_low =
          vmlal_u16(vmovl_u16(vget_low_u16(x)), vget_low_u16(y),
                    vget_low_u16(v_src_stride_));
      // Copy pixels from source
      dst[0] = src_rows_[vgetq_lane_u32(indices_low, 0)];
      dst[1] = src_rows_[vgetq_lane_u32(indices_low, 1)];
      dst[2] = src_rows_[vgetq_lane_u32(indices_low, 2)];
      dst[3] = src_rows_[vgetq_lane_u32(indices_low, 3)];
      uint32x4_t indices_high =
          vmlal_high_u16(vmovl_high_u16(x), y, v_src_stride_);
      dst[4] = src_rows_[vgetq_lane_u32(indices_high, 0)];
      dst[5] = src_rows_[vgetq_lane_u32(indices_high, 1)];
      dst[6] = src_rows_[vgetq_lane_u32(indices_high, 2)];
      dst[7] = src_rows_[vgetq_lane_u32(indices_high, 3)];
      mapxy += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };

    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

 private:
  Rows<const ScalarType> src_rows_;
  int16x8_t v_src_stride_;
  int16x8_t v_xmax_;
  int16x8_t v_ymax_;
};  // end of class RemapS16<uint8_t>

template <typename T>
kleidicv_error_t remap_s16(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels, const int16_t *mapxy, size_t mapxy_stride,
    kleidicv_border_type_t border_type,
    [[maybe_unused]] kleidicv_border_values_t border_values) {
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
  RemapS16<T> operation{src_rows, src_width, src_height};
  Rectangle rect{dst_width, dst_height};
  zip_rows(operation, rect, mapxy_rows, dst_rows);
  return KLEIDICV_OK;
}

template <typename ScalarType>
class RemapS16Point5;

template <>
class RemapS16Point5<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using MapVecTraits = neon::VecTraits<int16_t>;
  using MapVectorType = typename MapVecTraits::VectorType;
  using MapVector2Type = typename MapVecTraits::Vector2Type;
  using FracVecTraits = neon::VecTraits<uint16_t>;
  using FracVectorType = typename FracVecTraits::VectorType;

  RemapS16Point5(Rows<const ScalarType> src_rows, size_t src_width,
                 size_t src_height)
      : src_rows_{src_rows},
        v_src_stride_{vdupq_n_s16(static_cast<int16_t>(src_rows_.stride()))},
        v_xmax_{vdupq_n_s16(static_cast<int16_t>(src_width - 1))},
        v_ymax_{vdupq_n_s16(static_cast<int16_t>(src_height - 1))} {}

  void process_row(size_t width, Columns<const int16_t> mapxy,
                   Columns<const uint16_t> mapfrac, Columns<ScalarType> dst) {
    auto vector_path = [&](size_t step) {
      MapVector2Type xy = vld2q_s16(&mapxy[0]);
      FracVectorType frac = vld1q_u16(&mapfrac[0]);
      uint16x8_t xfrac =
          vbslq_u16(vcltq_s16(xy.val[0], vdupq_n_s16(0)), vdupq_n_u16(0),
                    vandq_u16(frac, vdupq_n_u16(REMAP16POINT5_FRAC_MAX - 1)));
      uint16x8_t yfrac =
          vbslq_u16(vcltq_s16(xy.val[1], vdupq_n_s16(0)), vdupq_n_u16(0),
                    vandq_u16(vshrq_n_u16(frac, REMAP16POINT5_FRAC_BITS),
                              vdupq_n_u16(REMAP16POINT5_FRAC_MAX - 1)));
      uint16x8_t nxfrac = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), xfrac);
      uint16x8_t nyfrac = vsubq_u16(vdupq_n_u16(REMAP16POINT5_FRAC_MAX), yfrac);

      // Clamp coordinates to within the dimensions of the source image
      uint16x8_t x0 = vreinterpretq_u16_s16(
          vmaxq_s16(vdupq_n_s16(0), vminq_s16(xy.val[0], v_xmax_)));
      uint16x8_t y0 = vreinterpretq_u16_s16(
          vmaxq_s16(vdupq_n_s16(0), vminq_s16(xy.val[1], v_ymax_)));

      // x1 = x0 + 1, except if it's already xmax
      uint16x8_t x1 = vsubq_u16(x0, vcltq_s16(xy.val[0], v_xmax_));
      uint16x8_t y1 = vsubq_u16(y0, vcltq_s16(xy.val[1], v_ymax_));

      uint16x4_t dst_low = load_and_interpolate(
          vmovl_u16(vget_low_u16(x0)), vget_low_u16(y0),
          vmovl_u16(vget_low_u16(x1)), vget_low_u16(y1), vget_low_u16(xfrac),
          vget_low_u16(yfrac), vget_low_u16(nxfrac), vget_low_u16(nyfrac));

      uint16x4_t dst_high = load_and_interpolate(
          vmovl_high_u16(x0), vget_high_u16(y0), vmovl_high_u16(x1),
          vget_high_u16(y1), vget_high_u16(xfrac), vget_high_u16(yfrac),
          vget_high_u16(nxfrac), vget_high_u16(nyfrac));

      vst1_u8(&dst[0], vuzp1_u8(dst_low, dst_high));
      mapxy += ptrdiff_t(step);
      mapfrac += ptrdiff_t(step);
      dst += ptrdiff_t(step);
    };
    LoopUnroll loop{width, MapVecTraits::num_lanes()};
    loop.unroll_once(vector_path);
    ptrdiff_t back_step = static_cast<ptrdiff_t>(loop.step()) -
                          static_cast<ptrdiff_t>(loop.remaining_length());
    mapxy -= back_step;
    mapfrac -= back_step;
    dst -= back_step;
    loop.remaining([&](size_t, size_t step) { vector_path(step); });
  }

 private:
  uint16x4_t load_and_interpolate(uint32x4_t x0, uint16x4_t y0, uint32x4_t x1,
                                  uint16x4_t y1, uint16x4_t xfrac,
                                  uint16x4_t yfrac, uint16x4_t nxfrac,
                                  uint16x4_t nyfrac) {
    // Calculate offsets from coordinates (y * stride + x)
    // a: top left, b: top right, c: bottom left, d: bottom right
    uint32x4_t offset = vmlal_u16(x0, y0, vget_low_u16(v_src_stride_));
    uint64_t acc =
        static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 0)]) |
        (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 1)]) << 16) |
        (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 2)]) << 32) |
        (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 3)]) << 48);
    uint16x4_t a = vreinterpret_u16_u64(vset_lane_u64(acc, vdup_n_u64(0), 0));

    offset = vmlal_u16(x1, y0, vget_low_u16(v_src_stride_));

    acc = static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 0)]) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 1)]) << 16) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 2)]) << 32) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 3)]) << 48);
    uint16x4_t b = vreinterpret_u16_u64(vset_lane_u64(acc, vdup_n_u64(0), 0));

    uint16x4_t line0 = vmla_u16(vmul_u16(xfrac, b), nxfrac, a);

    offset = vmlal_u16(x0, y1, vget_low_u16(v_src_stride_));

    acc = static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 0)]) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 1)]) << 16) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 2)]) << 32) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 3)]) << 48);
    uint16x4_t c = vreinterpret_u16_u64(vset_lane_u64(acc, vdup_n_u64(0), 0));

    uint32x4_t line0_lerpd = vmlal_u16(
        vdupq_n_u32(REMAP16POINT5_FRAC_MAX_SQUARE / 2), line0, nyfrac);

    offset = vmlal_u16(x1, y1, vget_low_u16(v_src_stride_));

    acc = static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 0)]) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 1)]) << 16) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 2)]) << 32) |
          (static_cast<uint64_t>(src_rows_[vgetq_lane_u32(offset, 3)]) << 48);
    uint16x4_t d = vreinterpret_u16_u64(vset_lane_u64(acc, vdup_n_u64(0), 0));

    uint16x4_t line1 = vmla_u16(vmul_u16(xfrac, d), nxfrac, c);
    return vshrn_n_u32(vmlal_u16(line0_lerpd, line1, yfrac),
                       2 * REMAP16POINT5_FRAC_BITS);
  }

  Rows<const ScalarType> src_rows_;
  int16x8_t v_src_stride_;
  int16x8_t v_xmax_;
  int16x8_t v_ymax_;
};  // end of class RemapS16Point5<uint8_t>

template <typename T>
kleidicv_error_t remap_s16point5(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels, const int16_t *mapxy, size_t mapxy_stride,
    const uint16_t *mapfrac, size_t mapfrac_stride,
    [[maybe_unused]] kleidicv_border_type_t border_type,
    [[maybe_unused]] kleidicv_border_values_t border_values) {
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
  RemapS16Point5<T> operation{src_rows, src_width, src_height};
  Rectangle rect{dst_width, dst_height};
  zip_rows(operation, rect, mapxy_rows, mapfrac_rows, dst_rows);
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(type)                          \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_s16<type>(          \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const int16_t *mapxy, size_t mapxy_stride,              \
      kleidicv_border_type_t border_type,                                      \
      kleidicv_border_values_t border_values)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(uint8_t);

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(type)                    \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_s16point5<type>(    \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const int16_t *mapxy, size_t mapxy_stride,              \
      const uint16_t *mapfrac, size_t mapfrac_stride,                          \
      kleidicv_border_type_t border_type,                                      \
      kleidicv_border_values_t border_values)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(uint8_t);

}  // namespace kleidicv::neon
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
