// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_THREAD_H
#define KLEIDICV_THREAD_H

#include "kleidicv/kleidicv.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Internal - not part of the public API and its direct use is not supported.
///
/// Signature of a function to be invoked for each thread.
/// @param begin index of first task.
/// @param end index of the last task + 1.
/// @param data opaque pointer to data internal to the callback implementation.
typedef kleidicv_error_t (*kleidicv_thread_callback)(unsigned begin,
                                                     unsigned end, void *data);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Signature of a function to invoke callbacks in parallel.
///
/// KleidiCV has no dependencies on any particular threading library.
/// Therefore, to allow multithreading to work an implementation of this
/// function must be provided to invoke callbacks across multiple threads via
/// the preferred threading library or framework.
///
/// @param callback the function to invoke in parallel.
/// @param callback_data opaque pointer to pass as the callback's data argument.
/// @param parallel_data pointer from kleidicv_thread_multithreading_t, for use
///                      by the implementation of this function.
/// @param task_count number of tasks to run.
typedef kleidicv_error_t (*kleidicv_thread_parallel)(
    kleidicv_thread_callback callback, void *callback_data, void *parallel_data,
    unsigned task_count);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Encapsulates the data required to run callbacks across multiple threads.
typedef struct {
  kleidicv_thread_parallel parallel;
  void *parallel_data;
} kleidicv_thread_multithreading;

#define KLEIDICV_THREAD_UNARY_OP(name, src_type, dst_type)                     \
  kleidicv_error_t name(const src_type *src, size_t src_stride, dst_type *dst, \
                        size_t dst_stride, size_t width, size_t height,        \
                        kleidicv_thread_multithreading)

KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_gray_to_rgb_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_gray_to_rgba_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_rgb_to_bgr_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_rgb_to_rgb_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_rgba_to_bgra_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_rgba_to_rgba_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_rgb_to_bgra_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_rgb_to_rgba_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_rgba_to_bgr_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_rgba_to_rgb_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_yuv_to_bgr_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_yuv_to_rgb_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_bgr_to_yuv_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_rgb_to_yuv_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_bgra_to_yuv_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_rgba_to_yuv_u8, uint8_t, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_exp_f32, float, float);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_f32_to_s8, float, int8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_f32_to_u8, float, uint8_t);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_s8_to_f32, int8_t, float);
KLEIDICV_THREAD_UNARY_OP(kleidicv_thread_u8_to_f32, uint8_t, float);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_yuv_sp_to_bgr_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_yuv_sp_to_bgr_u8(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, bool is_nv21, kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_resize_to_quarter_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_yuv_sp_to_bgra_u8(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, bool is_nv21, kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_resize_to_quarter_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_yuv_sp_to_rgb_u8(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, bool is_nv21, kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_resize_to_quarter_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_yuv_sp_to_rgba_u8(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, bool is_nv21, kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_min_max_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_min_max_u8(const uint8_t *src,
                                            size_t src_stride, size_t width,
                                            size_t height, uint8_t *min_value,
                                            uint8_t *max_value,
                                            kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_min_max_s8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_min_max_s8(const int8_t *src,
                                            size_t src_stride, size_t width,
                                            size_t height, int8_t *min_value,
                                            int8_t *max_value,
                                            kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_min_max_u16 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_min_max_u16(const uint16_t *src,
                                             size_t src_stride, size_t width,
                                             size_t height, uint16_t *min_value,
                                             uint16_t *max_value,
                                             kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_min_max_s16 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_min_max_s16(const int16_t *src,
                                             size_t src_stride, size_t width,
                                             size_t height, int16_t *min_value,
                                             int16_t *max_value,
                                             kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_min_max_s32 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_min_max_s32(const int32_t *src,
                                             size_t src_stride, size_t width,
                                             size_t height, int32_t *min_value,
                                             int32_t *max_value,
                                             kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_min_max_f32 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_min_max_f32(const float *src,
                                             size_t src_stride, size_t width,
                                             size_t height, float *min_value,
                                             float *max_value,
                                             kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_min_max_loc_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_min_max_loc_u8(
    const uint8_t *src, size_t src_stride, size_t width, size_t height,
    size_t *min_offset, size_t *max_offset, kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_resize_to_quarter_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_threshold_binary_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, uint8_t threshold, uint8_t value,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_resize_to_quarter_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_scale_u8(const uint8_t *src, size_t src_stride,
                                          uint8_t *dst, size_t dst_stride,
                                          size_t width, size_t height,
                                          float scale, float shift,
                                          kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_resize_to_quarter_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_scale_f32(const float *src, size_t src_stride,
                                           float *dst, size_t dst_stride,
                                           size_t width, size_t height,
                                           float scale, float shift,
                                           kleidicv_thread_multithreading);

#define KLEIDICV_THREAD_BINARY_OP(name, type)                              \
  kleidicv_error_t name(const type *src_a, size_t src_a_stride,            \
                        const type *src_b, size_t src_b_stride, type *dst, \
                        size_t dst_stride, size_t width, size_t height,    \
                        kleidicv_thread_multithreading)

#define KLEIDICV_THREAD_BINARY_OP_SCALE(name, type, scaletype)             \
  kleidicv_error_t name(const type *src_a, size_t src_a_stride,            \
                        const type *src_b, size_t src_b_stride, type *dst, \
                        size_t dst_stride, size_t width, size_t height,    \
                        scaletype scale, kleidicv_thread_multithreading)

KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_add_s8, int8_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_add_u8, uint8_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_add_s16, int16_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_add_u16, uint16_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_add_s32, int32_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_add_u32, uint32_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_add_s64, int64_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_add_u64, uint64_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_sub_s8, int8_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_sub_u8, uint8_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_sub_s16, int16_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_sub_u16, uint16_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_sub_s32, int32_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_sub_u32, uint32_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_sub_s64, int64_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_sub_u64, uint64_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_absdiff_u8, uint8_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_absdiff_s8, int8_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_absdiff_u16, uint16_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_absdiff_s16, int16_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_saturating_absdiff_s32, int32_t);
KLEIDICV_THREAD_BINARY_OP_SCALE(kleidicv_thread_saturating_multiply_u8, uint8_t,
                                double);
KLEIDICV_THREAD_BINARY_OP_SCALE(kleidicv_thread_saturating_multiply_s8, int8_t,
                                double);
KLEIDICV_THREAD_BINARY_OP_SCALE(kleidicv_thread_saturating_multiply_u16,
                                uint16_t, double);
KLEIDICV_THREAD_BINARY_OP_SCALE(kleidicv_thread_saturating_multiply_s16,
                                int16_t, double);
KLEIDICV_THREAD_BINARY_OP_SCALE(kleidicv_thread_saturating_multiply_s32,
                                int32_t, double);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_bitwise_and, uint8_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_compare_equal_u8, uint8_t);
KLEIDICV_THREAD_BINARY_OP(kleidicv_thread_compare_greater_u8, uint8_t);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of
/// kleidicv_saturating_add_abs_with_threshold_s16 - see the documentation of
/// that function for more details.
kleidicv_error_t kleidicv_thread_saturating_add_abs_with_threshold_s16(
    const int16_t *src_a, size_t src_a_stride, const int16_t *src_b,
    size_t src_b_stride, int16_t *dst, size_t dst_stride, size_t width,
    size_t height, int16_t threshold, kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_gaussian_blur_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_gaussian_blur_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, float sigma_x, float sigma_y,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_separable_filter_2d_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_separable_filter_2d_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint8_t *kernel_x,
    size_t kernel_width, const uint8_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_separable_filter_2d_u16 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_separable_filter_2d_u16(
    const uint16_t *src, size_t src_stride, uint16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint16_t *kernel_x,
    size_t kernel_width, const uint16_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_sobel_3x3_horizontal_s16_u8 - see
/// the documentation of that function for more details.
kleidicv_error_t kleidicv_thread_separable_filter_2d_s16(
    const int16_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const int16_t *kernel_x,
    size_t kernel_width, const int16_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_blur_and_downsample_u8 - see
/// the documentation of that function for more details.
kleidicv_error_t kleidicv_thread_blur_and_downsample_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t channels,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context,
    kleidicv_thread_multithreading mt);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_sobel_3x3_horizontal_s16_u8 - see
/// the documentation of that function for more details.
kleidicv_error_t kleidicv_thread_sobel_3x3_horizontal_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_sobel_3x3_vertical_s16_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_sobel_3x3_vertical_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_scharr_interleaved_s16_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_scharr_interleaved_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_resize_to_quarter_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_resize_to_quarter_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_resize_linear_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_resize_linear_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_resize_linear_f32 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_resize_linear_f32(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    float *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_remap_s16_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_remap_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels, const int16_t *mapxy, size_t mapxy_stride,
    kleidicv_border_type_t border_type, kleidicv_border_values_t border_values,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_remap_s16point5_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_remap_s16point5_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels, const int16_t *mapxy, size_t mapxy_stride,
    const uint16_t *mapfrac, size_t mapfrac_stride,
    kleidicv_border_type_t border_type, kleidicv_border_values_t border_values,
    kleidicv_thread_multithreading);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Multithreaded implementation of kleidicv_warp_perspective_u8 - see the
/// documentation of that function for more details.
kleidicv_error_t kleidicv_thread_warp_perspective_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    const float transformation[9], size_t channels,
    kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type, kleidicv_border_values_t border_values,
    kleidicv_thread_multithreading);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // KLEIDICV_THREAD_H
