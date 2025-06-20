// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_OPENCV_HAL_H
#define KLEIDICV_OPENCV_HAL_H

#include <cmath>
#include <type_traits>

#include "kleidicv/kleidicv.h"
#include "kleidicv_thread/kleidicv_thread.h"
#include "opencv2/core/hal/interface.h"

// Forward declarations of OpenCV internals.
struct cvhalFilter2D;

// Implemented HAL interfaces.
namespace kleidicv {
namespace hal {

// Macros to shorten repeated code.
#define KLEIDICV_HAL_API(api) (kleidicv::hal::api)

int gray_to_bgr(const uchar *src_data, size_t src_step, uchar *dst_data,
                size_t dst_step, int width, int height, int depth, int dcn);

int bgr_to_bgr(const uchar *src_data, size_t src_step, uchar *dst_data,
               size_t dst_step, int width, int height, int depth, int scn,
               int dcn, bool swapBlue);

int yuv_to_bgr_sp(const uchar *src_data, size_t src_step, uchar *dst_data,
                  size_t dst_step, int dst_width, int dst_height, int dcn,
                  bool swapBlue, int uIdx);

int yuv_to_bgr_sp_ex(const uchar *y_data, size_t y_step, const uchar *uv_data,
                     size_t uv_step, uchar *dst_data, size_t dst_step,
                     int dst_width, int dst_height, int dcn, bool swapBlue,
                     int uIdx);

int yuv_to_bgr(const uchar *src_data, size_t src_step, uchar *dst_data,
               size_t dst_step, int width, int height, int depth, int dcn,
               bool swapBlue, bool isCbCr);

int bgr_to_yuv(const uchar *src_data, size_t src_step, uchar *dst_data,
               size_t dst_step, int width, int height, int depth, int scn,
               bool swapBlue, bool isCbCr);

int threshold(const uchar *src_data, size_t src_step, uchar *dst_data,
              size_t dst_step, int width, int height, int depth, int cn,
              double thresh, double maxValue, int thresholdType);

int separable_filter_2d_init(cvhalFilter2D **context, int src_type,
                             int dst_type, int kernel_type, uchar *kernelx_data,
                             int kernelx_length, uchar *kernely_data,
                             int kernely_length, int anchor_x, int anchor_y,
                             double delta, int borderType);

int separable_filter_2d_operation(cvhalFilter2D *context, uchar *src_data,
                                  size_t src_step, uchar *dst_data,
                                  size_t dst_step, int width, int height,
                                  int full_width, int full_height, int offset_x,
                                  int offset_y);

int separable_filter_2d_free(cvhalFilter2D *context);

int gaussian_blur_binomial(const uchar *src_data, size_t src_step,
                           uchar *dst_data, size_t dst_step, int width,
                           int height, int depth, int cn, size_t margin_left,
                           size_t margin_top, size_t margin_right,
                           size_t margin_bottom, size_t kernel_size,
                           int border_type);

int gaussian_blur(const uchar *src_data, size_t src_step, uchar *dst_data,
                  size_t dst_step, int width, int height, int depth, int cn,
                  size_t margin_left, size_t margin_top, size_t margin_right,
                  size_t margin_bottom, size_t kernel_width,
                  size_t kernel_height, double sigma_x, double sigma_y,
                  int border_type);

int morphology_init(cvhalFilter2D **context, int operation, int src_type,
                    int dst_type, int max_width, int max_height,
                    int kernel_type, uchar *kernel_data, size_t kernel_step,
                    int kernel_width, int kernel_height, int anchor_x,
                    int anchor_y, int border_type,
                    const double border_values[4], int iterations,
                    bool allow_submatrix, bool allow_in_place);

int morphology_operation(cvhalFilter2D *context, uchar *src_data,
                         size_t src_step, uchar *dst_data, size_t dst_step,
                         int width, int height, int src_full_width,
                         int src_full_height, int src_roi_x, int src_roi_y,
                         int dst_full_width, int dst_full_height, int dst_roi_x,
                         int dst_roi_y);

int morphology_free(cvhalFilter2D *context);

int resize(int src_type, const uchar *src_data, size_t src_step, int src_width,
           int src_height, uchar *dst_data, size_t dst_step, int dst_width,
           int dst_height, double inv_scale_x, double inv_scale_y,
           int interpolation);

int sobel(const uchar *src_data, size_t src_step, uchar *dst_data,
          size_t dst_step, int width, int height, int src_depth, int dst_depth,
          int cn, int margin_left, int margin_top, int margin_right,
          int margin_bottom, int dx, int dy, int ksize, double scale,
          double delta, int border_type);

int compare_u8(const uchar *src1_data, size_t src1_step, const uchar *src2_data,
               size_t src2_step, uchar *dst_data, size_t dst_step, int width,
               int height, int operation);

#if KLEIDICV_EXPERIMENTAL_FEATURE_CANNY
int canny(const uchar *src_data, size_t src_step, uchar *dst_data,
          size_t dst_step, int width, int height, int cn, double lowThreshold,
          double highThreshold, int ksize, bool L2gradient);
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_CANNY

int pyrdown(const uchar *src_data, size_t src_step, int src_width,
            int src_height, uchar *dst_data, size_t dst_step, int dst_width,
            int dst_height, int depth, int cn, int border_type);

int transpose(const uchar *src_data, size_t src_step, uchar *dst_data,
              size_t dst_step, int src_width, int src_height, int element_size);

int min_max_idx(const uchar *src_data, size_t src_stride, int width, int height,
                int depth, double *min_value, double *max_value, int *min_index,
                int *max_index, uchar *mask);

int convertTo(const uchar *src_data, size_t src_step, int src_depth,
              uchar *dst_data, size_t dst_step, int dst_depth, int width,
              int height, double scale, double shift);

int exp32f(const float *src, float *dst, int len);

int inRange_u8(const uchar *src_data, size_t src_step, uchar *dst_data,
               size_t dst_step, int dst_depth, int width, int height, int cn,
               uchar lower_bound, uchar upper_bound);

int inRange_f32(const uchar *src_data, size_t src_step, uchar *dst_data,
                size_t dst_step, int dst_depth, int width, int height, int cn,
                double lower_bound, double upper_bound);

#if KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
int remap_s16(int src_type, const uchar *src_data, size_t src_step,
              int src_width, int src_height, uchar *dst_data, size_t dst_step,
              int dst_width, int dst_height, const int16_t *mapxy,
              size_t mapxy_step, int border_type, const double border_value[4]);

int remap_s16point5(int src_type, const uchar *src_data, size_t src_step,
                    int src_width, int src_height, uchar *dst_data,
                    size_t dst_step, int dst_width, int dst_height,
                    const int16_t *mapxy, size_t mapxy_step,
                    const uint16_t *mapfrac, size_t mapfrac_step,
                    int border_type, const double border_value[4]);
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_REMAP

#if KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
int warp_perspective(int src_type, const uchar *src_data, size_t src_step,
                     int src_width, int src_height, uchar *dst_data,
                     size_t dst_step, int dst_width, int dst_height,
                     const double transformation[9], int interpolation,
                     int borderType, const double borderValue[4]);
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE

int scharr_deriv(const uchar *src_data, size_t src_step, int16_t *dst_data,
                 size_t dst_step, int width, int height, int cn);

}  // namespace hal
}  // namespace kleidicv

// Other HAL implementations might require the cv namespace
namespace cv {

// If the KleidiCV function has a signature matching the OpenCV HAL interface
// AND it never returns KLEIDICV_NOT_IMPLEMENTED then we can call it directly
// and convert the return code.
#define KLEIDICV_HAL_FORWARD(kleidicv_impl, ...)               \
  (kleidicv_impl(__VA_ARGS__) == KLEIDICV_OK ? CV_HAL_ERROR_OK \
                                             : CV_HAL_ERROR_UNKNOWN)

#define KLEIDICV_HAL_FALLBACK_FORWARD(kleidicv_impl, fallback_hal_impl, ...) \
  (KLEIDICV_HAL_API(kleidicv_impl)(__VA_ARGS__) == CV_HAL_ERROR_OK           \
       ? CV_HAL_ERROR_OK                                                     \
       : fallback_hal_impl(__VA_ARGS__))

#ifdef OPENCV_IMGPROC_HAL_REPLACEMENT_HPP

// gray_to_bgr
static inline int kleidicv_gray_to_bgr_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int width, int height, int depth, int dcn) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(gray_to_bgr, cv_hal_cvtGraytoBGR,
                                       src_data, src_step, dst_data, dst_step,
                                       width, height, depth, dcn);
}
#undef cv_hal_cvtGraytoBGR
#define cv_hal_cvtGraytoBGR kleidicv_gray_to_bgr_with_fallback

// bgr_to_bgr
static inline int kleidicv_bgr_to_bgr_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int width, int height, int depth, int scn, int dcn, bool swapBlue) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(bgr_to_bgr, cv_hal_cvtBGRtoBGR, src_data,
                                       src_step, dst_data, dst_step, width,
                                       height, depth, scn, dcn, swapBlue);
}
#undef cv_hal_cvtBGRtoBGR
#define cv_hal_cvtBGRtoBGR kleidicv_bgr_to_bgr_with_fallback

// yuv_to_bgr_sp
static inline int kleidicv_yuv_to_bgr_sp_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int dst_width, int dst_height, int dcn, bool swapBlue, int uIdx) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      yuv_to_bgr_sp, cv_hal_cvtTwoPlaneYUVtoBGR, src_data, src_step, dst_data,
      dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);
}
#undef cv_hal_cvtTwoPlaneYUVtoBGR
#define cv_hal_cvtTwoPlaneYUVtoBGR kleidicv_yuv_to_bgr_sp_with_fallback

// yuv_to_bgr_sp_ex
static inline int kleidicv_yuv_to_bgr_sp_ex_with_fallback(
    const uchar *y_data, size_t y_step, const uchar *uv_data, size_t uv_step,
    uchar *dst_data, size_t dst_step, int dst_width, int dst_height, int dcn,
    bool swapBlue, int uIdx) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      yuv_to_bgr_sp_ex, cv_hal_cvtTwoPlaneYUVtoBGREx, y_data, y_step, uv_data,
      uv_step, dst_data, dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);
}
#undef cv_hal_cvtTwoPlaneYUVtoBGREx
#define cv_hal_cvtTwoPlaneYUVtoBGREx kleidicv_yuv_to_bgr_sp_ex_with_fallback

// yuv_to_bgr
static inline int kleidicv_yuv_to_bgr_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int width, int height, int depth, int dcn, bool swapBlue, bool isCbCr) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(yuv_to_bgr, cv_hal_cvtYUVtoBGR, src_data,
                                       src_step, dst_data, dst_step, width,
                                       height, depth, dcn, swapBlue, isCbCr);
}
#undef cv_hal_cvtYUVtoBGR
#define cv_hal_cvtYUVtoBGR kleidicv_yuv_to_bgr_with_fallback

// bgr_to_yuv
static inline int kleidicv_bgr_to_yuv_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int width, int height, int depth, int scn, bool swapBlue, bool isCbCr) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(bgr_to_yuv, cv_hal_cvtBGRtoYUV, src_data,
                                       src_step, dst_data, dst_step, width,
                                       height, depth, scn, swapBlue, isCbCr);
}
#undef cv_hal_cvtBGRtoYUV
#define cv_hal_cvtBGRtoYUV kleidicv_bgr_to_yuv_with_fallback

#if KLEIDICV_ENABLE_ALL_OPENCV_HAL
// threshold
static inline int kleidicv_threshold_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int width, int height, int depth, int cn, double thresh, double maxValue,
    int thresholdType) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      threshold, cv_hal_threshold, src_data, src_step, dst_data, dst_step,
      width, height, depth, cn, thresh, maxValue, thresholdType);
}
#undef cv_hal_threshold
#define cv_hal_threshold kleidicv_threshold_with_fallback
#endif  // KLEIDICV_ENABLE_ALL_OPENCV_HAL

// TODO: implement 3x3 kernels and 8UC1 -> 16SC1 data type
#if KLEIDICV_ENABLE_ALL_OPENCV_HAL

// separable_filter_2d_init
// no fallback, because it cannot be made sure that
// separable_filter_2d_operation also uses the fallback
#undef cv_hal_sepFilterInit
#define cv_hal_sepFilterInit kleidicv::hal::separable_filter_2d_init

// separable_filter_2d_operation
// no fallback, because it cannot be made sure that separable_filter_2d_init
// also uses the fallback
#undef cv_hal_sepFilter
#define cv_hal_sepFilter kleidicv::hal::separable_filter_2d_operation

// separable_filter_2d_free
// no fallback, because it cannot be made sure that separable_filter_2d_init
// also uses the fallback
#undef cv_hal_sepFilterFree
#define cv_hal_sepFilterFree kleidicv::hal::separable_filter_2d_free

#endif  // KLEIDICV_ENABLE_ALL_OPENCV_HAL

// gaussian_blur_binomial
static inline int kleidicv_gaussian_blur_binomial_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int width, int height, int depth, int cn, size_t margin_left,
    size_t margin_top, size_t margin_right, size_t margin_bottom,
    size_t kernel_size, int border_type) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      gaussian_blur_binomial, cv_hal_gaussianBlurBinomial, src_data, src_step,
      dst_data, dst_step, width, height, depth, cn, margin_left, margin_top,
      margin_right, margin_bottom, kernel_size, border_type);
}
#undef cv_hal_gaussianBlurBinomial
#define cv_hal_gaussianBlurBinomial \
  kleidicv_gaussian_blur_binomial_with_fallback

// gaussian_blur
static inline int kleidicv_gaussian_blur_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int width, int height, int depth, int cn, size_t margin_left,
    size_t margin_top, size_t margin_right, size_t margin_bottom,
    size_t kernel_width, size_t kernel_height, double sigma_x, double sigma_y,
    int border_type) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      gaussian_blur, cv_hal_gaussianBlur, src_data, src_step, dst_data,
      dst_step, width, height, depth, cn, margin_left, margin_top, margin_right,
      margin_bottom, kernel_width, kernel_height, sigma_x, sigma_y,
      border_type);
}
#undef cv_hal_gaussianBlur
#define cv_hal_gaussianBlur kleidicv_gaussian_blur_with_fallback

// morphology_init
// no fallback, because it cannot be made sure that morphology_operation also
// uses the fallback
#undef cv_hal_morphInit
#define cv_hal_morphInit kleidicv::hal::morphology_init

// morphology_operation
// no fallback, because it cannot be made sure that morphology_init also uses
// the fallback
#undef cv_hal_morph
#define cv_hal_morph kleidicv::hal::morphology_operation

// morphology_free
// no fallback, because it cannot be made sure that morphology_init also uses
// the fallback
#undef cv_hal_morphFree
#define cv_hal_morphFree kleidicv::hal::morphology_free

// resize
static inline int kleidicv_resize_with_fallback(
    int src_type, const uchar *src_data, size_t src_step, int src_width,
    int src_height, uchar *dst_data, size_t dst_step, int dst_width,
    int dst_height, double inv_scale_x, double inv_scale_y, int interpolation) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      resize, cv_hal_resize, src_type, src_data, src_step, src_width,
      src_height, dst_data, dst_step, dst_width, dst_height, inv_scale_x,
      inv_scale_y, interpolation);
}
#undef cv_hal_resize
#define cv_hal_resize kleidicv_resize_with_fallback

// sobel
static inline int kleidicv_sobel_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int width, int height, int src_depth, int dst_depth, int cn,
    int margin_left, int margin_top, int margin_right, int margin_bottom,
    int dx, int dy, int ksize, double scale, double delta, int border_type) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      sobel, cv_hal_sobel, src_data, src_step, dst_data, dst_step, width,
      height, src_depth, dst_depth, cn, margin_left, margin_top, margin_right,
      margin_bottom, dx, dy, ksize, scale, delta, border_type);
}
#undef cv_hal_sobel
#define cv_hal_sobel kleidicv_sobel_with_fallback

#if KLEIDICV_EXPERIMENTAL_FEATURE_CANNY
// canny
static inline int kleidicv_canny_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int width, int height, int cn, double lowThreshold, double highThreshold,
    int ksize, bool L2gradient) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      canny, cv_hal_canny, src_data, src_step, dst_data, dst_step, width,
      height, cn, lowThreshold, highThreshold, ksize, L2gradient);
}
#undef cv_hal_canny
#define cv_hal_canny kleidicv_canny_with_fallback
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_CANNY

#if KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
// remap
// This condition can be removed if this HAL macro is defined in all supported
// versions
#ifdef cv_hal_remap16s
static inline int kleidicv_remap_s16_with_fallback(
    int src_type, const uchar *src_data, size_t src_step, int src_width,
    int src_height, uchar *dst_data, size_t dst_step, int dst_width,
    int dst_height, const int16_t *mapxy, size_t mapxy_step, int border_type,
    const double border_value[4]) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      remap_s16, cv_hal_remap16s, src_type, src_data, src_step, src_width,
      src_height, dst_data, dst_step, dst_width, dst_height, mapxy, mapxy_step,
      border_type, border_value);
}
#undef cv_hal_remap16s
#define cv_hal_remap16s kleidicv_remap_s16_with_fallback
#endif  // cv_hal_remap16s

// This condition can be removed if this HAL macro is defined in all supported
// versions
#ifdef cv_hal_remap16s16u
static inline int kleidicv_remap_s16point5_with_fallback(
    int src_type, const uchar *src_data, size_t src_step, int src_width,
    int src_height, uchar *dst_data, size_t dst_step, int dst_width,
    int dst_height, const int16_t *mapxy, size_t mapxy_step,
    const uint16_t *mapfrac, size_t mapfrac_step, int border_type,
    const double border_value[4]) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      remap_s16point5, cv_hal_remap16s16u, src_type, src_data, src_step,
      src_width, src_height, dst_data, dst_step, dst_width, dst_height, mapxy,
      mapxy_step, mapfrac, mapfrac_step, border_type, border_value);
}

#undef cv_hal_remap16s16u
#define cv_hal_remap16s16u kleidicv_remap_s16point5_with_fallback
#endif  // cv_hal_remap16s16u

#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_REMAP

// pyrdown
static inline int kleidicv_pyrdown_with_fallback(
    const uchar *src_data, size_t src_step, int src_width, int src_height,
    uchar *dst_data, size_t dst_step, int dst_width, int dst_height, int depth,
    int cn, int border_type) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      pyrdown, cv_hal_pyrdown, src_data, src_step, src_width, src_height,
      dst_data, dst_step, dst_width, dst_height, depth, cn, border_type);
}
#undef cv_hal_pyrdown
#define cv_hal_pyrdown kleidicv_pyrdown_with_fallback

#if KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
// warp_perspective
static inline int kleidicv_warp_perspective_with_fallback(
    int src_type, const uchar *src_data, size_t src_step, int src_width,
    int src_height, uchar *dst_data, size_t dst_step, int dst_width,
    int dst_height, const double transformation[9], int interpolation,
    int borderType, const double borderValue[4]) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      warp_perspective, cv_hal_warpPerspective, src_type, src_data, src_step,
      src_width, src_height, dst_data, dst_step, dst_width, dst_height,
      transformation, interpolation, borderType, borderValue);
}

#undef cv_hal_warpPerspective
#define cv_hal_warpPerspective kleidicv_warp_perspective_with_fallback

#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE

#endif  // OPENCV_IMGPROC_HAL_REPLACEMENT_HPP

#ifdef OPENCV_CORE_HAL_REPLACEMENT_HPP

// transpose
static inline int kleidicv_transpose_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int src_width, int src_height, int element_size) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(transpose, cv_hal_transpose2d, src_data,
                                       src_step, dst_data, dst_step, src_width,
                                       src_height, element_size);
}
#undef cv_hal_transpose2d
#define cv_hal_transpose2d kleidicv_transpose_with_fallback

// min_max_idx
static inline int kleidicv_min_max_idx_with_fallback(
    const uchar *src_data, size_t src_stride, int width, int height, int depth,
    double *min_value, double *max_value, int *min_index, int *max_index,
    uchar *mask) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      min_max_idx, cv_hal_minMaxIdx, src_data, src_stride, width, height, depth,
      min_value, max_value, min_index, max_index, mask);
}
#undef cv_hal_minMaxIdx
#define cv_hal_minMaxIdx kleidicv_min_max_idx_with_fallback

#if defined(cv_hal_convertTo)
static inline int kleidicv_convertTo_with_fallback(
    const uchar *src_data, size_t src_step, int src_depth, uchar *dst_data,
    size_t dst_step, int dst_depth, int width, int height, double scale,
    double shift) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(convertTo, cv_hal_convertTo, src_data,
                                       src_step, src_depth, dst_data, dst_step,
                                       dst_depth, width, height, scale, shift);
}
#undef cv_hal_convertTo
#define cv_hal_convertTo kleidicv_convertTo_with_fallback
#endif  // defined(cv_hal_convertTo)

// exp32f
static inline int kleidicv_exp32f_with_fallback(const float *src, float *dst,
                                                int len) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(exp32f, cv_hal_exp32f, src, dst, len);
}
#undef cv_hal_exp32f
#define cv_hal_exp32f kleidicv_exp32f_with_fallback

#if KLEIDICV_ENABLE_ALL_OPENCV_HAL
// compare
static inline int kleidicv_compare_u8_with_fallback(
    const uchar *src1_data, size_t src1_step, const uchar *src2_data,
    size_t src2_step, uchar *dst_data, size_t dst_step, int width, int height,
    int operation) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      compare_u8, cv_hal_cmp8u, src1_data, src1_step, src2_data, src2_step,
      dst_data, dst_step, width, height, operation);
}
#undef cv_hal_cmp8u
#define cv_hal_cmp8u kleidicv_compare_u8_with_fallback
#endif  // KLEIDICV_ENABLE_ALL_OPENCV_HAL

// clang-format off
#undef cv_hal_add8s
#define cv_hal_add8s(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_add_s8, __VA_ARGS__)
#undef cv_hal_add8u
#define cv_hal_add8u(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_add_u8, __VA_ARGS__)
#undef cv_hal_add16s
#define cv_hal_add16s(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_add_s16, __VA_ARGS__)
#undef cv_hal_add16u
#define cv_hal_add16u(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_add_u16, __VA_ARGS__)

#undef cv_hal_sub8s
#define cv_hal_sub8s(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_sub_s8, __VA_ARGS__)
#undef cv_hal_sub8u
#define cv_hal_sub8u(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_sub_u8, __VA_ARGS__)
#undef cv_hal_sub16s
#define cv_hal_sub16s(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_sub_s16, __VA_ARGS__)
#undef cv_hal_sub16u
#define cv_hal_sub16u(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_sub_u16, __VA_ARGS__)

#undef cv_hal_absdiff8s
#define cv_hal_absdiff8s(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_absdiff_s8, __VA_ARGS__)
#undef cv_hal_absdiff8u
#define cv_hal_absdiff8u(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_absdiff_u8, __VA_ARGS__)
#undef cv_hal_absdiff16s
#define cv_hal_absdiff16s(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_absdiff_s16, __VA_ARGS__)
#undef cv_hal_absdiff16u
#define cv_hal_absdiff16u(...) KLEIDICV_HAL_FORWARD(kleidicv_saturating_absdiff_u16, __VA_ARGS__)

#undef cv_hal_and8u
#define cv_hal_and8u(...) KLEIDICV_HAL_FORWARD(kleidicv_bitwise_and, __VA_ARGS__)
// clang-format on

#define KLEIDICV_HAL_MUL(suffix, kleidicv_impl, T)                            \
  static inline int kleidicv_##suffix##_with_fallback(                        \
      const T *src_a, size_t src_a_stride, const T *src_b,                    \
      size_t src_b_stride, T *dst, size_t dst_stride, size_t width,           \
      size_t height, double scale) {                                          \
    if (scale != 1.0) {                                                       \
      return cv_hal_##suffix(src_a, src_a_stride, src_b, src_b_stride, dst,   \
                             dst_stride, width, height, scale);               \
    }                                                                         \
    return KLEIDICV_HAL_FORWARD(kleidicv_impl, src_a, src_a_stride, src_b,    \
                                src_b_stride, dst, dst_stride, width, height, \
                                scale);                                       \
  }

KLEIDICV_HAL_MUL(mul8u, kleidicv_saturating_multiply_u8, uint8_t);
#undef cv_hal_mul8u
#define cv_hal_mul8u kleidicv_mul8u_with_fallback

KLEIDICV_HAL_MUL(mul8s, kleidicv_saturating_multiply_s8, int8_t);
#undef cv_hal_mul8s
#define cv_hal_mul8s kleidicv_mul8s_with_fallback

KLEIDICV_HAL_MUL(mul16u, kleidicv_saturating_multiply_u16, uint16_t);
#undef cv_hal_mul16u
#define cv_hal_mul16u kleidicv_mul16u_with_fallback

KLEIDICV_HAL_MUL(mul16s, kleidicv_saturating_multiply_s16, int16_t);
#undef cv_hal_mul16s
#define cv_hal_mul16s kleidicv_mul16s_with_fallback

// inRange
// This condition can be removed if this HAL macro is defined in all supported
// versions
#ifdef cv_hal_inRange8u
static inline int kleidicv_in_range_u8_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int dst_depth, size_t width, size_t height, int cn, uchar lower_bound,
    uchar upper_bound) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      inRange_u8, cv_hal_inRange8u, src_data, src_step, dst_data, dst_step,
      dst_depth, width, height, cn, lower_bound, upper_bound);
}
#undef cv_hal_inRange8u
#define cv_hal_inRange8u kleidicv_in_range_u8_with_fallback
#endif  // cv_hal_inRange8u

// This condition can be removed if this HAL macro is defined in all supported
// versions
#ifdef cv_hal_inRange32f
static inline int kleidicv_in_range_f32_with_fallback(
    const uchar *src_data, size_t src_step, uchar *dst_data, size_t dst_step,
    int dst_depth, size_t width, size_t height, int cn, double lower_bound,
    double upper_bound) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(
      inRange_f32, cv_hal_inRange32f, src_data, src_step, dst_data, dst_step,
      dst_depth, width, height, cn, lower_bound, upper_bound);
}
#undef cv_hal_inRange32f
#define cv_hal_inRange32f kleidicv_in_range_f32_with_fallback
#endif  // cv_hal_inRange32f

#endif  // OPENCV_CORE_HAL_REPLACEMENT_HPP

#ifdef OPENCV_VIDEO_HAL_REPLACEMENT_HPP

// ScharrDeriv
// This condition can be removed if this HAL macro is defined in all supported
// versions
#ifdef cv_hal_ScharrDeriv
static inline int kleidicv_ScharrDeriv_with_fallback(const uchar *src_data,
                                                     size_t src_step,
                                                     int16_t *dst_data,
                                                     size_t dst_step, int width,
                                                     int height, int cn) {
  return KLEIDICV_HAL_FALLBACK_FORWARD(scharr_deriv, cv_hal_ScharrDeriv,
                                       src_data, src_step, dst_data, dst_step,
                                       width, height, cn);
}
#undef cv_hal_ScharrDeriv
#define cv_hal_ScharrDeriv kleidicv_ScharrDeriv_with_fallback
#endif  // cv_hal_ScharrDeriv

#endif  // OPENCV_VIDEO_HAL_REPLACEMENT_HPP

// Remove no longer needed macro definitions.
#undef KLEIDICV_HAL_FALLBACK_FORWARD

}  // namespace cv

#endif  // KLEIDICV_OPENCV_HAL_H
