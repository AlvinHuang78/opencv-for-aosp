// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv_hal.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

#include "kleidicv/filters/blur_and_downsample.h"
#include "kleidicv/filters/gaussian_blur.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv_thread/kleidicv_thread.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/hal/interface.h"

namespace kleidicv::hal {

// Images with fewer elements than this tend to perform better with KleidiCV's
// single-threaded implementation than its multi-threaded implementation.
enum {
  MULTITHREAD_MIN_ELEMENTS_GRAY_TO_RGB_U8 = 180000,
  MULTITHREAD_MIN_ELEMENTS_MIN_MAX_LOC_U8 = 100000,
  MULTITHREAD_MIN_ELEMENTS_MIN_MAX_S8 = 180000,
  MULTITHREAD_MIN_ELEMENTS_MIN_MAX_U8 = 180000,
  MULTITHREAD_MIN_ELEMENTS_MIN_MAX_S16 = 100000,
  MULTITHREAD_MIN_ELEMENTS_MIN_MAX_U16 = 100000,
  MULTITHREAD_MIN_ELEMENTS_MIN_MAX_S32 = 40000,
  MULTITHREAD_MIN_ELEMENTS_MIN_MAX_F32 = 40000,
  MULTITHREAD_MIN_ELEMENTS_RESIZE_TO_QUARTER_U8 = 150000,
  MULTITHREAD_MIN_ELEMENTS_RGB_TO_BGR_U8 = 180000,
  MULTITHREAD_MIN_ELEMENTS_RGBA_TO_BGRA_U8 = 11000,
  MULTITHREAD_MIN_ELEMENTS_SCALE_U8 = 5000,
  MULTITHREAD_MIN_ELEMENTS_SCALE_F32 = 20000,
};

static int convert_error(kleidicv_error_t e) {
  switch (e) {
    case KLEIDICV_OK:
      return CV_HAL_ERROR_OK;
    case KLEIDICV_ERROR_NOT_IMPLEMENTED:
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
    // Even if KleidiCV returns this error it's possible that another
    // implementation could handle the misalignment.
    case KLEIDICV_ERROR_ALIGNMENT:
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
    default:
      return CV_HAL_ERROR_UNKNOWN;
  }
}

// Returns the size in bytes of an OpenCV type.
static size_t get_type_size(int depth) {
  switch (depth) {
    case CV_8U:
    case CV_8S:
      return 1;
    case CV_16U:
    case CV_16S:
      return 2;
    case CV_32S:
      return 4;
    default:
      return SIZE_MAX;
  }
}

static kleidicv_error_t parallel(kleidicv_thread_callback callback,
                                 void *callback_data, void * /*parallel_data*/,
                                 unsigned task_count) {
  std::atomic<kleidicv_error_t> shared_result{KLEIDICV_OK};

  auto invoke_callback = [&](const cv::Range &range) {
    kleidicv_error_t result = callback(range.start, range.end, callback_data);
    if (result != KLEIDICV_OK) {
      shared_result.store(result);
    }
  };
  cv::parallel_for_(cv::Range(0, task_count), invoke_callback);
  return shared_result;
}

static kleidicv_thread_multithreading get_multithreading() {
  return kleidicv_thread_multithreading{parallel, nullptr};
}

// Note: 'dcn' is already accounted for in 'dst_step'.
int gray_to_bgr(const uchar *src_data, size_t src_step, uchar *dst_data,
                size_t dst_step, int width, int height, int depth, int dcn) {
  if (KLEIDICV_UNLIKELY((dcn != 3) && (dcn != 4))) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (depth == CV_8U) {
    if (dcn == 3) {
      return convert_error(
          width * height < MULTITHREAD_MIN_ELEMENTS_GRAY_TO_RGB_U8
              ? kleidicv_gray_to_rgb_u8(
                    reinterpret_cast<const uint8_t *>(src_data), src_step,
                    reinterpret_cast<uint8_t *>(dst_data), dst_step, width,
                    height)
              : kleidicv_thread_gray_to_rgb_u8(
                    reinterpret_cast<const uint8_t *>(src_data), src_step,
                    reinterpret_cast<uint8_t *>(dst_data), dst_step, width,
                    height, get_multithreading()));
    }
    return convert_error(
        width * height < MULTITHREAD_MIN_ELEMENTS_GRAY_TO_RGB_U8
            ? kleidicv_gray_to_rgba_u8(
                  reinterpret_cast<const uint8_t *>(src_data), src_step,
                  reinterpret_cast<uint8_t *>(dst_data), dst_step, width,
                  height)
            : kleidicv_thread_gray_to_rgba_u8(
                  reinterpret_cast<const uint8_t *>(src_data), src_step,
                  reinterpret_cast<uint8_t *>(dst_data), dst_step, width,
                  height, get_multithreading()));
  }

  return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int bgr_to_bgr(const uchar *src_data, size_t src_step, uchar *dst_data,
               size_t dst_step, int width, int height, int depth, int scn,
               int dcn, bool swapBlue) {
  if (KLEIDICV_UNLIKELY(((scn != 3) && (scn != 4)) ||
                        ((dcn != 3) && (dcn != 4)))) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (depth == CV_8U) {
    if (scn == 3 && dcn == 3) {
      if (swapBlue) {
        return convert_error(
            width * height < MULTITHREAD_MIN_ELEMENTS_RGB_TO_BGR_U8
                ? kleidicv_rgb_to_bgr_u8(
                      reinterpret_cast<const uint8_t *>(src_data), src_step,
                      reinterpret_cast<uint8_t *>(dst_data), dst_step, width,
                      height)
                : kleidicv_thread_rgb_to_bgr_u8(
                      reinterpret_cast<const uint8_t *>(src_data), src_step,
                      reinterpret_cast<uint8_t *>(dst_data), dst_step, width,
                      height, get_multithreading()));
      }
      return convert_error(kleidicv_rgb_to_rgb_u8(
          reinterpret_cast<const uint8_t *>(src_data), src_step,
          reinterpret_cast<uint8_t *>(dst_data), dst_step, width, height));
    }

    if (scn == 4 && dcn == 4) {
      if (swapBlue) {
        return convert_error(
            width * height < MULTITHREAD_MIN_ELEMENTS_RGBA_TO_BGRA_U8
                ? kleidicv_rgba_to_bgra_u8(
                      reinterpret_cast<const uint8_t *>(src_data), src_step,
                      reinterpret_cast<uint8_t *>(dst_data), dst_step, width,
                      height)
                : kleidicv_thread_rgba_to_bgra_u8(
                      reinterpret_cast<const uint8_t *>(src_data), src_step,
                      reinterpret_cast<uint8_t *>(dst_data), dst_step, width,
                      height, get_multithreading()));
      }
      return convert_error(kleidicv_rgba_to_rgba_u8(
          reinterpret_cast<const uint8_t *>(src_data), src_step,
          reinterpret_cast<uint8_t *>(dst_data), dst_step, width, height));
    }
  }

  return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int yuv_to_bgr_sp(const uchar *src_data, size_t src_step, uchar *dst_data,
                  size_t dst_step, int dst_width, int dst_height, int dcn,
                  bool swapBlue, int uIdx) {
  const uchar *uv_data =
      reinterpret_cast<const uint8_t *>(src_data) + dst_height * src_step;
  return yuv_to_bgr_sp_ex(src_data, src_step, uv_data, src_step, dst_data,
                          dst_step, dst_width, dst_height, dcn, swapBlue, uIdx);
}

int yuv_to_bgr_sp_ex(const uchar *y_data, size_t y_step, const uchar *uv_data,
                     size_t uv_step, uchar *dst_data, size_t dst_step,
                     int dst_width, int dst_height, int dcn, bool swapBlue,
                     int uIdx) {
  const bool is_bgr = !swapBlue;
  const bool is_nv21 = (uIdx != 0);

  auto mt = get_multithreading();

  if (dcn == 3) {
    if (is_bgr) {
      return convert_error(kleidicv_thread_yuv_sp_to_bgr_u8(
          reinterpret_cast<const uint8_t *>(y_data), y_step,
          reinterpret_cast<const uint8_t *>(uv_data), uv_step,
          reinterpret_cast<uint8_t *>(dst_data), dst_step, dst_width,
          dst_height, is_nv21, mt));
    }
    return convert_error(kleidicv_thread_yuv_sp_to_rgb_u8(
        reinterpret_cast<const uint8_t *>(y_data), y_step,
        reinterpret_cast<const uint8_t *>(uv_data), uv_step,
        reinterpret_cast<uint8_t *>(dst_data), dst_step, dst_width, dst_height,
        is_nv21, mt));
  }

  if (dcn == 4) {
    if (is_bgr) {
      return convert_error(kleidicv_thread_yuv_sp_to_bgra_u8(
          reinterpret_cast<const uint8_t *>(y_data), y_step,
          reinterpret_cast<const uint8_t *>(uv_data), uv_step,
          reinterpret_cast<uint8_t *>(dst_data), dst_step, dst_width,
          dst_height, is_nv21, mt));
    }
    return convert_error(kleidicv_thread_yuv_sp_to_rgba_u8(
        reinterpret_cast<const uint8_t *>(y_data), y_step,
        reinterpret_cast<const uint8_t *>(uv_data), uv_step,
        reinterpret_cast<uint8_t *>(dst_data), dst_step, dst_width, dst_height,
        is_nv21, mt));
  }

  return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int yuv_to_bgr(const uchar *src_data, size_t src_step, uchar *dst_data,
               size_t dst_step, int width, int height, int depth, int dcn,
               bool swapBlue, bool isCbCr) {
  const bool is_bgr = !swapBlue;

  if (depth != CV_8U || isCbCr || dcn != 3) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  auto mt = get_multithreading();

  if (is_bgr) {
    return convert_error(kleidicv_thread_yuv_to_bgr_u8(
        reinterpret_cast<const uint8_t *>(src_data), src_step,
        reinterpret_cast<uint8_t *>(dst_data), dst_step,
        static_cast<size_t>(width), static_cast<size_t>(height), mt));
  }
  return convert_error(kleidicv_thread_yuv_to_rgb_u8(
      reinterpret_cast<const uint8_t *>(src_data), src_step,
      reinterpret_cast<uint8_t *>(dst_data), dst_step,
      static_cast<size_t>(width), static_cast<size_t>(height), mt));
}

int bgr_to_yuv(const uchar *src_data, size_t src_step, uchar *dst_data,
               size_t dst_step, int width, int height, int depth, int scn,
               bool swapBlue, bool isCbCr) {
  const bool is_bgr = !swapBlue;

  if (depth != CV_8U || isCbCr) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  auto mt = get_multithreading();

  if (scn == 3) {
    if (is_bgr) {
      return convert_error(kleidicv_thread_bgr_to_yuv_u8(
          reinterpret_cast<const uint8_t *>(src_data), src_step,
          reinterpret_cast<uint8_t *>(dst_data), dst_step,
          static_cast<size_t>(width), static_cast<size_t>(height), mt));
    }
    return convert_error(kleidicv_thread_rgb_to_yuv_u8(
        reinterpret_cast<const uint8_t *>(src_data), src_step,
        reinterpret_cast<uint8_t *>(dst_data), dst_step,
        static_cast<size_t>(width), static_cast<size_t>(height), mt));
  }

  if (scn == 4) {
    if (is_bgr) {
      return convert_error(kleidicv_thread_bgra_to_yuv_u8(
          reinterpret_cast<const uint8_t *>(src_data), src_step,
          reinterpret_cast<uint8_t *>(dst_data), dst_step,
          static_cast<size_t>(width), static_cast<size_t>(height), mt));
    }
    return convert_error(kleidicv_thread_rgba_to_yuv_u8(
        reinterpret_cast<const uint8_t *>(src_data), src_step,
        reinterpret_cast<uint8_t *>(dst_data), dst_step,
        static_cast<size_t>(width), static_cast<size_t>(height), mt));
  }

  return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int threshold(const uchar *src_data, size_t src_step, uchar *dst_data,
              size_t dst_step, int width, int height, int depth, int cn,
              double thresh, double maxValue, int thresholdType) {
  auto mt = get_multithreading();

  if ((depth == CV_8U) && (thresholdType == 0 /* THRESH_BINARY */)) {
    size_t width_in_elements = width * cn;
    return convert_error(kleidicv_thread_threshold_binary_u8(
        reinterpret_cast<const uint8_t *>(src_data), src_step,
        reinterpret_cast<uint8_t *>(dst_data), dst_step, width_in_elements,
        height, static_cast<uint8_t>(thresh), static_cast<uint8_t>(maxValue),
        mt));
  }

  return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

// Converts an OpenCV border type to a KleidiCV border type.
static int from_opencv(int opencv_border_type,
                       kleidicv_border_type_t &border_type) {
  switch (opencv_border_type) {
    default:
      return 1;
    case CV_HAL_BORDER_CONSTANT:
      border_type = kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_CONSTANT;
      break;
    case CV_HAL_BORDER_REPLICATE:
      border_type = kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_REPLICATE;
      break;
    case CV_HAL_BORDER_REFLECT:
      border_type = kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_REFLECT;
      break;
    case CV_HAL_BORDER_REFLECT_101:
      border_type = kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_REVERSE;
      break;
    case CV_HAL_BORDER_WRAP:
      border_type = kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_WRAP;
      break;
    case CV_HAL_BORDER_TRANSPARENT:
      border_type = kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_TRANSPARENT;
      break;
    case CV_HAL_BORDER_ISOLATED:
      border_type = kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_NONE;
      break;
  }

  return 0;
}

// Converts an OpenCV interpolation type to a KleidiCV interpolation type.
static int from_opencv(int opencv_interpolation,
                       kleidicv_interpolation_type_t &interpolation_type) {
  switch (opencv_interpolation) {
    default:
      return 1;
    case CV_HAL_INTER_NEAREST:
      interpolation_type =
          kleidicv_interpolation_type_t::KLEIDICV_INTERPOLATION_NEAREST;
      break;
    case CV_HAL_INTER_LINEAR:
      interpolation_type =
          kleidicv_interpolation_type_t::KLEIDICV_INTERPOLATION_LINEAR;
      break;
  }

  return 0;
}

struct SeparableFilter2DParams {
  size_t channels;
  kleidicv_border_type_t border_type;
  int operation_depth;
  const uint8_t *kernel_x;
  size_t kernel_width;
  const uint8_t *kernel_y;
  size_t kernel_height;
  kleidicv_filter_context_t *cached_filter_context;
  size_t cached_max_image_width;
  size_t cached_max_image_height;
};

int separable_filter_2d_init(cvhalFilter2D **context, int src_type,
                             int dst_type, int kernel_type, uchar *kernelx_data,
                             int kernelx_length, uchar *kernely_data,
                             int kernely_length, int anchor_x, int anchor_y,
                             double delta, int borderType) {
  if (src_type != dst_type) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (CV_MAT_DEPTH(src_type) != CV_MAT_DEPTH(kernel_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (CV_MAT_CN(kernel_type) != 1) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  int operation_depth = CV_MAT_DEPTH(src_type);
  if (operation_depth != CV_8U && operation_depth != CV_16U &&
      operation_depth != CV_16S) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_border_type_t kleidicv_border_type;
  if (from_opencv(borderType, kleidicv_border_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (kleidicv_border_type !=
          kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_REPLICATE &&
      kleidicv_border_type !=
          kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_REFLECT &&
      kleidicv_border_type !=
          kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_WRAP &&
      kleidicv_border_type !=
          kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_REVERSE) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (anchor_x > -1 && anchor_x != (kernelx_length >> 1)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (anchor_y > -1 && anchor_y != (kernely_length >> 1)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (delta != 0.0) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  // Use std::unique_ptr<T> to make error returns safer.
  auto params = std::make_unique<SeparableFilter2DParams>();
  if (!params) {
    return CV_HAL_ERROR_UNKNOWN;
  }

  size_t type_size = get_type_size(operation_depth);

  uint8_t *kernel_x = new uint8_t[kernelx_length * type_size];
  uint8_t *kernel_y = new uint8_t[kernely_length * type_size];

  std::memcpy(kernel_x, kernelx_data, kernelx_length * type_size);
  std::memcpy(kernel_y, kernely_data, kernely_length * type_size);

  params->channels = (src_type >> CV_CN_SHIFT) + 1;
  params->border_type = kleidicv_border_type;
  params->operation_depth = operation_depth;

  params->kernel_x = kernel_x;
  params->kernel_width = static_cast<size_t>(kernelx_length);

  params->kernel_y = kernel_y;
  params->kernel_height = static_cast<size_t>(kernely_length);

  params->cached_filter_context = nullptr;

  *context = reinterpret_cast<cvhalFilter2D *>(params.release());
  return CV_HAL_ERROR_OK;
}

int separable_filter_2d_operation(cvhalFilter2D *context, uchar *src_data,
                                  size_t src_step, uchar *dst_data,
                                  size_t dst_step, int width, int height,
                                  int full_width, int full_height, int offset_x,
                                  int offset_y) {
  if (!context) {
    return CV_HAL_ERROR_UNKNOWN;
  }

  size_t margin_left = static_cast<size_t>(offset_x);
  size_t margin_top = static_cast<size_t>(offset_y);
  size_t margin_right = static_cast<size_t>(full_width - width - offset_x);
  size_t margin_bottom = static_cast<size_t>(full_height - height - offset_y);

  if (margin_left != 0 || margin_top != 0 || margin_right != 0 ||
      margin_bottom != 0) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  auto params = reinterpret_cast<SeparableFilter2DParams *>(context);
  size_t width_sz = static_cast<size_t>(width);
  size_t height_sz = static_cast<size_t>(height);

  kleidicv_filter_context_t *filter_context = params->cached_filter_context;
  if (filter_context && (width_sz > params->cached_max_image_width ||
                         height_sz > params->cached_max_image_height)) {
    if (kleidicv_error_t release_err =
            kleidicv_filter_context_release(params->cached_filter_context)) {
      return convert_error(release_err);
    }

    filter_context = nullptr;
  }

  if (!filter_context) {
    kleidicv_error_t create_err = kleidicv_filter_context_create(
        &filter_context, params->channels, params->kernel_width,
        params->kernel_height, width_sz, height_sz);
    if (create_err) {
      return convert_error(create_err);
    }
    params->cached_filter_context = filter_context;
    params->cached_max_image_width = width_sz;
    params->cached_max_image_height = height_sz;
  }

  auto mt = get_multithreading();

  kleidicv_error_t filter_err;
  switch (params->operation_depth) {
    case CV_8U:
      filter_err = kleidicv_thread_separable_filter_2d_u8(
          reinterpret_cast<const uint8_t *>(src_data), src_step,
          reinterpret_cast<uint8_t *>(dst_data), dst_step,
          static_cast<size_t>(width), static_cast<size_t>(height),
          params->channels, params->kernel_x, params->kernel_width,
          params->kernel_y, params->kernel_height, params->border_type,
          filter_context, mt);
      break;
    case CV_16U:
      filter_err = kleidicv_thread_separable_filter_2d_u16(
          reinterpret_cast<const uint16_t *>(src_data), src_step,
          reinterpret_cast<uint16_t *>(dst_data), dst_step,
          static_cast<size_t>(width), static_cast<size_t>(height),
          params->channels,
          reinterpret_cast<const uint16_t *>(params->kernel_x),
          params->kernel_width,
          reinterpret_cast<const uint16_t *>(params->kernel_y),
          params->kernel_height, params->border_type, filter_context, mt);
      break;
    case CV_16S:
      filter_err = kleidicv_thread_separable_filter_2d_s16(
          reinterpret_cast<const int16_t *>(src_data), src_step,
          reinterpret_cast<int16_t *>(dst_data), dst_step,
          static_cast<size_t>(width), static_cast<size_t>(height),
          params->channels, reinterpret_cast<const int16_t *>(params->kernel_x),
          params->kernel_width,
          reinterpret_cast<const int16_t *>(params->kernel_y),
          params->kernel_height, params->border_type, filter_context, mt);
      break;
    default:
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  return convert_error(filter_err);
}

int separable_filter_2d_free(cvhalFilter2D *context) {
  if (!context) {
    return CV_HAL_ERROR_UNKNOWN;
  }

  std::unique_ptr<SeparableFilter2DParams> params(
      reinterpret_cast<SeparableFilter2DParams *>(context));
  delete[] params->kernel_y;
  delete[] params->kernel_x;

  if (params->cached_filter_context) {
    kleidicv_error_t release_err =
        kleidicv_filter_context_release(params->cached_filter_context);
    return convert_error(release_err);
  }

  return CV_HAL_ERROR_OK;
}

int gaussian_blur_binomial(const uchar *src_data, size_t src_step,
                           uchar *dst_data, size_t dst_step, int width,
                           int height, int depth, int cn, size_t margin_left,
                           size_t margin_top, size_t margin_right,
                           size_t margin_bottom, size_t kernel_size,
                           int border_type) {
  if (src_data == dst_data) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (margin_left != 0 || margin_top != 0 || margin_right != 0 ||
      margin_bottom != 0) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  switch (depth) {
    case CV_8U:
      break;

    default:
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_border_type_t kleidicv_border_type;
  if (from_opencv(border_type, kleidicv_border_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  // Check for not-implemented before allocating a context
  if (!kleidicv::gaussian_blur_is_implemented(width, height, kernel_size,
                                              kernel_size, 0, 0) ||
      !kleidicv::get_fixed_border_type(kleidicv_border_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_filter_context_t *context;
  if (kleidicv_error_t create_err = kleidicv_filter_context_create(
          &context, cn, kernel_size, kernel_size, static_cast<size_t>(width),
          static_cast<size_t>(height))) {
    return convert_error(create_err);
  }

  auto mt = get_multithreading();
  kleidicv_error_t blur_err = kleidicv_thread_gaussian_blur_u8(
      reinterpret_cast<const uint8_t *>(src_data), src_step,
      reinterpret_cast<uint8_t *>(dst_data), dst_step, width, height, cn,
      kernel_size, kernel_size, 0.0, 0.0, kleidicv_border_type, context, mt);

  kleidicv_error_t release_err = kleidicv_filter_context_release(context);

  return convert_error(blur_err ? blur_err : release_err);
}

int gaussian_blur(const uchar *src_data, size_t src_step, uchar *dst_data,
                  size_t dst_step, int width, int height, int depth, int cn,
                  size_t margin_left, size_t margin_top, size_t margin_right,
                  size_t margin_bottom, size_t kernel_width,
                  size_t kernel_height, double sigma_x, double sigma_y,
                  int border_type) {
  if (src_data == dst_data) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (margin_left != 0 || margin_top != 0 || margin_right != 0 ||
      margin_bottom != 0) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  switch (depth) {
    case CV_8U:
      break;

    default:
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_border_type_t kleidicv_border_type;
  if (from_opencv(border_type, kleidicv_border_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  // Check for not-implemented before allocating a context
  if (!kleidicv::gaussian_blur_is_implemented(
          width, height, kernel_width, kernel_height, sigma_x, sigma_y) ||
      !kleidicv::get_fixed_border_type(kleidicv_border_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_filter_context_t *context;
  if (kleidicv_error_t create_err = kleidicv_filter_context_create(
          &context, cn, kernel_width, kernel_height, static_cast<size_t>(width),
          static_cast<size_t>(height))) {
    return convert_error(create_err);
  }

  auto mt = get_multithreading();
  kleidicv_error_t blur_err = kleidicv_thread_gaussian_blur_u8(
      reinterpret_cast<const uint8_t *>(src_data), src_step,
      reinterpret_cast<uint8_t *>(dst_data), dst_step, width, height, cn,
      kernel_width, kernel_height, sigma_x, sigma_y, kleidicv_border_type,
      context, mt);

  kleidicv_error_t release_err = kleidicv_filter_context_release(context);

  return convert_error(blur_err ? blur_err : release_err);
}

struct MorphologyParams {
  kleidicv_morphology_context_t *context;
  decltype(kleidicv_dilate_u8) impl;
};

int morphology_init(cvhalFilter2D **cvcontext, int operation, int src_type,
                    int dst_type, int max_width, int max_height,
                    int kernel_type, uchar *kernel_data, size_t kernel_step,
                    int kernel_width, int kernel_height, int anchor_x,
                    int anchor_y, int cvborder_type,
                    const double cvborder_values[4], int iterations,
                    bool allow_submatrix, bool allow_in_place) {
  // Some parameters are unused.
  (void)allow_in_place;

  if (src_type != dst_type) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (CV_MAT_DEPTH(src_type) != CV_8U) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (allow_submatrix) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_border_type_t border_type;
  if (from_opencv(cvborder_type, border_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (border_type != kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_CONSTANT &&
      border_type != kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_REPLICATE) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  size_t kernel_width_sz = static_cast<size_t>(kernel_width);
  size_t kernel_height_sz = static_cast<size_t>(kernel_height);
  size_t kernel_area = kernel_width_sz * kernel_height_sz;

#if !KLEIDICV_ENABLE_ALL_OPENCV_HAL
  // KleidiCV is not that fast on smaller kernels
  if (kernel_width_sz < 5 || kernel_height_sz < 5) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }
#endif

  switch (CV_MAT_DEPTH(kernel_type)) {
    case CV_8U: {
      size_t nonzero_count = 0;
      if (kleidicv_error_t err = kleidicv_count_nonzeros_u8(
              static_cast<uint8_t *>(kernel_data), kernel_step, kernel_width_sz,
              kernel_height_sz, &nonzero_count)) {
        return convert_error(err);
      }
      if (nonzero_count != kernel_area) {
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
      }
      break;
    }
    default:
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  // Use std::unique_ptr<T> to make error returns safer.
  auto params = std::make_unique<MorphologyParams>();
  if (!params) {
    return CV_HAL_ERROR_UNKNOWN;
  }

  size_t channels = (src_type >> CV_CN_SHIFT) + 1;
  size_t type_size = get_type_size(CV_MAT_DEPTH(src_type));
  if (SIZE_MAX == type_size) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_border_values_t border_values = {};
  if (border_type == kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_CONSTANT) {
    border_values.top = cvborder_values[0];
    border_values.left = cvborder_values[1];
    border_values.bottom = cvborder_values[2];
    border_values.right = cvborder_values[3];

    // In case of default border values for dilate() '0' should be used.
    auto border_max_value =
        std::numeric_limits<decltype(border_values.top)>::max();
    if ((operation == CV_HAL_MORPH_DILATE) &&
        (border_values.top == border_max_value) &&
        (border_values.left == border_max_value) &&
        (border_values.bottom == border_max_value) &&
        (border_values.right == border_max_value)) {
      border_values.top = 0;
      border_values.left = 0;
      border_values.bottom = 0;
      border_values.right = 0;
    }
  }

  switch (operation) {
    case CV_HAL_MORPH_DILATE:
      params->impl = kleidicv_dilate_u8;
      break;
    case CV_HAL_MORPH_ERODE:
      params->impl = kleidicv_erode_u8;
      break;
    default:
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (kleidicv_error_t err = kleidicv_morphology_create(
          &params->context,
          kleidicv_rectangle_t{static_cast<size_t>(kernel_width),
                               static_cast<size_t>(kernel_height)},
          kleidicv_point_t{static_cast<size_t>(anchor_x),
                           static_cast<size_t>(anchor_y)},
          border_type, border_values, channels, iterations, type_size,
          kleidicv_rectangle_t{static_cast<size_t>(max_width),
                               static_cast<size_t>(max_height)})) {
    return convert_error(err);
  }

  *cvcontext = reinterpret_cast<cvhalFilter2D *>(params.release());
  return CV_HAL_ERROR_OK;
}

int morphology_operation(cvhalFilter2D *cvcontext, uchar *src_data,
                         size_t src_step, uchar *dst_data, size_t dst_step,
                         int width, int height, int src_full_width,
                         int src_full_height, int src_roi_x, int src_roi_y,
                         int dst_full_width, int dst_full_height, int dst_roi_x,
                         int dst_roi_y) {
  // Some parameters are unused.
  (void)src_full_width;
  (void)src_full_height;
  (void)src_roi_x;
  (void)src_roi_y;
  (void)dst_full_width;
  (void)dst_full_height;
  (void)dst_roi_x;
  (void)dst_roi_y;

  auto params = reinterpret_cast<MorphologyParams *>(cvcontext);
  return convert_error(
      params->impl(reinterpret_cast<const uint8_t *>(src_data), src_step,
                   reinterpret_cast<uint8_t *>(dst_data), dst_step,
                   static_cast<size_t>(width), static_cast<size_t>(height),
                   params->context));
}

int morphology_free(cvhalFilter2D *cvcontext) {
  std::unique_ptr<MorphologyParams> params(
      reinterpret_cast<MorphologyParams *>(cvcontext));
  return convert_error(kleidicv_morphology_release(params->context));
}

int resize(int src_type, const uchar *src_data, size_t src_step, int src_width,
           int src_height, uchar *dst_data, size_t dst_step, int dst_width,
           int dst_height, double inv_scale_x, double inv_scale_y,
           int interpolation) {
  if (src_data == dst_data) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  size_t channels = (src_type >> CV_CN_SHIFT) + 1;
  if (channels != 1) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (CV_MAT_DEPTH(src_type) == CV_8U && inv_scale_x == 0.5 &&
      inv_scale_y == 0.5 &&
      (interpolation == CV_HAL_INTER_LINEAR ||
       interpolation == CV_HAL_INTER_AREA)) {
    if (src_width * src_height <
        MULTITHREAD_MIN_ELEMENTS_RESIZE_TO_QUARTER_U8) {
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
    } else {
      return convert_error(kleidicv_thread_resize_to_quarter_u8(
          src_data, src_step, src_width, src_height, dst_data, dst_step,
          dst_width, dst_height, get_multithreading()));
    }
  }

  if (interpolation != CV_HAL_INTER_LINEAR) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if ((inv_scale_x != 0 &&
       std::abs(inv_scale_x * src_width - dst_width) > FLT_EPSILON) ||
      (inv_scale_y != 0 &&
       std::abs(inv_scale_y * src_height - dst_height) > FLT_EPSILON)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  switch (CV_MAT_DEPTH(src_type)) {
    case CV_8U:
      return convert_error(kleidicv_thread_resize_linear_u8(
          src_data, src_step, src_width, src_height, dst_data, dst_step,
          dst_width, dst_height, get_multithreading()));
    case CV_32F:
      if (inv_scale_x <= 2.1 && inv_scale_y <= 2.1) {
        return convert_error(kleidicv_thread_resize_linear_f32(
            reinterpret_cast<const float *>(src_data), src_step, src_width,
            src_height, reinterpret_cast<float *>(dst_data), dst_step,
            dst_width, dst_height, get_multithreading()));
      } else {
        // Bigger resize algorithms (4x4 and 8x8) don't perform well with
        // multiple threads
#if KLEIDICV_ENABLE_ALL_OPENCV_HAL
        return convert_error(kleidicv_thread_resize_linear_f32(
            reinterpret_cast<const float *>(src_data), src_step, src_width,
            src_height, reinterpret_cast<float *>(dst_data), dst_step,
            dst_width, dst_height, get_multithreading()));
#else
        if (cv::getNumThreads() == 1) {
          return convert_error(kleidicv_resize_linear_f32(
              reinterpret_cast<const float *>(src_data), src_step, src_width,
              src_height, reinterpret_cast<float *>(dst_data), dst_step,
              dst_width, dst_height));
        }
#endif  // KLEIDICV_ENABLE_ALL_OPENCV_HAL
      }
  }
  return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int sobel(const uchar *src_data, size_t src_step, uchar *dst_data,
          size_t dst_step, int width, int height, int src_depth, int dst_depth,
          int cn, int margin_left, int margin_top, int margin_right,
          int margin_bottom, int dx, int dy, int ksize, double scale,
          double delta, int border_type) {
  if (src_data == dst_data) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (src_depth != CV_8U) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (dst_depth != CV_16S) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (ksize != 3) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (width < 3 || height < 3) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (scale != 1.0) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (delta != 0.0) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (border_type != CV_HAL_BORDER_REPLICATE) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (margin_left || margin_top || margin_right || margin_bottom) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  auto mt = get_multithreading();

  if (dx == 1 && dy == 0) {
    return convert_error(kleidicv_thread_sobel_3x3_horizontal_s16_u8(
        src_data, src_step, reinterpret_cast<int16_t *>(dst_data), dst_step,
        width, height, cn, mt));
  }

  if (dx == 0 && dy == 1) {
    return convert_error(kleidicv_thread_sobel_3x3_vertical_s16_u8(
        src_data, src_step, reinterpret_cast<int16_t *>(dst_data), dst_step,
        width, height, cn, mt));
  }

  return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#if KLEIDICV_EXPERIMENTAL_FEATURE_CANNY
int canny(const uchar *src_data, size_t src_step, uchar *dst_data,
          size_t dst_step, int width, int height, int cn, double lowThreshold,
          double highThreshold, int ksize, bool L2gradient) {
  if (ksize != 3) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (width < 3 || height < 3) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (L2gradient != false) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (cn != 1) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  return convert_error(
      kleidicv_canny_u8(reinterpret_cast<const uint8_t *>(src_data), src_step,
                        reinterpret_cast<uint8_t *>(dst_data), dst_step,
                        static_cast<size_t>(width), static_cast<size_t>(height),
                        lowThreshold, highThreshold));
}
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_CANNY

int transpose(const uchar *src_data, size_t src_step, uchar *dst_data,
              size_t dst_step, int src_width, int src_height,
              int element_size) {
  if ((element_size != 1) && (element_size != 2)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  // Inplace transpose is only implemented if width and height is equal
  if (src_data == dst_data && src_width != src_height) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  return convert_error(kleidicv_transpose(
      reinterpret_cast<const void *>(src_data), src_step,
      reinterpret_cast<void *>(dst_data), dst_step,
      static_cast<size_t>(src_width), static_cast<size_t>(src_height),
      static_cast<size_t>(element_size)));
}

template <typename T, typename SingleThreadFunc, typename MultithreadFunc>
kleidicv_error_t call_min_max(SingleThreadFunc min_max_func_st,
                              MultithreadFunc min_max_func_mt,
                              int multithread_min_elements,
                              const uchar *src_data, size_t src_stride,
                              int width, int height, double *min_value,
                              double *max_value,
                              kleidicv_thread_multithreading mt) {
  T tmp_min_value, tmp_max_value;
  T *p_min_value = min_value ? &tmp_min_value : nullptr;
  T *p_max_value = max_value ? &tmp_max_value : nullptr;
  kleidicv_error_t err =
      width * height < multithread_min_elements
          ? min_max_func_st(reinterpret_cast<const T *>(src_data), src_stride,
                            static_cast<size_t>(width),
                            static_cast<size_t>(height), p_min_value,
                            p_max_value)
          : min_max_func_mt(reinterpret_cast<const T *>(src_data), src_stride,
                            static_cast<size_t>(width),
                            static_cast<size_t>(height), p_min_value,
                            p_max_value, mt);
  if (min_value) {
    *min_value = static_cast<double>(tmp_min_value);
  }
  if (max_value) {
    *max_value = static_cast<double>(tmp_max_value);
  }
  return err;
}

template <typename T, typename SingleThreadFunc, typename MultithreadFunc>
kleidicv_error_t call_min_max_loc(
    SingleThreadFunc min_max_loc_func_st, MultithreadFunc min_max_loc_func_mt,
    int multithread_min_elements, const uchar *src_data, size_t src_stride,
    int width, int height, double *min_value, double *max_value, int *min_index,
    int *max_index, kleidicv_thread_multithreading mt) {
  size_t tmp_min_offset = 0, tmp_max_offset = 0;
  size_t *p_min_offset = (min_value || min_index) ? &tmp_min_offset : nullptr;
  size_t *p_max_offset = (max_value || max_index) ? &tmp_max_offset : nullptr;

  kleidicv_error_t err =
      width * height < multithread_min_elements
          ? min_max_loc_func_st(reinterpret_cast<const T *>(src_data),
                                src_stride, static_cast<size_t>(width),
                                static_cast<size_t>(height), p_min_offset,
                                p_max_offset)
          : min_max_loc_func_mt(reinterpret_cast<const T *>(src_data),
                                src_stride, static_cast<size_t>(width),
                                static_cast<size_t>(height), p_min_offset,
                                p_max_offset, mt);
  if (min_value) {
    *min_value = static_cast<double>(src_data[tmp_min_offset]);
  }
  if (max_value) {
    *max_value = static_cast<double>(src_data[tmp_max_offset]);
  }

  // Convert from offset into array to x,y coordinates.
  auto offset_to_coords = [](int *coords, size_t offset, size_t src_stride) {
    if (!coords) return;
    if (src_stride) {
      /* row */ coords[0] = offset / src_stride;
      /* col */ coords[1] = (offset % src_stride) / sizeof(T);
    } else {
      /* row */ coords[0] = 0;
      /* col */ coords[1] = offset / sizeof(T);
    }
  };
  offset_to_coords(min_index, tmp_min_offset, src_stride);
  offset_to_coords(max_index, tmp_max_offset, src_stride);

  return err;
}

int min_max_idx(const uchar *src_data, size_t src_step, int width, int height,
                int depth, double *minVal, double *maxVal, int *minIdx,
                int *maxIdx, uchar *mask) {
  if (mask) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if (minIdx || maxIdx) {
    if (depth == CV_8U) {
      return convert_error(call_min_max_loc<uint8_t>(
          kleidicv_min_max_loc_u8, kleidicv_thread_min_max_loc_u8,
          MULTITHREAD_MIN_ELEMENTS_MIN_MAX_LOC_U8, src_data, src_step, width,
          height, minVal, maxVal, minIdx, maxIdx, get_multithreading()));
    }
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  switch (depth) {
    case CV_8S:
      return convert_error(call_min_max<int8_t>(
          kleidicv_min_max_s8, kleidicv_thread_min_max_s8,
          MULTITHREAD_MIN_ELEMENTS_MIN_MAX_S8, src_data, src_step, width,
          height, minVal, maxVal, get_multithreading()));
    case CV_8U:
      return convert_error(call_min_max<uint8_t>(
          kleidicv_min_max_u8, kleidicv_thread_min_max_u8,
          MULTITHREAD_MIN_ELEMENTS_MIN_MAX_U8, src_data, src_step, width,
          height, minVal, maxVal, get_multithreading()));
    case CV_16S:
      return convert_error(call_min_max<int16_t>(
          kleidicv_min_max_s16, kleidicv_thread_min_max_s16,
          MULTITHREAD_MIN_ELEMENTS_MIN_MAX_S16, src_data, src_step, width,
          height, minVal, maxVal, get_multithreading()));
    case CV_16U:
      return convert_error(call_min_max<uint16_t>(
          kleidicv_min_max_u16, kleidicv_thread_min_max_u16,
          MULTITHREAD_MIN_ELEMENTS_MIN_MAX_U16, src_data, src_step, width,
          height, minVal, maxVal, get_multithreading()));
    case CV_32S:
      return convert_error(call_min_max<int32_t>(
          kleidicv_min_max_s32, kleidicv_thread_min_max_s32,
          MULTITHREAD_MIN_ELEMENTS_MIN_MAX_S32, src_data, src_step, width,
          height, minVal, maxVal, get_multithreading()));
    case CV_32F:
      return convert_error(call_min_max<float>(
          kleidicv_min_max_f32, kleidicv_thread_min_max_f32,
          MULTITHREAD_MIN_ELEMENTS_MIN_MAX_F32, src_data, src_step, width,
          height, minVal, maxVal, get_multithreading()));
    default:
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }
}

int convertTo(const uchar *src_data, size_t src_step, int src_depth,
              uchar *dst_data, size_t dst_step, int dst_depth, int width,
              int height, double scale, double shift) {
  auto mt = get_multithreading();

  // scaling only
  if (src_depth == dst_depth) {
    // no scaling, no advantage
    if (fabs(scale - 1.0) < std::numeric_limits<double>::epsilon() &&
        fabs(shift) < std::numeric_limits<double>::epsilon()) {
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    switch (src_depth) {
      case CV_8U:
        return convert_error(
            width * height < MULTITHREAD_MIN_ELEMENTS_SCALE_U8
                ? kleidicv_scale_u8(
                      reinterpret_cast<const uint8_t *>(src_data), src_step,
                      reinterpret_cast<uint8_t *>(dst_data), dst_step, width,
                      height, static_cast<float>(scale),
                      static_cast<float>(shift))
                : kleidicv_thread_scale_u8(
                      reinterpret_cast<const uint8_t *>(src_data), src_step,
                      reinterpret_cast<uint8_t *>(dst_data), dst_step, width,
                      height, static_cast<float>(scale),
                      static_cast<float>(shift), mt));
      case CV_32F:
        return convert_error(
            width * height < MULTITHREAD_MIN_ELEMENTS_SCALE_F32
                ? kleidicv_scale_f32(
                      reinterpret_cast<const float *>(src_data), src_step,
                      reinterpret_cast<float *>(dst_data), dst_step, width,
                      height, static_cast<float>(scale),
                      static_cast<float>(shift))
                : kleidicv_thread_scale_f32(
                      reinterpret_cast<const float *>(src_data), src_step,
                      reinterpret_cast<float *>(dst_data), dst_step, width,
                      height, static_cast<float>(scale),
                      static_cast<float>(shift), mt));
      default:
        break;
    }
  }

  // type conversion only
  if (scale == 1.0 && shift == 0.0) {
    // float32 to int8
    if (src_depth == CV_32F && dst_depth == CV_8S) {
      return convert_error(kleidicv_thread_f32_to_s8(
          reinterpret_cast<const float *>(src_data), src_step,
          reinterpret_cast<int8_t *>(dst_data), dst_step, width, height, mt));
    }
    // float32 to uint8
    if (src_depth == CV_32F && dst_depth == CV_8U) {
      return convert_error(kleidicv_thread_f32_to_u8(
          reinterpret_cast<const float *>(src_data), src_step,
          reinterpret_cast<uint8_t *>(dst_data), dst_step, width, height, mt));
    }
    // int8 to float32
    if (src_depth == CV_8S && dst_depth == CV_32F) {
      return convert_error(kleidicv_thread_s8_to_f32(
          reinterpret_cast<const int8_t *>(src_data), src_step,
          reinterpret_cast<float *>(dst_data), dst_step, width, height, mt));
    }
    // uint8 to float32
    if (src_depth == CV_8U && dst_depth == CV_32F) {
      return convert_error(kleidicv_thread_u8_to_f32(
          reinterpret_cast<const uint8_t *>(src_data), src_step,
          reinterpret_cast<float *>(dst_data), dst_step, width, height, mt));
    }
  }
  return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int exp32f(const float *src, float *dst, int len) {
  auto mt = get_multithreading();

  return convert_error(kleidicv_thread_exp_f32(
      src, len * sizeof(float), dst, len * sizeof(float), len, 1, mt));
}

int compare_u8(const uchar *src1_data, size_t src1_step, const uchar *src2_data,
               size_t src2_step, uchar *dst_data, size_t dst_step, int width,
               int height, int operation) {
  auto mt = get_multithreading();

  switch (operation) {
    case cv::CMP_EQ:
      return convert_error(kleidicv_thread_compare_equal_u8(
          src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width,
          height, mt));
    case cv::CMP_GT:
      return convert_error(kleidicv_thread_compare_greater_u8(
          src1_data, src1_step, src2_data, src2_step, dst_data, dst_step, width,
          height, mt));
    default:
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }
}

int inRange_u8(const uchar *src_data, size_t src_step, uchar *dst_data,
               size_t dst_step, int dst_depth, int width, int height, int cn,
               uchar lower_bound, uchar upper_bound) {
  if (dst_depth != CV_8U || cn != 1) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }
  return convert_error(kleidicv_in_range_u8(
      reinterpret_cast<const uint8_t *>(src_data), src_step,
      reinterpret_cast<uint8_t *>(dst_data), dst_step, width, height,
      static_cast<uint8_t>(lower_bound), static_cast<uint8_t>(upper_bound)));
}

int inRange_f32(const uchar *src_data, size_t src_step, uchar *dst_data,
                size_t dst_step, int dst_depth, int width, int height, int cn,
                double lower_bound, double upper_bound) {
  if (dst_depth != CV_8U || cn != 1) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }
  return convert_error(kleidicv_in_range_f32(
      reinterpret_cast<const float *>(src_data), src_step,
      reinterpret_cast<uint8_t *>(dst_data), dst_step, width, height,
      static_cast<float>(lower_bound), static_cast<float>(upper_bound)));
}

#if KLEIDICV_EXPERIMENTAL_FEATURE_REMAP

int remap_s16(int src_type, const uchar *src_data, size_t src_step,
              int src_width, int src_height, uchar *dst_data, size_t dst_step,
              int dst_width, int dst_height, const int16_t *mapxy,
              size_t mapxy_step, int border_type,
              const double border_value[4]) {
  kleidicv_border_type_t kleidicv_border_type;
  if (from_opencv(border_type, kleidicv_border_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_border_values_t border_values = {};
  if (border_type == kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_CONSTANT) {
    border_values.top = border_value[0];
    border_values.left = border_value[1];
    border_values.bottom = border_value[2];
    border_values.right = border_value[3];
  }
  auto mt = get_multithreading();

  if (src_type == CV_8UC1) {
    return convert_error(kleidicv_thread_remap_s16_u8(
        src_data, src_step, static_cast<size_t>(src_width),
        static_cast<size_t>(src_height), dst_data, dst_step,
        static_cast<size_t>(dst_width), static_cast<size_t>(dst_height),
        CV_MAT_CN(src_type), mapxy, mapxy_step, kleidicv_border_type,
        border_values, mt));
  }

  return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int remap_s16point5(int src_type, const uchar *src_data, size_t src_step,
                    int src_width, int src_height, uchar *dst_data,
                    size_t dst_step, int dst_width, int dst_height,
                    const int16_t *mapxy, size_t mapxy_step,
                    const uint16_t *mapfrac, size_t mapfrac_step,
                    int border_type, const double border_value[4]) {
  kleidicv_border_type_t kleidicv_border_type;
  if (from_opencv(border_type, kleidicv_border_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_border_values_t border_values = {};
  if (border_type == kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_CONSTANT) {
    border_values.top = border_value[0];
    border_values.left = border_value[1];
    border_values.bottom = border_value[2];
    border_values.right = border_value[3];
  }
  auto mt = get_multithreading();

  if (src_type == CV_8UC1) {
    return convert_error(kleidicv_thread_remap_s16point5_u8(
        src_data, src_step, static_cast<size_t>(src_width),
        static_cast<size_t>(src_height), dst_data, dst_step,
        static_cast<size_t>(dst_width), static_cast<size_t>(dst_height), 1,
        mapxy, mapxy_step, mapfrac, mapfrac_step, kleidicv_border_type,
        border_values, mt));
  }

  return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_REMAP

int pyrdown(const uchar *src_data, size_t src_step, int src_width,
            int src_height, uchar *dst_data, size_t dst_step, int dst_width,
            int dst_height, int depth, int cn, int border_type) {
  if (src_data == dst_data) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  switch (depth) {
    case CV_8U:
      break;

    default:
      return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  if ((dst_width != (src_width + 1) / 2) ||
      (dst_height != (src_height + 1) / 2)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_border_type_t kleidicv_border_type;
  if (from_opencv(border_type, kleidicv_border_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  // Check for not-implemented before allocating a context
  if (!kleidicv::blur_and_downsample_is_implemented(src_width, src_height,
                                                    cn) ||
      !kleidicv::get_fixed_border_type(kleidicv_border_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_filter_context_t *context;
  if (kleidicv_error_t create_err = kleidicv_filter_context_create(
          &context, cn, 5, 5, static_cast<size_t>(src_width),
          static_cast<size_t>(src_height))) {
    return convert_error(create_err);
  }

  auto mt = get_multithreading();
  kleidicv_error_t blur_err = kleidicv_thread_blur_and_downsample_u8(
      reinterpret_cast<const uint8_t *>(src_data), src_step, src_width,
      src_height, reinterpret_cast<uint8_t *>(dst_data), dst_step, cn,
      kleidicv_border_type, context, mt);

  kleidicv_error_t release_err = kleidicv_filter_context_release(context);

  return convert_error(blur_err ? blur_err : release_err);
}

int scharr_deriv(const uchar *src_data, size_t src_step, int16_t *dst_data,
                 size_t dst_step, int width, int height, int cn) {
  // OpenCV provides the source pointer in a way that out-of-bounds reads are
  // possible to handle borders. On the other hand, KleidiCV expects that the
  // source pointer points to the top left pixel to be read by the algorithm.
  const uint8_t *src =
      reinterpret_cast<const uint8_t *>(src_data - src_step) - cn;

  auto mt = get_multithreading();
  return convert_error(kleidicv_thread_scharr_interleaved_s16_u8(
      src, src_step, width + 2, height + 2, cn, dst_data, dst_step, mt));
}

#if KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
int warp_perspective(int src_type, const uchar *src_data, size_t src_step,
                     int src_width, int src_height, uchar *dst_data,
                     size_t dst_step, int dst_width, int dst_height,
                     const double transformation[9], int interpolation,
                     int border_type, const double border_value[4]) {
  kleidicv_border_type_t kleidicv_border_type;
  if (from_opencv(border_type, kleidicv_border_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  float float_transformation[9];
  for (size_t i = 0; i < 9; ++i) {
    float_transformation[i] = static_cast<float>(transformation[i]);
  }

  kleidicv_interpolation_type_t kleidicv_interpolation_type;
  if (from_opencv(interpolation, kleidicv_interpolation_type)) {
    return CV_HAL_ERROR_NOT_IMPLEMENTED;
  }

  kleidicv_border_values_t border_values = {};
  if (kleidicv_border_type ==
      kleidicv_border_type_t::KLEIDICV_BORDER_TYPE_CONSTANT) {
    border_values.top = border_value[0];
    border_values.left = border_value[1];
    border_values.bottom = border_value[2];
    border_values.right = border_value[3];
  }
  auto mt = get_multithreading();

  if (src_type == CV_8UC1) {
    return convert_error(kleidicv_thread_warp_perspective_u8(
        src_data, src_step, static_cast<size_t>(src_width),
        static_cast<size_t>(src_height), dst_data, dst_step,
        static_cast<size_t>(dst_width), static_cast<size_t>(dst_height),
        float_transformation, 1, kleidicv_interpolation_type,
        kleidicv_border_type, border_values, mt));
  }

  return CV_HAL_ERROR_NOT_IMPLEMENTED;
}
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE

}  // namespace kleidicv::hal
