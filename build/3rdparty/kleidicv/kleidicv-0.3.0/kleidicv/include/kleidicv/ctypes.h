// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

/// @file ctypes.h
/// @brief Helper type definitions

#ifndef KLEIDICV_CTYPES_H
#define KLEIDICV_CTYPES_H

#ifdef __cplusplus
#include <cinttypes>
#include <cstddef>
#else  // __cplusplus
#include "inttypes.h"
#include "stddef.h"
#endif  // __cplusplus

#include "kleidicv/config.h"

/// Error values reported by KleidiCV
typedef enum KLEIDICV_NODISCARD {
  /// Success.
  KLEIDICV_OK = 0,
  /// Requested operation is not implemented.
  KLEIDICV_ERROR_NOT_IMPLEMENTED,
  /// Null pointer was passed as an argument.
  KLEIDICV_ERROR_NULL_POINTER,
  /// A value was encountered outside the representable or valid range.
  KLEIDICV_ERROR_RANGE,
  /// Could not allocate memory.
  KLEIDICV_ERROR_ALLOCATION,
  /// A value did not meet alignment requirements.
  KLEIDICV_ERROR_ALIGNMENT,
  /// The provided context (like @ref kleidicv_morphology_context_t) is not
  /// compatible with the operation.
  KLEIDICV_ERROR_CONTEXT_MISMATCH,
} kleidicv_error_t;

/// Struct to represent a point
typedef struct {
  /// x coordinate
  size_t x;
  /// y coordinate
  size_t y;
} kleidicv_point_t;

/// Struct to represent a rectangle
typedef struct {
  /// Width of the rectangle
  size_t width;
  /// Height of the rectangle
  size_t height;
} kleidicv_rectangle_t;

/// Struct to store border values
typedef struct {
  /// Top border value
  double top;
  /// Left border value
  double left;
  /// Bottom border value
  double bottom;
  /// Right border value
  double right;
} kleidicv_border_values_t;

/// KleidiCV border types
typedef enum {
  /// The border is a constant value.
  KLEIDICV_BORDER_TYPE_CONSTANT,
  /// The border is the value of the first/last element.
  KLEIDICV_BORDER_TYPE_REPLICATE,
  /// The border is the mirrored value of the first/last elements.
  KLEIDICV_BORDER_TYPE_REFLECT,
  /// The border simply acts as a "wrap around" to the beginning/end.
  KLEIDICV_BORDER_TYPE_WRAP,
  /// Like KLEIDICV_BORDER_TYPE_REFLECT, but the first/last elements are
  /// ignored.
  KLEIDICV_BORDER_TYPE_REVERSE,
  /// The border is the "continuation" of the input rows. It is the caller's
  /// responsibility to provide the input data (and an appropriate stride value)
  /// in a way that the rows can be under and over read. E.g. can be used when
  /// executing an operation on a region of a picture.
  KLEIDICV_BORDER_TYPE_TRANSPARENT,
  /// The border is a hard border, there are no additional values to use.
  KLEIDICV_BORDER_TYPE_NONE,
} kleidicv_border_type_t;

/// KleidiCV interpolation types
typedef enum {
  /** Nearest neighbour interpolation */
  KLEIDICV_INTERPOLATION_NEAREST,
  /** Bilinear interpolation */
  KLEIDICV_INTERPOLATION_LINEAR,
} kleidicv_interpolation_type_t;

/// Internal structure where morphology operations store their state
typedef struct kleidicv_morphology_context_t_ kleidicv_morphology_context_t;

/// Internal structure where filter operations store their state
typedef struct kleidicv_filter_context_t_ kleidicv_filter_context_t;

#endif  // KLEIDICV_CTYPES_H
