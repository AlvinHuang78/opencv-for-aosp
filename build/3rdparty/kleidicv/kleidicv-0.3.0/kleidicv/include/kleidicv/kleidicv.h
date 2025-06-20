// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

/// @mainpage
///
/// For documentation of the KleidiCV C API see @ref kleidicv.h
///
/// Project page: https://gitlab.arm.com/kleidi/kleidicv
///
/// KleidiCV shall use <a href="https://semver.org/spec/v2.0.0.html">Semantic
/// Versioning 2.0.0</a>. The public API is defined according to what is
/// included in the Doxygen-generated documentation as well as the content of
/// the project's Markdown files. Features without such documentation are not
/// part of the public API and should not be relied upon.
///
/// @see <a href="coverage/coverage_report.html">Code Coverage Report</a>

/// @file
/// @brief For an overview of the functions and their supported types see
/// @ref doc/functionality.md.

#ifndef KLEIDICV_H
#define KLEIDICV_H

#include "kleidicv/config.h"
#include "kleidicv/ctypes.h"

#ifndef __aarch64__
#error "KleidiCV is only supported for aarch64"
#endif

#ifdef __aarch64__
/// Maximum image size in pixels the library accepts.
///
/// In case of AArch64 it is limited to (almost) 256 terapixels. This way 16 bit
/// is left for any arithmetic operations around image size or width or height.
#define KLEIDICV_MAX_IMAGE_PIXELS ((1ULL << 48) - 1)
#endif

/// Size in bytes of the largest possible element type
#define KLEIDICV_MAXIMUM_TYPE_SIZE (8)

/// Maximum number of channels
#define KLEIDICV_MAXIMUM_CHANNEL_COUNT (8)

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifdef DOXYGEN
#define KLEIDICV_API_DECLARATION(name, ...) kleidicv_error_t name(__VA_ARGS__)
#else
#define KLEIDICV_API_DECLARATION(name, ...) \
  extern kleidicv_error_t (*name)(__VA_ARGS__)
#endif

#define KLEIDICV_BINARY_OP(name, type)                                        \
  KLEIDICV_API_DECLARATION(name, const type *src_a, size_t src_a_stride,      \
                           const type *src_b, size_t src_b_stride, type *dst, \
                           size_t dst_stride, size_t width, size_t height)

#define KLEIDICV_BINARY_OP_SCALE(name, type, scaletype)                       \
  KLEIDICV_API_DECLARATION(name, const type *src_a, size_t src_a_stride,      \
                           const type *src_b, size_t src_b_stride, type *dst, \
                           size_t dst_stride, size_t width, size_t height,    \
                           scaletype scale)

/// Adds the values of the corresponding elements in `src_a` and `src_b`, and
/// puts the result into `dst`.
///
/// The addition is saturated, i.e. the result is the largest number of the
/// type of the element if the addition result would overflow. Source data
/// length (in bytes) is `stride * height`. Width and height are the same
/// for the two sources and for the destination. Number of elements is limited
/// to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s8, int8_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u8, uint8_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s16, int16_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u16, uint16_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s32, int32_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u32, uint32_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s64, int64_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u64, uint64_t);

/// Subtracts the value of the corresponding element in `src_b` from `src_a`,
/// and puts the result into `dst`.
///
/// The subtraction is saturated, i.e. the result is 0 (unsigned) or the
/// smallest possible value of the type of the element if the subtraction result
/// would underflow. Source data length (in bytes) is `stride * height`.
/// Width and height are the same for the two sources and for the destination.
/// Number of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s8, int8_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u8, uint8_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s16, int16_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u16, uint16_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s32, int32_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u32, uint32_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s64, int64_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u64, uint64_t);

/// From the corresponding elements in `src_a` and `src_b`, subtracts the lower
/// one from the higher one, and puts the result into `dst`.
///
/// The subtraction is saturated, i.e. the result is the largest number of the
/// type of the element if the result would overflow (it is only possible with
/// signed types). Source data length (in bytes) is `stride * height`. Width
/// and height are the same for the two sources and for the destination. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_u8, uint8_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_s8, int8_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_u16, uint16_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_s16, int16_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_s32, int32_t);

/// Multiplies the values of the corresponding elements in `src_a` and `src_b`,
/// and puts the result into `dst`.
///
/// The multiplication is saturated, i.e. the result is the largest number of
/// the type of the element if the multiplication result would overflow. Source
/// data length (in bytes) is `stride * height`. Width and height are the
/// same for the two sources and for the destination. Number of elements is
/// limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param scale        Currently unused parameter.
///
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_u8, uint8_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_s8, int8_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_u16, uint16_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_s16, int16_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_s32, int32_t, double);

/// Adds the absolute values of the corresponding elements in `src_a` and
/// `src_b`. Then, performs a comparison of each element's value in the result
/// with respect to a caller defined threshold. The strictly larger elements
/// remain unchanged and the rest are set to 0.
///
/// The addition is saturated, i.e. the result is the largest number of the
/// type of the element if the addition result would overflow. Source data
/// length (in bytes) is `stride * height`. Width and height are the same
/// for the two sources and for the destination. Number of elements is limited
/// to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param threshold    The value that the elements of the addition result
///                     are compared to.
///
KLEIDICV_API_DECLARATION(kleidicv_saturating_add_abs_with_threshold_s16,
                         const int16_t *src_a, size_t src_a_stride,
                         const int16_t *src_b, size_t src_b_stride,
                         int16_t *dst, size_t dst_stride, size_t width,
                         size_t height, int16_t threshold);

/// Bitwise-ands the values of the corresponding elements in `src_a` and
/// `src_b`, and puts the result into `dst`.
///
/// Source data length (in bytes) is `stride * height`. Width and height are
/// the same for the two sources and for the destination. Number of elements is
/// limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_BINARY_OP(kleidicv_bitwise_and, uint8_t);

/// Converts a grayscale image to RGB. All channels are 8-bit wide.
///
/// Destination data is filled as follows: `R = G = B = Gray`
/// resulting in `| R,G,B | R,G,B | R,G,B | ...` image
/// where each letter represents one byte of data, and one pixel is represented
/// by 3 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_gray_to_rgb_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts a grayscale image to RGBA. All channels are 8-bit wide.
///
/// Destination data is filled as follows: `R = G = B = Gray`, `A = 0xFF`
/// resulting in `| R,G,B,A | R,G,B,A | R,G,B,A | ...` image
/// where each letter represents one byte of data, and one pixel is represented
/// by 4 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_gray_to_rgba_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGB image to BGR. All channels are 8-bit wide.
///
/// Destination data is filled as follows:
/// `| B,G,R | B,G,R | B,G,R | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 3 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_bgr_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Copies a source RBG image to destination buffer.
/// All channels are 8-bit wide.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_rgb_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGBA image to BGRA. All channels are 8-bit wide.
///
/// Destination data is filled as follows:
/// `| B,G,R,A | B,G,R,A | B,G,R,A | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 4 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_bgra_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Copies a source RBGA image to destination buffer.
/// All channels are 8-bit wide.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_rgba_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGB image to BGRA. All channels are 8-bit wide.
///
/// Corresponding colors are set while Alpha channel is set to 0xFF.
/// Destination data is filled as follows:
/// `| B,G,R,A | B,G,R,A | B,G,R,A | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 4 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_bgra_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGB image to RGBA. All channels are 8-bit wide.
///
/// Corresponding colors are set while Alpha channel is set to 0xFF.
/// Destination data is filled as follows:
/// `| R,G,B,A | R,G,B,A | R,G,B,A | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 4 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_rgba_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGBA image to BGR. All channels are 8-bit wide.
///
/// Corresponding colors are set while Alpha channel is discarded.
/// Destination data is filled as follows:
/// `| B,G,R | B,G,R | B,G,R | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 3 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_bgr_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGBA image to RGB. All channels are 8-bit wide.
///
/// Corresponding colors are set while Alpha channel is discarded.
/// Destination data is filled as follows:
/// `| R,G,B | R,G,B | R,G,B | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 3 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_rgb_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an NV12 or NV21 (Semi-Planar) YUV image to RGB. All channels are
/// 8-bit wide.
///
/// Destination data is filled liked this:
/// `| R,G,B | R,G,B | R,G,B | ...`
/// Where each letter represents one byte of data, and one pixel is represented
/// by 3 bytes. There is no padding between the pixels.
/// If 4-byte alignment is required then @ref kleidicv_yuv_sp_to_rgba_u8 can be
/// used.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_y         Pointer to the input's Y component. Must be non-null.
/// @param src_y_stride  Distance in bytes from the start of one row to the
///                      start of the next row for the input's Y component.
///                      Must not be less than `width * sizeof(u8)`, except for
///                      single-row images.
/// @param src_uv        Pointer to the input's interleaved UV components.
///                      Must be non-null. If the width parameter is odd, the
///                      width of this input stream still needs to be even.
/// @param src_uv_stride Distance in bytes from the start of one row to the
///                      start of the next row for the input's UV components.
///                      Must not be less than
///                      `__builtin_align_up(width, 2) * sizeof(u8)`, except for
///                      single-row images.
/// @param dst           Pointer to the destination data. Must be non-null.
/// @param dst_stride    Distance in bytes from the start of one row to the
///                      start of the next row for the destination data. Must
///                      not be less than `width * 3 * sizeof(type)`, except for
///                      single-row images.
/// @param width         Number of pixels in a row.
/// @param height        Number of rows in the data.
/// @param is_nv21       If true, input is treated as NV21, otherwise treated
///                      as NV12.
///
KLEIDICV_API_DECLARATION(kleidicv_yuv_sp_to_rgb_u8, const uint8_t *src_y,
                         size_t src_y_stride, const uint8_t *src_uv,
                         size_t src_uv_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, bool is_nv21);

/// Converts an NV12 or NV21 (Semi-Planar) YUV image to BGR. All channels are
/// 8-bit wide.
///
/// Destination data is filled liked this:
/// `| B,G,R | B,G,R | B,G,R | ...`
/// Where each letter represents one byte of data, and one pixel is represented
/// by 3 bytes. There is no padding between the pixels.
/// If 4-byte alignment is required then @ref kleidicv_yuv_sp_to_bgra_u8 can be
/// used.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_y         Pointer to the input's Y component. Must be non-null.
/// @param src_y_stride  Distance in bytes from the start of one row to the
///                      start of the next row for the input's Y component.
///                      Must not be less than `width * sizeof(u8)`, except for
///                      single-row images.
/// @param src_uv        Pointer to the input's interleaved UV components.
///                      Must be non-null. If the width parameter is odd, the
///                      width of this input stream still needs to be even.
/// @param src_uv_stride Distance in bytes from the start of one row to the
///                      start of the next row for the input's UV components.
///                      Must not be less than
///                      `__builtin_align_up(width, 2) * sizeof(u8)`, except for
///                      single-row images.
/// @param dst           Pointer to the destination data. Must be non-null.
/// @param dst_stride    Distance in bytes from the start of one row to the
///                      start of the next row for the destination data. Must
///                      not be less than `width * 3 * sizeof(type)`, except for
///                      single-row images.
/// @param width         Number of pixels in a row.
/// @param height        Number of rows in the data.
/// @param is_nv21       If true, input is treated as NV21, otherwise treated
///                      as NV12.
///
KLEIDICV_API_DECLARATION(kleidicv_yuv_sp_to_bgr_u8, const uint8_t *src_y,
                         size_t src_y_stride, const uint8_t *src_uv,
                         size_t src_uv_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, bool is_nv21);

/// Converts an NV12 or NV21 (Semi-Planar) YUV image to RGBA. All channels are
/// 8-bit wide. Alpha channel is set to 0xFF.
///
/// Destination data is filled liked this:
/// `| R,G,B,A | R,G,B,A | R,G,B,A | ...`
/// Where each letter represents one byte of data, and one pixel is represented
/// by 4 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_y         Pointer to the input's Y component. Must be non-null.
/// @param src_y_stride  Distance in bytes from the start of one row to the
///                      start of the next row for the input's Y component.
///                      Must not be less than `width * sizeof(u8)`, except for
///                      single-row images.
/// @param src_uv        Pointer to the input's interleaved UV components.
///                      Must be non-null. If the width parameter is odd, the
///                      width of this input stream still needs to be even.
/// @param src_uv_stride Distance in bytes from the start of one row to the
///                      start of the next row for the input's UV components.
///                      Must not be less than
///                      `__builtin_align_up(width, 2) * sizeof(u8)`, except for
///                      single-row images.
/// @param dst           Pointer to the destination data. Must be non-null.
/// @param dst_stride    Distance in bytes from the start of one row to the
///                      start of the next row for the destination data. Must
///                      not be less than `width * 4 * sizeof(type)`, except for
///                      single-row images.
/// @param width         Number of pixels in a row.
/// @param height        Number of rows in the data.
/// @param is_nv21       If true, input is treated as NV21, otherwise treated
///                      as NV12.
///
KLEIDICV_API_DECLARATION(kleidicv_yuv_sp_to_rgba_u8, const uint8_t *src_y,
                         size_t src_y_stride, const uint8_t *src_uv,
                         size_t src_uv_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, bool is_nv21);

/// Converts an NV12 or NV21 (Semi-Planar) YUV image to BGRA. All channels are
/// 8-bit wide. Alpha channel is set to 0xFF.
///
/// Destination data is filled liked this:
/// `| B,G,R,A | B,G,R,A | B,G,R,A | ...`
/// Where each letter represents one byte of data, and one pixel is represented
/// by 4 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_y         Pointer to the input's Y component. Must be non-null.
/// @param src_y_stride  Distance in bytes from the start of one row to the
///                      start of the next row for the input's Y component.
///                      Must not be less than `width * sizeof(u8)`, except for
///                      single-row images.
/// @param src_uv        Pointer to the input's interleaved UV components.
///                      Must be non-null. If the width parameter is odd, the
///                      width of this input stream still needs to be even.
/// @param src_uv_stride Distance in bytes from the start of one row to the
///                      start of the next row for the input's UV components.
///                      Must not be less than
///                      `__builtin_align_up(width, 2) * sizeof(u8)`, except for
///                      single-row images.
/// @param dst           Pointer to the destination data. Must be non-null.
/// @param dst_stride    Distance in bytes from the start of one row to the
///                      start of the next row for the destination data. Must
///                      not be less than `width * 4 * sizeof(type)`, except for
///                      single-row images.
/// @param width         Number of pixels in a row.
/// @param height        Number of rows in the data.
/// @param is_nv21       If true, input is treated as NV21, otherwise treated
///                      as NV12.
///
KLEIDICV_API_DECLARATION(kleidicv_yuv_sp_to_bgra_u8, const uint8_t *src_y,
                         size_t src_y_stride, const uint8_t *src_uv,
                         size_t src_uv_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, bool is_nv21);

/// Converts a YUV image to RGB, pixel by pixel. All channels are 8-bit wide.
///
/// Source data has 3 channels like this:
/// `| Y,U,V | Y,U,V | Y,U,V | ...`
/// One pixel is represented by 3 bytes. There is no padding between the pixels.
///
/// Destination data has 3 channels:
/// - R,G,B
/// - B,G,R
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_yuv_to_bgr_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_yuv_to_bgr_u8
KLEIDICV_API_DECLARATION(kleidicv_yuv_to_rgb_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGB image to YUV, pixel by pixel. All channels are 8-bit wide.
///
/// Source data can have 3 channels or 4 if there is an alpha channel:
/// - R,G,B
/// - R,G,B,Alpha
/// - B,G,R
/// - B,G,R,Alpha
///
/// Destination data is filled like this:
/// `| Y,U,V | Y,U,V | Y,U,V | ...`
/// One pixel is represented by 3 bytes. There is no padding between the pixels.
/// Alpha channel is not used in the conversion.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `width * (number of channels) *
///                    sizeof(uint8)`, except for single-row images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_bgr_to_yuv_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_bgr_to_yuv_u8
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_yuv_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_bgr_to_yuv_u8
KLEIDICV_API_DECLARATION(kleidicv_bgra_to_yuv_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_bgr_to_yuv_u8
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_yuv_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Performs a comparison of each element's value in `src` with respect to a
/// caller defined threshold. The strictly larger elements are set to
/// `value` and the rest to 0.
///
/// Width and height are the same for the source and for the destination. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data. Must
///                     not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param dst          Pointer to the first destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data. Must
///                     not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param threshold    The value that the elements of the source data are
///                     compared to.
/// @param value        The value that the larger elements are set to.
///
KLEIDICV_API_DECLARATION(kleidicv_threshold_binary_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, uint8_t threshold,
                         uint8_t value);

/// Performs an 'equal to' comparison of each element's value in `src_a` with
/// respect to the corresponding element's value in `src_b`.
///
/// If the result of the comparison is true then the corresponding element in
/// `dst` is set to 255, otherwise to 0.
///
/// Width and height are the same for the source and for the destination. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the first destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data. Must
///                     not be less than `width * sizeof(type)`.
///                     Must be a multiple of `sizeof(type)`.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_compare_equal_u8, const uint8_t *src_a,
                         size_t src_a_stride, const uint8_t *src_b,
                         size_t src_b_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Performs a 'strictly greater than' comparison of each element's value in
/// `src_a` with respect to the corresponding element's value in `src_b`.
///
/// If the result of the comparison is true then the corresponding element in
/// `dst` is set to 255, otherwise to 0.
///
/// Width and height are the same for the source and for the destination. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the first destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_compare_greater_u8, const uint8_t *src_a,
                         size_t src_a_stride, const uint8_t *src_b,
                         size_t src_b_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Creates a morphology context according to the parameters.
///
/// Before a @ref kleidicv_dilate_u8 "dilate" or @ref kleidicv_erode_u8
/// "erode" operation, this initialization is needed. After the operation is
/// finished, the context needs to be released using @ref
/// kleidicv_morphology_release.
///
/// @param context       Pointer where to return the created context's address.
/// @param kernel        Width and height of the kernel. Its size must not be
///                      more than @ref KLEIDICV_MAX_IMAGE_PIXELS.
/// @param anchor        Location in the kernel which is aligned to the actual
///                      point in the source data. Must not point out of the
///                      kernel.
/// @param border_type   Way of handling the border. The supported border types
///                      are: \n
///                         - @ref KLEIDICV_BORDER_TYPE_CONSTANT \n
///                         - @ref KLEIDICV_BORDER_TYPE_REPLICATE
/// @param border_values Border values if the border_type is
///                      @ref KLEIDICV_BORDER_TYPE_CONSTANT.
/// @param channels      Number of channels in the data. Must be not more than
///                      @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
/// @param iterations    Number of times to do the morphology operation.
/// @param type_size     Element size in bytes. Must not be more than
///                      @ref KLEIDICV_MAXIMUM_TYPE_SIZE.
/// @param image         Image dimensions. Its size must not be more than
///                      @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
kleidicv_error_t kleidicv_morphology_create(
    kleidicv_morphology_context_t **context, kleidicv_rectangle_t kernel,
    kleidicv_point_t anchor, kleidicv_border_type_t border_type,
    kleidicv_border_values_t border_values, size_t channels, size_t iterations,
    size_t type_size, kleidicv_rectangle_t image);

/// Releases a morphology context that was previously created using @ref
/// kleidicv_morphology_create.
///
/// @param context      Pointer to morphology context. Must not be nullptr.
///
kleidicv_error_t kleidicv_morphology_release(
    kleidicv_morphology_context_t *context);

/// Calculates maximum (dilate) or minimum (erode) element value of `src`
/// values using a given kernel which has a rectangular shape, and puts the
/// result into `dst`.
///
/// Width and height are the same for the source and the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// The kernel has an anchor point, it is usually the center of the kernel.
/// The algorithm takes a rectangle from the source data using the kernel and
/// the anchor, and calculates the max/min value in that rectangle.
///
/// Example for dilate:
/// ```
///    [ 2, 8, 9, 3, 6 ]  kernel: (3, 3)        [ 8, 9, 9, 9, 6 ]
///    [ 7, 2, 4, 1, 5 ]  anchor: (1, 1)     -> [ 8, 9, 9, 9, 8 ]
///    [ 4, 3, 6, 8, 1 ]  border: replicate     [ 7, 7, 8, 8, 8 ]
///    [ 1, 2, 5, 3, 7 ]                        [ 4, 6, 8, 8, 8 ]
/// ```
/// Usage:
///
/// Before using this function, a context must be created using
/// @ref kleidicv_morphology_create, and after finished, it has to be
/// released using @ref kleidicv_morphology_release.
/// The context must be created with the same image dimensions as `width` and
/// `height` parameters, with `sizeof(uint8)` as `type_size`, and with the
/// channel number of the data as `channels`.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must not be less than `width * channels *
///                     sizeof(uint8)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data. Must
///                     not be less than `width * channels * sizeof(uint8)`,
///                     except for single-row images.
/// @param width        Number of pixels in a row.
/// @param height       Number of rows in the data.
/// @param context      Pointer to morphology context.
///
KLEIDICV_API_DECLARATION(kleidicv_dilate_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height,
                         kleidicv_morphology_context_t *context);

/// @copydoc kleidicv_dilate_u8
///
KLEIDICV_API_DECLARATION(kleidicv_erode_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height,
                         kleidicv_morphology_context_t *context);

/// Counts how many nonzero elements are in the source data. Number of elements
/// is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param count        Pointer to variable to store result. Must be non-null.
///
KLEIDICV_API_DECLARATION(kleidicv_count_nonzeros_u8, const uint8_t *src,
                         size_t src_stride, size_t width, size_t height,
                         size_t *count);

/// Resizes source data by averaging 4 elements to one.
/// In-place operation not supported.
///
/// For even source dimensions `(2*N, 2*M)` destination dimensions should be
/// `(N, M)`.
/// In case of odd source dimensions `(2*N+1, 2*M+1)` destination
/// dimensions could be either `(N+1, M+1)` or `(N, M)` or combination of both.
/// For later cases last respective row or column of source data will not be
/// processed. Currently only supports single-channel data. Number of pixels in
/// the source is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// Even dimension example of 2x2 to 1x1 conversion:
/// ```
/// | a | b | --> | (a+b+c+d)/4 |
/// | c | d |
/// ```
/// Odd dimension example of 3x3 to 2x2 conversion:
/// ```
/// | a | b | c |     | (a+b+c+d)/4 | (c+f)/2 |
/// | d | e | f | --> |   (g+h)/2   |    i    |
/// | g | h | i |
/// ```
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_width    Number of elements in the source row.
/// @param src_height   Number of rows in the source data.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst_width    Number of elements in the destination row.
///                     Must be `src_width / 2` for even src_width.
///                     For odd `src_width` it must be either `src_width / 2`
///                     or `(src_width / 2) + 1`.
/// @param dst_height   Number of rows in the destination data.
///                     Must be `src_height / 2` for even src_height.
///                     For odd `src_height` it must be either `src_height / 2`
///                     or `(src_height / 2) + 1`.
///
KLEIDICV_API_DECLARATION(kleidicv_resize_to_quarter_u8, const uint8_t *src,
                         size_t src_stride, size_t src_width, size_t src_height,
                         uint8_t *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height);

/// Resize image using linear interpolation.
/// In-place operation not supported.
///
/// At present only 2x2 and 4x4 upsizing is supported, and 8x8 for float data.
/// For other ratios KLEIDICV_ERROR_NOT_IMPLEMENTED
/// will be returned.
/// The total number of pixels in the destination is limited to
/// @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_width    Number of elements in the source row.
/// @param src_height   Number of rows in the source data.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst_width    Number of elements in the destination row.
///                     Must be inline with the choosen upsizing operation, for
///                     example `src_width * 2` in case of 2x2.
/// @param dst_height   Number of rows in the destination data.
///                     Must be inline with the choosen upsizing operation, for
///                     example `src_height * 2` in case of 2x2.
///
KLEIDICV_API_DECLARATION(kleidicv_resize_linear_u8, const uint8_t *src,
                         size_t src_stride, size_t src_width, size_t src_height,
                         uint8_t *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height);

/// @copydoc kleidicv_resize_linear_u8
KLEIDICV_API_DECLARATION(kleidicv_resize_linear_f32, const float *src,
                         size_t src_stride, size_t src_width, size_t src_height,
                         float *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height);

/// Calculates vertical derivative approximation with Sobel filter.
///
/// The used convolution kernel is:
/// ```
/// [  1  2  1 ]
/// [  0  0  0 ]
/// [ -1 -2 -1 ]
/// ```
/// Note, that the kernel is mirrored both vertically and horizontally during
/// the convolution.
///
/// The only supported border type is @ref KLEIDICV_BORDER_TYPE_REPLICATE
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(src type)` and no less than `width *
///                     sizeof(src type) * channels`, except for single-row
///                     images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the destination data. Must be a
///                     multiple of `sizeof(dst type)` and no less than `width *
///                     sizeof(dst type) * channels`, except for single-row
///                     images.
/// @param width        Number of columns in the data. (One column consists of
///                     `channels` number of elements.)
/// @param height       Number of rows in the data.
/// @param channels     Number of channels in the data. Must be not more than
///                     @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
///
kleidicv_error_t kleidicv_sobel_3x3_vertical_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels);

/// Calculates horizontal derivative approximation with Sobel filter.
///
/// The used convolution kernel is:
/// ```
/// [  1  0 -1 ]
/// [  2  0 -2 ]
/// [  1  0 -1 ]
/// ```
/// Note, that the kernel is mirrored both vertically and horizontally during
/// the convolution.
///
/// The only supported border type is @ref KLEIDICV_BORDER_TYPE_REPLICATE
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(src type)` and no less than `width *
///                     sizeof(src type) * channels`, except for single-row
///                     images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the destination data. Must be a
///                     multiple of `sizeof(dst type)` and no less than `width *
///                     sizeof(dst type) * channels`, except for single-row
///                     images.
/// @param width        Number of columns in the data. (One column consists of
///                     `channels` number of elements.)
/// @param height       Number of rows in the data.
/// @param channels     Number of channels in the data. Must be not more than
///                     @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
///
kleidicv_error_t kleidicv_sobel_3x3_horizontal_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels);

#if KLEIDICV_EXPERIMENTAL_FEATURE_CANNY
/// Canny edge detector for uint8_t grayscale input. Output is also a `uint8_t`
/// grayscale image. Width and height are the same for input and output. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// The steps:
///  - Execute horizontal and vertical Sobel filtering with 3*3 kernels to
///    calculate the gradient approximation in both directions.
///  - Calculate magnitude approximation (by summing horizontal and vertical
///    gradient approximations) and apply lower threshold.
///  - Perform non-maxima suppression and high thresholding.
///  - Perform hysteresis: promote weak edges, which are connected to strong
///    edges, to strong edges.
///  - Suppress remaining weak edges.
///
/// @param src            Pointer to the source data. Must be non-null.
/// @param src_stride     Distance in bytes from the start of one row to the
///                       start of the next row in the source data. Must not be
///                       less than `width`, except for single-row images.
/// @param dst            Pointer to the destination data. Must be non-null.
/// @param dst_stride     Distance in bytes from the start of one row to the
///                       start of the next row in the destination data. Must
///                       not be less than `width`, except for single-row
///                       images.
/// @param width          Number of elements in a row.
/// @param height         Number of rows in the data.
/// @param low_threshold  Low threshold for the edge detector algorithm.
/// @param high_threshold High threshold for the edge detector algorithm.
///
KLEIDICV_API_DECLARATION(kleidicv_canny_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, double low_threshold,
                         double high_threshold);
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_CANNY

/// Creates a filter context according to the parameters.
///
/// Before a Gaussian blur operation, this initialization is needed.
/// After the operation is finished, the context needs to be released
/// using @ref kleidicv_filter_context_release.
///
/// @param context           Pointer where to return the created context's
///                          address.
/// @param max_channels      Maximum number of channels in the data. Must not be
///                          more than @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
/// @param max_kernel_width  Maximum width of the Gaussian blur kernel.
/// @param max_kernel_height Maximum height of the Gaussian blur kernel.
/// @param max_image_width   Maximum image width. `max_image_width *
///                          max_image_height` must not be more than @ref
///                          KLEIDICV_MAX_IMAGE_PIXELS.
/// @param max_image_height  Maximum image height. m`ax_image_width *
///                          max_image_height` must not be more than @ref
///                          KLEIDICV_MAX_IMAGE_PIXELS.
///
kleidicv_error_t kleidicv_filter_context_create(
    kleidicv_filter_context_t **context, size_t max_channels,
    size_t max_kernel_width, size_t max_kernel_height, size_t max_image_width,
    size_t max_image_height);

/// Releases a filter context that was previously created using @ref
/// kleidicv_filter_context_create.
///
/// @param context      Pointer to filter context. Must not be nullptr.
///
kleidicv_error_t kleidicv_filter_context_release(
    kleidicv_filter_context_t *context);

/// Applies a two-dimensional separable filter to the source image using the
/// specified parameters. In-place filtering is not supported.
///
/// Width and height are assumed to be the same for the source and for the
/// destination. The number of elements is limited to @ref
/// KLEIDICV_MAX_IMAGE_PIXELS.
///
/// Usage:
///
/// Before using this function, a context must be created using
/// @ref kleidicv_filter_context_create, and when finished, it has to be
/// released using @ref kleidicv_filter_context_release. Please ensure that your
/// filter context parameters are large enough, otherwise this API will return
/// with an error.
///
/// Note, from the border types only KLEIDICV_BORDER_TYPE_REPLICATE is
/// supported.
///
/// @param src           Pointer to the source data. Must be non-null.
/// @param src_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the source data. Must be a
///                      multiple of `sizeof(type)` and no less than `width *
///                      sizeof(type) * channels`, except for single-row images.
/// @param dst           Pointer to the destination data. Must be non-null.
/// @param dst_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the destination data. Must be
///                      a multiple of `sizeof(type)` and no less than `width *
///                      sizeof(type) * channels`, except for single-row images.
/// @param width         Number of columns in the data. (One column consists of
///                      `channels` number of elements.)
/// @param height        Number of rows in the data.
/// @param channels      Number of channels in the data. Must be not more than
///                      @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
/// @param kernel_x      Pointer to the horizontal 2D kernel values.
/// @param kernel_width  Size of the horizontal 2D kernel.
/// @param kernel_y      Pointer to the vertical 2D kernel values.
/// @param kernel_height Size of the vertical 2D kernel.
/// @param border_type   Way of handling the border.
/// @param context       Pointer to filter context.
///
kleidicv_error_t kleidicv_separable_filter_2d_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint8_t *kernel_x,
    size_t kernel_width, const uint8_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context);
/// @copydoc kleidicv_separable_filter_2d_u8
kleidicv_error_t kleidicv_separable_filter_2d_u16(
    const uint16_t *src, size_t src_stride, uint16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint16_t *kernel_x,
    size_t kernel_width, const uint16_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context);
/// @copydoc kleidicv_separable_filter_2d_u8
kleidicv_error_t kleidicv_separable_filter_2d_s16(
    const int16_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const int16_t *kernel_x,
    size_t kernel_width, const int16_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context);

/// Applies Gaussian blur to the source image using the specified parameters.
/// In-place filtering is not supported.
///
/// Width and height are assumed to be the same for the source and for the
/// destination. The number of elements is limited to @ref
/// KLEIDICV_MAX_IMAGE_PIXELS.
///
/// Usage:
///
/// Before using this function, a context must be created using
/// @ref kleidicv_filter_context_create, and when finished, it has to be
/// released using @ref kleidicv_filter_context_release. Please ensure that your
/// filter context parameters are large enough, otherwise this API will return
/// with an error.
///
/// Note, from the border types only these are supported:
///                       - @ref KLEIDICV_BORDER_TYPE_REPLICATE
///                       - @ref KLEIDICV_BORDER_TYPE_REFLECT
///                       - @ref KLEIDICV_BORDER_TYPE_WRAP
///                       - @ref KLEIDICV_BORDER_TYPE_REVERSE
///
/// @param src           Pointer to the source data. Must be non-null.
/// @param src_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the source data. Must be a
///                      multiple of `sizeof(type)` and no less than `width *
///                      sizeof(type) * channels`, except for single-row images.
/// @param dst           Pointer to the destination data. Must be non-null.
/// @param dst_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the destination data. Must be
///                      a multiple of `sizeof(type)` and no less than `width *
///                      sizeof(type) * channels`, except for single-row images.
/// @param width         Number of columns in the data. (One column consists of
///                      `channels` number of elements.)
/// @param height        Number of rows in the data.
/// @param channels      Number of channels in the data. Must be not more than
///                      @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
/// @param kernel_width  Width of the Gaussian kernel.
/// @param kernel_height Height of the Gaussian kernel.
/// @param sigma_x       Horizontal sigma (standard deviation) value. If equal
///                      to 0.0, Gaussian filter is approximated by the
///                      probability mass function of the binomial distribution
///                      in the horizontal direction.
/// @param sigma_y       Vertical sigma (standard deviation) value. If equal
///                      to 0.0, Gaussian filter is approximated by the
///                      probability mass function of the binomial distribution
///                      in the vertical direction.
/// @param border_type   Way of handling the border.
/// @param context       Pointer to filter context.
///
kleidicv_error_t kleidicv_gaussian_blur_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, float sigma_x, float sigma_y,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context);

#ifndef DOXYGEN
/// Internal - not part of the public API and its direct use is not supported.
///
/// Applies 5x5 binomial Gaussian blur to the source image and downsaples the
/// result by keeping odd rows and columns only.
/// This function can be used to generate an Image Pyramid.
/// In-place operation is not supported.
///
/// The number of elements in the source is limited to @ref
/// KLEIDICV_MAX_IMAGE_PIXELS.
///
/// Width and height of the destination is calculated as:
///   - `dst_width = (src_width + 1) / 2`
///   - `dst_height = (src_height + 1) / 2`
///
/// Usage:
///
/// Before using this function, a context must be created using @ref
/// kleidicv_filter_context_create, and when finished, it has to be released
/// using @ref kleidicv_filter_context_release. Please ensure that your filter
/// context parameters are large enough (max_kernel_width and max_kernel_height
/// must be at least 5), otherwise this API will return with an error.
///
/// Note, from the border types only these are supported:
///                       - @ref KLEIDICV_BORDER_TYPE_REPLICATE
///                       - @ref KLEIDICV_BORDER_TYPE_REFLECT
///                       - @ref KLEIDICV_BORDER_TYPE_WRAP
///                       - @ref KLEIDICV_BORDER_TYPE_REVERSE
///
/// @param src           Pointer to the source data. Must be non-null.
/// @param src_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the source data. Must be a
///                      multiple of `sizeof(type)` and no less than `src_width
///                      * sizeof(type) * channels`, except for single-row
///                      images.
/// @param src_width     Number of columns in the source data. (One column
///                      consists of `channels` number of elements.)
/// @param src_height    Number of rows in the source data.
/// @param dst           Pointer to the destination data. Must be non-null.
/// @param dst_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the destination data. Must be
///                      a multiple of `sizeof(type)` and no less than
///                      `dst_width * sizeof(type) * channels`, except for
///                      single-row images.
/// @param channels      Number of channels in the data. Must be equal to 1.
/// @param border_type   Way of handling the border.
/// @param context       Pointer to filter context.
///
kleidicv_error_t kleidicv_blur_and_downsample_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t channels,
    kleidicv_border_type_t border_type, kleidicv_filter_context_t *context);
#endif

/// Splits a multi channel source stream into separate 1-channel streams. Width
/// and height are the same for the source stream and for all the destination
/// streams. Number of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_data     Pointer to the source data. Must be non-null.
///                     Must be aligned to `element_size`.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `element_size` and no less than `width *
///                     element_size * channels`, except for single-row images.
/// @param dst_data     A C style array of pointers to the destination data.
///                     Number of pointers in the array must be the same as the
///                     channel number. All pointers must be non-null.
///                     All pointers must be aligned to `element_size`.
/// @param dst_strides  A C style array of stride values for the destination
///                     streams. A stride value represents the distance in
///                     bytes from the start of one row to the start of the
///                     next row in the given destination stream. Number of
///                     stride values in the array must be the same as the
///                     channel number. All stride values must be a multiple of
///                     `element_siz`e and no less than `width * element_size`,
///                     except for single-row images.
/// @param width        Number of pixels in one row of the source data. (One
///                     pixel consists of `channels` number of elements.)
/// @param height       Number of rows in the source data.
/// @param channels     Number of channels in the source data. Must be 2, 3 or
///                     4.
/// @param element_size Size of one element in bytes. Must be 1, 2, 4 or 8.
///
KLEIDICV_API_DECLARATION(kleidicv_split, const void *src_data,
                         size_t src_stride, void **dst_data,
                         const size_t *dst_strides, size_t width, size_t height,
                         size_t channels, size_t element_size);

/// Matrix transpose operation.
/// In-place transpose (`src == dst`) is only supported for
/// square matrixes (`src_width == src_height`).
///
/// Example for `src[4,3]` to `dst[3,4]`:
/// ```
/// | 0 | 2 | 2 | 2 |    | 0 | 1 | 1 |
/// | 1 | 0 | 2 | 2 | -> | 2 | 0 | 1 |
/// | 1 | 1 | 0 | 0 |    | 2 | 2 | 0 |
///                      | 2 | 2 | 2 |
/// ```
///
/// Number of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
///                     Must be aligned to `element_size`.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must be a multiple of `element_size` and no less than
///                     `width * element_size`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
///                     Can be the same as source data for in-place operation.
///                     Must be aligned to `element_size`.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `element_size` and no less than
///                     `height * element_size`, except for single-column
///                     images.
/// @param src_width    Number of elements in a row.
/// @param src_height   Number of rows in the data.
/// @param element_size Size of one element in bytes. Must be 1, 2, 4 or 8.
///
KLEIDICV_API_DECLARATION(kleidicv_transpose, const void *src, size_t src_stride,
                         void *dst, size_t dst_stride, size_t src_width,
                         size_t src_height, size_t element_size);

/// Merges separate 1-channel source streams to one multi channel stream. Width
/// and height are the same for all the source streams and for the destination.
/// Number of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param srcs         A C style array of pointers to the source data.
///                     Number of pointers in the array must be the same as the
///                     channel number. All pointers must be non-null.
///                     All pointers must be aligned to `element_size`.
/// @param src_strides  A C style array of stride values for the source
///                     streams. A stride value represents the distance in
///                     bytes from the start of one row to the start of the
///                     next row in the given source stream. Number of
///                     stride values in the array must be the same as the
///                     channel number. All stride values must be a multiple of
///                     `element_size` and no less than `width * element_size`,
///                     except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
///                     Must be aligned to element_size.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the destination data. Must be a
///                     multiple of `element_size` and no less than `width *
///                     element_size * channels`, except for single-row images.
/// @param width        Number of elements in a row for the source streams,
///                     number of pixels in a row for the destination data.
/// @param height       Number of rows in the data.
/// @param channels     Number of channels in the destination data. Must be 2,
///                     3 or 4.
/// @param element_size Size of one element in bytes. Must be 1, 2, 4 or 8.
///
KLEIDICV_API_DECLARATION(kleidicv_merge, const void **srcs,
                         const size_t *src_strides, void *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t channels, size_t element_size);

/// Calculates minimum and maximum element value across the source data. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(type)` and no less than `width *
///                     sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row. Must be greater than 0.
/// @param height       Number of rows in the data. Must be greater than 0.
/// @param min_value    Pointer to save result minimum value to, or `nullptr` if
///                     minimum is not to be calculated.
/// @param max_value    Pointer to save result maximum value to, or `nullptr` if
///                     maximum is not to be calculated.
///
KLEIDICV_API_DECLARATION(kleidicv_min_max_u8, const uint8_t *src,
                         size_t src_stride, size_t width, size_t height,
                         uint8_t *min_value, uint8_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_s8, const int8_t *src,
                         size_t src_stride, size_t width, size_t height,
                         int8_t *min_value, int8_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_u16, const uint16_t *src,
                         size_t src_stride, size_t width, size_t height,
                         uint16_t *min_value, uint16_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_s16, const int16_t *src,
                         size_t src_stride, size_t width, size_t height,
                         int16_t *min_value, int16_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_s32, const int32_t *src,
                         size_t src_stride, size_t width, size_t height,
                         int32_t *min_value, int32_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_f32, const float *src,
                         size_t src_stride, size_t width, size_t height,
                         float *min_value, float *max_value);

/// Finds minimum and maximum element value across the source data,
/// and returns their location in the source data as offset in bytes
/// from the source beginning. Number of elements is limited to
/// @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(type)` and no less than `width *
///                     sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row. Must be greater than 0.
/// @param height       Number of rows in the data. Must be greater than 0.
/// @param min_offset   Pointer to save result offset of minimum value to, or
///                     `nullptr` if minimum is not to be calculated.
/// @param max_offset   Pointer to save result offset of maximum value to, or
///                     `nullptr` if maximum is not to be calculated.
///
KLEIDICV_API_DECLARATION(kleidicv_min_max_loc_u8, const uint8_t *src,
                         size_t src_stride, size_t width, size_t height,
                         size_t *min_offset, size_t *max_offset);

/// Returns the sum of element values across the source data.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(type)` and no less than `width *
///                     sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param sum          Pointer to save the sum value to. Must be non-null.
///
KLEIDICV_API_DECLARATION(kleidicv_sum_f32, const float *src, size_t src_stride,
                         size_t width, size_t height, float *sum);

/// Multiplies the elements in `src` by `scale`, then adds `shift` to the
/// result and stores it in `dst`.
///
/// The result is saturated, i.e. it is the smallest/largest number of the
/// type of the element if the result would underflow/overflow. Source data
/// length (in bytes) is `stride * height`. Width and height are the same
/// for the source and destination. Number of elements is limited to
/// @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param scale        Value to multiply the input by.
/// @param shift        Value to add to the result.
///
KLEIDICV_API_DECLARATION(kleidicv_scale_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, float scale, float shift);
/// @copydoc kleidicv_scale_u8
KLEIDICV_API_DECLARATION(kleidicv_scale_f32, const float *src,
                         size_t src_stride, float *dst, size_t dst_stride,
                         size_t width, size_t height, float scale, float shift);

/// Exponential function, input is the elements in `src`, output is the elements
/// in `dst`.
///
/// In case of `float` type the maximum error is 0.36565+0.5 ULP, or the error
/// of the toolchain's expf implementation, if it is bigger.
///
/// Source and destination data length is `width * height`. Number of elements
/// is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data. Must be a
///                     multiple of `sizeof(type)` and no less than `width *
///                     sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data. Must be
///                     a multiple of `sizeof(type)` and no less than `width *
///                     sizeof(type)`, except for single-row images.
/// @param width        Number of pixels in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_exp_f32, const float *src, size_t src_stride,
                         float *dst, size_t dst_stride, size_t width,
                         size_t height);

/// Converts the elements in `src` from a floating-point type to an integer
/// type, then stores the result in `dst`.
///
/// Each resulting element is saturated, i.e. it is the smallest/largest
/// number of the type of the element if the `src` data type cannot be
/// represented as the `dst` type. In case of some special values, such as the
/// different variations of `NaN`, the result is `0`. Source and destination
/// data length is `width * height`. Number of elements is limited to @ref
/// KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must not be less than `width * sizeof(type)`.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must not be less than `width * sizeof(type)`.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_f32_to_s8, const float *src,
                         size_t src_stride, int8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_f32_to_s8
KLEIDICV_API_DECLARATION(kleidicv_f32_to_u8, const float *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts the elements in `src` from an integer type to a floating-point
/// type, then stores the result in `dst`.
///
/// Each resulting element is saturated, i.e. it is the smallest/largest
/// number of the type of the element if the `src` data type cannot be
/// represented as the `dst` type. Source and destination data length is
/// `width * height`. Number of elements is limited to @ref
/// KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data. Must
///                     not be less than `width * sizeof(type)`.
///                     Must be a multiple of `sizeof(type)`.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data. Must
///                     not be less than `width * sizeof(type)`.
///                     Must be a multiple of `sizeof(type)`.
/// @param width        Number of pixels in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_s8_to_f32, const int8_t *src,
                         size_t src_stride, float *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_s8_to_f32
KLEIDICV_API_DECLARATION(kleidicv_u8_to_f32, const uint8_t *src,
                         size_t src_stride, float *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Performs a per element comparison in `src` with respect to caller defined
/// lower and upper bounds. For the elements exceeding these bounds, the
/// corresponding elements in `dst` are set to 0 and elements within to 255.
///
/// Width and height are the same for the source and for the destination. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data. Must
///                     not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data. Must
///                     not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param lower_bound  The lower bound of the interval.
/// @param upper_bound  The upper bound of the interval.
///
KLEIDICV_API_DECLARATION(kleidicv_in_range_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, uint8_t lower_bound,
                         uint8_t upper_bound);
/// @copydoc kleidicv_in_range_u8
KLEIDICV_API_DECLARATION(kleidicv_in_range_f32, const float *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, float lower_bound,
                         float upper_bound);

#if KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
/// Transforms the `src` image by taking the pixels specified by the coordinates
/// from the `mapxy` image.
///
/// Width and height are the same for `mapxy` and for `dst`. `src` dimensions
/// may be different, but due to the 16-bit signed format, its width and height
/// must not be bigger than 32767. Coordinates outside of `src` dimensions are
/// considered border. In case of @ref KLEIDICV_BORDER_TYPE_REPLICATE, that
/// means that negative coordinates map to the first row/column (zero), and
/// those bigger than height/width - 1 map to the last row/column.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data. Must
///                     not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param src_width    Number of elements in the source row.
/// @param src_height   Number of rows in the source data.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst_width    Number of elements in the destination row.
/// @param dst_height   Number of rows in the destination data.
/// @param mapxy        Pointer to the mapping data. Must be non-null.
/// @param mapxy_stride Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(int16_t)` and no less than
///                     `width * sizeof(int16_t)`, except for single-row images.
/// @param channels      Number of channels in the data. Must be 1.
/// @param border_type   Way of handling the border. The supported border types
///                      are: \n
///                         - @ref KLEIDICV_BORDER_TYPE_REPLICATE
/// @param border_values Border values if the border_type is
///                      @ref KLEIDICV_BORDER_TYPE_CONSTANT.
KLEIDICV_API_DECLARATION(kleidicv_remap_s16_u8, const uint8_t *src,
                         size_t src_stride, size_t src_width, size_t src_height,
                         uint8_t *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height, size_t channels,
                         const int16_t *mapxy, size_t mapxy_stride,
                         kleidicv_border_type_t border_type,
                         kleidicv_border_values_t border_values);

#ifndef DOXYGEN
/// Internal - not part of the public API and its direct use is not supported.
/// Functionality is similar to @ref kleidicv_remap_s16_u8 , the difference is
/// in the data format: it contains a fractional part with 5+5 bits (`mapfrac`).
KLEIDICV_API_DECLARATION(kleidicv_remap_s16point5_u8, const uint8_t *src,
                         size_t src_stride, size_t src_width, size_t src_height,
                         uint8_t *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height, size_t channels,
                         const int16_t *mapxy, size_t mapxy_stride,
                         const uint16_t *mapfrac, size_t mapfrac_stride,
                         kleidicv_border_type_t border_type,
                         kleidicv_border_values_t border_values);
#endif  // DOXYGEN
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_REMAP

#ifndef DOXYGEN
/// Internal - not part of the public API and its direct use is not supported.
///
/// Calculates horizontal and vertical derivative approximation with Scharr
/// filter and store the results interleaved.
///
/// The horizontal convolution kernel is:
/// ```
/// [  3   0  -3 ]
/// [ 10   0 -10 ]
/// [  3   0  -3 ]
/// ```
///
/// The vertical convolution kernel is:
/// ```
/// [  3  10   3 ]
/// [  0   0   0 ]
/// [ -3 -10  -3 ]
/// ```
///
/// Note, that the kernels are mirrored both vertically and horizontally during
/// the convolution.
///
/// This API does not handle borders, so the result's width and height is `width
/// - 2` and `height - 2`, respectively. Number of pixels in the source is
/// limited to @ref KLEIDICV_MAX_IMAGE_PIXELS. Result's channel count is the
/// double of the source' channel count, as the calculated derivative
/// approximations are stored interleaved:
/// ```
/// | dx,dy | dx,dy | dx,dy | ...
/// ```
/// Where `dx` is the horizontal derivative approximation and `dy` is the
/// vertical derivative approximation.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(src type)` and no less than `width *
///                     sizeof(src type) * channels`.
/// @param src_width    Number of columns in the source. Must be more than 2.
///                     (One column consists of `channels` number of elements.)
/// @param src_height   Number of rows in the source. Must be more than 2.
/// @param src_channels Number of channels in the source data. Must be equal
///                     to 1.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the destination data. Must be a
///                     multiple of `sizeof(dst type)` and no less than `(width
///                     - 2) * sizeof(dst type) * channels`.
///
kleidicv_error_t kleidicv_scharr_interleaved_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride);
#endif  // DOXYGEN

#if KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
/// Performs a perspective transformation on an image.
/// For each pixel in `dst` take a pixel from `src` specified by
/// the transformed x and y coordinates, and optionally doing a bilinear
/// interpolation.
///
/// `src` and `dst` dimensions may be different. Number of elements is limited
/// to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src            Pointer to the source data. Must be non-null.
/// @param src_stride     Distance in bytes from the start of one row to the
///                       start of the next row for the source data. Must
///                       not be less than `width * sizeof(type)`, except for
///                       single-row images. Must be less than 2^32.
/// @param src_width      Number of elements in the source row. Must be less
///                       than 2^24.
/// @param src_height     Number of rows in the source data. Must be less than
///                       2^24.
/// @param dst            Pointer to the destination data. Must be non-null.
/// @param dst_stride     Distance in bytes from the start of one row to the
///                       start of the next row for the destination data.
///                       Must be a multiple of `sizeof(type)` and no less than
///                       `width * sizeof(type)`, except for single-row images.
/// @param dst_width      Number of elements in the destination row. Must be at
///                       least 8.
/// @param dst_height     Number of rows in the destination data.
/// @param transformation Pointer to the transformation matrix of 9 values.
/// @param channels       Number of channels in the data. Must be 1.
/// @param interpolation  Interpolation algorithm. Supported types: \n
///                         - @ref KLEIDICV_INTERPOLATION_NEAREST
/// @param border_type    Way of handling the border. The supported border types
///                       are: \n
///                         - @ref KLEIDICV_BORDER_TYPE_REPLICATE
/// @param border_values  Border values if the border_type is
///                       @ref KLEIDICV_BORDER_TYPE_CONSTANT.
kleidicv_error_t kleidicv_warp_perspective_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    const float transformation[9], size_t channels,
    kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type, kleidicv_border_values_t border_values);
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // KLEIDICV_H
