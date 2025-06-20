// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <thread>

#include "framework/array.h"
#include "framework/generator.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv_thread/kleidicv_thread.h"
#include "multithreading_fake.h"

// Tuple of width, height, thread count.
typedef std::tuple<unsigned, unsigned, unsigned> P;

class Thread : public testing::TestWithParam<P> {
 public:
  template <typename SrcT, typename DstT, typename SingleThreadedFunc,
            typename MultithreadedFunc, typename... Args>
  void check_unary_op(SingleThreadedFunc single_threaded_func,
                      MultithreadedFunc multithreaded_func, size_t src_channels,
                      size_t dst_channels, Args... args) {
    unsigned width = 0, height = 0, thread_count = 0;
    std::tie(width, height, thread_count) = GetParam();
    test::Array2D<SrcT> src(size_t{width} * src_channels, height);
    test::Array2D<DstT> dst_single(size_t{width} * dst_channels, height),
        dst_multi(size_t{width} * dst_channels, height);

    test::PseudoRandomNumberGenerator<SrcT> generator;
    src.fill(generator);

    kleidicv_error_t single_result =
        single_threaded_func(src.data(), src.stride(), dst_single.data(),
                             dst_single.stride(), width, height, args...);

    kleidicv_error_t multi_result = multithreaded_func(
        src.data(), src.stride(), dst_multi.data(), dst_multi.stride(), width,
        height, args..., get_multithreading_fake(thread_count));

    EXPECT_EQ(single_result, multi_result);
    if (KLEIDICV_OK == single_result) {
      EXPECT_EQ_ARRAY2D(dst_multi, dst_single);
    }
  }

  template <typename SrcT, typename DstT, typename SingleThreadedFunc,
            typename MultithreadedFunc, typename... Args>
  void check_binary_op(SingleThreadedFunc single_threaded_func,
                       MultithreadedFunc multithreaded_func,
                       size_t src_channels, size_t dst_channels, Args... args) {
    unsigned width = 0, height = 0, thread_count = 0;
    std::tie(width, height, thread_count) = GetParam();
    test::Array2D<SrcT> src_a(size_t{width} * src_channels, height),
        src_b(size_t{width} * src_channels, height);
    test::Array2D<DstT> dst_single(size_t{width} * dst_channels, height),
        dst_multi(size_t{width} * dst_channels, height);

    test::PseudoRandomNumberGenerator<SrcT> generator;
    src_a.fill(generator);
    src_b.fill(generator);

    kleidicv_error_t single_result = single_threaded_func(
        src_a.data(), src_a.stride(), src_b.data(), src_b.stride(),
        dst_single.data(), dst_single.stride(), width, height, args...);

    kleidicv_error_t multi_result = multithreaded_func(
        src_a.data(), src_a.stride(), src_b.data(), src_b.stride(),
        dst_multi.data(), dst_multi.stride(), width, height, args...,
        get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_OK, single_result);
    EXPECT_EQ(KLEIDICV_OK, multi_result);
    EXPECT_EQ_ARRAY2D(dst_multi, dst_single);
  }

  template <typename T, typename SingleThreadedFunc, typename MultithreadedFunc>
  void check_separable_filter_2d(SingleThreadedFunc single_threaded_func,
                                 MultithreadedFunc multithreaded_func) {
    unsigned width = 0, height = 0, thread_count = 0;
    std::tie(width, height, thread_count) = GetParam();
    (void)thread_count;
    size_t channels = 1;
    const size_t kernel_width = 5;
    const size_t kernel_height = kernel_width;

    test::Array2D<T> kernel_x{kernel_width, 1};
    kernel_x.set(0, 0, {1, 2, 3, 4, 5});
    test::Array2D<T> kernel_y{kernel_height, 1};
    kernel_y.set(0, 0, {5, 6, 7, 8, 9});

    kleidicv_border_type_t border_type = KLEIDICV_BORDER_TYPE_REPLICATE;
    kleidicv_filter_context_t *context = nullptr;
    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_filter_context_create(&context, channels, kernel_width,
                                             kernel_height, width, height));
    check_unary_op<T, T>(
        single_threaded_func, multithreaded_func, channels /*src_channels*/,
        channels /*dst_channels*/,
        /*remaining arguments passed to separable_filter_2d_... functions*/
        channels, kernel_x.data(), kernel_width, kernel_y.data(), kernel_height,
        border_type, context);
    ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
  }

  template <typename T, typename SingleThreadedFunc, typename MultithreadedFunc,
            typename... Args>
  void check_remap_s16(SingleThreadedFunc single_threaded_func,
                       MultithreadedFunc multithreaded_func, size_t channels,
                       Args... args) {
    unsigned test_width = 0, height = 0, thread_count = 0;
    std::tie(test_width, height, thread_count) = GetParam();
    const unsigned src_width = 300, src_height = 300;
    // width < 8 are not supported, that's not tested here
    size_t width = test_width + 8;
    test::Array2D<T> src(size_t{src_width} * channels, src_height);
    test::Array2D<int16_t> mapxy(width * 2, height);
    test::Array2D<T> dst_single(width * channels, height),
        dst_multi(width * channels, height);

    test::PseudoRandomNumberGenerator<T> src_generator;
    src.fill(src_generator);
    test::PseudoRandomNumberGeneratorIntRange<int16_t> coord_generator{
        static_cast<int16_t>(-src_width / 4),
        static_cast<int16_t>(src_width * 4 / 3)};
    mapxy.fill(coord_generator);

    kleidicv_error_t single_result = single_threaded_func(
        src.data(), src.stride(), src_width, src_height, dst_single.data(),
        dst_single.stride(), width, height, channels, mapxy.data(),
        mapxy.stride(), args...);

    kleidicv_error_t multi_result = multithreaded_func(
        src.data(), src.stride(), src_width, src_height, dst_multi.data(),
        dst_multi.stride(), width, height, channels, mapxy.data(),
        mapxy.stride(), args..., get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_OK, single_result);
    EXPECT_EQ(KLEIDICV_OK, multi_result);
    EXPECT_EQ_ARRAY2D(dst_multi, dst_single);
  }

  template <typename T, typename MultithreadedFunc, typename... Args>
  void check_remap_s16_not_implemented(MultithreadedFunc multithreaded_func,
                                       size_t channels, Args... args) {
    unsigned test_width = 0, height = 0, thread_count = 0;
    std::tie(test_width, height, thread_count) = GetParam();
    const unsigned src_width = 300, src_height = 300;
    // width < 8 are not supported!
    size_t width = test_width + 8;
    test::Array2D<T> src(size_t{src_width} * channels, src_height);
    test::Array2D<int16_t> mapxy(width * 2, height);
    test::Array2D<T> dst_small(test_width * channels, height),
        dst(width * channels, height);

    kleidicv_error_t result = multithreaded_func(
        src.data(), src.stride(), src_width, src_height, dst.data(),
        dst.stride(), width, height, channels, mapxy.data(), mapxy.stride(),
        args..., get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED, result);

    result = multithreaded_func(
        src.data(), src.stride(), src_width, src_height, dst_small.data(),
        dst_small.stride(), test_width, height, channels, mapxy.data(),
        mapxy.stride(), args..., get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED, result);
  }

  template <typename T, typename SingleThreadedFunc, typename MultithreadedFunc,
            typename... Args>
  void check_remap_s16point5(SingleThreadedFunc single_threaded_func,
                             MultithreadedFunc multithreaded_func,
                             size_t channels, Args... args) {
    unsigned test_width = 0, height = 0, thread_count = 0;
    std::tie(test_width, height, thread_count) = GetParam();
    const unsigned src_width = 300, src_height = 300;
    // width < 8 are not supported, that's not tested here
    size_t width = test_width + 8;
    test::Array2D<T> src(size_t{src_width} * channels, src_height);
    test::Array2D<int16_t> mapxy(width * 2, height);
    test::Array2D<uint16_t> mapfrac(width, height);
    test::Array2D<T> dst_single(width * channels, height),
        dst_multi(width * channels, height);

    test::PseudoRandomNumberGenerator<T> src_generator;
    src.fill(src_generator);
    test::PseudoRandomNumberGeneratorIntRange<int16_t> coord_generator{
        0, std::min(static_cast<int16_t>(src_height - 1),
                    static_cast<int16_t>(src_width - 1))};
    mapxy.fill(coord_generator);
    test::PseudoRandomNumberGeneratorIntRange<uint16_t> coordfrac_generator{
        0, (1 << 10) - 1};
    mapfrac.fill(coordfrac_generator);

    kleidicv_error_t single_result = single_threaded_func(
        src.data(), src.stride(), src_width, src_height, dst_single.data(),
        dst_single.stride(), width, height, channels, mapxy.data(),
        mapxy.stride(), mapfrac.data(), mapfrac.stride(), args...);

    kleidicv_error_t multi_result = multithreaded_func(
        src.data(), src.stride(), src_width, src_height, dst_multi.data(),
        dst_multi.stride(), width, height, channels, mapxy.data(),
        mapxy.stride(), mapfrac.data(), mapfrac.stride(), args...,
        get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_OK, single_result);
    EXPECT_EQ(KLEIDICV_OK, multi_result);
    EXPECT_EQ_ARRAY2D(dst_multi, dst_single);
  }

  template <typename T, typename MultithreadedFunc, typename... Args>
  void check_remap_s16point5_not_implemented(
      MultithreadedFunc multithreaded_func, size_t channels, Args... args) {
    unsigned test_width = 0, height = 0, thread_count = 0;
    std::tie(test_width, height, thread_count) = GetParam();
    const unsigned src_width = 300, src_height = 300;
    // width < 8 are not supported!
    size_t width = test_width + 8;
    test::Array2D<T> src(size_t{src_width} * channels, src_height);
    test::Array2D<int16_t> mapxy(width * 2, height);
    test::Array2D<uint16_t> mapfrac(width, height);
    test::Array2D<T> dst_small(test_width * channels, height),
        dst(width * channels, height);

    kleidicv_error_t result = multithreaded_func(
        src.data(), src.stride(), src_width, src_height, dst.data(),
        dst.stride(), width, height, channels, mapxy.data(), mapxy.stride(),
        mapfrac.data(), mapfrac.stride(), args...,
        get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED, result);

    result = multithreaded_func(
        src.data(), src.stride(), src_width, src_height, dst_small.data(),
        dst_small.stride(), test_width, height, channels, mapxy.data(),
        mapxy.stride(), mapfrac.data(), mapfrac.stride(), args...,
        get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED, result);
  }

  template <typename T, typename SingleThreadedFunc, typename MultithreadedFunc,
            typename... Args>
  void check_warp_perspective(SingleThreadedFunc single_threaded_func,
                              MultithreadedFunc multithreaded_func,
                              size_t channels, Args... args) {
    unsigned test_width = 0, height = 0, thread_count = 0;
    std::tie(test_width, height, thread_count) = GetParam();
    const unsigned src_width = 300, src_height = 300;
    // width < 8 are not supported, that's not tested here
    size_t width = test_width + 8;
    test::Array2D<T> src(size_t{src_width} * channels, src_height);
    test::Array2D<T> dst_single(width * channels, height),
        dst_multi(width * channels, height);
    // clang-format off
    const float transform[] = {
      0.8, 0.1, 2,
      0.1, 0.8, -2,
      0.001, 0.001, 1.0
    };
    // clang-format on
    kleidicv_error_t single_result = single_threaded_func(
        src.data(), src.stride(), src_width, src_height, dst_single.data(),
        dst_single.stride(), width, height, transform, channels, args...);

    kleidicv_error_t multi_result = multithreaded_func(
        src.data(), src.stride(), src_width, src_height, dst_multi.data(),
        dst_multi.stride(), width, height, transform, channels, args...,
        get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_OK, single_result);
    EXPECT_EQ(KLEIDICV_OK, multi_result);
    EXPECT_EQ_ARRAY2D(dst_multi, dst_single);
  }

  template <typename T, typename MultithreadedFunc, typename... Args>
  void check_warp_perspective_not_implemented(
      MultithreadedFunc multithreaded_func, size_t channels, Args... args) {
    unsigned test_width = 0, height = 0, thread_count = 0;
    std::tie(test_width, height, thread_count) = GetParam();
    const unsigned src_width = 300, src_height = 300;
    // width < 8 are not supported!
    size_t width = test_width + 8;
    test::Array2D<T> src(size_t{src_width} * channels, src_height);
    test::Array2D<T> dst_small(test_width * channels, height),
        dst(width * channels, height);
    // clang-format off
    const float transform[] = {
      0.8, 0.1, 2,
      0.1, 0.8, -2,
      0.001, 0.001, 1.0
    };
    // clang-format on

    kleidicv_error_t result = multithreaded_func(
        src.data(), src.stride(), src_width, src_height, dst.data(),
        dst.stride(), width, height, transform, channels, args...,
        get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED, result);

    result = multithreaded_func(src.data(), src.stride(), src_width, src_height,
                                dst_small.data(), dst_small.stride(),
                                test_width, height, transform, channels,
                                args..., get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED, result);
  }
};

#define TEST_UNARY_OP(suffix, SrcT, DstT, ...)                              \
  TEST_P(Thread, suffix) {                                                  \
    check_unary_op<SrcT, DstT>(kleidicv_##suffix, kleidicv_thread_##suffix, \
                               __VA_ARGS__);                                \
  }

#define TEST_BINARY_OP(suffix, T, ...)                                 \
  TEST_P(Thread, suffix) {                                             \
    check_binary_op<T, T>(kleidicv_##suffix, kleidicv_thread_##suffix, \
                          __VA_ARGS__);                                \
  }

TEST_UNARY_OP(gray_to_rgb_u8, uint8_t, uint8_t, 1, 3);
TEST_UNARY_OP(gray_to_rgba_u8, uint8_t, uint8_t, 1, 4);
TEST_UNARY_OP(rgb_to_bgr_u8, uint8_t, uint8_t, 3, 3);
TEST_UNARY_OP(rgb_to_rgb_u8, uint8_t, uint8_t, 3, 3);
TEST_UNARY_OP(rgba_to_bgra_u8, uint8_t, uint8_t, 4, 4);
TEST_UNARY_OP(rgba_to_rgba_u8, uint8_t, uint8_t, 4, 4);
TEST_UNARY_OP(rgb_to_bgra_u8, uint8_t, uint8_t, 3, 4);
TEST_UNARY_OP(rgb_to_rgba_u8, uint8_t, uint8_t, 3, 4);
TEST_UNARY_OP(rgba_to_bgr_u8, uint8_t, uint8_t, 4, 3);
TEST_UNARY_OP(rgba_to_rgb_u8, uint8_t, uint8_t, 4, 3);
TEST_UNARY_OP(yuv_to_bgr_u8, uint8_t, uint8_t, 3, 3);
TEST_UNARY_OP(yuv_to_rgb_u8, uint8_t, uint8_t, 3, 3);
TEST_UNARY_OP(bgr_to_yuv_u8, uint8_t, uint8_t, 3, 3);
TEST_UNARY_OP(rgb_to_yuv_u8, uint8_t, uint8_t, 3, 3);
TEST_UNARY_OP(bgra_to_yuv_u8, uint8_t, uint8_t, 4, 3);
TEST_UNARY_OP(rgba_to_yuv_u8, uint8_t, uint8_t, 4, 3);
TEST_UNARY_OP(threshold_binary_u8, uint8_t, uint8_t, 1, 1, 100, 200);
TEST_UNARY_OP(scale_u8, uint8_t, uint8_t, 1, 1, 0.5F, 3.5F);
TEST_UNARY_OP(scale_f32, float, float, 1, 1, 0.123F, 45.6789F);
TEST_UNARY_OP(exp_f32, float, float, 1, 1);
TEST_UNARY_OP(f32_to_s8, float, int8_t, 1, 1);
TEST_UNARY_OP(f32_to_u8, float, uint8_t, 1, 1);
TEST_UNARY_OP(s8_to_f32, int8_t, float, 1, 1);
TEST_UNARY_OP(u8_to_f32, uint8_t, float, 1, 1);

TEST_BINARY_OP(saturating_add_s8, int8_t, 1, 1);
TEST_BINARY_OP(saturating_add_u8, uint8_t, 1, 1);
TEST_BINARY_OP(saturating_add_s16, int16_t, 1, 1);
TEST_BINARY_OP(saturating_add_u16, uint16_t, 1, 1);
TEST_BINARY_OP(saturating_add_s32, int32_t, 1, 1);
TEST_BINARY_OP(saturating_add_u32, uint32_t, 1, 1);
TEST_BINARY_OP(saturating_add_s64, int64_t, 1, 1);
TEST_BINARY_OP(saturating_add_u64, uint64_t, 1, 1);
TEST_BINARY_OP(saturating_sub_s8, int8_t, 1, 1);
TEST_BINARY_OP(saturating_sub_u8, uint8_t, 1, 1);
TEST_BINARY_OP(saturating_sub_s16, int16_t, 1, 1);
TEST_BINARY_OP(saturating_sub_u16, uint16_t, 1, 1);
TEST_BINARY_OP(saturating_sub_s32, int32_t, 1, 1);
TEST_BINARY_OP(saturating_sub_u32, uint32_t, 1, 1);
TEST_BINARY_OP(saturating_sub_s64, int64_t, 1, 1);
TEST_BINARY_OP(saturating_sub_u64, uint64_t, 1, 1);
TEST_BINARY_OP(saturating_absdiff_u8, uint8_t, 1, 1);
TEST_BINARY_OP(saturating_absdiff_s8, int8_t, 1, 1);
TEST_BINARY_OP(saturating_absdiff_u16, uint16_t, 1, 1);
TEST_BINARY_OP(saturating_absdiff_s16, int16_t, 1, 1);
TEST_BINARY_OP(saturating_absdiff_s32, int32_t, 1, 1);
TEST_BINARY_OP(saturating_multiply_u8, uint8_t, 1, 1, 1.23);
TEST_BINARY_OP(saturating_multiply_s8, int8_t, 1, 1, -2.34);
TEST_BINARY_OP(saturating_multiply_u16, uint16_t, 1, 1, 0.321);
TEST_BINARY_OP(saturating_multiply_s16, int16_t, 1, 1, -0.543);
TEST_BINARY_OP(saturating_multiply_s32, int32_t, 1, 1, -0.0123);
TEST_BINARY_OP(bitwise_and, uint8_t, 1, 1);
TEST_BINARY_OP(compare_equal_u8, uint8_t, 1, 1);
TEST_BINARY_OP(compare_greater_u8, uint8_t, 1, 1);
TEST_BINARY_OP(saturating_add_abs_with_threshold_s16, int16_t, 1, 1, 123);

TEST_P(Thread, gaussian_blur_u8) {
  unsigned width = 0, height = 0, thread_count = 0;
  std::tie(width, height, thread_count) = GetParam();
  (void)thread_count;
  size_t channels = 1;
  size_t kernel_width = 5;
  size_t kernel_height = kernel_width;
  float sigma_x = 0.0F, sigma_y = 0.0F;
  kleidicv_border_type_t border_type = KLEIDICV_BORDER_TYPE_REPLICATE;
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_filter_context_create(&context, channels, kernel_width,
                                           kernel_height, width, height));
  check_unary_op<uint8_t, uint8_t>(
      kleidicv_gaussian_blur_u8, kleidicv_thread_gaussian_blur_u8,
      channels /*src_channels*/, channels /*dst_channels*/,
      /*remaining arguments passed to gaussian_blur_u8 functions*/ channels,
      kernel_width, kernel_height, sigma_x, sigma_y, border_type, context);
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TEST(ThreadGaussianBlur, NotImplemented) {
  unsigned max_width = 10, max_height = 10;
  size_t channels = 1;
  size_t kernel_width = 5;
  size_t kernel_height = kernel_width;
  float sigma_x = 0.0F, sigma_y = 0.0F;
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(
                             &context, channels, kernel_width, kernel_height,
                             max_width, max_height));

  uint8_t src[1] = {}, dst[1] = {};
  // Image too small
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_thread_gaussian_blur_u8(
                src, 1, dst, 1, 1, 1, channels, kernel_width, kernel_height,
                sigma_x, sigma_y, KLEIDICV_BORDER_TYPE_REPLICATE, context,
                get_multithreading_fake(2)));
  // Border not supported
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      kleidicv_thread_gaussian_blur_u8(
          src, 1, dst, 1, max_width, max_height, channels, kernel_width,
          kernel_height, sigma_x, sigma_y, KLEIDICV_BORDER_TYPE_TRANSPARENT,
          context, get_multithreading_fake(2)));

  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TEST_P(Thread, blur_and_downsample_u8) {
  unsigned src_width = 0, src_height = 0, thread_count = 0;
  std::tie(src_width, src_height, thread_count) = GetParam();
  size_t channels = 1;
  size_t kernel_width = 5;
  size_t kernel_height = kernel_width;
  kleidicv_border_type_t border_type = KLEIDICV_BORDER_TYPE_REPLICATE;
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(
                             &context, channels, kernel_width, kernel_height,
                             src_width, src_height));

  test::Array2D<uint8_t> src(size_t{src_width} * channels, src_height);
  test::Array2D<uint8_t> dst_single(size_t{(src_width + 1) / 2} * channels,
                                    (src_height + 1) / 2),
      dst_multi(size_t{(src_width + 1) / 2} * channels, (src_height + 1) / 2);

  test::PseudoRandomNumberGenerator<uint8_t> generator;
  src.fill(generator);

  kleidicv_error_t single_result = kleidicv_blur_and_downsample_u8(
      src.data(), src.stride(), src_width, src_height, dst_single.data(),
      dst_single.stride(), channels, border_type, context);

  kleidicv_error_t multi_result = kleidicv_thread_blur_and_downsample_u8(
      src.data(), src.stride(), src_width, src_height, dst_multi.data(),
      dst_multi.stride(), channels, border_type, context,
      get_multithreading_fake(thread_count));

  EXPECT_EQ(single_result, multi_result);
  if (KLEIDICV_OK == single_result) {
    EXPECT_EQ_ARRAY2D(dst_multi, dst_single);
  }

  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TEST(ThreadBlurAndDownsample, NotImplemented) {
  unsigned max_width = 10, max_height = 10;
  size_t channels = 1;
  size_t kernel_width = 5;
  size_t kernel_height = kernel_width;
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(
                             &context, channels, kernel_width, kernel_height,
                             max_width, max_height));

  uint8_t src[1] = {}, dst[1] = {};
  // Image too small
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_thread_blur_and_downsample_u8(
                src, 1, 1, 1, dst, 1, channels, KLEIDICV_BORDER_TYPE_REPLICATE,
                context, get_multithreading_fake(2)));
  // Border not supported
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_thread_blur_and_downsample_u8(
                src, 1, max_width, max_height, dst, 1, channels,
                KLEIDICV_BORDER_TYPE_TRANSPARENT, context,
                get_multithreading_fake(2)));

  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TEST_P(Thread, scharr_interleaved_s16_u8) {
  unsigned src_width = 0, src_height = 0, thread_count = 0;
  std::tie(src_width, src_height, thread_count) = GetParam();

  // Minimal width and height is 3
  src_width += 2;
  src_height += 2;

  size_t src_channels = 1;

  test::Array2D<uint8_t> src(size_t{src_width} * src_channels, src_height);
  test::Array2D<int16_t> dst_single(size_t{src_width - 2} * src_channels * 2,
                                    size_t{src_height - 2}),
      dst_multi(size_t{src_width - 2} * src_channels * 2,
                size_t{src_height - 2});

  test::PseudoRandomNumberGenerator<uint8_t> generator;
  src.fill(generator);

  kleidicv_error_t single_result = kleidicv_scharr_interleaved_s16_u8(
      src.data(), src.stride(), src_width, src_height, src_channels,
      dst_single.data(), dst_single.stride());

  kleidicv_error_t multi_result = kleidicv_thread_scharr_interleaved_s16_u8(
      src.data(), src.stride(), src_width, src_height, src_channels,
      dst_multi.data(), dst_multi.stride(),
      get_multithreading_fake(thread_count));

  EXPECT_EQ(single_result, multi_result);
  if (KLEIDICV_OK == single_result) {
    EXPECT_EQ_ARRAY2D(dst_multi, dst_single);
  }
}

TEST(ThreadScharrInterleaved, NotImplemented) {
  uint8_t src[1] = {};
  int16_t dst[1] = {};
  // Multichannel input
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_thread_scharr_interleaved_s16_u8(
                src, 1, 1, 1, 2, dst, sizeof(dst), get_multithreading_fake(2)));
}

TEST_P(Thread, separable_filter_2d_u8) {
  check_separable_filter_2d<uint8_t>(kleidicv_separable_filter_2d_u8,
                                     kleidicv_thread_separable_filter_2d_u8);
}

TEST_P(Thread, separable_filter_2d_u16) {
  check_separable_filter_2d<uint16_t>(kleidicv_separable_filter_2d_u16,
                                      kleidicv_thread_separable_filter_2d_u16);
}

TEST_P(Thread, separable_filter_2d_s16) {
  check_separable_filter_2d<int16_t>(kleidicv_separable_filter_2d_s16,
                                     kleidicv_thread_separable_filter_2d_s16);
}

template <typename T, typename MultithreadedFunc>
void check_separable_filter_2d_not_implemented(
    MultithreadedFunc multithreaded_func) {
  unsigned max_width = 10, max_height = 10;
  size_t channels = 1;
  const size_t kernel_width = 5;
  const size_t kernel_height = kernel_width;

  test::Array2D<T> kernel_x{kernel_width, 1};
  kernel_x.set(0, 0, {1, 2, 3, 4, 5});
  test::Array2D<T> kernel_y{kernel_height, 1};
  kernel_y.set(0, 0, {5, 6, 7, 8, 9});

  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(
                             &context, channels, kernel_width, kernel_height,
                             max_width, max_height));
  T src[1] = {}, dst[1] = {};
  // Image too small
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            multithreaded_func(src, 1, dst, 1, 1, 1, channels, kernel_x.data(),
                               kernel_width, kernel_y.data(), kernel_height,
                               KLEIDICV_BORDER_TYPE_REPLICATE, context,
                               get_multithreading_fake(2)));
  // Border not supported
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            multithreaded_func(src, 1, dst, 1, max_width, max_height, channels,
                               kernel_x.data(), kernel_width, kernel_y.data(),
                               kernel_height, KLEIDICV_BORDER_TYPE_TRANSPARENT,
                               context, get_multithreading_fake(2)));

  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TEST(ThreadSeparableFilter2D, NotImplemented) {
  check_separable_filter_2d_not_implemented<uint8_t>(
      kleidicv_thread_separable_filter_2d_u8);
  check_separable_filter_2d_not_implemented<int16_t>(
      kleidicv_thread_separable_filter_2d_s16);
  check_separable_filter_2d_not_implemented<uint16_t>(
      kleidicv_thread_separable_filter_2d_u16);
}

#if KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
TEST_P(Thread, remap_s16_u8_border_replicate) {
  check_remap_s16<uint8_t>(kleidicv_remap_s16_u8, kleidicv_thread_remap_s16_u8,
                           1, KLEIDICV_BORDER_TYPE_REPLICATE,
                           kleidicv_border_values_t{});
}

TEST_P(Thread, remap_s16_u8_not_implemented) {
  check_remap_s16_not_implemented<uint8_t>(kleidicv_thread_remap_s16_u8, 2,
                                           KLEIDICV_BORDER_TYPE_REPLICATE,
                                           kleidicv_border_values_t{});
  check_remap_s16_not_implemented<uint8_t>(kleidicv_thread_remap_s16_u8, 1,
                                           KLEIDICV_BORDER_TYPE_CONSTANT,
                                           kleidicv_border_values_t{});
}

TEST_P(Thread, remap_s16point5_u8_border_replicate) {
  kleidicv_border_values_t border_values = {};
  check_remap_s16point5<uint8_t>(kleidicv_remap_s16point5_u8,
                                 kleidicv_thread_remap_s16point5_u8, 1,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, border_values);
}

TEST_P(Thread, remap_s16point5_u8_not_implemented) {
  check_remap_s16point5_not_implemented<uint8_t>(
      kleidicv_thread_remap_s16point5_u8, 2, KLEIDICV_BORDER_TYPE_REPLICATE,
      kleidicv_border_values_t{});
  check_remap_s16point5_not_implemented<uint8_t>(
      kleidicv_thread_remap_s16point5_u8, 1, KLEIDICV_BORDER_TYPE_CONSTANT,
      kleidicv_border_values_t{});
}
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_REMAP

#if KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
TEST_P(Thread, warp_perspective_u8_border_replicate) {
  check_warp_perspective<uint8_t>(
      kleidicv_warp_perspective_u8, kleidicv_thread_warp_perspective_u8, 1,
      KLEIDICV_INTERPOLATION_NEAREST, KLEIDICV_BORDER_TYPE_REPLICATE,
      kleidicv_border_values_t{});
}

TEST_P(Thread, warp_perspective_u8_not_implemented) {
  check_warp_perspective_not_implemented<uint8_t>(
      kleidicv_thread_warp_perspective_u8, 1, KLEIDICV_INTERPOLATION_LINEAR,
      KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{});
  check_warp_perspective_not_implemented<uint8_t>(
      kleidicv_thread_warp_perspective_u8, 2, KLEIDICV_INTERPOLATION_NEAREST,
      KLEIDICV_BORDER_TYPE_REPLICATE, kleidicv_border_values_t{});
  check_warp_perspective_not_implemented<uint8_t>(
      kleidicv_thread_warp_perspective_u8, 1, KLEIDICV_INTERPOLATION_NEAREST,
      KLEIDICV_BORDER_TYPE_CONSTANT, kleidicv_border_values_t{});
}
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE

TEST_P(Thread, SobelHorizontal1Channel) {
  check_unary_op<uint8_t, int16_t>(kleidicv_sobel_3x3_horizontal_s16_u8,
                                   kleidicv_thread_sobel_3x3_horizontal_s16_u8,
                                   1, 1, 1);
}

TEST_P(Thread, SobelHorizontal3Channels) {
  check_unary_op<uint8_t, int16_t>(kleidicv_sobel_3x3_horizontal_s16_u8,
                                   kleidicv_thread_sobel_3x3_horizontal_s16_u8,
                                   3, 3, 3);
}

TEST_P(Thread, SobelVertical1Channel) {
  check_unary_op<uint8_t, int16_t>(kleidicv_sobel_3x3_vertical_s16_u8,
                                   kleidicv_thread_sobel_3x3_vertical_s16_u8, 1,
                                   1, 1);
}

TEST_P(Thread, SobelVertical3Channels) {
  check_unary_op<uint8_t, int16_t>(kleidicv_sobel_3x3_vertical_s16_u8,
                                   kleidicv_thread_sobel_3x3_vertical_s16_u8, 3,
                                   3, 3);
}

TEST(ThreadSobel, NotImplemented) {
  uint8_t src[1] = {};
  int16_t dst[1] = {};
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_thread_sobel_3x3_vertical_s16_u8(
                src, sizeof(src), dst, sizeof(dst), 1, 1, 1,
                get_multithreading_fake(2)));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_thread_sobel_3x3_horizontal_s16_u8(
                src, sizeof(src), dst, sizeof(dst), 1, 1, 1,
                get_multithreading_fake(2)));
}

INSTANTIATE_TEST_SUITE_P(
    , Thread,
    testing::Values(P{1, 1, 1}, P{1, 2, 1}, P{1, 2, 2}, P{2, 1, 2}, P{2, 2, 1},
                    P{1, 3, 2}, P{2, 3, 1}, P{6, 4, 1}, P{4, 5, 2}, P{2, 6, 3},
                    P{1, 7, 4}, P{12, 34, 5}, P{1, 16, 1}, P{1, 32, 1},
                    P{1, 32, 2}, P{2, 16, 2}, P{2, 32, 1}, P{1, 48, 2},
                    P{2, 48, 1}, P{6, 64, 1}, P{4, 80, 2}, P{2, 96, 3},
                    P{1, 112, 4}, P{12, 34, 5}));

// Operations in the Neon backend have both a vector path and a scalar path.
// The vector path is used to process most data and the scalar path is used to
// process the parts of the data that don't fit into the vector width.
// For floating point operations in particular, the results may be very slightly
// different between vector and scalar paths.
// When using multithreading, images are divided into parts to be processed by
// each thread, and this can change which parts of the data end up being
// processed by the vector and scalar paths. Since the threading may be
// non-deterministic in how it divides up the image, this non-determinism could
// leak through in the values of the output. This could cause subtle bugs and
// must be avoided.
// Prior to the fix, these tests were found to trigger the bug when run via
// qemu-aarch64 8.0.4 on an x86 machine, but passed in other environments.
class SingleMultiThreadInconsistency : public testing::TestWithParam<P> {};

TEST_P(SingleMultiThreadInconsistency, ScaleF32) {
  const auto [width, height, thread_count] = GetParam();

  const float src_val = -407.727905F, scale = 0.123F, shift = 45.6789F;
  test::Array2D<float> src(size_t{width}, height),
      dst_single(size_t{width}, height), dst_multi(size_t{width}, height);

  src.fill(src_val);

  kleidicv_error_t single_result =
      kleidicv_scale_f32(src.data(), src.stride(), dst_single.data(),
                         dst_single.stride(), width, height, scale, shift);

  kleidicv_error_t multi_result = kleidicv_thread_scale_f32(
      src.data(), src.stride(), dst_multi.data(), dst_multi.stride(), width,
      height, scale, shift, get_multithreading_fake(thread_count));

  EXPECT_EQ(KLEIDICV_OK, multi_result);
  EXPECT_EQ(KLEIDICV_OK, single_result);
  EXPECT_EQ_ARRAY2D(dst_multi, dst_single);
}

TEST_P(SingleMultiThreadInconsistency, ExpF32) {
  const auto [width, height, thread_count] = GetParam();

  test::Array2D<float> src(size_t{width}, height),
      dst_single(size_t{width}, height), dst_multi(size_t{width}, height);

  // Approximately -1.900039
  unsigned value_bits = 0xBFF3347D;
  float value = 0;
  memcpy(&value, &value_bits, sizeof(value));

  src.fill(value);

  kleidicv_error_t single_result =
      kleidicv_exp_f32(src.data(), src.stride(), dst_single.data(),
                       dst_single.stride(), width, height);

  kleidicv_error_t multi_result = kleidicv_thread_exp_f32(
      src.data(), src.stride(), dst_multi.data(), dst_multi.stride(), width,
      height, get_multithreading_fake(thread_count));

  EXPECT_EQ(KLEIDICV_OK, multi_result);
  EXPECT_EQ(KLEIDICV_OK, single_result);
  EXPECT_EQ_ARRAY2D(dst_multi, dst_single);
}

INSTANTIATE_TEST_SUITE_P(, SingleMultiThreadInconsistency,
                         testing::Values(P{1, 7, 4}, P{1, 17, 17}, P{1, 33, 33},
                                         P{2, 2, 2}, P{6, 3, 2}));
