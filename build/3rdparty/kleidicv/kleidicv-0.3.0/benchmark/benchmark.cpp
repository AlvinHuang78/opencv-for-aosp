// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <random>

#include "kleidicv/kleidicv.h"

// These variables can be set runtime, from command line
extern size_t image_width, image_height;

// Setting up buffers can be time-consuming and is not of interest when
// benchmarking. Therefore endeavour to reuse buffers where possible. This
// function creates a single shared buffer for each pair of template arguments.
// All bytes of the buffer are filled with the Value argument.
template <int PixelSize, int Value>
uint8_t* get_buffer() {
  static std::vector<uint8_t> result(image_width * image_height * PixelSize,
                                     Value);
  return result.data();
}

// Get a buffer suitable for using as the first input buffer.
template <typename T, int Channels = 1>
const T* get_source_buffer_a() {
  return reinterpret_cast<T*>(get_buffer<sizeof(T) * Channels, 0xA3>());
}

// Get a buffer suitable for using as the second input buffer.
template <typename T, int Channels = 1>
const T* get_source_buffer_b() {
  return reinterpret_cast<T*>(get_buffer<sizeof(T) * Channels, 0x9E>());
}

// Get a buffer suitable for using as the destination buffer.
template <typename T, int Channels = 1>
T* get_destination_buffer() {
  // Value argument is only used here to differentiate from the source buffers.
  return reinterpret_cast<T*>(get_buffer<sizeof(T) * Channels, 0xC1>());
}

// Warms up the functor then benchmarks it.
template <typename F>
void bench_functor(benchmark::State& state, F functor) {
  // warm up
  functor();
  for (auto _ : state) {
    // This code gets benchmarked
    functor();
  }
}

template <typename T, typename Function>
static void bench_binary_op(Function f, benchmark::State& state) {
  bench_functor(state, [f]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T),
            get_source_buffer_b<T>(), image_width * sizeof(T),
            get_destination_buffer<T>(), image_width * sizeof(T), image_width,
            image_height);
  });
}

#define BENCH_BINARY_OP(name, type)                \
  static void name(benchmark::State& state) {      \
    bench_binary_op<type>(kleidicv_##name, state); \
  }                                                \
  BENCHMARK(name)

BENCH_BINARY_OP(saturating_add_s8, int8_t);
BENCH_BINARY_OP(saturating_sub_u16, uint16_t);
BENCH_BINARY_OP(saturating_absdiff_s32, int32_t);
BENCH_BINARY_OP(bitwise_and, uint8_t);
BENCH_BINARY_OP(compare_equal_u8, uint8_t);
BENCH_BINARY_OP(compare_greater_u8, uint8_t);

template <typename I, typename O, size_t InChannels, size_t OutChannels,
          typename Function>
static void bench_unary_op(Function f, benchmark::State& state) {
  bench_functor(state, [f]() {
    (void)f(get_source_buffer_a<I, InChannels>(),
            image_width * InChannels * sizeof(I),
            get_destination_buffer<O, OutChannels>(),
            image_width * OutChannels * sizeof(O), image_width, image_height);
  });
}

#define BENCH_UNARY_OP(name, channels, type)                                \
  static void name(benchmark::State& state) {                               \
    bench_unary_op<type, type, channels, channels>(kleidicv_##name, state); \
  }                                                                         \
  BENCHMARK(name)

BENCH_UNARY_OP(exp_f32, 1, float);

#define BENCH_UNARY_OP_DIFFERENT_IO_TYPES(name, itype, otype)   \
  static void name(benchmark::State& state) {                   \
    bench_unary_op<itype, otype, 1, 1>(kleidicv_##name, state); \
  }                                                             \
  BENCHMARK(name)

BENCH_UNARY_OP_DIFFERENT_IO_TYPES(f32_to_s8, float, int8_t);
BENCH_UNARY_OP_DIFFERENT_IO_TYPES(f32_to_u8, float, uint8_t);
BENCH_UNARY_OP_DIFFERENT_IO_TYPES(s8_to_f32, int8_t, float);
BENCH_UNARY_OP_DIFFERENT_IO_TYPES(u8_to_f32, uint8_t, float);

#define BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(name, in_channels,         \
                                                out_channels, type)        \
  static void name(benchmark::State& state) {                              \
    bench_unary_op<type, type, in_channels, out_channels>(kleidicv_##name, \
                                                          state);          \
  }                                                                        \
  BENCHMARK(name)

BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgb_to_yuv_u8, 3, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgba_to_yuv_u8, 4, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(bgr_to_yuv_u8, 3, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(bgra_to_yuv_u8, 4, 3, uint8_t);

BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(gray_to_rgb_u8, 1, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(gray_to_rgba_u8, 1, 4, uint8_t);

BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgb_to_bgr_u8, 3, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgb_to_rgb_u8, 3, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgba_to_bgra_u8, 4, 4, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgba_to_rgba_u8, 4, 4, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgb_to_bgra_u8, 3, 4, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgb_to_rgba_u8, 3, 4, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgba_to_bgr_u8, 4, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(rgba_to_rgb_u8, 4, 3, uint8_t);

BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(yuv_to_rgb_u8, 3, 3, uint8_t);
BENCH_UNARY_OP_DIFFERENT_CHANNEL_NUMBER(yuv_to_bgr_u8, 3, 3, uint8_t);

static void min_max_loc_u8(benchmark::State& state) {
  bench_functor(state, []() {
    size_t min_offset, max_offset;
    (void)kleidicv_min_max_loc_u8(get_source_buffer_a<uint8_t>(),
                                  image_width * sizeof(uint8_t), image_width,
                                  image_height, &min_offset, &max_offset);
  });
}
BENCHMARK(min_max_loc_u8);

static void sum_f32(benchmark::State& state) {
  bench_functor(state, []() {
    float total;
    (void)kleidicv_sum_f32(get_source_buffer_a<float>(),
                           image_width * sizeof(float), image_width,
                           image_height, &total);
  });
}
BENCHMARK(sum_f32);

template <typename T, typename Function>
static void scale(Function f, float factor, float shift,
                  benchmark::State& state) {
  bench_functor(state, [f, factor, shift]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T),
            get_destination_buffer<T>(), image_width * sizeof(T), image_width,
            image_height, factor, shift);
  });
}

#define BENCH_SCALE(benchname, name, factor, shift, type) \
  static void benchname(benchmark::State& state) {        \
    scale<type>(kleidicv_##name, factor, shift, state);   \
  }                                                       \
  BENCHMARK(benchname)

BENCH_SCALE(scale_u8_1, scale_u8, 1.0, 4.567, uint8_t);
BENCH_SCALE(scale_u8_generic, scale_u8, 1.234, 4.567, uint8_t);
BENCH_SCALE(scale_f32_1, scale_f32, 1.0, 4.567, float);
BENCH_SCALE(scale_f32_generic, scale_f32, 1.234, 4.567, float);

template <typename T, typename F>
static void min_max(F f, benchmark::State& state) {
  bench_functor(state, [f]() {
    T min_value = 0, max_value = 0;
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T), image_width,
            image_height, &min_value, &max_value);
  });
}

#define BENCH_MIN_MAX(name, type)             \
  static void name(benchmark::State& state) { \
    min_max<type>(kleidicv_##name, state);    \
  }                                           \
  BENCHMARK(name)

BENCH_MIN_MAX(min_max_s8, int8_t);
BENCH_MIN_MAX(min_max_u8, uint8_t);
BENCH_MIN_MAX(min_max_s16, int16_t);
BENCH_MIN_MAX(min_max_u16, uint16_t);
BENCH_MIN_MAX(min_max_s32, int32_t);
BENCH_MIN_MAX(min_max_f32, float);

template <typename T, typename F>
static void resize(F f, size_t src_width, size_t src_height, size_t dst_width,
                   size_t dst_height, benchmark::State& state) {
  bench_functor(state, [f, src_width, src_height, dst_width, dst_height]() {
    (void)f(get_source_buffer_a<T>(), src_width * sizeof(T), src_width,
            src_height, get_destination_buffer<T>(), dst_width * sizeof(T),
            dst_width, dst_height);
  });
}

template <typename T, typename F>
static void resize_upscale(F f, size_t scale_x, size_t scale_y,
                           benchmark::State& state) {
  size_t src_width = image_width / scale_x;
  size_t src_height = image_height / scale_y;
  resize<T>(f, src_width, src_height, src_width * scale_x, src_height * scale_y,
            state);
}

template <typename T, typename F>
static void resize_downscale(F f, size_t scale_x, size_t scale_y,
                             benchmark::State& state) {
  size_t dst_width = image_width / scale_x;
  size_t dst_height = image_height / scale_y;
  resize<T>(f, dst_width * scale_x, dst_height * scale_y, dst_width, dst_height,
            state);
}

static void resize_quarter_u8(benchmark::State& state) {
  resize_downscale<uint8_t>(kleidicv_resize_to_quarter_u8, 2, 2, state);
}
BENCHMARK(resize_quarter_u8);

static void resize_linear_2x2_u8(benchmark::State& state) {
  resize_upscale<uint8_t>(kleidicv_resize_linear_u8, 2, 2, state);
}
BENCHMARK(resize_linear_2x2_u8);

static void resize_linear_4x4_u8(benchmark::State& state) {
  resize_upscale<uint8_t>(kleidicv_resize_linear_u8, 4, 4, state);
}
BENCHMARK(resize_linear_4x4_u8);

static void resize_linear_2x2_f32(benchmark::State& state) {
  resize_upscale<float>(kleidicv_resize_linear_f32, 2, 2, state);
}
BENCHMARK(resize_linear_2x2_f32);

static void resize_linear_4x4_f32(benchmark::State& state) {
  resize_upscale<float>(kleidicv_resize_linear_f32, 4, 4, state);
}
BENCHMARK(resize_linear_4x4_f32);

static void resize_linear_8x8_f32(benchmark::State& state) {
  resize_upscale<float>(kleidicv_resize_linear_f32, 8, 8, state);
}
BENCHMARK(resize_linear_8x8_f32);

template <typename T, size_t KernelSize, int Channels, typename F>
static void separable_filter_2d(benchmark::State& state, F function) {
  kleidicv_filter_context_t* context;
  kleidicv_error_t err = kleidicv_filter_context_create(
      &context, Channels, KernelSize, KernelSize, image_width, image_height);
  if (err != KLEIDICV_OK) {
    state.SkipWithError(
        "Could not initialize SeparableFilter2D filter context.");
    return;
  }

  std::vector<T> kernel(KernelSize, 2);

  bench_functor(state, [context, kernel, function]() {
    (void)function(
        get_source_buffer_a<T, Channels>(), image_width * Channels * sizeof(T),
        get_destination_buffer<T, Channels>(),
        image_width * Channels * sizeof(T), image_width, image_height, Channels,
        kernel.data(), KernelSize, kernel.data(), KernelSize,
        KLEIDICV_BORDER_TYPE_REPLICATE, context);
  });

  (void)kleidicv_filter_context_release(context);
}

static void separable_filter_2d_u8_5x5_1ch(benchmark::State& state) {
  separable_filter_2d<uint8_t, 5, 1>(state, kleidicv_separable_filter_2d_u8);
}
BENCHMARK(separable_filter_2d_u8_5x5_1ch);

static void separable_filter_2d_u8_5x5_3ch(benchmark::State& state) {
  separable_filter_2d<uint8_t, 5, 3>(state, kleidicv_separable_filter_2d_u8);
}
BENCHMARK(separable_filter_2d_u8_5x5_3ch);

static void separable_filter_2d_u16_5x5_1ch(benchmark::State& state) {
  separable_filter_2d<uint16_t, 5, 1>(state, kleidicv_separable_filter_2d_u16);
}
BENCHMARK(separable_filter_2d_u16_5x5_1ch);

static void separable_filter_2d_u16_5x5_3ch(benchmark::State& state) {
  separable_filter_2d<uint16_t, 5, 3>(state, kleidicv_separable_filter_2d_u16);
}
BENCHMARK(separable_filter_2d_u16_5x5_3ch);

static void separable_filter_2d_s16_5x5_1ch(benchmark::State& state) {
  separable_filter_2d<int16_t, 5, 1>(state, kleidicv_separable_filter_2d_s16);
}
BENCHMARK(separable_filter_2d_s16_5x5_1ch);

static void separable_filter_2d_s16_5x5_3ch(benchmark::State& state) {
  separable_filter_2d<int16_t, 5, 3>(state, kleidicv_separable_filter_2d_s16);
}
BENCHMARK(separable_filter_2d_s16_5x5_3ch);

template <typename T, size_t KernelSize, int Channels, bool Binomial>
static void gaussian_blur(benchmark::State& state) {
  kleidicv_filter_context_t* context;
  kleidicv_error_t err = kleidicv_filter_context_create(
      &context, Channels, KernelSize, KernelSize, image_width, image_height);
  if (err != KLEIDICV_OK) {
    state.SkipWithError("Could not initialize Gaussian blur filter context.");
    return;
  }

  bench_functor(state, [context]() {
    (void)kleidicv_gaussian_blur_u8(
        get_source_buffer_a<T, Channels>(), image_width * Channels * sizeof(T),
        get_destination_buffer<T, Channels>(),
        image_width * Channels * sizeof(T), image_width, image_height, Channels,
        KernelSize, KernelSize, (Binomial ? 0.0 : 2.0), (Binomial ? 0.0 : 2.0),
        KLEIDICV_BORDER_TYPE_REFLECT, context);
  });

  (void)kleidicv_filter_context_release(context);
}

#define BENCH_GAUSSIAN_BLUR(kernel_size, channel_number)                                    \
  static void                                                                               \
      gaussian_blur_binomial_u8##_##kernel_size##x##kernel_size##_##channel_number##ch(     \
          benchmark::State& state) {                                                        \
    gaussian_blur<uint8_t, kernel_size, channel_number, true>(state);                       \
  }                                                                                         \
  BENCHMARK(                                                                                \
      gaussian_blur_binomial_u8##_##kernel_size##x##kernel_size##_##channel_number##ch);    \
                                                                                            \
  static void                                                                               \
      gaussian_blur_custom_sigma_u8##_##kernel_size##x##kernel_size##_##channel_number##ch( \
          benchmark::State& state) {                                                        \
    gaussian_blur<uint8_t, kernel_size, channel_number, false>(state);                      \
  }                                                                                         \
  BENCHMARK(                                                                                \
      gaussian_blur_custom_sigma_u8##_##kernel_size##x##kernel_size##_##channel_number##ch);

BENCH_GAUSSIAN_BLUR(3, 1);
BENCH_GAUSSIAN_BLUR(3, 3);
BENCH_GAUSSIAN_BLUR(5, 1);
BENCH_GAUSSIAN_BLUR(5, 3);
BENCH_GAUSSIAN_BLUR(7, 1);
BENCH_GAUSSIAN_BLUR(7, 3);
BENCH_GAUSSIAN_BLUR(15, 1);
BENCH_GAUSSIAN_BLUR(15, 3);

template <typename Function>
static void sobel_filter(Function f, benchmark::State& state) {
  bench_functor(state, [f]() {
    (void)f(get_source_buffer_a<uint8_t, 1>(), image_width * sizeof(uint8_t),
            get_destination_buffer<int16_t, 1>(), image_width * sizeof(int16_t),
            image_width, image_height, 1);
  });
}

static void sobel_filter_vertical(benchmark::State& state) {
  sobel_filter(kleidicv_sobel_3x3_vertical_s16_u8, state);
}
BENCHMARK(sobel_filter_vertical);

static void sobel_filter_horizontal(benchmark::State& state) {
  sobel_filter(kleidicv_sobel_3x3_horizontal_s16_u8, state);
}
BENCHMARK(sobel_filter_horizontal);

template <size_t OutChannels, typename Function>
static void yuv_sp(Function f, benchmark::State& state) {
  bench_functor(state, [f]() {
    (void)f(get_source_buffer_a<uint8_t, 1>(), image_width * sizeof(uint8_t),
            get_source_buffer_b<uint8_t, 2>(),
            (image_width / 2) * sizeof(uint8_t),
            get_destination_buffer<uint8_t, OutChannels>(),
            image_width * sizeof(uint8_t), image_width, image_height, true);
  });
}

static void yuv_sp_to_rgb(benchmark::State& state) {
  yuv_sp<3>(kleidicv_yuv_sp_to_rgb_u8, state);
}
BENCHMARK(yuv_sp_to_rgb);

static void yuv_sp_to_bgr(benchmark::State& state) {
  yuv_sp<3>(kleidicv_yuv_sp_to_bgr_u8, state);
}
BENCHMARK(yuv_sp_to_bgr);

static void yuv_sp_to_rgba(benchmark::State& state) {
  yuv_sp<4>(kleidicv_yuv_sp_to_rgba_u8, state);
}
BENCHMARK(yuv_sp_to_rgba);

static void yuv_sp_to_bgra(benchmark::State& state) {
  yuv_sp<4>(kleidicv_yuv_sp_to_bgra_u8, state);
}
BENCHMARK(yuv_sp_to_bgra);

template <typename T, size_t KernelSize, typename Function>
static void morphology(Function f, benchmark::State& state) {
  kleidicv_morphology_context_t* context = nullptr;
  kleidicv_error_t err = kleidicv_morphology_create(
      &context, kleidicv_rectangle_t{KernelSize, KernelSize},
      kleidicv_point_t{0, 0}, KLEIDICV_BORDER_TYPE_REPLICATE,
      kleidicv_border_values_t{0, 0, 0, 0}, 1, 1, sizeof(T),
      kleidicv_rectangle_t{image_width, image_height});
  if (err != KLEIDICV_OK) {
    state.SkipWithError("Could not initialize morphology context.");
    return;
  }

  bench_functor(state, [f, context]() {
    (void)f(get_source_buffer_a<T, 1>(), image_width * sizeof(T),
            get_destination_buffer<T, 1>(), image_width * sizeof(T),
            image_width, image_height, context);
  });

  (void)kleidicv_morphology_release(context);
}

#define BENCH_MORPHOLOGY(name, kernel_size)                                   \
  static void name##_##kernel_size##x##kernel_size(benchmark::State& state) { \
    morphology<uint8_t, kernel_size>(kleidicv_##name##_u8, state);            \
  }                                                                           \
  BENCHMARK(name##_##kernel_size##x##kernel_size)

BENCH_MORPHOLOGY(dilate, 3);
BENCH_MORPHOLOGY(dilate, 5);
BENCH_MORPHOLOGY(dilate, 17);
BENCH_MORPHOLOGY(erode, 3);
BENCH_MORPHOLOGY(erode, 5);
BENCH_MORPHOLOGY(erode, 17);

template <typename T, typename Function>
static void in_range(Function f, T lower_bound, T upper_bound,
                     benchmark::State& state) {
  bench_functor(state, [f, lower_bound, upper_bound]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T),
            get_destination_buffer<uint8_t>(), image_width * sizeof(uint8_t),
            image_width, image_height, lower_bound, upper_bound);
  });
}

#define BENCH_IN_RANGE(benchname, name, lower_bound, upper_bound, type) \
  static void benchname(benchmark::State& state) {                      \
    in_range<type>(kleidicv_##name, lower_bound, upper_bound, state);   \
  }                                                                     \
  BENCHMARK(benchname)

BENCH_IN_RANGE(in_range_u8, in_range_u8, 1, 2, uint8_t);
BENCH_IN_RANGE(in_range_f32, in_range_f32, 1.111, 1.112, float);

static void blur_and_downsample_u8(benchmark::State& state) {
  kleidicv_filter_context_t* context;
  kleidicv_error_t err = kleidicv_filter_context_create(
      &context, 1, 5, 5, image_width, image_height);
  if (err != KLEIDICV_OK) {
    state.SkipWithError(
        "Could not initialize filter context for Blur and Downsample");
    return;
  }

  bench_functor(state, [context]() {
    (void)kleidicv_blur_and_downsample_u8(
        get_source_buffer_a<uint8_t>(), image_width * sizeof(uint8_t),
        image_width, image_height, get_destination_buffer<uint8_t>(),
        ((image_width + 1) / 2) * sizeof(uint8_t), 1,
        KLEIDICV_BORDER_TYPE_REFLECT, context);
  });

  (void)kleidicv_filter_context_release(context);
}
BENCHMARK(blur_and_downsample_u8);

static void scharr_interleaved_s16_u8(benchmark::State& state) {
  bench_functor(state, []() {
    (void)kleidicv_scharr_interleaved_s16_u8(
        get_source_buffer_a<uint8_t>(), image_width * sizeof(uint8_t),
        image_width, image_height, 1, get_destination_buffer<int16_t>(),
        (image_width - 2) * sizeof(int16_t));
  });
}
BENCHMARK(scharr_interleaved_s16_u8);

#if KLEIDICV_EXPERIMENTAL_FEATURE_REMAP
template <class ScalarType>
static const ScalarType* get_random_mapxy() {
  auto generate_mapxy = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 bytes padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe * 2);
    std::mt19937_64 rng;
    std::uniform_int_distribution<ScalarType> dist_x(0, image_width),
        dist_y(0, image_height);
    for (size_t row = 0; row < image_height; ++row) {
      for (size_t column = 0; column < image_width; ++column) {
        size_t index = row * image_stripe + column;
        v[2 * index] = dist_x(rng);
        v[2 * index + 1] = dist_y(rng);
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapxy = generate_mapxy();
  return mapxy.data();
}

template <class ScalarType>
static const ScalarType* get_blend_mapxy() {
  auto generate_mapxy = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 bytes padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe * 2);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        // Use a second degree function to add a nonlinear blend to the image
        v[2 * index] =
            static_cast<int16_t>(column * 2 - column * column / image_width);
        v[2 * index + 1] =
            static_cast<int16_t>(row * (image_width - column) / image_width +
                                 4 * row / image_height);
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapxy = generate_mapxy();
  return mapxy.data();
}

template <class ScalarType>
static const ScalarType* get_flip_mapxy() {
  auto generate_mapxy = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 bytes padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe * 2);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        v[2 * index] = static_cast<int16_t>(image_width - column - 1);
        v[2 * index + 1] = static_cast<int16_t>(row);
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapxy = generate_mapxy();
  return mapxy.data();
}

template <class ScalarType>
static const ScalarType* get_identity_mapxy() {
  auto generate_mapxy = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 bytes padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<ScalarType> v(image_height * image_stripe * 2);
    for (int row = 0; row < static_cast<int>(image_height); ++row) {
      for (int column = 0; column < static_cast<int>(image_width); ++column) {
        size_t index = row * image_stripe + column;
        v[2 * index] = static_cast<int16_t>(column);
        v[2 * index + 1] = static_cast<int16_t>(row);
      }
    }
    return v;
  };
  static std::vector<ScalarType> mapxy = generate_mapxy();
  return mapxy.data();
}

static const uint16_t* get_random_mapfrac() {
  static const uint16_t FRAC_BITS = 5;
  static const uint16_t REMAP16POINT5_FRAC_MAX = 1 << FRAC_BITS;

  auto generate_mapfrac = [&]() {
    // Prevent KleidiCV from flattening the image, it affects the performance
    // Add 4 bytes padding, so the image won't be processed as a single row
    const size_t image_stripe = image_width + 4;
    std::vector<uint16_t> v(image_height * image_stripe);
    std::mt19937_64 rng;
    std::uniform_int_distribution<uint16_t> dist_x(0,
                                                   REMAP16POINT5_FRAC_MAX - 1),
        dist_y(0, REMAP16POINT5_FRAC_MAX - 1);
    for (size_t row = 0; row < image_height; ++row) {
      for (size_t column = 0; column < image_width; ++column) {
        v[row * image_stripe + column] =
            dist_x(rng) | (dist_y(rng) << FRAC_BITS);
      }
    }
    return v;
  };
  static std::vector<uint16_t> mapfrac = generate_mapfrac();
  return mapfrac.data();
}

template <typename T, typename Function, typename MapFunc>
static void remap_s16(Function f, MapFunc mf, size_t channels,
                      kleidicv_border_type_t border_type,
                      benchmark::State& state) {
  bench_functor(state, [f, mf, channels, border_type]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T), image_width,
            image_height, get_destination_buffer<T>(), image_width * sizeof(T),
            image_width, image_height, channels, mf(),
            image_width * 2 * sizeof(int16_t), border_type,
            kleidicv_border_values_t{});
  });
}

#define BENCH_REMAP_S16(benchname, name, mapfunc, channels, border_type, type) \
  static void benchname(benchmark::State& state) {                             \
    remap_s16<type>(kleidicv_##name, mapfunc, channels, border_type, state);   \
  }                                                                            \
  BENCHMARK(benchname)

BENCH_REMAP_S16(remap_s16_u8_random, remap_s16_u8, get_random_mapxy<int16_t>, 1,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16(remap_s16_u8_blend, remap_s16_u8, get_blend_mapxy<int16_t>, 1,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16(remap_s16_u8_flip, remap_s16_u8, get_flip_mapxy<int16_t>, 1,
                KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16(remap_s16_u8_identity, remap_s16_u8,
                get_identity_mapxy<int16_t>, 1, KLEIDICV_BORDER_TYPE_REPLICATE,
                uint8_t);

template <typename T, typename Function, typename MapFunc>
static void remap_s16point5(Function f, MapFunc mf, size_t channels,
                            kleidicv_border_type_t border_type,
                            benchmark::State& state) {
  bench_functor(state, [f, mf, channels, border_type]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T), image_width,
            image_height, get_destination_buffer<T>(), image_width * sizeof(T),
            image_width, image_height, channels, mf(),
            image_width * 2 * sizeof(int16_t), get_random_mapfrac(),
            image_width * sizeof(uint16_t), border_type,
            kleidicv_border_values_t{});
  });
}

#define BENCH_REMAP_S16POINT5(benchname, name, mapfunc, channels, border_type, \
                              type)                                            \
  static void benchname(benchmark::State& state) {                             \
    remap_s16point5<type>(kleidicv_##name, mapfunc, channels, border_type,     \
                          state);                                              \
  }                                                                            \
  BENCHMARK(benchname)

BENCH_REMAP_S16POINT5(remap_s16point5_u8_random, remap_s16point5_u8,
                      get_random_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16POINT5(remap_s16point5_u8_blend, remap_s16point5_u8,
                      get_blend_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16POINT5(remap_s16point5_u8_flip, remap_s16point5_u8,
                      get_flip_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_REMAP_S16POINT5(remap_s16point5_u8_identity, remap_s16point5_u8,
                      get_identity_mapxy<int16_t>, 1,
                      KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_REMAP

#if KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
// clang-format off
static const float transform_identity[] = {
  1.0, 0, 0,
  0, 1.0, 0,
  0, 0, 1.0
};

static const float transform_small[] = {
  0.8, 0.1, 2,
  0.1, 0.8, -2,
  0.001, 0.001, 1.1
};

static const float transform_bend[] = {
  200, 0, 0,
  0, 200, -2,
  0.01, 0.01, 10
};

// designed for source 512 x 512 (rotate center is at [256,256])
// rotate by 30 degrees and upscale to 2.2
static const float transform_rotate[] = {
 0.3685617397559249, -0.2421336247796208, 201.2709623094595,
 0.2158651377567587, 0.3868901230951353, 91.52518789972362,
 -0.0001175503101026296, -6.963609976572184e-05, 0.9431277488336682};

// a near transformation made by OpenCV getPerspectiveTransform
static const float transform_near[] = {
  1.015917119306434, -0.03848938238648505, 2.925193061372864,
 0.04162265799405287, 1.030708451905362, -78.33384234480749,
 -2.763770366739386e-06, -4.019862712379178e-05, 1.003055095661408};
// clang-format on

template <typename T, typename Function>
static void warp_perspective(Function f, const float transform[9],
                             size_t channels,
                             kleidicv_interpolation_type_t interpolation,
                             kleidicv_border_type_t border_type,
                             benchmark::State& state) {
  bench_functor(state, [f, transform, channels, interpolation, border_type]() {
    (void)f(get_source_buffer_a<T>(), image_width * sizeof(T), image_width,
            image_height, get_destination_buffer<T>(), image_width * sizeof(T),
            image_width, image_height, transform, channels, interpolation,
            border_type, kleidicv_border_values_t{});
  });
}

#define BENCH_WARP_PERSPECTIVE(benchname, name, transform, channels, \
                               interpolation, border_type, type)     \
  static void benchname(benchmark::State& state) {                   \
    warp_perspective<type>(kleidicv_##name, transform, channels,     \
                           interpolation, border_type, state);       \
  }                                                                  \
  BENCHMARK(benchname)

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_identity, warp_perspective_u8,
                       transform_identity, 1, KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_small, warp_perspective_u8,
                       transform_small, 1, KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_bend, warp_perspective_u8,
                       transform_bend, 1, KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_rotate, warp_perspective_u8,
                       transform_rotate, 1, KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);

BENCH_WARP_PERSPECTIVE(warp_perspective_u8_near, warp_perspective_u8,
                       transform_near, 1, KLEIDICV_INTERPOLATION_NEAREST,
                       KLEIDICV_BORDER_TYPE_REPLICATE, uint8_t);
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_WARP_PERSPECTIVE
