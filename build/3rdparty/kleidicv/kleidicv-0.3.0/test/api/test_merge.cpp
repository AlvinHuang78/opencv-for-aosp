// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

template <typename ElementType, size_t Channels>
class MergeTest final {
 public:
  // Shorthand for internal data layout representation.
  using ArrayType = test::Array2D<ElementType>;

  MergeTest() : output_padding_{0} { inputs_padding_.fill(0); }

  // Sets the number of padding bytes at the end of rows.
  MergeTest& with_paddings(std::initializer_list<size_t> inputs_padding,
                           size_t output_padding) {
    output_padding_ = output_padding;
    size_t i = 0;
    for (size_t p : inputs_padding) {
      inputs_padding_[i++] = p;
    }
    return *this;
  }

  // Sets the number of elements in a row.
  MergeTest& with_width(size_t width) {
    width_ = width;
    return *this;
  }

  // Executes the test
  void test() {
    size_t vector_lanes = test::Options::vector_lanes<ElementType>();
    size_t input_width = width_;
    size_t output_width = input_width * Channels;
    size_t height = 2;

    // Create input and output arrays
    std::array<ArrayType, Channels> inputs;
    for (size_t i = 0; i < Channels; ++i) {
      inputs[i] = ArrayType{input_width, height, inputs_padding_[i]};
      inputs[i].fill(ElementType{0});
    }

    ArrayType expected_output{output_width, height, output_padding_};
    expected_output.fill(ElementType{0});
    ArrayType actual_output{output_width, height, output_padding_};
    // Prefill actual_outputs with a different value than expected
    actual_output.fill(ElementType{1});

    // Place 4 set of special values in input and expected output. The size of
    // the set is equals to 'Channels'. In the expected output 2 set is placed
    // at the beginning of rows, 2 set at the end of the rows.
    ElementType running_test_value = test::Options::seed() % 50;
    for (size_t i = 0; i < Channels; ++i) {
      inputs[i].set(0, 0, {running_test_value});
      expected_output.set(0, i, {running_test_value});
      running_test_value++;

      inputs[i].set(0, vector_lanes * 2, {running_test_value});
      expected_output.set(0, (vector_lanes * 2 * Channels) + i,
                          {running_test_value});
      running_test_value++;

      inputs[i].set(1, 0, {running_test_value});
      expected_output.set(1, i, {running_test_value});
      running_test_value++;

      inputs[i].set(1, vector_lanes * 2, {running_test_value});
      expected_output.set(1, (vector_lanes * 2 * Channels) + i,
                          {running_test_value});
      running_test_value++;
    }

    // Call the function to be tested
    const void* input_raw_pointers[Channels];
    for (size_t i = 0; i < Channels; ++i) {
      input_raw_pointers[i] = inputs[i].data();
    }
    size_t strides[Channels];
    for (size_t i = 0; i < Channels; ++i) {
      strides[i] = inputs[i].stride();
    }
    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_merge(input_raw_pointers, strides, actual_output.data(),
                             actual_output.stride(), input_width, height,
                             Channels, sizeof(ElementType)));

    // Compare the results
    for (size_t i = 0; i < Channels; ++i) {
      EXPECT_EQ_ARRAY2D(expected_output, actual_output);
    }
  }

 private:
  // Number of padding bytes at the end of rows.
  std::array<size_t, Channels> inputs_padding_;
  size_t output_padding_;
  // Tested number of elements in a row.
  // Sufficient number of elements to exercise both vector and scalar paths.
  size_t width_{2 * test::Options::vector_lanes<ElementType>() + 1};
};

template <typename ElementType, int kChannels>
static void test_not_implemented(
    kleidicv_error_t expected = KLEIDICV_ERROR_NOT_IMPLEMENTED) {
  const size_t width = 1, height = 1;
  ElementType src_arrays[kChannels][width * height] = {{234}};
  ElementType dst[kChannels * width * height] = {123};
  size_t src_strides[kChannels];
  const void* srcs[kChannels];
  for (int i = 0; i < kChannels; ++i) {
    srcs[i] = src_arrays[i];
    src_strides[i] = width * sizeof(ElementType);
  }
  size_t dst_stride = kChannels * width * sizeof(ElementType);

  ASSERT_EQ(expected, kleidicv_merge(srcs, src_strides, dst, dst_stride, width,
                                     height, kChannels, sizeof(ElementType)));

  // Destination should not be modified.
  EXPECT_EQ(123, dst[0]);
}

template <typename ElementType>
class Merge : public testing::Test {};

using ElementTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(Merge, ElementTypes);

TYPED_TEST(Merge, TwoChannels) {
  MergeTest<TypeParam, 2>().test();
  MergeTest<TypeParam, 2>().with_paddings({0, 0}, 1).test();
  MergeTest<TypeParam, 2>().with_paddings({0, 1}, 0).test();
  MergeTest<TypeParam, 2>().with_paddings({0, 1}, 1).test();
  MergeTest<TypeParam, 2>().with_paddings({1, 0}, 0).test();
  MergeTest<TypeParam, 2>().with_paddings({1, 0}, 1).test();
  MergeTest<TypeParam, 2>().with_paddings({1, 1}, 0).test();
  MergeTest<TypeParam, 2>().with_paddings({1, 1}, 1).test();
  MergeTest<TypeParam, 2>()
      .with_width(4 * test::Options::vector_lanes<TypeParam>())
      .test();
}

TYPED_TEST(Merge, ThreeChannels) {
  MergeTest<TypeParam, 3>().test();
  MergeTest<TypeParam, 3>().with_paddings({0, 0, 0}, 1).test();
  MergeTest<TypeParam, 3>().with_paddings({0, 0, 1}, 0).test();
  MergeTest<TypeParam, 3>().with_paddings({0, 0, 1}, 1).test();
  MergeTest<TypeParam, 3>().with_paddings({0, 1, 0}, 0).test();
  MergeTest<TypeParam, 3>().with_paddings({0, 1, 0}, 1).test();
  MergeTest<TypeParam, 3>().with_paddings({0, 1, 1}, 0).test();
  MergeTest<TypeParam, 3>().with_paddings({0, 1, 1}, 1).test();
  MergeTest<TypeParam, 3>().with_paddings({1, 0, 0}, 0).test();
  MergeTest<TypeParam, 3>().with_paddings({1, 0, 0}, 1).test();
  MergeTest<TypeParam, 3>().with_paddings({1, 0, 1}, 0).test();
  MergeTest<TypeParam, 3>().with_paddings({1, 0, 1}, 1).test();
  MergeTest<TypeParam, 3>().with_paddings({1, 1, 0}, 0).test();
  MergeTest<TypeParam, 3>().with_paddings({1, 1, 0}, 1).test();
  MergeTest<TypeParam, 3>().with_paddings({1, 1, 1}, 0).test();
  MergeTest<TypeParam, 3>().with_paddings({1, 1, 1}, 1).test();
  MergeTest<TypeParam, 3>()
      .with_width(4 * test::Options::vector_lanes<TypeParam>())
      .test();
}

TYPED_TEST(Merge, FourChannels) {
  MergeTest<TypeParam, 4>().test();
  MergeTest<TypeParam, 4>().with_paddings({0, 0, 0, 0}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 0, 0, 1}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 0, 0, 1}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 0, 1, 0}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 0, 1, 0}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 0, 1, 1}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 0, 1, 1}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 1, 0, 0}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 1, 0, 0}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 1, 0, 1}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 1, 0, 1}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 1, 1, 0}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 1, 1, 0}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 1, 1, 1}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({0, 1, 1, 1}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 0, 0, 0}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 0, 0, 0}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 0, 0, 1}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 0, 0, 1}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 0, 1, 0}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 0, 1, 0}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 0, 1, 1}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 0, 1, 1}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 1, 0, 0}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 1, 0, 0}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 1, 0, 1}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 1, 0, 1}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 1, 1, 0}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 1, 1, 0}, 1).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 1, 1, 1}, 0).test();
  MergeTest<TypeParam, 4>().with_paddings({1, 1, 1, 1}, 1).test();
  MergeTest<TypeParam, 4>()
      .with_width(4 * test::Options::vector_lanes<TypeParam>())
      .test();
}

TYPED_TEST(Merge, OneChannelOutOfRange) {
  test_not_implemented<TypeParam, 1>(KLEIDICV_ERROR_RANGE);
}

TYPED_TEST(Merge, FiveChannelsNotImplemented) {
  test_not_implemented<TypeParam, 5>();
}

TEST(Merge128, NotImplemented) { test_not_implemented<__uint128_t, 2>(); }

TYPED_TEST(Merge, NullPointer) {
  const size_t kChannels = 4;
  TypeParam src_arrays[kChannels];
  TypeParam dst[kChannels];
  size_t src_strides[kChannels] = {sizeof(TypeParam), sizeof(TypeParam),
                                   sizeof(TypeParam), sizeof(TypeParam)};
  const void* srcs[kChannels] = {src_arrays, src_arrays + 1, src_arrays + 2,
                                 src_arrays + 3};
  size_t dst_stride = kChannels * sizeof(TypeParam);
  test::test_null_args(kleidicv_merge, srcs, src_strides, dst, dst_stride, 1, 1,
                       kChannels, sizeof(TypeParam));

  for (int channels = 2; channels <= 4; ++channels) {
    for (int null_src = 0; null_src < channels; ++null_src) {
      for (int i = 0; i < channels; ++i) {
        srcs[i] = (i == null_src) ? nullptr : src_arrays + i;
      }
      EXPECT_EQ(KLEIDICV_ERROR_NULL_POINTER,
                kleidicv_merge(srcs, src_strides, dst, dst_stride, 1, 1,
                               channels, sizeof(TypeParam)));
    }
  }
}

TYPED_TEST(Merge, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }

  const size_t kChannels = 4;
  // A size comfortably large enough to hold the data, taking into account the
  // various offsets that this test will make.
  const size_t kBufSize = kChannels * sizeof(TypeParam) * 4;
  alignas(TypeParam) char src_arrays[kBufSize] = {};
  alignas(TypeParam) char dst[kBufSize] = {};
  size_t src_strides[kChannels] = {};
  const char* srcs[kChannels] = {};
  const size_t dst_stride = kChannels * sizeof(TypeParam);

  auto init = [&]() {
    for (size_t i = 0; i < kChannels; ++i) {
      srcs[i] = src_arrays + sizeof(TypeParam) * i;
      src_strides[i] = sizeof(TypeParam);
    }
  };

  auto check_merge = [&](int channels, void* dst_maybe_misaligned,
                         size_t dst_stride_maybe_misaligned) {
    EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
              kleidicv_merge(reinterpret_cast<const void**>(srcs), src_strides,
                             dst_maybe_misaligned, dst_stride_maybe_misaligned,
                             1, 2, channels, sizeof(TypeParam)));
  };

  for (size_t channels = 2; channels <= kChannels; ++channels) {
    init();

    // Misaligned destination pointer
    check_merge(channels, dst + 1, dst_stride);

    // Misaligned destination stride
    check_merge(channels, dst, dst_stride + 1);

    for (size_t misaligned_channel = 0; misaligned_channel < channels;
         ++misaligned_channel) {
      // Misaligned source pointer
      init();
      ++srcs[misaligned_channel];
      check_merge(channels, dst, dst_stride);

      // Misaligned source stride
      init();
      ++src_strides[misaligned_channel];
      check_merge(channels, dst, dst_stride);
    }
  }
}

TYPED_TEST(Merge, ZeroImageSize) {
  const size_t kChannels = 2;
  TypeParam src1[1] = {}, src2[1] = {}, dst[1];
  const void* srcs[kChannels] = {src1, src2};
  size_t src_strides[kChannels] = {sizeof(TypeParam), sizeof(TypeParam)};
  const size_t dst_stride = kChannels * sizeof(TypeParam);

  EXPECT_EQ(KLEIDICV_OK, kleidicv_merge(srcs, src_strides, dst, dst_stride, 0,
                                        1, kChannels, sizeof(TypeParam)));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_merge(srcs, src_strides, dst, dst_stride, 1,
                                        0, kChannels, sizeof(TypeParam)));
}

TYPED_TEST(Merge, OversizeImage) {
  const size_t kChannels = 2;
  TypeParam src1[1], src2[1], dst[1];
  const void* srcs[kChannels] = {src1, src2};
  size_t src_strides[kChannels] = {sizeof(TypeParam), sizeof(TypeParam)};
  const size_t dst_stride = kChannels * sizeof(TypeParam);

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_merge(srcs, src_strides, dst, dst_stride,
                           KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, kChannels,
                           sizeof(TypeParam)));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_merge(srcs, src_strides, dst, dst_stride,
                           KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS,
                           kChannels, sizeof(TypeParam)));
}
