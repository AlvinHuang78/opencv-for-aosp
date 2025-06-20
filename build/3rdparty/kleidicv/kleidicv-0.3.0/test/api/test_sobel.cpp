// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/kernel.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

#define KLEIDICV_SOBEL_3X3_HORIZONTAL(type, suffix)                          \
  KLEIDICV_API(sobel_3x3_horizontal, kleidicv_sobel_3x3_horizontal_##suffix, \
               type)

#define KLEIDICV_SOBEL_3X3_VERTICAL(type, suffix) \
  KLEIDICV_API(sobel_3x3_vertical, kleidicv_sobel_3x3_vertical_##suffix, type)

KLEIDICV_SOBEL_3X3_HORIZONTAL(uint8_t, s16_u8);

KLEIDICV_SOBEL_3X3_VERTICAL(uint8_t, s16_u8);

// Implements KernelTestParams for Sobel operators.
template <typename ElementType, bool IsHorizontal>
struct SobelKernelTestParams;

template <bool IsHorizontal>
struct SobelKernelTestParams<uint8_t, IsHorizontal> {
  using InputType = uint8_t;
  using IntermediateType = int16_t;
  using OutputType = int16_t;

  static constexpr bool kIsHorizontal = IsHorizontal;
};  // end of struct SobelKernelTestParams<uint8_t, IsHorizontal>

static constexpr std::array<kleidicv_border_type_t, 1> kSupportedBorders = {
    KLEIDICV_BORDER_TYPE_REPLICATE,
};

// Test for Sobel 3x3 operator.
template <class KernelTestParams>
class Sobel3x3Test : public test::KernelTest<KernelTestParams> {
  using Base = test::KernelTest<KernelTestParams>;
  using typename Base::InputType;
  using typename Base::IntermediateType;
  using typename Base::OutputType;

  kleidicv_error_t call_api(const test::Array2D<InputType> *input,
                            test::Array2D<OutputType> *output,
                            kleidicv_border_type_t,
                            kleidicv_border_values_t) override {
    auto api = KernelTestParams::kIsHorizontal
                   ? sobel_3x3_horizontal<InputType>()
                   : sobel_3x3_vertical<InputType>();
    return api(input->data(), input->stride(), output->data(), output->stride(),
               input->width() / input->channels(), input->height(),
               input->channels());
  }

 public:
  void test(const test::Array2D<IntermediateType> &mask) {
    test::Kernel kernel{mask};
    // Use the default array layouts for testing.
    auto array_layouts =
        test::default_array_layouts(mask.width() - 1, mask.height() - 1);
    // Create generators and execute test.
    test::SequenceGenerator tested_array_layouts{array_layouts};
    test::SequenceGenerator tested_borders{kSupportedBorders};
    test::SequenceGenerator tested_border_values{test::default_border_values()};
    test::PseudoRandomNumberGenerator<InputType> element_generator;
    Base::test(kernel, tested_array_layouts, tested_borders,
               tested_border_values, element_generator);
  }
};  // end of class Sobel3x3Test<KernelTestParams>

template <typename TypeParam>
class Sobel : public testing::Test {};

using ElementTypes = ::testing::Types<uint8_t>;
TYPED_TEST_SUITE(Sobel, ElementTypes);

// Tests sobel_3x3_horizontal_<output_type>_<input_type> API.
TYPED_TEST(Sobel, Horizontal3x3) {
  using KernelTestParams = SobelKernelTestParams<TypeParam, true>;
  // Horizontal 3x3 Sobel operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{3, 3};
  mask.set(0, 0, {-1, 0, 1});
  mask.set(1, 0, {-2, 0, 2});
  mask.set(2, 0, {-1, 0, 1});
  Sobel3x3Test<KernelTestParams>{}.test(mask);

  typename KernelTestParams::InputType src[1] = {};
  typename KernelTestParams::OutputType dst[1];
  test::test_null_args(sobel_3x3_horizontal<TypeParam>(), src, sizeof(src), dst,
                       sizeof(dst), 3, 3, 1);
}

// Tests sobel_3x3_vertical_<output_type>_<input_type> API.
TYPED_TEST(Sobel, Vertical3x3) {
  using KernelTestParams = SobelKernelTestParams<TypeParam, false>;
  // Vertical 3x3 Sobel operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{3, 3};
  // clang-format off
  mask.set(0, 0, {-1, -2, -1});
  mask.set(1, 0, { 0,  0,  0});
  mask.set(2, 0, { 1,  2,  1});
  // clang-format on
  Sobel3x3Test<KernelTestParams>{}.test(mask);

  typename KernelTestParams::InputType src[1] = {};
  typename KernelTestParams::OutputType dst[1];
  test::test_null_args(sobel_3x3_vertical<TypeParam>(), src, sizeof(src), dst,
                       sizeof(dst), 3, 3, 1);
}

TYPED_TEST(Sobel, MisalignmentHorizontal) {
  using KernelTestParams = SobelKernelTestParams<TypeParam, true>;
  typename KernelTestParams::InputType src[2] = {};
  typename KernelTestParams::OutputType dst[2];
  if (sizeof(typename KernelTestParams::InputType) != 1) {
    EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
              sobel_3x3_horizontal<TypeParam>()(src, sizeof(src) + 1, dst,
                                                sizeof(dst), 3, 3, 1));
  }
  if (sizeof(typename KernelTestParams::OutputType) != 1) {
    EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
              sobel_3x3_horizontal<TypeParam>()(src, sizeof(src), dst,
                                                sizeof(dst) + 1, 3, 3, 1));
  }
}

TYPED_TEST(Sobel, MisalignmentVertical) {
  using KernelTestParams = SobelKernelTestParams<TypeParam, false>;
  typename KernelTestParams::InputType src[2] = {};
  typename KernelTestParams::OutputType dst[2];
  if (sizeof(typename KernelTestParams::InputType) != 1) {
    EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
              sobel_3x3_vertical<TypeParam>()(src, sizeof(src) + 1, dst,
                                              sizeof(dst), 3, 3, 1));
  }
  if (sizeof(typename KernelTestParams::OutputType) != 1) {
    EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
              sobel_3x3_vertical<TypeParam>()(src, sizeof(src), dst,
                                              sizeof(dst) + 1, 3, 3, 1));
  }
}

TYPED_TEST(Sobel, ZeroImageSizeHorizontal) {
  using KernelTestParams = SobelKernelTestParams<TypeParam, true>;
  typename KernelTestParams::InputType src[1] = {};
  typename KernelTestParams::OutputType dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            sobel_3x3_horizontal<TypeParam>()(src, sizeof(src), dst,
                                              sizeof(dst), 0, 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            sobel_3x3_horizontal<TypeParam>()(src, sizeof(src), dst,
                                              sizeof(dst), 1, 0, 1));
}

TYPED_TEST(Sobel, ZeroImageSizeVertical) {
  using KernelTestParams = SobelKernelTestParams<TypeParam, false>;
  typename KernelTestParams::InputType src[1] = {};
  typename KernelTestParams::OutputType dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            sobel_3x3_vertical<TypeParam>()(src, sizeof(src), dst, sizeof(dst),
                                            0, 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            sobel_3x3_vertical<TypeParam>()(src, sizeof(src), dst, sizeof(dst),
                                            1, 0, 1));
}

TYPED_TEST(Sobel, ValidImageSize) {
  using KernelTestParams = SobelKernelTestParams<TypeParam, true>;
  size_t validSize = 2;

  test::Array2D<typename KernelTestParams::InputType> src{
      validSize, validSize, test::Options::vector_length()};
  src.set(0, 0, {1, 2});
  src.set(1, 0, {1, 2});

  test::Array2D<typename KernelTestParams::OutputType> dst{
      validSize, validSize, test::Options::vector_length()};
  EXPECT_EQ(KLEIDICV_OK, sobel_3x3_horizontal<TypeParam>()(
                             src.data(), src.stride(), dst.data(), dst.stride(),
                             validSize, validSize, 1));
  EXPECT_EQ(KLEIDICV_OK, sobel_3x3_vertical<TypeParam>()(
                             src.data(), src.stride(), dst.data(), dst.stride(),
                             validSize, validSize, 1));
}

TYPED_TEST(Sobel, UndersizeImage) {
  using KernelTestParams = SobelKernelTestParams<TypeParam, false>;
  typename KernelTestParams::InputType src[1] = {};
  typename KernelTestParams::OutputType dst[1];
  size_t underSize = 1;
  size_t validWidth = 13;
  size_t validHeight = 8;

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            sobel_3x3_horizontal<TypeParam>()(
                src, sizeof(src), dst, sizeof(dst), underSize, underSize, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            sobel_3x3_vertical<TypeParam>()(src, sizeof(src), dst, sizeof(dst),
                                            underSize, underSize, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            sobel_3x3_horizontal<TypeParam>()(
                src, sizeof(src), dst, sizeof(dst), underSize, validHeight, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            sobel_3x3_vertical<TypeParam>()(src, sizeof(src), dst, sizeof(dst),
                                            underSize, validHeight, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            sobel_3x3_horizontal<TypeParam>()(
                src, sizeof(src), dst, sizeof(dst), validWidth, underSize, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            sobel_3x3_vertical<TypeParam>()(src, sizeof(src), dst, sizeof(dst),
                                            validWidth, underSize, 1));
}

#ifdef KLEIDICV_ALLOCATION_TESTS
TYPED_TEST(Sobel, CannotAllocateImageHorizontal) {
  MockMallocToFail::enable();
  using KernelTestParams = SobelKernelTestParams<TypeParam, true>;
  typename KernelTestParams::InputType src[1] = {};
  typename KernelTestParams::OutputType dst[1];
  size_t validSize = 2;

  EXPECT_EQ(
      KLEIDICV_ERROR_ALLOCATION,
      sobel_3x3_horizontal<TypeParam>()(
          src, sizeof(src), dst, sizeof(dst), KLEIDICV_MAX_IMAGE_PIXELS / 2,
          validSize, KLEIDICV_MAXIMUM_CHANNEL_COUNT));

  EXPECT_EQ(
      KLEIDICV_ERROR_ALLOCATION,
      sobel_3x3_vertical<TypeParam>()(src, sizeof(src), dst, sizeof(dst),
                                      KLEIDICV_MAX_IMAGE_PIXELS / 2, validSize,
                                      KLEIDICV_MAXIMUM_CHANNEL_COUNT));
  MockMallocToFail::disable();
}
#endif

TYPED_TEST(Sobel, OversizeImageHorizontal) {
  using KernelTestParams = SobelKernelTestParams<TypeParam, true>;
  typename KernelTestParams::InputType src[1] = {};
  typename KernelTestParams::OutputType dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE, sobel_3x3_horizontal<TypeParam>()(
                                      src, sizeof(src), dst, sizeof(dst),
                                      KLEIDICV_MAX_IMAGE_PIXELS + 1, 3, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            sobel_3x3_horizontal<TypeParam>()(
                src, sizeof(src), dst, sizeof(dst), KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS, 1));
}

TYPED_TEST(Sobel, OversizeImageVertical) {
  using KernelTestParams = SobelKernelTestParams<TypeParam, false>;
  typename KernelTestParams::InputType src[1] = {};
  typename KernelTestParams::OutputType dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_RANGE, sobel_3x3_vertical<TypeParam>()(
                                      src, sizeof(src), dst, sizeof(dst),
                                      KLEIDICV_MAX_IMAGE_PIXELS + 1, 3, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            sobel_3x3_vertical<TypeParam>()(src, sizeof(src), dst, sizeof(dst),
                                            KLEIDICV_MAX_IMAGE_PIXELS,
                                            KLEIDICV_MAX_IMAGE_PIXELS, 1));
}

TYPED_TEST(Sobel, ChannelNumber) {
  using KernelTestParams = SobelKernelTestParams<TypeParam, true>;
  typename KernelTestParams::InputType src[1] = {};
  typename KernelTestParams::OutputType dst[1];
  size_t validSize = 2;

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            sobel_3x3_horizontal<TypeParam>()(
                src, sizeof(src), dst, sizeof(dst), validSize, validSize,
                KLEIDICV_MAXIMUM_CHANNEL_COUNT + 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            sobel_3x3_vertical<TypeParam>()(
                src, sizeof(src), dst, sizeof(dst), validSize, validSize,
                KLEIDICV_MAXIMUM_CHANNEL_COUNT + 1));
}
