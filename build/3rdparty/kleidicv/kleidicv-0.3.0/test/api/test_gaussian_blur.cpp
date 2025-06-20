// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/kernel.h"
#include "framework/utils.h"
#include "kleidicv/filters/sigma.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

#define KLEIDICV_GAUSSIAN_BLUR(type, type_suffix) \
  KLEIDICV_API(gaussian_blur, kleidicv_gaussian_blur_##type_suffix, type)

KLEIDICV_GAUSSIAN_BLUR(uint8_t, u8);

// Implements KernelTestParams for Gaussian Blur operators.
template <typename ElementType, size_t KernelSize>
struct GaussianBlurKernelTestParams;

template <size_t KernelSize>
struct GaussianBlurKernelTestParams<uint8_t, KernelSize> {
  using InputType = uint8_t;
  using IntermediateType = uint64_t;
  using OutputType = uint8_t;

  static constexpr size_t kKernelSize = KernelSize;
};  // end of struct GaussianBlurKernelTestParams<uint8_t, KernelSize>

static constexpr std::array<kleidicv_border_type_t, 1> kReplicateBorder = {
    KLEIDICV_BORDER_TYPE_REPLICATE};

static constexpr std::array<kleidicv_border_type_t, 4> kAllBorders = {
    KLEIDICV_BORDER_TYPE_REPLICATE,
    KLEIDICV_BORDER_TYPE_REFLECT,
    KLEIDICV_BORDER_TYPE_WRAP,
    KLEIDICV_BORDER_TYPE_REVERSE,
};

// Test for GaussianBlur operator.
template <class KernelTestParams,
          typename ArrayLayoutsGetterType = decltype(test::small_array_layouts),
          typename BorderContainerType = decltype(kAllBorders)>
class GaussianBlurTest : public test::KernelTest<KernelTestParams> {
  using Base = test::KernelTest<KernelTestParams>;
  using typename test::KernelTest<KernelTestParams>::InputType;
  using typename test::KernelTest<KernelTestParams>::IntermediateType;
  using typename test::KernelTest<KernelTestParams>::OutputType;
  using ArrayContainerType =
      std::invoke_result_t<ArrayLayoutsGetterType, size_t, size_t>;

 public:
  explicit GaussianBlurTest(
      KernelTestParams,
      ArrayLayoutsGetterType array_layouts_getter = test::small_array_layouts,
      BorderContainerType border_types = kAllBorders)
      : array_layouts_{array_layouts_getter(KernelTestParams::kKernelSize - 1,
                                            KernelTestParams::kKernelSize - 1)},
        border_types_{border_types},
        array_layout_generator_{array_layouts_},
        border_type_generator_{border_types_} {}

  void test(const test::Array2D<IntermediateType> &mask) {
    test::Kernel kernel{mask};
    // Create generators and execute test.
    test::SequenceGenerator tested_border_values{test::default_border_values()};
    test::PseudoRandomNumberGenerator<InputType> element_generator;
    Base::test(kernel, array_layout_generator_, border_type_generator_,
               tested_border_values, element_generator);
  }

 private:
  kleidicv_error_t call_api(const test::Array2D<InputType> *input,
                            test::Array2D<OutputType> *output,
                            kleidicv_border_type_t border_type,
                            kleidicv_border_values_t) override {
    kleidicv_filter_context_t *context = nullptr;
    auto ret = kleidicv_filter_context_create(
        &context, input->channels(), KernelTestParams::kKernelSize,
        KernelTestParams::kKernelSize, input->width() / input->channels(),
        input->height());
    if (ret != KLEIDICV_OK) {
      return ret;
    }

    ret = gaussian_blur<InputType>()(
        input->data(), input->stride(), output->data(), output->stride(),
        input->width() / input->channels(), input->height(), input->channels(),
        KernelTestParams::kKernelSize, KernelTestParams::kKernelSize, 0.0, 0.0,
        border_type, context);
    auto releaseRet = kleidicv_filter_context_release(context);
    if (releaseRet != KLEIDICV_OK) {
      return releaseRet;
    }

    return ret;
  }

  // Apply rounding to nearest integer division.
  IntermediateType scale_result(const test::Kernel<IntermediateType> &kernel,
                                IntermediateType result) override {
    // NOLINTBEGIN(readability-avoid-nested-conditional-operator)
    return kernel.width() == 3   ? ((result + 8) / 16)
           : kernel.width() == 5 ? ((result + 128) / 256)
           : kernel.width() == 7 ? ((result + 2048) / 4096)
                                 : ((result + 524288) / 1048576);
    // NOLINTEND(readability-avoid-nested-conditional-operator)
  }

  const ArrayContainerType array_layouts_;
  const BorderContainerType border_types_;
  test::SequenceGenerator<ArrayContainerType> array_layout_generator_;
  test::SequenceGenerator<BorderContainerType> border_type_generator_;
};  // end of class GaussianBlurTest<KernelTestParams, ArrayLayoutsGetterType,
    // BorderContainerType>

using ElementTypes = ::testing::Types<uint8_t>;

template <typename ElementType>
class GaussianBlur : public testing::Test {};

TYPED_TEST_SUITE(GaussianBlur, ElementTypes);

// Tests gaussian_blur_3x3_<input_type> API.
TYPED_TEST(GaussianBlur, 3x3Small) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 3>;

  // 3x3 GaussianBlur operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{3, 3};
  // clang-format off
  mask.set(0, 0, { 1, 2, 1});
  mask.set(1, 0, { 2, 4, 2});
  mask.set(2, 0, { 1, 2, 1});
  // clang-format on
  GaussianBlurTest{KernelTestParams{}}.test(mask);
}

TYPED_TEST(GaussianBlur, 3x3Default) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 3>;
  // 3x3 GaussianBlur operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{3, 3};
  // clang-format off
  mask.set(0, 0, { 1, 2, 1});
  mask.set(1, 0, { 2, 4, 2});
  mask.set(2, 0, { 1, 2, 1});
  // clang-format on
  GaussianBlurTest{KernelTestParams{}, test::default_array_layouts,
                   kReplicateBorder}
      .test(mask);
}

// Tests gaussian_blur_5x5_<input_type> API.
TYPED_TEST(GaussianBlur, 5x5) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 5>;
  // 5x5 GaussianBlur operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{5, 5};
  // clang-format off
  mask.set(0, 0, { 1,  4,  6,  4, 1});
  mask.set(1, 0, { 4, 16, 24, 16, 4});
  mask.set(2, 0, { 6, 24, 36, 24, 6});
  mask.set(3, 0, { 4, 16, 24, 16, 4});
  mask.set(4, 0, { 1,  4,  6,  4, 1});
  // clang-format on
  GaussianBlurTest{KernelTestParams{}}.test(mask);
}

// Tests gaussian_blur_7x7_<input_type> API.
TYPED_TEST(GaussianBlur, 7x7) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 7>;
  // 7x7 GaussianBlur operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{7, 7};
  // clang-format off
  mask.set(0, 0, {  4,  14,  28,  36,  28,  14,  4 });
  mask.set(1, 0, { 14,  49,  98, 126,  98,  49, 14 });
  mask.set(2, 0, { 28,  98, 196, 252, 196,  98, 28 });
  mask.set(3, 0, { 36, 126, 252, 324, 252, 126, 36 });
  mask.set(4, 0, { 28,  98, 196, 252, 196,  98, 28 });
  mask.set(5, 0, { 14,  49,  98, 126,  98,  49, 14 });
  mask.set(6, 0, {  4,  14,  28,  36,  28,  14,  4 });
  // clang-format on
  GaussianBlurTest{KernelTestParams{}}.test(mask);
}

// Tests gaussian_blur_15x15_<input_type> API.
TYPED_TEST(GaussianBlur, 15x15) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  // 15x15 GaussianBlur operator.
  test::Array2D<typename KernelTestParams::IntermediateType> mask{15, 15};
  // clang-format off
  mask.set(0, 0,  {  16,   44,  100,  192,   324,   472,   584,   632,   584,   472,   324,  192,  100,   44,  16 });
  mask.set(1, 0,  {  44,  121,  275,  528,   891,  1298,  1606,  1738,  1606,  1298,   891,  528,  275,  121,  44 });
  mask.set(2, 0,  { 100,  275,  625, 1200,  2025,  2950,  3650,  3950,  3650,  2950,  2025, 1200,  625,  275, 100 });
  mask.set(3, 0,  { 192,  528, 1200, 2304,  3888,  5664,  7008,  7584,  7008,  5664,  3888, 2304, 1200,  528, 192 });
  mask.set(4, 0,  { 324,  891, 2025, 3888,  6561,  9558, 11826, 12798, 11826,  9558,  6561, 3888, 2025,  891, 324 });
  mask.set(5, 0,  { 472, 1298, 2950, 5664,  9558, 13924, 17228, 18644, 17228, 13924,  9558, 5664, 2950, 1298, 472 });
  mask.set(6, 0,  { 584, 1606, 3650, 7008, 11826, 17228, 21316, 23068, 21316, 17228, 11826, 7008, 3650, 1606, 584 });
  mask.set(7, 0,  { 632, 1738, 3950, 7584, 12798, 18644, 23068, 24964, 23068, 18644, 12798, 7584, 3950, 1738, 632 });
  mask.set(8, 0,  { 584, 1606, 3650, 7008, 11826, 17228, 21316, 23068, 21316, 17228, 11826, 7008, 3650, 1606, 584 });
  mask.set(9, 0,  { 472, 1298, 2950, 5664,  9558, 13924, 17228, 18644, 17228, 13924,  9558, 5664, 2950, 1298, 472 });
  mask.set(10, 0, { 324,  891, 2025, 3888,  6561,  9558, 11826, 12798, 11826,  9558,  6561, 3888, 2025,  891, 324 });
  mask.set(11, 0, { 192,  528, 1200, 2304,  3888,  5664,  7008,  7584,  7008,  5664,  3888, 2304, 1200,  528, 192 });
  mask.set(12, 0, { 100,  275,  625, 1200,  2025,  2950,  3650,  3950,  3650,  2950,  2025, 1200,  625,  275, 100 });
  mask.set(13, 0, {  44,  121,  275,  528,   891,  1298,  1606,  1738,  1606,  1298,   891,  528,  275,  121,  44 });
  mask.set(14, 0, {  16,   44,  100,  192,   324,   472,   584,   632,   584,   472,   324,  192,  100,   44,  16 });
  // clang-format on
  GaussianBlurTest{KernelTestParams{}}.test(mask);
}

TYPED_TEST(GaussianBlur, 3x3_CustomSigma) {
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_filter_context_create(&context, 1, 15, 15, 18, 8));
  test::Array2D<TypeParam> src{18, 8, test::Options::vector_length()};
  // clang-format off
  src.set(0, 0, { 11, 22,  33,  44,  55,  66,  77,  88,  99, 111, 222, 33, 44, 55, 66, 77, 88, 99});
  src.set(1, 0, { 22, 33,  44,  55,  66,  77,  88,  99, 111, 222,  33, 44, 55, 66, 77, 88, 99, 11});
  src.set(2, 0, { 33, 44,  55,  66,  77,  88,  99, 111, 222,  33,  44, 55, 66, 77, 88, 99, 11, 22});
  src.set(3, 0, { 44, 55,  66,  77,  88,  99, 111, 222,  33,  44,  55, 66, 77, 88, 99, 11, 22, 33});
  src.set(4, 0, { 55, 66,  77,  88,  99, 111, 222,  33,  44,  55,  66, 77, 88, 99, 11, 22, 33, 44});
  src.set(5, 0, { 66, 77,  88,  99, 111, 222,  33,  44,  55,  66,  77, 88, 99, 11, 22, 33, 44, 55});
  src.set(6, 0, { 77, 88,  99, 111, 222,  33,  44,  55,  66,  77,  88, 99, 11, 22, 33, 44, 55, 66});
  src.set(7, 0, { 88, 99, 111, 222,  33,  44,  55,  66,  77,  88,  99, 11, 22, 33, 44, 55, 66, 77});
  // clang-format on

  test::Array2D<TypeParam> dst{18, 8, test::Options::vector_length()};
  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                       dst.stride(), 18, 8, 1, 3, 3, 4.56, 4.56,
                                       KLEIDICV_BORDER_TYPE_WRAP, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));

  test::Array2D<TypeParam> dst_expected{18, 8, test::Options::vector_length()};
  // clang-format off
  dst_expected.set(0, 0, { 51, 51,  73,  74,  73,  62,  73,  84, 107, 118, 96, 63, 40, 51, 62, 73, 73, 62});
  dst_expected.set(1, 0, { 33, 33,  44,  55,  66,  77,  88, 110, 122, 122, 89, 66, 55, 66, 77, 77, 66, 44});
  dst_expected.set(2, 0, { 33, 44,  55,  66,  77,  88, 110, 122, 122,  89, 66, 55, 66, 77, 77, 66, 44, 33});
  dst_expected.set(3, 0, { 44, 55,  66,  77,  88, 110, 122, 122,  89,  66, 55, 66, 77, 77, 66, 44, 33, 33});
  dst_expected.set(4, 0, { 55, 66,  77,  88, 110, 122, 122,  89,  66,  55, 66, 77, 77, 66, 44, 33, 33, 44});
  dst_expected.set(5, 0, { 66, 77,  88, 110, 122, 122,  89,  66,  55,  66, 77, 77, 66, 44, 33, 33, 44, 55});
  dst_expected.set(6, 0, { 77, 88, 110, 122, 122,  89,  66,  55,  66,  77, 77, 66, 44, 33, 33, 44, 55, 66});
  dst_expected.set(7, 0, { 70, 70,  92, 103,  92,  70,  59,  70,  81, 103, 92, 70, 37, 37, 48, 59, 70, 70});
  // clang-format on
  EXPECT_EQ_ARRAY2D(dst_expected, dst);
}

TYPED_TEST(GaussianBlur, 5x5_CustomSigma) {
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_filter_context_create(&context, 1, 15, 15, 20, 8));
  test::Array2D<TypeParam> src{20, 8, test::Options::vector_length()};
  // clang-format off
  src.set(0, 0, { 11, 22,  33,  44,  55,  66,  77,  88,  99, 111, 222, 33, 44, 55, 66, 77, 88, 99, 11, 22});
  src.set(1, 0, { 22, 33,  44,  55,  66,  77,  88,  99, 111, 222,  33, 44, 55, 66, 77, 88, 99, 11, 22, 33});
  src.set(2, 0, { 33, 44,  55,  66,  77,  88,  99, 111, 222,  33,  44, 55, 66, 77, 88, 99, 11, 22, 33, 44});
  src.set(3, 0, { 44, 55,  66,  77,  88,  99, 111, 222,  33,  44,  55, 66, 77, 88, 99, 11, 22, 33, 44, 55});
  src.set(4, 0, { 55, 66,  77,  88,  99, 111, 222,  33,  44,  55,  66, 77, 88, 99, 11, 22, 33, 44, 55, 66});
  src.set(5, 0, { 66, 77,  88,  99, 111, 222,  33,  44,  55,  66,  77, 88, 99, 11, 22, 33, 44, 55, 66, 77});
  src.set(6, 0, { 77, 88,  99, 111, 222,  33,  44,  55,  66,  77,  88, 99, 11, 22, 33, 44, 55, 66, 77, 88});
  src.set(7, 0, { 88, 99, 111, 222,  33,  44,  55,  66,  77,  88,  99, 11, 22, 33, 44, 55, 66, 77, 88, 99});
  // clang-format on

  test::Array2D<TypeParam> dst{20, 8, test::Options::vector_length()};
  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                       dst.stride(), 20, 8, 1, 5, 5, 4.56, 4.56,
                                       KLEIDICV_BORDER_TYPE_WRAP, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));

  test::Array2D<TypeParam> dst_expected{20, 8, test::Options::vector_length()};
  // clang-format off
  dst_expected.set(0, 0, { 54, 65, 72,  75,  78,  81,  84,  88,  96, 91, 82, 68, 59, 54, 58, 61, 60, 59, 54, 52});
  dst_expected.set(1, 0, { 48, 58, 61,  68,  75,  86,  90,  98, 101, 92, 79, 70, 64, 60, 64, 62, 57, 52, 47, 45});
  dst_expected.set(2, 0, { 42, 48, 55,  66,  81,  92, 100, 103, 102, 89, 80, 74, 70, 66, 65, 59, 51, 45, 40, 39});
  dst_expected.set(3, 0, { 53, 59, 66,  81,  92, 100, 103, 102,  89, 80, 74, 70, 66, 65, 59, 51, 45, 44, 43, 46});
  dst_expected.set(4, 0, { 64, 70, 81,  92, 100, 103, 102,  89,  80, 74, 70, 66, 65, 59, 51, 45, 44, 48, 51, 57});
  dst_expected.set(5, 0, { 75, 85, 92, 100, 103, 102,  89,  80,  74, 70, 66, 65, 59, 51, 45, 44, 48, 55, 62, 68});
  dst_expected.set(6, 0, { 69, 79, 87,  94,  97,  91,  82,  76,  79, 76, 75, 69, 60, 48, 46, 50, 53, 61, 63, 66});
  dst_expected.set(7, 0, { 62, 73, 80,  87,  86,  84,  79,  81,  86, 85, 80, 71, 58, 49, 52, 56, 59, 62, 61, 59});
  // clang-format on
  EXPECT_EQ_ARRAY2D(dst_expected, dst);
}

TYPED_TEST(GaussianBlur, 7x7_CustomSigma) {
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_filter_context_create(&context, 1, 15, 15, 23, 8));
  test::Array2D<TypeParam> src{23, 8, test::Options::vector_length()};
  // clang-format off
  src.set(0, 0, { 11, 22,  33,  44,  55,  66,  77,  88,  99, 111, 222, 33, 44, 55, 66, 77, 88, 99, 11, 22,  33,  44,  55});
  src.set(1, 0, { 22, 33,  44,  55,  66,  77,  88,  99, 111, 222,  33, 44, 55, 66, 77, 88, 99, 11, 22, 33,  44,  55,  66});
  src.set(2, 0, { 33, 44,  55,  66,  77,  88,  99, 111, 222,  33,  44, 55, 66, 77, 88, 99, 11, 22, 33, 44,  55,  66,  77});
  src.set(3, 0, { 44, 55,  66,  77,  88,  99, 111, 222,  33,  44,  55, 66, 77, 88, 99, 11, 22, 33, 44, 55,  66,  77,  88});
  src.set(4, 0, { 55, 66,  77,  88,  99, 111, 222,  33,  44,  55,  66, 77, 88, 99, 11, 22, 33, 44, 55, 66,  77,  88,  99});
  src.set(5, 0, { 66, 77,  88,  99, 111, 222,  33,  44,  55,  66,  77, 88, 99, 11, 22, 33, 44, 55, 66, 77,  88,  99, 111});
  src.set(6, 0, { 77, 88,  99, 111, 222,  33,  44,  55,  66,  77,  88, 99, 11, 22, 33, 44, 55, 66, 77, 88,  99, 111, 222});
  src.set(7, 0, { 88, 99, 111, 222,  33,  44,  55,  66,  77,  88,  99, 11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222,  33});
  // clang-format on

  test::Array2D<TypeParam> dst{23, 8, test::Options::vector_length()};
  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(src.data(), src.stride(), dst.data(),
                                       dst.stride(), 23, 8, 1, 7, 7, 4.56, 4.56,
                                       KLEIDICV_BORDER_TYPE_WRAP, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));

  test::Array2D<TypeParam> dst_expected{23, 8, test::Options::vector_length()};
  // clang-format off
  dst_expected.set(0, 0, { 76, 78, 77, 76, 82, 87, 90, 90, 85, 81, 77, 71, 65, 60, 56, 55, 56, 58, 62, 67, 68, 69, 71});
  dst_expected.set(1, 0, { 73, 75, 73, 75, 83, 88, 91, 91, 87, 83, 77, 72, 66, 61, 56, 55, 55, 56, 60, 65, 65, 66, 69});
  dst_expected.set(2, 0, { 69, 70, 72, 76, 84, 89, 92, 92, 89, 83, 78, 72, 67, 62, 57, 55, 54, 54, 58, 61, 61, 62, 65});
  dst_expected.set(3, 0, { 69, 73, 77, 78, 86, 91, 93, 93, 88, 83, 78, 72, 67, 61, 57, 53, 52, 52, 55, 62, 63, 64, 67});
  dst_expected.set(4, 0, { 82, 85, 85, 86, 91, 93, 93, 88, 83, 78, 72, 67, 61, 57, 53, 52, 52, 55, 62, 70, 72, 76, 78});
  dst_expected.set(5, 0, { 82, 85, 85, 85, 90, 92, 90, 88, 83, 77, 72, 67, 62, 58, 53, 52, 53, 56, 64, 72, 74, 76, 78});
  dst_expected.set(6, 0, { 81, 84, 83, 83, 87, 88, 89, 88, 83, 78, 73, 68, 64, 58, 54, 53, 54, 57, 65, 72, 73, 74, 77});
  dst_expected.set(7, 0, { 78, 81, 80, 80, 83, 87, 89, 88, 84, 79, 74, 70, 64, 59, 54, 54, 55, 58, 65, 70, 70, 72, 74});
  // clang-format on
  EXPECT_EQ_ARRAY2D(dst_expected, dst);
}

TYPED_TEST(GaussianBlur, 15x15_CustomSigma) {
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_filter_context_create(&context, 1, 15, 15, 40, 22));
  test::Array2D<TypeParam> src{40, 22, test::Options::vector_length()};
  // clang-format off
  src.set(0, 0,  {  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,
                    33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44});
  src.set(1, 0,  {  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,
                    44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55});
  src.set(2, 0,  {  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,
                    55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66});
  src.set(3, 0,  {  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,
                    66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77});
  src.set(4, 0,  {  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,
                    77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88});
  src.set(5, 0,  {  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,
                    88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99});
  src.set(6, 0,  {  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,
                    99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111});
  src.set(7, 0,  {  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99,
                   111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222});
  src.set(8, 0,  {  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111,
                   222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33});
  src.set(9, 0,  { 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,
                    33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44});
  src.set(10, 0, { 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,
                    44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55});
  src.set(11, 0, {  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,
                    55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66});
  src.set(12, 0, {  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,
                    66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77});
  src.set(13, 0, {  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,
                    77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88});
  src.set(14, 0, {  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,
                    88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99});
  src.set(15, 0, {  77,  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,
                    99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11});
  src.set(16, 0, {  88,  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,
                    11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22});
  src.set(17, 0, {  99,  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,
                    22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33});
  src.set(18, 0, {  11,  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,
                    33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44});
  src.set(19, 0, {  22,  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,
                    44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55});
  src.set(20, 0, {  33,  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,
                    55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66});
  src.set(21, 0, {  44,  55,  66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,
                    66,  77,  88,  99, 111, 222,  33,  44,  55,  66,  77,  88,  99,  11,  22,  33,  44,  55,  66,  77});
  // clang-format on

  test::Array2D<TypeParam> dst{40, 22, test::Options::vector_length()};
  EXPECT_EQ(KLEIDICV_OK,
            gaussian_blur<TypeParam>()(
                src.data(), src.stride(), dst.data(), dst.stride(), 40, 22, 1,
                15, 15, 4.56, 4.56, KLEIDICV_BORDER_TYPE_WRAP, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));

  test::Array2D<TypeParam> dst_expected{40, 22, test::Options::vector_length()};
  // clang-format off
  dst_expected.set(0, 0,  { 60, 63, 67, 71, 74, 77, 79, 81, 81, 81, 79, 77, 73, 70, 66, 64, 62, 61, 63, 65,
                            68, 71, 74, 76, 79, 81, 81, 81, 79, 77, 73, 70, 66, 63, 60, 57, 56, 56, 56, 58});
  dst_expected.set(1, 0,  { 63, 66, 70, 74, 77, 79, 80, 81, 81, 80, 78, 75, 71, 68, 65, 63, 62, 62, 64, 66,
                            69, 73, 75, 78, 79, 81, 81, 80, 78, 75, 71, 68, 65, 62, 59, 58, 57, 58, 59, 60});
  dst_expected.set(2, 0,  { 67, 70, 73, 76, 79, 80, 81, 80, 80, 78, 76, 73, 70, 67, 64, 63, 62, 63, 65, 68,
                            71, 74, 76, 78, 80, 80, 80, 78, 76, 73, 70, 67, 64, 62, 60, 59, 59, 60, 62, 64});
  dst_expected.set(3, 0,  { 71, 74, 76, 79, 80, 80, 80, 79, 78, 76, 74, 71, 68, 66, 64, 63, 63, 65, 67, 70,
                            73, 76, 77, 79, 79, 79, 78, 76, 74, 71, 68, 66, 64, 62, 61, 62, 62, 64, 66, 68});
  dst_expected.set(4, 0,  { 74, 77, 79, 80, 80, 80, 79, 78, 77, 74, 72, 69, 67, 65, 64, 64, 65, 67, 69, 72,
                            75, 77, 78, 79, 79, 78, 77, 74, 72, 69, 67, 65, 64, 63, 63, 64, 66, 68, 70, 72});
  dst_expected.set(5, 0,  { 77, 79, 80, 80, 80, 79, 78, 76, 74, 72, 70, 68, 66, 65, 65, 66, 67, 69, 71, 74,
                            76, 77, 78, 78, 77, 76, 74, 72, 70, 68, 66, 65, 65, 65, 66, 67, 69, 71, 73, 75});
  dst_expected.set(6, 0,  { 79, 80, 81, 80, 79, 78, 76, 74, 72, 70, 68, 67, 66, 66, 66, 67, 69, 71, 73, 75,
                            77, 77, 77, 77, 76, 74, 72, 70, 68, 67, 66, 66, 66, 67, 68, 70, 72, 74, 76, 78});
  dst_expected.set(7, 0,  { 81, 81, 80, 79, 78, 76, 74, 72, 70, 69, 67, 66, 66, 67, 68, 70, 71, 73, 75, 76,
                            77, 77, 76, 75, 74, 72, 70, 69, 67, 66, 66, 67, 68, 69, 71, 73, 75, 77, 79, 80});
  dst_expected.set(8, 0,  { 81, 81, 80, 78, 77, 74, 72, 70, 69, 67, 66, 66, 67, 68, 70, 71, 73, 75, 76, 77,
                            77, 76, 75, 74, 72, 70, 69, 67, 66, 66, 67, 68, 70, 71, 73, 76, 78, 79, 81, 81});
  dst_expected.set(9, 0,  { 81, 80, 78, 76, 74, 72, 70, 69, 67, 66, 66, 67, 68, 70, 71, 73, 75, 76, 77, 77,
                            76, 75, 74, 72, 70, 69, 67, 66, 66, 67, 68, 70, 71, 73, 76, 78, 79, 81, 81, 81});
  dst_expected.set(10, 0, { 79, 78, 76, 74, 72, 70, 68, 67, 66, 66, 67, 68, 70, 71, 73, 75, 76, 77, 77, 76,
                            75, 74, 72, 70, 69, 67, 66, 66, 67, 68, 70, 71, 73, 75, 77, 79, 80, 81, 81, 80});
  dst_expected.set(11, 0, { 77, 75, 73, 71, 69, 68, 67, 66, 66, 67, 68, 70, 71, 73, 75, 76, 77, 77, 76, 75,
                            74, 72, 70, 69, 67, 66, 66, 67, 68, 70, 71, 73, 75, 77, 78, 79, 80, 80, 79, 78});
  dst_expected.set(12, 0, { 73, 71, 70, 68, 67, 66, 66, 66, 67, 68, 70, 71, 73, 75, 76, 77, 77, 76, 75, 74,
                            72, 70, 69, 67, 66, 66, 67, 68, 70, 71, 73, 75, 76, 78, 78, 79, 78, 78, 76, 75});
  dst_expected.set(13, 0, { 70, 68, 67, 66, 65, 65, 66, 67, 68, 70, 71, 73, 75, 76, 77, 77, 76, 75, 74, 72,
                            70, 69, 67, 66, 66, 67, 68, 70, 71, 73, 75, 76, 77, 78, 78, 77, 76, 75, 73, 72});
  dst_expected.set(14, 0, { 66, 65, 64, 64, 64, 65, 66, 68, 70, 71, 73, 75, 76, 77, 77, 76, 75, 74, 72, 70,
                            69, 67, 66, 66, 67, 68, 70, 71, 73, 75, 76, 77, 77, 77, 76, 75, 74, 72, 70, 68});
  dst_expected.set(15, 0, { 63, 62, 62, 62, 63, 65, 67, 69, 71, 73, 75, 77, 78, 78, 77, 76, 74, 72, 70, 68,
                            67, 66, 66, 66, 68, 69, 71, 73, 75, 77, 78, 78, 77, 76, 74, 72, 70, 68, 66, 64});
  dst_expected.set(16, 0, { 60, 59, 60, 61, 63, 66, 68, 71, 73, 76, 77, 78, 78, 78, 76, 74, 72, 70, 68, 66,
                            66, 65, 66, 67, 69, 71, 73, 76, 77, 78, 78, 78, 76, 74, 72, 69, 67, 64, 62, 60});
  dst_expected.set(17, 0, { 57, 58, 59, 62, 64, 67, 70, 73, 76, 78, 79, 79, 79, 77, 75, 73, 70, 67, 66, 65,
                            65, 65, 66, 68, 70, 73, 76, 78, 79, 79, 79, 77, 75, 72, 69, 66, 63, 61, 59, 58});
  dst_expected.set(18, 0, { 56, 57, 59, 62, 66, 69, 72, 75, 78, 79, 80, 80, 78, 76, 74, 71, 68, 65, 64, 64,
                            64, 66, 67, 70, 72, 75, 78, 79, 80, 80, 78, 76, 74, 70, 67, 63, 60, 58, 57, 56});
  dst_expected.set(19, 0, { 56, 58, 60, 64, 68, 71, 74, 77, 79, 81, 81, 80, 78, 75, 72, 68, 65, 63, 63, 63,
                            64, 66, 69, 71, 74, 77, 79, 81, 81, 80, 78, 75, 72, 68, 64, 61, 58, 56, 55, 55});
  dst_expected.set(20, 0, { 56, 59, 62, 66, 70, 73, 76, 79, 81, 81, 81, 79, 76, 73, 70, 66, 64, 62, 62, 63,
                            65, 68, 70, 73, 76, 79, 81, 81, 81, 79, 76, 73, 70, 66, 62, 59, 57, 55, 55, 55});
  dst_expected.set(21, 0, { 58, 60, 64, 68, 72, 75, 78, 80, 81, 81, 80, 78, 75, 72, 68, 65, 63, 62, 62, 64,
                            66, 69, 72, 75, 78, 80, 81, 81, 80, 78, 75, 72, 68, 64, 60, 58, 56, 55, 55, 55});
  // clang-format on
  EXPECT_EQ_ARRAY2D(dst_expected, dst);
}

TYPED_TEST(GaussianBlur, UnsupportedBorderType3x3) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 3>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 3, 3,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              gaussian_blur<TypeParam>()(
                  src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                  validSize, 1, 3, 3, 0.0, 0.0, border, context));
  }
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, UnsupportedBorderType5x5) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 5>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 5, 5,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              gaussian_blur<TypeParam>()(
                  src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                  validSize, 1, 5, 5, 0.0, 0.0, border, context));
  }
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, UnsupportedBorderType7x7) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 7>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 7, 7,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              gaussian_blur<TypeParam>()(
                  src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                  validSize, 1, 7, 7, 0.0, 0.0, border, context));
  }
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, UnsupportedBorderType15x15) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];
  for (kleidicv_border_type_t border : {
           KLEIDICV_BORDER_TYPE_CONSTANT,
           KLEIDICV_BORDER_TYPE_TRANSPARENT,
           KLEIDICV_BORDER_TYPE_NONE,
       }) {
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              gaussian_blur<TypeParam>()(
                  src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                  validSize, 1, 15, 15, 0.0, 0.0, border, context));
  }
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, DifferentKernelSize) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 5, 15, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));

  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, NonZeroSigma) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 15, 15, 1.23, 4.56,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 15, 15, 1.23, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 15, 15, 0.0, 4.56,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));

  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, UnsupportedKernelSize) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 33, 33, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));

  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, NullPointer) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];
  test::test_null_args(gaussian_blur<TypeParam>(), src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), validSize, validSize, 1, 3, 3, 0.0,
                       0.0, KLEIDICV_BORDER_TYPE_REPLICATE, context);
  test::test_null_args(gaussian_blur<TypeParam>(), src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), validSize, validSize, 1, 5, 5, 0.0,
                       0.0, KLEIDICV_BORDER_TYPE_REPLICATE, context);
  test::test_null_args(gaussian_blur<TypeParam>(), src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), validSize, validSize, 1, 7, 7, 0.0,
                       0.0, KLEIDICV_BORDER_TYPE_REPLICATE, context);
  test::test_null_args(gaussian_blur<TypeParam>(), src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), validSize, validSize, 1, 15, 15, 0.0,
                       0.0, KLEIDICV_BORDER_TYPE_REPLICATE, context);
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];

  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam) + 1, dst, sizeof(TypeParam), validSize,
                validSize, 1, 3, 3, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE,
                context));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam) + 1, validSize,
                validSize, 1, 3, 3, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE,
                context));

  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam) + 1, dst, sizeof(TypeParam), validSize,
                validSize, 1, 5, 5, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE,
                context));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam) + 1, validSize,
                validSize, 1, 5, 5, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE,
                context));

  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam) + 1, dst, sizeof(TypeParam), validSize,
                validSize, 1, 7, 7, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE,
                context));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam) + 1, validSize,
                validSize, 1, 7, 7, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE,
                context));

  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam) + 1, dst, sizeof(TypeParam), validSize,
                validSize, 1, 15, 15, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE,
                context));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam) + 1, validSize,
                validSize, 1, 15, 15, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REPLICATE,
                context));

  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, ZeroImageSize3x3) {
  TypeParam src[1] = {}, dst[1];
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_filter_context_create(&context, 1, 3, 3, 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 0, 3, 1, 3, 3,
                0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 3, 0, 1, 3, 3,
                0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, ZeroImageSize5x5) {
  TypeParam src[1] = {}, dst[1];
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_filter_context_create(&context, 1, 5, 5, 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 0, 5, 1, 5, 5,
                0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 5, 0, 1, 5, 5,
                0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, ZeroImageSize7x7) {
  TypeParam src[1] = {}, dst[1];
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_filter_context_create(&context, 1, 7, 7, 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 0, 7, 1, 7, 7,
                0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 7, 0, 1, 7, 7,
                0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, ZeroImageSize15x15) {
  TypeParam src[1] = {}, dst[1];
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_filter_context_create(&context, 1, 15, 15, 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 0, 15, 1, 15,
                15, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), 15, 0, 1, 15,
                15, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, ValidImageSize3x3) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 3>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 3, 3,
                                                        validSize, validSize));
  test::Array2D<TypeParam> src{validSize, validSize,
                               test::Options::vector_length()};
  src.set(0, 0, {1, 2});
  src.set(1, 0, {4, 3});

  test::Array2D<TypeParam> dst{validSize, validSize,
                               test::Options::vector_length()};
  test::Array2D<TypeParam> expected{validSize, validSize,
                                    test::Options::vector_length()};

  EXPECT_EQ(KLEIDICV_OK, gaussian_blur<TypeParam>()(
                             src.data(), src.stride(), dst.data(), dst.stride(),
                             validSize, validSize, 1, 3, 3, 0.0, 0.0,
                             KLEIDICV_BORDER_TYPE_REVERSE, context));
  expected.set(0, 0, {3, 3});
  expected.set(1, 0, {3, 3});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK, gaussian_blur<TypeParam>()(
                             src.data(), src.stride(), dst.data(), dst.stride(),
                             validSize, validSize, 1, 3, 3, 2.25, 2.25,
                             KLEIDICV_BORDER_TYPE_REVERSE, context));
  expected.set(0, 0, {3, 3});
  expected.set(1, 0, {2, 2});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, ValidImageSize5x5) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 5>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 5, 5,
                                                        validSize, validSize));
  test::Array2D<TypeParam> src{validSize, validSize,
                               test::Options::vector_length()};
  src.set(0, 0, {1, 2, 3, 4});
  src.set(1, 0, {8, 7, 6, 5});
  src.set(2, 0, {1, 2, 3, 4});
  src.set(3, 0, {16, 27, 38, 49});

  test::Array2D<TypeParam> dst{validSize, validSize,
                               test::Options::vector_length()};
  test::Array2D<TypeParam> expected{validSize, validSize,
                                    test::Options::vector_length()};

  EXPECT_EQ(KLEIDICV_OK, gaussian_blur<TypeParam>()(
                             src.data(), src.stride(), dst.data(), dst.stride(),
                             validSize, validSize, 1, 5, 5, 0.0, 0.0,
                             KLEIDICV_BORDER_TYPE_REVERSE, context));
  expected.set(0, 0, {5, 5, 5, 5});
  expected.set(1, 0, {6, 6, 6, 7});
  expected.set(2, 0, {9, 10, 12, 13});
  expected.set(3, 0, {11, 13, 16, 18});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK, gaussian_blur<TypeParam>()(
                             src.data(), src.stride(), dst.data(), dst.stride(),
                             validSize, validSize, 1, 5, 5, 2.25, 2.25,
                             KLEIDICV_BORDER_TYPE_REVERSE, context));
  expected.set(0, 0, {4, 4, 4, 4});
  expected.set(1, 0, {8, 9, 9, 10});
  expected.set(2, 0, {9, 9, 10, 11});
  expected.set(3, 0, {10, 11, 12, 12});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, ValidImageSize7x7) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 7>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 7, 7,
                                                        validSize, validSize));
  test::Array2D<TypeParam> src{validSize, validSize,
                               test::Options::vector_length()};
  src.set(0, 0, {1, 2, 3, 4, 5, 6});
  src.set(1, 0, {12, 11, 10, 9, 8, 7});
  src.set(2, 0, {1, 2, 3, 4, 5, 6});
  src.set(3, 0, {1, 2, 3, 4, 5, 6});
  src.set(4, 0, {11, 22, 33, 44, 55, 66});
  src.set(5, 0, {127, 67, 37, 27, 17, 7});

  test::Array2D<TypeParam> dst{validSize, validSize,
                               test::Options::vector_length()};
  test::Array2D<TypeParam> expected{validSize, validSize,
                                    test::Options::vector_length()};

  EXPECT_EQ(KLEIDICV_OK, gaussian_blur<TypeParam>()(
                             src.data(), src.stride(), dst.data(), dst.stride(),
                             validSize, validSize, 1, 7, 7, 0.0, 0.0,
                             KLEIDICV_BORDER_TYPE_REVERSE, context));
  expected.set(0, 0, {6, 6, 6, 6, 6, 6});
  expected.set(1, 0, {6, 6, 7, 7, 8, 8});
  expected.set(2, 0, {9, 9, 10, 10, 11, 12});
  expected.set(3, 0, {16, 16, 16, 17, 18, 19});
  expected.set(4, 0, {26, 26, 25, 26, 27, 27});
  expected.set(5, 0, {32, 31, 29, 29, 30, 30});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK, gaussian_blur<TypeParam>()(
                             src.data(), src.stride(), dst.data(), dst.stride(),
                             validSize, validSize, 1, 7, 7, 2.25, 2.25,
                             KLEIDICV_BORDER_TYPE_REVERSE, context));
  expected.set(0, 0, {5, 5, 6, 6, 6, 6});
  expected.set(1, 0, {7, 7, 8, 9, 9, 10});
  expected.set(2, 0, {13, 13, 13, 13, 13, 13});
  expected.set(3, 0, {18, 18, 19, 19, 19, 19});
  expected.set(4, 0, {22, 22, 23, 23, 23, 23});
  expected.set(5, 0, {24, 24, 24, 24, 24, 24});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, ValidImageSize15x15) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  test::Array2D<TypeParam> src{validSize, validSize,
                               test::Options::vector_length()};
  src.set(0, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  src.set(1, 0, {28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15});
  src.set(2, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  src.set(3, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  src.set(4, 0, {28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15});
  src.set(5, 0, {28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15});
  src.set(6, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  src.set(7, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  src.set(8, 0, {28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15});
  src.set(9, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  src.set(10, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  src.set(11, 0, {247, 207, 167, 127, 87, 47, 7, 3, 7, 47, 87, 127, 167, 207});
  src.set(12, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  src.set(
      13, 0,
      {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242});

  test::Array2D<TypeParam> dst{validSize, validSize,
                               test::Options::vector_length()};
  test::Array2D<TypeParam> expected{validSize, validSize,
                                    test::Options::vector_length()};

  EXPECT_EQ(KLEIDICV_OK, gaussian_blur<TypeParam>()(
                             src.data(), src.stride(), dst.data(), dst.stride(),
                             validSize, validSize, 1, 15, 15, 0.0, 0.0,
                             KLEIDICV_BORDER_TYPE_REVERSE, context));
  expected.set(0, 0, {13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14});
  expected.set(1, 0, {13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14});
  expected.set(2, 0, {13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14});
  expected.set(3, 0, {13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14});
  expected.set(4, 0, {14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15});
  expected.set(5, 0, {15, 15, 15, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15});
  expected.set(6, 0, {17, 17, 17, 16, 16, 15, 15, 15, 16, 16, 17, 17, 18, 18});
  expected.set(7, 0, {21, 21, 20, 20, 19, 18, 17, 17, 18, 19, 20, 21, 21, 22});
  expected.set(8, 0, {29, 29, 28, 26, 24, 22, 21, 21, 22, 23, 25, 27, 28, 29});
  expected.set(9, 0, {40, 40, 38, 35, 32, 30, 28, 28, 29, 31, 33, 36, 38, 38});
  expected.set(10, 0, {54, 53, 50, 47, 43, 39, 37, 36, 38, 40, 44, 47, 49, 50});
  expected.set(11, 0, {67, 66, 63, 58, 54, 50, 47, 46, 47, 50, 54, 58, 61, 62});
  expected.set(12, 0, {76, 75, 72, 67, 62, 57, 54, 53, 54, 58, 62, 67, 70, 71});
  expected.set(13, 0, {80, 79, 76, 71, 65, 60, 57, 56, 57, 61, 66, 70, 73, 75});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK, gaussian_blur<TypeParam>()(
                             src.data(), src.stride(), dst.data(), dst.stride(),
                             validSize, validSize, 1, 15, 15, 2.25, 2.25,
                             KLEIDICV_BORDER_TYPE_REVERSE, context));
  expected.set(0, 0, {13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14});
  expected.set(1, 0, {13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14});
  expected.set(2, 0, {13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14});
  expected.set(3, 0, {13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14});
  expected.set(4, 0, {14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14});
  expected.set(5, 0, {15, 15, 15, 15, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15});
  expected.set(6, 0, {15, 15, 15, 15, 14, 14, 14, 14, 14, 15, 15, 16, 16, 16});
  expected.set(7, 0, {19, 19, 19, 18, 17, 16, 16, 16, 16, 17, 18, 19, 20, 20});
  expected.set(8, 0, {26, 26, 25, 23, 21, 19, 18, 18, 19, 20, 22, 24, 26, 26});
  expected.set(9, 0, {38, 37, 35, 32, 29, 26, 24, 24, 25, 27, 30, 34, 36, 37});
  expected.set(10, 0, {55, 54, 51, 46, 42, 37, 35, 34, 35, 39, 43, 47, 51, 52});
  expected.set(11, 0, {71, 70, 66, 61, 55, 49, 46, 45, 47, 51, 56, 61, 65, 67});
  expected.set(12, 0, {85, 83, 79, 73, 66, 60, 57, 55, 57, 61, 67, 73, 77, 79});
  expected.set(13, 0, {89, 88, 83, 77, 71, 65, 61, 60, 62, 66, 72, 77, 82, 83});
  EXPECT_EQ_ARRAY2D(expected, dst);

  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, UndersizeImage3x3) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 3>;
  kleidicv_filter_context_t *context = nullptr;
  size_t underSize = KernelTestParams::kKernelSize - 2;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 3, 3,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 underSize, underSize, 1, 3, 3, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 underSize, validSize, 1, 3, 3, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, underSize, 1, 3, 3, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, UndersizeImage5x5) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 5>;
  kleidicv_filter_context_t *context = nullptr;
  size_t underSize = KernelTestParams::kKernelSize - 2;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 5, 5,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 underSize, underSize, 1, 5, 5, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 underSize, validSize, 1, 5, 5, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, underSize, 1, 5, 5, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, UndersizeImage7x7) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 7>;
  kleidicv_filter_context_t *context = nullptr;
  size_t underSize = KernelTestParams::kKernelSize - 2;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 7, 7,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 underSize, underSize, 1, 7, 7, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 underSize, validSize, 1, 7, 7, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, underSize, 1, 7, 7, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, UndersizeImage15x15) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t underSize = KernelTestParams::kKernelSize - 2;
  size_t validSize = KernelTestParams::kKernelSize - 1;
  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 underSize, underSize, 1, 15, 15, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 underSize, validSize, 1, 15, 15, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, underSize, 1, 15, 15, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REPLICATE, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, OversizeImage) {
  kleidicv_filter_context_t *context = nullptr;
  ASSERT_EQ(KLEIDICV_OK,
            kleidicv_filter_context_create(&context, 1, 15, 15, 1, 1));
  TypeParam src[1], dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS + 1, 15, 1, 15, 15, 0.0, 0.0,
                KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam),
                KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, 15, 15,
                15, 0.0, 0.0, KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, ChannelNumber) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;

  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  TypeParam src[1], dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), validSize,
                validSize, KLEIDICV_MAXIMUM_CHANNEL_COUNT + 1, 15, 15, 0.0, 0.0,
                KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, InvalidContextMaxChannels) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;

  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  TypeParam src[1], dst[1];
  EXPECT_EQ(
      KLEIDICV_ERROR_CONTEXT_MISMATCH,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 2, 15, 15, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, InvalidContextImageSize) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;

  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  TypeParam src[1], dst[1];
  EXPECT_EQ(
      KLEIDICV_ERROR_CONTEXT_MISMATCH,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize + 1, validSize, 1, 15, 15, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_CONTEXT_MISMATCH,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize + 1, 1, 15, 15, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_CONTEXT_MISMATCH,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize + 1, validSize + 1, 1, 15, 15, 0.0,
                                 0.0, KLEIDICV_BORDER_TYPE_REFLECT, context));

  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, InvalidKernelSize) {
  kleidicv_filter_context_t *context = nullptr;
  size_t kernel_size = 17;

  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(
                             &context, 1, 15, 15, kernel_size, kernel_size));
  TypeParam src[1], dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), kernel_size,
                kernel_size, 1, kernel_size, kernel_size, 0.0, 0.0,
                KLEIDICV_BORDER_TYPE_REFLECT, context));
  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            gaussian_blur<TypeParam>()(
                src, sizeof(TypeParam), dst, sizeof(TypeParam), kernel_size,
                kernel_size, 1, kernel_size, kernel_size, 1.0, 1.0,
                KLEIDICV_BORDER_TYPE_REFLECT, context));

  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

TYPED_TEST(GaussianBlur, InvalidBorderType) {
  using KernelTestParams = GaussianBlurKernelTestParams<TypeParam, 15>;
  kleidicv_filter_context_t *context = nullptr;
  size_t validSize = KernelTestParams::kKernelSize - 1;

  ASSERT_EQ(KLEIDICV_OK, kleidicv_filter_context_create(&context, 1, 15, 15,
                                                        validSize, validSize));
  TypeParam src[1], dst[1];
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 15, 15, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_CONSTANT, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 15, 15, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_TRANSPARENT, context));
  EXPECT_EQ(
      KLEIDICV_ERROR_NOT_IMPLEMENTED,
      gaussian_blur<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                                 validSize, validSize, 1, 15, 15, 0.0, 0.0,
                                 KLEIDICV_BORDER_TYPE_NONE, context));
  EXPECT_EQ(KLEIDICV_OK, kleidicv_filter_context_release(context));
}

template <size_t Size>
static std::array<uint16_t, Size> generate_reference_kernel(float sigma) {
  std::array<float, Size> float_kernel{};

  for (size_t i = 0; i < Size; ++i) {
    float_kernel[i] = std::exp(-1 * std::pow(i, 2) / (2 * std::pow(sigma, 2)));
  }

  float sum = 0;
  for (auto val : float_kernel) {
    sum += val;
  }

  // Sum needs to be corrected to contain all kernel values
  sum = (sum * 2) - 1;

  for (auto &val : float_kernel) {
    val = val / sum;
    // Multiplication is needed as the results are fixed point values on
    // uint16_t type
    val *= 256;
  }

  std::array<uint16_t, Size> kernel_to_return{};
  // Conversion with rounding error diffusion
  float last_rounding_error = 0.0;
  for (size_t i = 0; i < Size; ++i) {
    float corrected_value = float_kernel[Size - 1 - i] - last_rounding_error;
    float rounded_value = std::round(corrected_value);
    last_rounding_error = rounded_value - corrected_value;
    kernel_to_return[i] = rounded_value;
  }

  return kernel_to_return;
}
template <size_t Size>
void test_sigma() {
  const std::array<uint16_t, Size> expected_half_kernel =
      generate_reference_kernel<Size>(3.0);
  const std::array<uint16_t, Size> actual_half_kernel =
      kleidicv::generate_gaussian_half_kernel<Size>(3.0);

  EXPECT_EQ(expected_half_kernel, actual_half_kernel);

  const std::array<uint16_t, Size> expected_half_kernel1 =
      generate_reference_kernel<Size>(((Size * 2) - 1) * 0.15 + 0.35);
  const std::array<uint16_t, Size> actual_half_kernel1 =
      kleidicv::generate_gaussian_half_kernel<Size>(0.0);

  EXPECT_EQ(expected_half_kernel1, actual_half_kernel1);
}

TYPED_TEST(GaussianBlur, KernelGenerationFromSigma) {
  test_sigma<2>();
  test_sigma<3>();
  test_sigma<4>();
  test_sigma<8>();
}
