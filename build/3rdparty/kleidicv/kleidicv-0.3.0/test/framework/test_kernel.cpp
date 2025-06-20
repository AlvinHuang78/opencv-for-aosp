// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>

#include "framework/kernel.h"
#include "framework/types.h"
#include "kleidicv/kleidicv.h"

// Tests that the constructor of test::Kernel<T> works for odd width and
// height.
TEST(Kernel, ConstructOdd) {
  using ElementType = uint8_t;

  test::Array2D<ElementType> mask{3, 3};
  mask.fill(0);
  mask.set(0, 0, {1, 2, 3});
  mask.set(2, 0, {4, 5, 6});

  test::Kernel<ElementType> kernel{mask};
  EXPECT_EQ(kernel.width(), 3);
  EXPECT_EQ(kernel.height(), 3);
  EXPECT_EQ(kernel.channels(), 1);
  EXPECT_EQ(kernel.anchor().x, 1);
  EXPECT_EQ(kernel.anchor().y, 1);

  EXPECT_EQ(kernel.top(), 1);
  EXPECT_EQ(kernel.left(), 1);
  EXPECT_EQ(kernel.bottom(), 1);
  EXPECT_EQ(kernel.right(), 1);

  EXPECT_EQ(kernel.at(0, 0)[0], 1);
  EXPECT_EQ(kernel.at(0, 1)[0], 2);
  EXPECT_EQ(kernel.at(0, 2)[0], 3);
  EXPECT_EQ(kernel.at(1, 0)[0], 0);
  EXPECT_EQ(kernel.at(1, 1)[0], 0);
  EXPECT_EQ(kernel.at(1, 2)[0], 0);
  EXPECT_EQ(kernel.at(2, 0)[0], 4);
  EXPECT_EQ(kernel.at(2, 1)[0], 5);
  EXPECT_EQ(kernel.at(2, 2)[0], 6);
}

// Tests that the constructor of test::Kernel<T> works for even width and
// height.
TEST(Kernel, ConstructEven) {
  using ElementType = uint8_t;

  test::Array2D<ElementType> mask{4, 4};
  test::Kernel<ElementType> kernel{mask};
  EXPECT_EQ(kernel.width(), 4);
  EXPECT_EQ(kernel.height(), 4);
  EXPECT_EQ(kernel.channels(), 1);
  EXPECT_EQ(kernel.anchor().x, 2);
  EXPECT_EQ(kernel.anchor().y, 2);

  EXPECT_EQ(kernel.top(), 2);
  EXPECT_EQ(kernel.left(), 2);
  EXPECT_EQ(kernel.bottom(), 1);
  EXPECT_EQ(kernel.right(), 1);
}

// Tests that the default constructor of test::Kernel<T> works with all
// zero arguments.
TEST(Kernel, ConstructEmpty) {
  using ElementType = uint8_t;

  test::Array2D<ElementType> mask{0, 0};
  test::Kernel<ElementType> kernel{mask};
  EXPECT_EQ(kernel.width(), 0);
  EXPECT_EQ(kernel.height(), 0);
  EXPECT_EQ(kernel.channels(), 1);
  EXPECT_EQ(kernel.anchor().x, 0);
  EXPECT_EQ(kernel.anchor().y, 0);
  EXPECT_EQ(kernel.top(), 0);
  EXPECT_EQ(kernel.left(), 0);
  EXPECT_EQ(kernel.bottom(), 0);
  EXPECT_EQ(kernel.right(), 0);
}

class KernelTestParams {
 public:
  using InputType = uint8_t;
  using IntermediateType = int32_t;
  using OutputType = int16_t;
};  // end of class KernelTestParams

template <class KernelTestParams>
class ExampleKernelTest : public test::KernelTest<KernelTestParams> {
  using InputType = typename KernelTestParams::InputType;
  using IntermediateType = typename KernelTestParams::IntermediateType;
  using OutputType = typename KernelTestParams::OutputType;

  kleidicv_error_t call_api(const test::Array2D<InputType> *input,
                            test::Array2D<OutputType> *output,
                            kleidicv_border_type_t border_type,
                            kleidicv_border_values_t border_values) override {
    // Check the expected border type.
    EXPECT_EQ(border_type, kBorders[border_count_ % kBorders.size()]);
    // Check the expected border value.
    auto act = border_values;
    auto exp = kBorderValues[border_value_count_ % kBorderValues.size()];
    EXPECT_EQ(act.top, exp.top);
    EXPECT_EQ(act.right, exp.right);
    EXPECT_EQ(act.bottom, exp.bottom);
    EXPECT_EQ(act.left, exp.left);

    // Check the expected layout.
    const test::ArrayLayout &expected_array_layout =
        kArrayLayouts[array_layouts_ % kArrayLayouts.size()];

    EXPECT_EQ(expected_array_layout.width, input->width());
    EXPECT_EQ(expected_array_layout.height, input->height());
    EXPECT_EQ(expected_array_layout.padding,
              input->stride() - input->width() * sizeof(InputType));
    EXPECT_EQ(expected_array_layout.channels, input->channels());

    EXPECT_EQ(input->width(), output->width());
    EXPECT_EQ(input->height(), output->height());
    // NOTE: This is not expected.
    // EXPECT_EQ(input->stride(), output->stride());
    EXPECT_EQ(input->channels(), output->channels());

    ++api_calls_;
    ++border_value_count_;
    if (border_value_count_ == kBorderValues.size()) {
      border_value_count_ = 0;
      ++border_count_;
      if (border_count_ == kBorders.size()) {
        border_count_ = 0;
        ++array_layouts_;
      }
    }

    // Fake some result.
    output->fill(OutputType{0});

    return KLEIDICV_OK;
  }

 public:
  static const std::array<test::Kernel<IntermediateType>, 3> kKernels;

  static constexpr std::array<test::ArrayLayout, 3> kArrayLayouts = {{
      {3, 2, 10, 1},
      {6, 5, 11, 2},
      {9, 7, 11, 3},
  }};

  static constexpr std::array<kleidicv_border_type_t, 2> kBorders = {
      KLEIDICV_BORDER_TYPE_REPLICATE, KLEIDICV_BORDER_TYPE_CONSTANT};

  static constexpr std::array<kleidicv_border_values_t, 2> kBorderValues = {
      {{0, 0, 0, 0}, {1, 2, 3, 4}}};

  size_t api_calls_{0};
  size_t array_layouts_{0};
  size_t border_count_{0};
  size_t border_value_count_{0};
};  // end of class class ExampleKernelTest<KernelTestParams>

// Disable check for possible exceptions thrown outside main.
// The check isn't important in a test program.
// NOLINTBEGIN(cert-err58-cpp)
template <class KernelTestParams>
test::Array2D<typename KernelTestParams::IntermediateType> fill_kernel(
    size_t width, size_t height,
    typename KernelTestParams::IntermediateType value) {
  test::Array2D<typename KernelTestParams::IntermediateType> array{width,
                                                                   height};
  array.fill(value);
  return array;
}

template <class KernelTestParams>
const std::array<test::Kernel<typename KernelTestParams::IntermediateType>, 3>
    ExampleKernelTest<KernelTestParams>::kKernels = {
        test::Kernel{fill_kernel<KernelTestParams>(3, 3, 0)},
        test::Kernel{fill_kernel<KernelTestParams>(4, 4, 0)},
        test::Kernel{fill_kernel<KernelTestParams>(8, 4, 0)},
};
// NOLINTEND(cert-err58-cpp)

// Tests that KernelTest::test() works.
TEST(KernelTest, Test) {
  test::SequenceGenerator tested_kernels{
      ExampleKernelTest<KernelTestParams>::kKernels};
  test::SequenceGenerator tested_array_layouts{
      ExampleKernelTest<KernelTestParams>::kArrayLayouts};
  test::SequenceGenerator tested_borders{
      ExampleKernelTest<KernelTestParams>::kBorders};
  test::SequenceGenerator tested_border_values{
      ExampleKernelTest<KernelTestParams>::kBorderValues};
  test::PseudoRandomNumberGenerator<typename KernelTestParams::InputType>
      element_generator;

  ExampleKernelTest<KernelTestParams> kernel_test;
  kernel_test.test(tested_kernels, tested_array_layouts, tested_borders,
                   tested_border_values, element_generator);

  EXPECT_EQ(kernel_test.api_calls_,
            ExampleKernelTest<KernelTestParams>::kKernels.size() *
                ExampleKernelTest<KernelTestParams>::kArrayLayouts.size() *
                ExampleKernelTest<KernelTestParams>::kBorders.size() *
                ExampleKernelTest<KernelTestParams>::kBorderValues.size());
  EXPECT_EQ(kernel_test.array_layouts_,
            ExampleKernelTest<KernelTestParams>::kKernels.size() *
                ExampleKernelTest<KernelTestParams>::kArrayLayouts.size());
}
