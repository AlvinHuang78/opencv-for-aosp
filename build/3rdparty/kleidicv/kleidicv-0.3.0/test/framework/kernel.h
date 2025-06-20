// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TEST_FRAMEWORK_KERNEL_H_
#define KLEIDICV_TEST_FRAMEWORK_KERNEL_H_

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>

#include "framework/abstract.h"
#include "framework/array.h"
#include "framework/border.h"
#include "framework/generator.h"
#include "framework/types.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"

namespace test {

// Represents a kernel operator.
template <typename ElementType>
class Kernel : protected Array2D<ElementType>, public Bordered {
 public:
  // Make some features of Array2D<ElementType> base class public.
  using Array2D<ElementType>::at;
  using Array2D<ElementType>::channels;
  using Array2D<ElementType>::height;
  using Array2D<ElementType>::width;

  explicit Kernel(const Array2D<ElementType>& mask)
      : Kernel(mask, {mask.width() / 2, mask.height() / 2}) {}

  explicit Kernel(const Array2D<ElementType>& mask, Point anchor)
      : Array2D<ElementType>(mask), anchor_{anchor} {}

  // Returns the anchor point of the kernel.
  Point anchor() const { return anchor_; }

  // Returns the number of elements to the left of the anchor point.
  size_t left() const override { return anchor().x; }

  // Returns the number of elements above the anchor point.
  size_t top() const override { return anchor().y; }

  // Returns the number of elements to the right of the anchor point.
  size_t right() const override {
    return (width() > 0) ? width() - anchor().x - 1 : 0;
  }

  // Returns the number of elements above the anchor point.
  size_t bottom() const override {
    return (height() > 0) ? height() - anchor().y - 1 : 0;
  }

 private:
  // The anchor point of the kernel.
  Point anchor_;
};  // end of class Kernel<ElementType>

// Abstract class to help implement a kernel operation.
//
// Required:
//   - KernelTestParams::InputType: Input type to the operation.
//   - KernelTestParams::IntermediateType: Type which is used during element
//   calculations.
//   - KernelTestParams::OutputType: Output type of the operation.
template <class KernelTestParams>
class KernelTest {
 public:
  using InputType = typename KernelTestParams::InputType;
  using IntermediateType = typename KernelTestParams::IntermediateType;
  using OutputType = typename KernelTestParams::OutputType;

  KernelTest() : debug_{false} {}

  // Enables debug mode.
  KernelTest<KernelTestParams>& with_debug() {
    debug_ = true;
    return *this;
  }

  void test(Generator<Kernel<IntermediateType>>& kernel_generator,
            Generator<ArrayLayout>& array_layout_generator,
            Generator<kleidicv_border_type_t>& border_type_generator,
            Generator<kleidicv_border_values_t>& border_values_generator,
            Generator<InputType>& element_generator) {
    kernel_generator.reset();

    std::optional<Kernel<IntermediateType>> maybe_kernel;
    while ((maybe_kernel = kernel_generator.next()) != std::nullopt) {
      test(*maybe_kernel, array_layout_generator, border_type_generator,
           border_values_generator, element_generator);
      ASSERT_NO_FAILURES();
    }
  }

  void test(const Kernel<IntermediateType>& kernel,
            Generator<ArrayLayout>& array_layout_generator,
            Generator<kleidicv_border_type_t>& border_type_generator,
            Generator<kleidicv_border_values_t>& border_values_generator,
            Generator<InputType>& element_generator) {
    array_layout_generator.reset();

    std::optional<ArrayLayout> maybe_array_layout;
    while ((maybe_array_layout = array_layout_generator.next()) !=
           std::nullopt) {
      ArrayLayout array_layout = *maybe_array_layout;
      create_arrays(kernel, array_layout);
      ASSERT_NO_FAILURES();
      test(kernel, array_layout, border_type_generator, border_values_generator,
           element_generator);
      ASSERT_NO_FAILURES();
    }
  }

  void test(const Kernel<IntermediateType>& kernel, ArrayLayout array_layout,
            Generator<kleidicv_border_type_t>& border_type_generator,
            Generator<kleidicv_border_values_t>& border_values_generator,
            Generator<InputType>& element_generator) {
    border_type_generator.reset();

    std::optional<kleidicv_border_type_t> maybe_border_type;
    while ((maybe_border_type = border_type_generator.next()) != std::nullopt) {
      test(kernel, array_layout, *maybe_border_type, border_values_generator,
           element_generator);
      ASSERT_NO_FAILURES();
    }
  }

  void test(const Kernel<IntermediateType>& kernel, ArrayLayout array_layout,
            kleidicv_border_type_t border_type,
            Generator<kleidicv_border_values_t>& border_values_generator,
            Generator<InputType>& element_generator) {
    border_values_generator.reset();
    std::optional<kleidicv_border_values_t> maybe_border_values;
    while ((maybe_border_values = border_values_generator.next()) !=
           std::nullopt) {
      test(kernel, array_layout, border_type, *maybe_border_values,
           element_generator);
      ASSERT_NO_FAILURES();
    }
  }

  void test(const Kernel<IntermediateType>& kernel, ArrayLayout array_layout,
            kleidicv_border_type_t border_type,
            kleidicv_border_values_t border_values,
            Generator<InputType>& element_generator) {
    prepare_source(element_generator);
    prepare_expected(kernel, array_layout, border_type, border_values);
    prepare_actual();
    check_results(
        this->call_api(&input_, &actual_, border_type, border_values));
  }

 protected:
  // Calls the API-under-test in the appropriate way.
  //
  // The arguments are never nullptr.
  virtual kleidicv_error_t call_api(const Array2D<InputType>* input,
                                    Array2D<OutputType>* output,
                                    kleidicv_border_type_t border_type,
                                    kleidicv_border_values_t border_values) = 0;

  // Calculates the expected output.
  virtual void calculate_expected(const Kernel<IntermediateType>& kernel,
                                  const TwoDimensional<InputType>& source) {
    for (size_t row = 0; row < expected_.height(); ++row) {
      for (size_t column = 0; column < expected_.width(); ++column) {
        IntermediateType result;
        result = calculate_expected_at(kernel, source, row, column);
        expected_.at(row, column)[0] =
            static_cast<OutputType>(scale_result(kernel, result));
      }
    }
  }

  // Calculates the expected element at a given position.
  virtual IntermediateType calculate_expected_at(
      const Kernel<IntermediateType>& kernel,
      const TwoDimensional<InputType>& source, size_t row, size_t column) {
    IntermediateType result{0};
    for (size_t height = 0; height < kernel.height(); ++height) {
      for (size_t width = 0; width < kernel.width(); ++width) {
        IntermediateType coefficient = kernel.at(height, width)[0];
        InputType value =
            source.at(row + height, column + width * source.channels())[0];
        result = saturating_add(
            result,
            saturating_mul(coefficient, static_cast<IntermediateType>(value)));
      }
    }

    return result;
  }

  virtual IntermediateType scale_result(const Kernel<IntermediateType>&,
                                        IntermediateType result) {
    return result;
  }

  // Creates arrays for a given layout.
  virtual void create_arrays(const Kernel<IntermediateType>& kernel,
                             const ArrayLayout& array_layout) {
    input_ = Array2D<InputType>{array_layout};
    ASSERT_TRUE(input_.valid());

    expected_ = Array2D<OutputType>{array_layout};
    ASSERT_TRUE(expected_.valid());

    actual_ = Array2D<OutputType>{array_layout};
    ASSERT_TRUE(actual_.valid());

    input_with_borders_ = Array2D<InputType>{
        array_layout.width +
            (kernel.left() + kernel.right()) * array_layout.channels,
        array_layout.height + kernel.top() + kernel.bottom(), 0,
        array_layout.channels};
    ASSERT_TRUE(input_with_borders_.valid());
  }

  // Prepares input to the kernel-based operation.
  void prepare_source(Generator<InputType>& element_generator) {
    element_generator.reset();
    input_.fill(element_generator);

    if (debug_) {
      std::cout << "[source]" << std::endl;
      dump(&input_);
    }
  }

  // Computes expected output of the kernel-based operation.
  virtual void prepare_expected(const Kernel<IntermediateType>& kernel,
                                const ArrayLayout& array_layout,
                                kleidicv_border_type_t border_type,
                                kleidicv_border_values_t border_values) {
    input_with_borders_.set(kernel.anchor().y,
                            kernel.anchor().x * array_layout.channels, &input_);

    if (debug_) {
      std::cout << "[input_with_borders without borders]" << std::endl;
      dump(&input_with_borders_);
    }

    prepare_borders<InputType>(border_type, border_values, &kernel,
                               &input_with_borders_);

    if (debug_) {
      std::cout << "[input_with_borders with borders]" << std::endl;
      dump(&input_with_borders_);
    }

    calculate_expected(kernel, input_with_borders_);

    if (debug_) {
      std::cout << "[expected]" << std::endl;
      dump(&expected_);
    }
  }

  // Prepares the actual output of the kernel-based operation.
  void prepare_actual() { actual_.fill(42); }

  // Checks that the actual output matches the expectations.
  void check_results(kleidicv_error_t err) {
    if (debug_) {
      std::cout << "[actual]" << std::endl;
      dump(&actual_);
    }

    EXPECT_EQ(KLEIDICV_OK, err);

    // Check that the actual result matches the expectation.
    EXPECT_EQ_ARRAY2D(expected_, actual_);
  }

  // Input operand for the operation.
  Array2D<InputType> input_;
  // Input operand with borders, used to calculate expected values.
  Array2D<InputType> input_with_borders_;
  // Expected result of the operation.
  Array2D<OutputType> expected_;
  // Actual result of the operation.
  Array2D<OutputType> actual_;
  // Enables debug mode.
  bool debug_;
};  // end of class KernelTest<KernelTestParams>

}  // namespace test

#endif  // KLEIDICV_TEST_FRAMEWORK_KERNEL_H_
