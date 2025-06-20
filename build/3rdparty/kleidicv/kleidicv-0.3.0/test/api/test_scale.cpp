// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/operation.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

template <typename DestinationType, typename SourceType>
static DestinationType saturating_cast(SourceType value) {
  if (value >
      static_cast<SourceType>(std::numeric_limits<DestinationType>::max())) {
    return std::numeric_limits<DestinationType>::max();
  }
  if (value < std::numeric_limits<DestinationType>::lowest()) {
    return std::numeric_limits<DestinationType>::lowest();
  }
  return static_cast<DestinationType>(value);
}

uint8_t scalar_scale_u8(uint8_t x, float scale, float shift) {
  float result = static_cast<float>(x) * scale + shift;
  if (result < std::numeric_limits<uint8_t>::lowest()) {
    return std::numeric_limits<uint8_t>::lowest();
  }
  if (result > std::numeric_limits<uint8_t>::max()) {
    return std::numeric_limits<uint8_t>::max();
  }
  return static_cast<uint8_t>(lrintf(result));
}

float scalar_scale_f32(float x, float scale, float shift) {
  return x * scale + shift;
}

#define KLEIDICV_SCALE_API(type, suffix) \
  KLEIDICV_API(scale_api, kleidicv_scale_##suffix, type)

#define KLEIDICV_SCALE_OPERATION(type, suffix) \
  KLEIDICV_API(scale_operation, &scalar_scale_##suffix, type)

KLEIDICV_SCALE_API(uint8_t, u8);
KLEIDICV_SCALE_OPERATION(uint8_t, u8);
KLEIDICV_SCALE_API(float, f32);
KLEIDICV_SCALE_OPERATION(float, f32);

template <typename ElementType>
class ScaleTestBase : public UnaryOperationTest<ElementType> {
 protected:
  using UnaryOperationTest<ElementType>::lowest;
  using UnaryOperationTest<ElementType>::max;
  using typename UnaryOperationTest<ElementType>::Elements;

  // Calls the API-under-test in the appropriate way.
  kleidicv_error_t call_api() override {
    return scale_api<ElementType>()(
        this->inputs_[0].data(), this->inputs_[0].stride(),
        this->actual_[0].data(), this->actual_[0].stride(), this->width(),
        this->height(), this->scale(), this->shift());
  }
  virtual float scale() = 0;
  virtual float shift() = 0;

  // Prepares expected outputs for the operation.
  void setup() override {
    this->expected_[0].fill(
        scale_operation<ElementType>()(0, scale(), shift()));
    UnaryOperationTest<ElementType>::setup();
  }

  void fill_expected(std::vector<Elements>& elements) {
    for (size_t i = 0; i < elements.size(); ++i) {
      elements[i].values[1] = scale_operation<ElementType>()(
          elements[i].values[0], scale(), shift());
    }
  }
};  // end of class ScaleTestBase<ElementType>

template <typename ElementType>
class ScaleTestLinearBase {
 public:
  // Sets the number of padding bytes at the end of rows.
  ScaleTestLinearBase& with_padding(size_t padding) {
    padding_ = padding;
    return *this;
  }

  // minimum_size set by caller to trigger the 'big' scale path.
  void test_scalar(size_t minimum_size = 1) {
    size_t width = test::Options::vector_length() - 1;
    test_linear(width, minimum_size);
  }

  void test_vector(size_t minimum_size = 1) {
    size_t width = test::Options::vector_length() * 2;
    test_linear(width, minimum_size);
  }

 protected:
  static constexpr ElementType
  min_if_integral_else_smallest_positive_normalized() {
    return std::numeric_limits<ElementType>::min();
  }
  static constexpr ElementType max() {
    return std::numeric_limits<ElementType>::max();
  }
  static constexpr ElementType lowest() {
    return std::numeric_limits<ElementType>::lowest();
  }
  virtual float scale() = 0;
  virtual float shift() = 0;

 private:
  class GenerateLinearSeries : public test::Generator<ElementType> {
   public:
    explicit GenerateLinearSeries(ElementType start_from, ElementType step)
        : counter_{start_from}, step_{step} {}

    std::optional<ElementType> next() override { return counter_ + step_; }

   private:
    ElementType counter_, step_;
  };  // end of class GenerateLinearSeries

  // Number of padding bytes at the end of rows.
  size_t padding_{0};

  void test_linear(size_t width, size_t minimum_size) {
    size_t image_size = std::max(
        minimum_size,
        std::min(saturating_cast<size_t, ElementType>(max() - lowest()),
                 10000UL));
    size_t step =
        std::max(static_cast<ElementType>(image_size / (max() - lowest())),
                 static_cast<ElementType>(1));
    size_t height = image_size / width + 1;
    test::Array2D<ElementType> source(width, height, padding_, 1);
    test::Array2D<ElementType> expected(width, height, padding_, 1);
    test::Array2D<ElementType> actual =
        test::Array2D<ElementType>(width, height, padding_, 1);

    GenerateLinearSeries generator(
        min_if_integral_else_smallest_positive_normalized(), step);

    source.fill(generator);

    calculate_expected(source, expected);

    ASSERT_EQ(KLEIDICV_OK,
              scale_api<ElementType>()(source.data(), source.stride(),
                                       actual.data(), actual.stride(), width,
                                       height, scale(), shift()));

    EXPECT_EQ_ARRAY2D(expected, actual);
  }

 protected:
  void calculate_expected(const test::Array2D<ElementType>& source,
                          test::Array2D<ElementType>& expected) {
    for (size_t hindex = 0; hindex < source.height(); ++hindex) {
      for (size_t vindex = 0; vindex < source.width(); ++vindex) {
        // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
        *expected.at(hindex, vindex) = scale_operation<ElementType>()(
            *source.at(hindex, vindex), scale(), shift());
        // NOLINTEND(clang-analyzer-core.uninitialized.Assign)
      }
    }
  }
};  // end of class ScaleTestLinearBase

template <typename ElementType>
class ScaleTestLinear1 final : public ScaleTestLinearBase<ElementType> {
  float scale() override { return 10; };
  float shift() override { return 13; };
};

template <typename ElementType>
class ScaleTestLinear2 final : public ScaleTestLinearBase<ElementType> {
  float scale() override { return 14.69; };
  float shift() override { return 10.13; };
};

template <typename ElementType>
class ScaleTestLinear3 final : public ScaleTestLinearBase<ElementType> {
  float scale() override { return 0.18; };
  float shift() override { return 1.41; };
};

template <typename ElementType>
class ScaleTestAdd final : public ScaleTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;

  float scale() override { return 1; }
  float shift() override { return 6; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      { 8, 14},
      {12, 18},
        // clang-format on
    };
    ScaleTestBase<ElementType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename ElementType>
class ScaleTestSubtract final : public ScaleTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;

  float scale() override { return 8; }
  float shift() override { return -3; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      { 6,  45},
      { 20, 157},
        // clang-format on
    };
    ScaleTestBase<ElementType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename ElementType>
class ScaleTestDivide final : public ScaleTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;

  float scale() override { return 0.25; }
  float shift() override { return 3; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      { 252, 0},
      { 255, 0},
        // clang-format on
    };
    ScaleTestBase<ElementType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename ElementType>
class ScaleTestMultiply final : public ScaleTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;

  float scale() override { return 3.14; }
  float shift() override { return 2.72; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      { 60, 0},
      { 45, 0},
        // clang-format on
    };
    ScaleTestBase<ElementType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename ElementType>
class ScaleTestZero final : public ScaleTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;
  using UnaryOperationTest<
      ElementType>::min_if_integral_else_smallest_positive_normalized;
  using UnaryOperationTest<ElementType>::max;
  using UnaryOperationTest<ElementType>::lowest;

  float scale() override { return 0; }
  float shift() override { return 0; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      { lowest(), 0},
      { min_if_integral_else_smallest_positive_normalized(), 0},
      {     0, 0},
      { max(), 0},
        // clang-format on
    };
    ScaleTestBase<ElementType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename ElementType>
class ScaleTestUnderflowByShift final : public ScaleTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;
  using UnaryOperationTest<
      ElementType>::min_if_integral_else_smallest_positive_normalized;
  using UnaryOperationTest<ElementType>::lowest;

  float scale() override { return 1; }
  float shift() override { return -1; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      {lowest() + 1, 0},
      {    lowest(), 0},
      {min_if_integral_else_smallest_positive_normalized() + 1, 0},
      {    min_if_integral_else_smallest_positive_normalized(), 0},
      {           0, 0},
      {           1, 0},
      {           2, 0},
        // clang-format on
    };
    ScaleTestBase<ElementType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename ElementType>
class ScaleTestOverflowByShift final : public ScaleTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;
  using UnaryOperationTest<ElementType>::max;

  float scale() override { return 1; }
  float shift() override { return 1; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      {max() - 1, 0},
      {    max(), 0},
        // clang-format on
    };
    ScaleTestBase<ElementType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename ElementType>
class ScaleTestUnderflowByScale final : public ScaleTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;
  using UnaryOperationTest<ElementType>::max;

  float scale() override { return -2; }
  float shift() override { return 0; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      { max(), 0},
        // clang-format on
    };
    ScaleTestBase<ElementType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename ElementType>
class ScaleTestOverflowByScale final : public ScaleTestBase<ElementType> {
  using Elements = typename UnaryOperationTest<ElementType>::Elements;
  using UnaryOperationTest<ElementType>::max;

  float scale() override { return 2; }
  float shift() override { return 0; }

  const std::vector<Elements>& test_elements() override {
    static std::vector<Elements> kTestElements = {
        // clang-format off
      { max(), 0},
        // clang-format on
    };
    ScaleTestBase<ElementType>::fill_expected(kTestElements);
    return kTestElements;
  }
};

template <typename ElementType>
class ScaleTest : public testing::Test {};

using ElementTypes = ::testing::Types<uint8_t, float>;

TYPED_TEST_SUITE(ScaleTest, ElementTypes);

TYPED_TEST(ScaleTest, TestScalar1) {
  ScaleTestLinear1<TypeParam>{}.test_scalar();
  ScaleTestLinear1<TypeParam>{}.with_padding(1).test_scalar();
}
TYPED_TEST(ScaleTest, TestVector1) {
  ScaleTestLinear1<TypeParam>{}.test_vector();
  ScaleTestLinear1<TypeParam>{}.with_padding(1).test_vector();
}

TYPED_TEST(ScaleTest, TestScalar1Tbx) {
  ScaleTestLinear1<TypeParam>{}.test_scalar(2500);
  ScaleTestLinear1<TypeParam>{}.with_padding(1).test_scalar(2500);
}
TYPED_TEST(ScaleTest, TestVector1Tbx) {
  ScaleTestLinear1<TypeParam>{}.test_vector(2500);
  ScaleTestLinear1<TypeParam>{}.with_padding(1).test_vector(2500);
}

TYPED_TEST(ScaleTest, TestScalar2) {
  ScaleTestLinear2<TypeParam>{}.test_scalar();
}
TYPED_TEST(ScaleTest, TestVector2) {
  ScaleTestLinear2<TypeParam>{}.test_vector();
}

TYPED_TEST(ScaleTest, TestScalar2Tbx) {
  ScaleTestLinear2<TypeParam>{}.test_scalar(2500);
}
TYPED_TEST(ScaleTest, TestVector2Tbx) {
  ScaleTestLinear2<TypeParam>{}.test_vector(2500);
}

TYPED_TEST(ScaleTest, TestScalar3) {
  ScaleTestLinear3<TypeParam>{}.test_scalar();
}
TYPED_TEST(ScaleTest, TestVector3) {
  ScaleTestLinear3<TypeParam>{}.test_vector();
}

TYPED_TEST(ScaleTest, TestScalar3Tbx) {
  ScaleTestLinear3<TypeParam>{}.test_scalar(2500);
}
TYPED_TEST(ScaleTest, TestVector3Tbx) {
  ScaleTestLinear3<TypeParam>{}.test_vector(2500);
}

TYPED_TEST(ScaleTest, TestAdd) {
  ScaleTestAdd<TypeParam>{}.test();
  ScaleTestAdd<TypeParam>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<TypeParam>() - 1)
      .test();
}

TYPED_TEST(ScaleTest, TestSubtract) {
  ScaleTestSubtract<TypeParam>{}.test();
  ScaleTestSubtract<TypeParam>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<TypeParam>() - 1)
      .test();
}

TYPED_TEST(ScaleTest, TestDivide) {
  ScaleTestDivide<TypeParam>{}.test();
  ScaleTestDivide<TypeParam>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<TypeParam>() - 1)
      .test();
}

TYPED_TEST(ScaleTest, TestMultiply) {
  ScaleTestMultiply<TypeParam>{}.test();
  ScaleTestMultiply<TypeParam>{}
      .with_padding(1)
      .with_width(test::Options::vector_lanes<TypeParam>() - 1)
      .test();
}

TYPED_TEST(ScaleTest, TestZero) { ScaleTestZero<TypeParam>{}.test(); }

TYPED_TEST(ScaleTest, TestUnderflowByShift) {
  ScaleTestUnderflowByShift<TypeParam>{}.test();
}

TYPED_TEST(ScaleTest, TestOverflowByShift) {
  ScaleTestOverflowByShift<TypeParam>{}.test();
}

TYPED_TEST(ScaleTest, TestUnderflowByScale) {
  ScaleTestUnderflowByScale<TypeParam>{}.test();
}

TYPED_TEST(ScaleTest, TestOverflowByScale) {
  ScaleTestOverflowByScale<TypeParam>{}.test();
}

TYPED_TEST(ScaleTest, NullPointer) {
  TypeParam src[1] = {}, dst[1];
  test::test_null_args(scale_api<TypeParam>(), src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), 1, 1, 2, 0);
}

TYPED_TEST(ScaleTest, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[2] = {}, dst[2];
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            scale_api<TypeParam>()(src, sizeof(TypeParam) + 1, dst,
                                   sizeof(TypeParam), 1, 2, 2, 0));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            scale_api<TypeParam>()(src, sizeof(TypeParam), dst,
                                   sizeof(TypeParam) + 1, 1, 2, 2, 0));
}

TYPED_TEST(ScaleTest, ZeroImageSize) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_OK, scale_api<TypeParam>()(src, sizeof(TypeParam), dst,
                                                sizeof(TypeParam), 0, 1, 2, 0));
  EXPECT_EQ(KLEIDICV_OK, scale_api<TypeParam>()(src, sizeof(TypeParam), dst,
                                                sizeof(TypeParam), 1, 0, 2, 0));
}

TYPED_TEST(ScaleTest, OversizeImage) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      scale_api<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                             KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 2, 0));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            scale_api<TypeParam>()(src, sizeof(TypeParam), dst,
                                   sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS,
                                   KLEIDICV_MAX_IMAGE_PIXELS, 2, 0));
}
