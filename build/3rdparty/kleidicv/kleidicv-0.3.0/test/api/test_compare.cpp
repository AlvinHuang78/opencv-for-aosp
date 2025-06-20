// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/operation.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

template <typename ElementType>
bool is_equal(ElementType a, ElementType b) {
  return a == b;
}

template <typename ElementType>
bool is_greater(ElementType a, ElementType b) {
  return a > b;
}

#define KLEIDICV_PARAMS(name, impl, type, op)                                 \
  template <typename ElementType,                                             \
            std::enable_if_t<std::is_same_v<ElementType, type>, bool> = true> \
  class name {                                                                \
   public:                                                                    \
    static decltype(auto) api() { return impl; }                              \
    static decltype(auto) operation() {                                       \
      return [](type a, type b) { return op<type>(a, b); };                   \
    }                                                                         \
  };

KLEIDICV_PARAMS(CompareEqualParams, kleidicv_compare_equal_u8, uint8_t,
                is_equal);
KLEIDICV_PARAMS(CompareGreaterParams, kleidicv_compare_greater_u8, uint8_t,
                is_greater);

template <class ElementType, template <typename, auto...> class OperationParams>
class CompareTestLinear final {
 public:
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
  static constexpr ElementType lowest() {
    return std::numeric_limits<ElementType>::lowest();
  }
  static constexpr ElementType max() {
    return std::numeric_limits<ElementType>::max();
  }

 private:
  class GenerateLinearSeries : public test::Generator<ElementType> {
   public:
    explicit GenerateLinearSeries(ElementType start_from)
        : counter_{start_from} {}

    std::optional<ElementType> next() override { return counter_++; }

   private:
    ElementType counter_;
  };  // end of class GenerateLinearSeries

  // Number of padding bytes at the end of rows.
  size_t padding_{0};

  void test_linear(size_t width, size_t minimum_size) {
    size_t image_size =
        std::max(minimum_size, static_cast<size_t>(max() - lowest()));
    size_t height = image_size / width + 1;
    test::Array2D<ElementType> source_a(width, height, padding_, 1);
    test::Array2D<ElementType> source_b(width, height, padding_, 1);
    test::Array2D<ElementType> expected(width, height, padding_, 1);
    test::Array2D<ElementType> actual =
        test::Array2D<ElementType>(width, height, padding_, 1);

    GenerateLinearSeries generator_a(lowest());
    GenerateLinearSeries generator_b(128);

    source_a.fill(generator_a);
    source_b.fill(generator_b);
    expected.fill(0);

    calculate_expected(source_a, source_b, expected);

    ASSERT_EQ(KLEIDICV_OK, (OperationParams<ElementType>::api()(
                               source_a.data(), source_a.stride(),
                               source_b.data(), source_b.stride(),
                               actual.data(), actual.stride(), width, height)));

    EXPECT_EQ_ARRAY2D(expected, actual);
  }

 protected:
  void calculate_expected(const test::Array2D<ElementType>& source_a,
                          const test::Array2D<ElementType>& source_b,
                          test::Array2D<ElementType>& expected) {
    for (size_t hindex = 0; hindex < source_a.height(); ++hindex) {
      for (size_t vindex = 0; vindex < source_a.width(); ++vindex) {
        ElementType result =
            OperationParams<ElementType>::operation()(
                source_a.at(hindex, vindex)[0], source_b.at(hindex, vindex)[0])
                ? 255
                : 0;
        // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
        *expected.at(hindex, vindex) = result;
        // NOLINTEND(clang-analyzer-core.uninitialized.Assign)
      }
    }
  }
};  // end of class CompareTestLinearBase

template <typename ElementType,
          template <typename, auto...> class OperationParams>
class CompareTestBase : public BinaryOperationTest<ElementType> {
 protected:
  // Needed to initialize value()
  using BinaryOperationTest<ElementType>::max;

  // Calls the API-under-test in the appropriate way.
  kleidicv_error_t call_api() override {
    return OperationParams<ElementType>::api()(
        this->inputs_[0].data(), this->inputs_[0].stride(),
        this->inputs_[1].data(), this->inputs_[1].stride(),
        this->actual_[0].data(), this->actual_[0].stride(), this->width(),
        this->height());
  }

  // Filling with max() value only necessary for CompareEqual operation
  void setup() override {
    if (OperationParams<ElementType>::operation()(0, 0)) {
      ElementType expected = max();
      this->expected_[0].fill(expected);
      BinaryOperationTest<ElementType>::setup();
    }
  }
  static constexpr ElementType value() { return 255; }
};  // end of class CompareTestBase<ElementType>

template <typename ElementType>
class CompareEqTest final
    : public CompareTestBase<ElementType, CompareEqualParams> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      { 12, 13, 0},
      { 13, 13, this->value()},
      { 14, 13, 0},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class CompareEqTestMin final
    : public CompareTestBase<ElementType, CompareEqualParams> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;
  using BinaryOperationTest<ElementType>::lowest;

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      {    lowest(), lowest(), this->value()},
      {lowest() + 1, lowest(), 0},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class CompareEqTestZero final
    : public CompareTestBase<ElementType, CompareEqualParams> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      { 0, 0, this->value()},
      { 1, 0, 0},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class CompareEqTestMax final
    : public CompareTestBase<ElementType, CompareEqualParams> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;
  using BinaryOperationTest<ElementType>::max;

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      { max(),     max(), this->value()},
      { max(), max() - 1, 0},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class CompareGtTest final
    : public CompareTestBase<ElementType, CompareGreaterParams> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      { 12, 13, 0},
      { 13, 13, 0},
      { 14, 13, this->value()},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class CompareGtTestMin final
    : public CompareTestBase<ElementType, CompareGreaterParams> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;
  using BinaryOperationTest<ElementType>::lowest;

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      {    lowest(), lowest(),             0},
      {lowest() + 1, lowest(), this->value()},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class CompareGtTestZero final
    : public CompareTestBase<ElementType, CompareGreaterParams> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      { 0, 0, 0},
      { 1, 0, this->value()},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class CompareGtTestMax final
    : public CompareTestBase<ElementType, CompareGreaterParams> {
  using Elements = typename BinaryOperationTest<ElementType>::Elements;
  using BinaryOperationTest<ElementType>::max;

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
      { max(),     max(), 0},
      { max(), max() - 1, this->value()},
        // clang-format on
    };
    return kTestElements;
  }
};

template <typename ElementType>
class Compare : public testing::Test {};

using ElementTypes = ::testing::Types<uint8_t>;

TYPED_TEST_SUITE(Compare, ElementTypes);

TYPED_TEST(Compare, TestScalarEq) {
  CompareTestLinear<TypeParam, CompareEqualParams>{}.test_scalar();
}

TYPED_TEST(Compare, TestScalarGt) {
  CompareTestLinear<TypeParam, CompareGreaterParams>{}.test_scalar();
}

TYPED_TEST(Compare, TestVectorEq) {
  CompareTestLinear<TypeParam, CompareEqualParams>{}.test_vector();
}

TYPED_TEST(Compare, TestVectorGt) {
  CompareTestLinear<TypeParam, CompareGreaterParams>{}.test_vector();
}

TYPED_TEST(Compare, Test) {
  CompareEqTest<TypeParam>{}.test();
  CompareGtTest<TypeParam>{}.test();
}

// Tests various padding combinations.
TYPED_TEST(Compare, Padding) {
  CompareEqTest<TypeParam>{}.with_paddings({0}, {0}).test();
  CompareEqTest<TypeParam>{}.with_paddings({0}, {1}).test();
  CompareEqTest<TypeParam>{}.with_paddings({1}, {0}).test();
  CompareEqTest<TypeParam>{}.with_paddings({1}, {1}).test();

  CompareGtTest<TypeParam>{}.with_paddings({0}, {0}).test();
  CompareGtTest<TypeParam>{}.with_paddings({0}, {1}).test();
  CompareGtTest<TypeParam>{}.with_paddings({1}, {0}).test();
  CompareGtTest<TypeParam>{}.with_paddings({1}, {1}).test();
}

TYPED_TEST(Compare, TestMin) {
  CompareEqTestMin<TypeParam>{}.test();
  CompareGtTestMin<TypeParam>{}.test();
}

TYPED_TEST(Compare, TestZero) {
  CompareEqTestZero<TypeParam>{}.test();
  CompareGtTestZero<TypeParam>{}.test();
}

TYPED_TEST(Compare, TestMax) {
  CompareEqTestMax<TypeParam>{}.test();
  CompareGtTestMax<TypeParam>{}.test();
}

TYPED_TEST(Compare, EqualNullPointer) {
  const TypeParam src_a[1] = {};
  const TypeParam src_b[1] = {};
  TypeParam dst[1];
  test::test_null_args(CompareEqualParams<TypeParam>::api(), src_a,
                       sizeof(TypeParam), src_b, sizeof(TypeParam), dst,
                       sizeof(TypeParam), 1, 1);
}

TYPED_TEST(Compare, GreaterNullPointer) {
  const TypeParam src_a[1] = {};
  const TypeParam src_b[1] = {};
  TypeParam dst[1];
  test::test_null_args(CompareGreaterParams<TypeParam>::api(), src_a,
                       sizeof(TypeParam), src_b, sizeof(TypeParam), dst,
                       sizeof(TypeParam), 1, 1);
}

TYPED_TEST(Compare, EqualMisalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src_a[2] = {}, src_b[2] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            (CompareEqualParams<TypeParam>::api()(src_a, sizeof(TypeParam) + 1,
                                                  src_b, sizeof(TypeParam), dst,
                                                  sizeof(TypeParam), 1, 2)));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            (CompareEqualParams<TypeParam>::api()(
                src_a, sizeof(TypeParam), src_b, sizeof(TypeParam), dst,
                sizeof(TypeParam) + 1, 1, 2)));
}

TYPED_TEST(Compare, GreaterMisalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src_a[2] = {}, src_b[2] = {}, dst[2];
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            (CompareGreaterParams<TypeParam>::api()(
                src_a, sizeof(TypeParam) + 1, src_b, sizeof(TypeParam), dst,
                sizeof(TypeParam), 1, 2)));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            (CompareGreaterParams<TypeParam>::api()(
                src_a, sizeof(TypeParam), src_b, sizeof(TypeParam), dst,
                sizeof(TypeParam) + 1, 1, 2)));
}

TYPED_TEST(Compare, EqualZeroImageSize) {
  TypeParam src_a[1] = {}, src_b[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_OK, (CompareEqualParams<TypeParam>::api()(
                             src_a, sizeof(TypeParam), src_b, sizeof(TypeParam),
                             dst, sizeof(TypeParam), 0, 1)));
  EXPECT_EQ(KLEIDICV_OK, (CompareEqualParams<TypeParam>::api()(
                             src_a, sizeof(TypeParam), src_b, sizeof(TypeParam),
                             dst, sizeof(TypeParam), 1, 0)));
}

TYPED_TEST(Compare, GreaterZeroImageSize) {
  TypeParam src_a[1] = {}, src_b[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_OK, (CompareGreaterParams<TypeParam>::api()(
                             src_a, sizeof(TypeParam), src_b, sizeof(TypeParam),
                             dst, sizeof(TypeParam), 0, 1)));
  EXPECT_EQ(KLEIDICV_OK, (CompareGreaterParams<TypeParam>::api()(
                             src_a, sizeof(TypeParam), src_b, sizeof(TypeParam),
                             dst, sizeof(TypeParam), 1, 0)));
}

TYPED_TEST(Compare, EqualOversizeImage) {
  TypeParam src_a[1] = {}, src_b[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            (CompareEqualParams<TypeParam>::api()(
                src_a, sizeof(TypeParam), src_b, sizeof(TypeParam), dst,
                sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS + 1, 1)));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            (CompareEqualParams<TypeParam>::api()(
                src_a, sizeof(TypeParam), src_b, sizeof(TypeParam), dst,
                sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS)));
}

TYPED_TEST(Compare, GreaterOversizeImage) {
  TypeParam src_a[1] = {}, src_b[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            (CompareGreaterParams<TypeParam>::api()(
                src_a, sizeof(TypeParam), src_b, sizeof(TypeParam), dst,
                sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS + 1, 1)));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            (CompareGreaterParams<TypeParam>::api()(
                src_a, sizeof(TypeParam), src_b, sizeof(TypeParam), dst,
                sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS)));
}
